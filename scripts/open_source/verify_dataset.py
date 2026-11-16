#!/usr/bin/env python3
"""Verify the files required by this M3Eval release without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

DATASET_REPOSITORY = "PKU-VaLuE-Lab/m3eval"
MANIFEST_PATH = "metadata/dataset_manifest.json"
MANIFEST_SHA256 = "4a814d31911fb33caf93976f87df85bd44ed875befafe76a5554635c323498df"
TASKS = ("memory_interference", "split_screen", "interleaved", "nback")


def file_sha256(path: Path) -> str:
    """Hash a file with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_path(root: Path, relative: str) -> Path:
    """Resolve a manifest path, rejecting traversal and escaping symlinks."""
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts or not part.parts:
        raise ValueError(f"Unsafe dataset path: {relative}")
    candidate = root / part
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes destination: {relative}")
    return candidate


def load_manifest(content: bytes, expected_sha256: str = MANIFEST_SHA256) -> dict[str, Any]:
    """Check the release identity and validate each file record."""
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Dataset manifest differs from this code release: expected {expected_sha256}, received {actual}. Use matching code and data revisions.")
    manifest = json.loads(content)
    if manifest.get("schema_version") not in {1, 2} or manifest.get("repository") != DATASET_REPOSITORY:
        raise ValueError("Unsupported dataset manifest")
    seen: set[str] = set()
    for record in manifest["files"]:
        relative = record["path"]
        dataset_path(Path("/dataset"), relative)
        if relative in seen:
            raise ValueError(f"Duplicate manifest path: {relative}")
        seen.add(relative)
        if not isinstance(record["size"], int) or record["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError(f"Invalid manifest hash or size: {relative}")
        if not set(record["tasks"]).issubset(TASKS) or not set(record["scopes"]).issubset({"paper", "all"}):
            raise ValueError(f"Invalid task selection: {relative}")
    if manifest["file_count"] != len(seen) or manifest["total_bytes"] != sum(record["size"] for record in manifest["files"]):
        raise ValueError("Manifest totals disagree with its file records")
    if manifest["schema_version"] == 2:
        validate_archives(manifest, seen)
    return manifest


def validate_archives(manifest: dict[str, Any], file_paths: set[str] | None = None) -> None:
    """Validate archive coverage without opening the archive payloads."""
    file_paths = file_paths or {record["path"] for record in manifest["files"]}
    records = {record["path"]: record for record in manifest["files"]}
    archives = manifest.get("archives")
    if not isinstance(archives, list) or not archives:
        raise ValueError("Schema 2 manifest must contain archives")
    archive_paths: set[str] = set()
    covered: list[str] = []
    for archive in archives:
        if not isinstance(archive, dict):
            raise ValueError("Invalid archive record")
        path = archive.get("path", "")
        if not isinstance(path, str):
            raise ValueError("Invalid archive path")
        try:
            dataset_path(Path("/dataset"), path)
        except ValueError as error:
            raise ValueError(f"Unsafe archive path: {path}") from error
        if path in archive_paths or not path.startswith("archives/") or not path.endswith(".zip"):
            raise ValueError(f"Invalid or duplicate archive path: {path}")
        archive_paths.add(path)
        if not isinstance(archive.get("size"), int) or archive["size"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", archive.get("sha256", "")):
            raise ValueError(f"Invalid archive hash or size: {path}")
        members = archive.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError(f"Archive has no members: {path}")
        member_records = []
        for member in members:
            if not isinstance(member, str) or member not in records or not member.endswith(".mp4"):
                raise ValueError(f"Archive member is not a manifest video: {path}/{member}")
            member_records.append(records[member])
            covered.append(member)
        if len(set(members)) != len(members):
            raise ValueError(f"Duplicate archive member: {path}")
        first = (tuple(member_records[0]["tasks"]), tuple(member_records[0]["scopes"]))
        if any((tuple(record["tasks"]), tuple(record["scopes"])) != first for record in member_records):
            raise ValueError(f"Archive members have mixed selection scopes: {path}")
        if tuple(archive.get("tasks", [])) != first[0] or tuple(archive.get("scopes", [])) != first[1]:
            raise ValueError(f"Archive selection scope disagrees with members: {path}")
    videos = {path for path in file_paths if path.endswith(".mp4")}
    if set(covered) != videos or len(covered) != len(videos):
        raise ValueError("Schema 2 archives do not cover every video exactly once")


def select_files(manifest: dict[str, Any], tasks: list[str] | None, include_prefix_videos: bool) -> list[dict[str, Any]]:
    """Choose shared metadata and the requested task assets."""
    selected_tasks = set(TASKS if not tasks or "all" in tasks else tasks)
    scope = "all" if include_prefix_videos else "paper"
    return [record for record in manifest["files"] if scope in record["scopes"] and (not record["tasks"] or selected_tasks.intersection(record["tasks"]))]


def select_download_entries(manifest: dict[str, Any], tasks: list[str] | None, include_prefix_videos: bool) -> list[dict[str, Any]]:
    """Select direct files and archives needed for the requested task scope."""
    selected = select_files(manifest, tasks, include_prefix_videos)
    if manifest.get("schema_version") != 2:
        return [{**record, "_entry_type": "file"} for record in selected]
    selected_paths = {record["path"] for record in selected}
    entries = [{**record, "_entry_type": "file"} for record in selected if not record["path"].endswith(".mp4")]
    for archive in manifest["archives"]:
        if selected_paths.intersection(archive["members"]):
            entries.append({**archive, "_entry_type": "archive"})
    return entries


def verify_file(root: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Validate size, digest, and file stability during hashing."""
    path = dataset_path(root, expected["path"])
    record: dict[str, Any] = {"path": expected["path"], "ok": False}
    if not path.is_file():
        return {**record, "error": "missing"}
    before = path.stat()
    record["size"] = before.st_size
    if expected["path"] == ".gitattributes":
        record["sha256"] = file_sha256(path)
        record["ok"] = before.st_size > 0
        if not record["ok"]:
            record["error"] = "empty platform-generated metadata"
        return record
    if before.st_size != expected["size"]:
        return {**record, "error": "size differs"}
    record["sha256"] = file_sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        return {**record, "error": "file changed during verification"}
    if record["sha256"] != expected["sha256"]:
        return {**record, "error": "SHA256 differs"}
    return {**record, "ok": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task", action="append", choices=("all",) + TASKS)
    parser.add_argument("--include-prefix-videos", action="store_true", help="Also verify the original public concatenated N-back videos.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, help="Optional JSON verification report.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.data_root.expanduser().resolve()
    manifest = load_manifest((root / MANIFEST_PATH).read_bytes())
    expected = select_files(manifest, args.task, args.include_prefix_videos)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(lambda item: verify_file(root, item), expected), 1):
            results.append(record)
            if index % 25 == 0:
                print(f"Verified {index}/{len(expected)} files; mismatches={sum(not x['ok'] for x in results)}", flush=True)
    errors = [record for record in results if not record["ok"]]
    report = {
        "verified": not errors,
        "release": manifest["release"],
        "manifest_sha256": MANIFEST_SHA256,
        "tasks": args.task or ["all"],
        "include_prefix_videos": args.include_prefix_videos,
        "files": len(expected),
        "expected_bytes": sum(record["size"] for record in expected),
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
