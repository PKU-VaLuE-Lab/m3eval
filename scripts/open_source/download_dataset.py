#!/usr/bin/env python3
"""Download the matching public M3Eval files, resuming and checking each file."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from verify_dataset import (
    DATASET_REPOSITORY,
    MANIFEST_PATH,
    TASKS,
    dataset_path,
    file_sha256,
    load_manifest,
    select_download_entries,
    select_files,
)


def download_file(url: str, target: Path, expected: dict[str, Any], retries: int = 5) -> dict[str, Any]:
    """Resume a Range-capable transfer and only install verified complete bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # Hugging Face generates this transport metadata file when publishing a
    # large folder. Its contents are repository-specific, so the release
    # manifest records the local source copy while downloads accept the
    # generated remote copy.
    if expected["path"] == ".gitattributes":
        if target.is_file() and target.stat().st_size > 0:
            return {"path": expected["path"], "status": "platform_generated", "size": target.stat().st_size}
        partial = target.with_name(target.name + ".part")
        for attempt in range(retries):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "m3eval-release-downloader", "Accept-Encoding": "identity"})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as sink:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        sink.write(chunk)
                if partial.stat().st_size <= 0:
                    raise ValueError("Hugging Face generated .gitattributes is empty")
                size = partial.stat().st_size
                os.replace(partial, target)
                return {"path": expected["path"], "status": "platform_generated", "size": size}
            except (OSError, ValueError, urllib.error.URLError, http.client.IncompleteRead):
                if attempt + 1 < retries:
                    time.sleep(min(2**attempt, 16))
        raise RuntimeError("Could not download Hugging Face generated .gitattributes")
    if target.is_file() and target.stat().st_size == expected["size"] and file_sha256(target) == expected["sha256"]:
        return {"path": expected["path"], "status": "already_verified", "size": expected["size"]}
    partial = target.with_name(target.name + ".part")
    if partial.is_symlink():
        raise ValueError(f"Refusing a symlink download partial: {partial}")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected["size"]:
                partial.unlink()
                offset = 0
            if offset == expected["size"] and partial.exists():
                if file_sha256(partial) == expected["sha256"]:
                    os.replace(partial, target)
                    return {"path": expected["path"], "status": "downloaded", "size": expected["size"]}
                partial.unlink()
                offset = 0
            headers = {"User-Agent": "m3eval-release-downloader", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = response.status
                if offset and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or content_range.rsplit("/", 1)[-1] != str(expected["size"]):
                        raise ValueError(f"Incorrect resume range: {content_range}")
                    mode = "ab"
                elif status == 200:
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected HTTP status: {status}")
                with partial.open(mode) as sink:
                    for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        sink.write(chunk)
            if partial.stat().st_size != expected["size"]:
                raise ValueError(f"Incomplete download: {expected['path']} ({partial.stat().st_size}/{expected['size']} bytes)")
            if file_sha256(partial) != expected["sha256"]:
                partial.unlink()
                raise ValueError(f"SHA256 differs: {expected['path']}; retrying from the beginning")
            os.replace(partial, target)
            return {"path": expected["path"], "status": "downloaded", "size": expected["size"]}
        except (OSError, ValueError, urllib.error.URLError, http.client.IncompleteRead) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"Could not download {expected['path']} after {retries} attempts: {last_error}")


def verified_file(root: Path, expected: dict[str, Any]) -> bool:
    """Return whether an extracted file already matches its manifest record."""
    try:
        path = dataset_path(root, expected["path"])
    except ValueError:
        return False
    if not path.is_file() or path.is_symlink():
        return False
    try:
        return path.stat().st_size == expected["size"] and file_sha256(path) == expected["sha256"]
    except OSError:
        return False


def _zip_member_is_symlink(info: Any) -> bool:
    """Detect Unix and DOS symlink entries in a ZIP central directory."""
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode) or bool(info.external_attr & 0x10 and info.filename.endswith("/"))


def _safe_zip_member(info: Any, expected_paths: set[str]) -> None:
    """Reject unsafe or unexpected ZIP members before writing anything."""
    name = info.filename
    if not name or name not in expected_paths:
        raise ValueError(f"Unexpected archive member: {name!r}")
    if name.startswith("/") or "\\" in name or ".." in Path(name).parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if info.is_dir() or _zip_member_is_symlink(info):
        raise ValueError(f"Archive member is not a regular file: {name}")


def extract_archive(archive_path: Path, root: Path, archive: dict[str, Any], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Safely extract and verify every member of one downloaded archive."""
    expected_paths = set(archive["members"])
    seen: set[str] = set()
    extracted: list[dict[str, Any]] = []
    try:
        with ZipFile(archive_path, "r") as source:
            infos = source.infolist()
            if len(infos) != len(expected_paths):
                raise ValueError(f"Archive member count differs: {archive['path']}")
            for info in infos:
                _safe_zip_member(info, expected_paths)
                if info.filename in seen:
                    raise ValueError(f"Duplicate archive member: {info.filename}")
                seen.add(info.filename)
            if seen != expected_paths:
                missing = sorted(expected_paths - seen)
                raise ValueError(f"Archive is missing members: {missing[:3]}")
            for name in sorted(seen):
                expected = records[name]
                target = dataset_path(root, name)
                if verified_file(root, expected):
                    extracted.append({"path": name, "status": "already_verified", "size": expected["size"]})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + ".part")
                if partial.is_symlink():
                    raise ValueError(f"Refusing a symlink extraction partial: {partial}")
                if partial.exists():
                    partial.unlink()
                info = source.getinfo(name)
                digest = hashlib.sha256()
                written = 0
                with source.open(info, "r") as source_member, partial.open("wb") as sink:
                    for chunk in iter(lambda: source_member.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        written += len(chunk)
                        sink.write(chunk)
                if written != expected["size"] or digest.hexdigest() != expected["sha256"]:
                    try:
                        partial.unlink()
                    except FileNotFoundError:
                        pass
                    raise ValueError(f"Extracted member hash or size differs: {name}")
                os.replace(partial, target)
                extracted.append({"path": name, "status": "extracted", "size": expected["size"]})
    except (BadZipFile, EOFError, OSError) as error:
        raise ValueError(f"Could not extract archive {archive['path']}: {error}") from error
    return {"path": archive["path"], "status": "extracted", "members": extracted, "size": archive["size"]}


def download_archive(
    url: str,
    archive_path: Path,
    root: Path,
    archive: dict[str, Any],
    records: dict[str, dict[str, Any]],
    keep_archive: bool = False,
    retries: int = 5,
) -> dict[str, Any]:
    """Download one archive, extract it safely, and optionally retain the ZIP."""
    # Default downloads remove ZIPs after extraction. Verify the extracted
    # files before transferring them again; --keep-archives still fetches ZIPs.
    members = [records[name] for name in archive["members"]]
    if not keep_archive and all(verified_file(root, member) for member in members):
        return {
            "path": archive["path"],
            "status": "already_verified",
            "members": [
                {"path": member["path"], "status": "already_verified", "size": member["size"]}
                for member in members
            ],
            "size": archive["size"],
            "download_status": "not_needed",
        }
    result = download_file(url, archive_path, archive, retries=retries)
    extraction = extract_archive(archive_path, root, archive, records)
    if not keep_archive:
        try:
            archive_path.unlink()
        except FileNotFoundError:
            pass
    extraction["download_status"] = result["status"]
    return extraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main", help="Hugging Face dataset commit or tag; its manifest must match this code release.")
    parser.add_argument("--task", action="append", choices=("all",) + TASKS)
    parser.add_argument("--include-prefix-videos", action="store_true", help="Include the original public N-back concatenated prefixes (not used by the paper runner).")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, help="Optional JSON completion report.")
    parser.add_argument("--base-url", help="Optional resolve URL for validating a staged release before publication.")
    parser.add_argument("--archive-endpoint", help="Optional Hugging Face mirror endpoint for ZIP archives; metadata still uses HF_ENDPOINT.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded ZIP archives after successful extraction.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = args.local_dir.expanduser().resolve()
    revision = urllib.parse.quote(args.revision, safe="")
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    base = args.base_url.rstrip("/") + "/" if args.base_url else f"{endpoint}/datasets/{DATASET_REPOSITORY}/resolve/{revision}/"
    archive_base = f"{args.archive_endpoint.rstrip('/')}/datasets/{DATASET_REPOSITORY}/resolve/{revision}/" if args.archive_endpoint else base
    try:
        with urllib.request.urlopen(base + MANIFEST_PATH, timeout=60) as response:
            content = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Dataset revision {args.revision!r} does not provide the matching release manifest (HTTP {error.code}). Use the data revision listed with this code release.") from error
    manifest = load_manifest(content)
    target_manifest = dataset_path(root, MANIFEST_PATH)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_bytes(content)
    checksum_lines = [f"{record['sha256']}  {record['path']}" for record in manifest["files"]]
    if manifest.get("schema_version") == 2:
        checksum_lines.extend(f"{record['sha256']}  {record['path']}" for record in manifest["archives"])
    checksum_lines.append(f"{hashlib.sha256(content).hexdigest()}  {MANIFEST_PATH}")
    dataset_path(root, "checksums_sha256.txt").write_text("\n".join(checksum_lines) + "\n")
    selected_files = select_files(manifest, args.task, args.include_prefix_videos)
    entries = select_download_entries(manifest, args.task, args.include_prefix_videos)
    records_by_path = {record["path"]: record for record in manifest["files"]}
    total_bytes = sum(record["size"] for record in selected_files)
    print(f"Selected {len(selected_files)} files, {total_bytes / 1e9:.2f} GB. Existing correct files are reused; partial files resume.", flush=True)
    failures = []
    results = []
    archive_cache = root / ".m3eval_archives"
    archive_semaphore = threading.BoundedSemaphore(max(1, min(2, args.workers // 2 or 1)))

    def run_entry(entry: dict[str, Any]) -> dict[str, Any]:
        if entry["_entry_type"] == "file":
            return download_file(base + urllib.parse.quote(entry["path"], safe="/"), dataset_path(root, entry["path"]), entry)
        archive_path = dataset_path(archive_cache, entry["path"])
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_semaphore:
            return download_archive(
                archive_base + urllib.parse.quote(entry["path"], safe="/"),
                archive_path,
                root,
                entry,
                records_by_path,
                keep_archive=args.keep_archives,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_entry, entry): entry for entry in entries}
        for future in as_completed(futures):
            record = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"path": record["path"], "error": str(error)})
                print(f"FAILED {record['path']}: {error}", flush=True)
            completed = len(results) + len(failures)
            if completed % 10 == 0 or completed == len(entries):
                print(f"Verified {len(results)}/{len(entries)} download entries; failures={len(failures)}", flush=True)
    verified_paths: set[str] = set()
    for result in results:
        if result.get("path") in records_by_path:
            verified_paths.add(result["path"])
        for member in result.get("members", []):
            if member.get("path") in records_by_path:
                verified_paths.add(member["path"])
    report = {
        "complete": not failures,
        "repository": DATASET_REPOSITORY,
        "revision": args.revision,
        "source": base,
        "archive_source": archive_base,
        "release": manifest["release"],
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
        "tasks": args.task or ["all"],
        "include_prefix_videos": args.include_prefix_videos,
        "selected_files": len(selected_files),
        "download_entries": len(entries),
        "verified_files": len(verified_paths),
        "verified_bytes": sum(records_by_path[path]["size"] for path in verified_paths),
        "keep_archives": args.keep_archives,
        "failures": failures,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
