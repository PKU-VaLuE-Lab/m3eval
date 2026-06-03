from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

REGISTRY_ENV_VAR = "VIDEO_MEMORY_EVAL_SHARED_BAD_VIDEOS_REGISTRY"
REGISTRY_VERSION = 1
NON_REGISTRY_ERROR_TYPES = {
    "memoryerror",
    "outofmemoryerror",
}
NON_REGISTRY_ERROR_MESSAGE_PATTERNS = (
    "out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "shape mismatch",
    "cannot be broadcast to indexing result",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_shared_bad_videos_path() -> Path:
    override = os.environ.get(REGISTRY_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (_repo_root() / "data/ours/v4/intra_dataset_pairs/videos/analysis/shared_bad_videos.json").resolve()


def normalize_video_path(video_path: object) -> str:
    text = str(video_path or "").strip()
    if not text:
        return ""
    return str(Path(text).expanduser().resolve())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_registry_eligible_video_failure(error_type: object, error_message: object) -> bool:
    error_type_text = str(error_type or "").strip().lower()
    error_message_text = str(error_message or "").strip().lower()
    if error_type_text in NON_REGISTRY_ERROR_TYPES:
        return False
    return not any(pattern in error_message_text for pattern in NON_REGISTRY_ERROR_MESSAGE_PATTERNS)


def request_identity(request: Any, doc: Optional[dict] = None) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "task": "",
        "split": "",
        "doc_id": "",
        "pair_id": "",
    }
    if request is not None:
        arguments = getattr(request, "arguments", None) or ()
        if len(arguments) >= 6:
            meta["doc_id"] = arguments[3]
            meta["task"] = arguments[4]
            meta["split"] = arguments[5]
    if doc:
        meta["pair_id"] = doc.get("pair_id", "")
    doc_id = meta["doc_id"]
    if isinstance(doc_id, int) or str(doc_id).isdigit():
        meta["doc_id"] = int(doc_id)
    return meta


def build_video_failure_record(
    *,
    video_path: object,
    error_type: str,
    error_message: str,
    source: str,
    request: Any = None,
    doc: Optional[dict] = None,
    rank: Optional[int] = None,
    known_bad_video: bool = False,
) -> dict[str, Any]:
    meta = request_identity(request, doc=doc)
    record = {
        "task": meta["task"],
        "split": meta["split"],
        "doc_id": meta["doc_id"],
        "pair_id": meta["pair_id"],
        "video_path": normalize_video_path(video_path),
        "error_type": str(error_type or "VideoReadFailure"),
        "error_message": str(error_message or "").strip(),
        "rank": rank,
        "source": str(source or "").strip(),
    }
    if known_bad_video:
        record["known_bad_video"] = True
    return record


def append_request_video_failure(request: Any, failure: dict[str, Any]) -> None:
    existing = list(getattr(request, "video_read_failures", []) or [])
    existing.append(failure)
    request.video_read_failures = existing


class SharedBadVideoRegistry:
    def __init__(self, registry_path: Optional[os.PathLike[str] | str] = None):
        self.path = Path(registry_path) if registry_path else default_shared_bad_videos_path()
        self.path = self.path.expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._cache_mtime_ns: Optional[int] = None
        self._cache_entries: dict[str, dict[str, Any]] = {}

    def _empty_payload(self) -> dict[str, Any]:
        return {
            "meta": {
                "registry_version": REGISTRY_VERSION,
                "updated_at": "",
                "total_bad_videos": 0,
            },
            "videos": [],
        }

    def _normalize_entry(self, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        video_path = normalize_video_path(item.get("video_path", ""))
        if not video_path:
            return None
        sources = sorted({str(src).strip() for src in (item.get("sources") or []) if str(src).strip()})
        first_seen = str(item.get("first_seen", "") or "")
        last_seen = str(item.get("last_seen", "") or "")
        latest_error_type = str(item.get("latest_error_type", "") or "")
        latest_error_message = str(item.get("latest_error_message", "") or "")
        return {
            "video_path": video_path,
            "sources": sources,
            "first_seen": first_seen,
            "last_seen": last_seen,
            "latest_error_type": latest_error_type,
            "latest_error_message": latest_error_message,
        }

    def _load_payload_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._empty_payload()
        with self.path.open(encoding="utf-8") as f:
            payload = json.load(f)
        videos = payload.get("videos") or []
        normalized = []
        for item in videos:
            if not isinstance(item, dict):
                continue
            entry = self._normalize_entry(item)
            if entry is not None:
                normalized.append(entry)
        return {
            "meta": payload.get("meta") or {},
            "videos": normalized,
        }

    def _entries_from_payload(self, payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for item in payload.get("videos") or []:
            entry = self._normalize_entry(item)
            if entry is None:
                continue
            entries[entry["video_path"]] = entry
        return entries

    def _payload_from_entries(self, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
        items = [entries[path] for path in sorted(entries)]
        error_type_counts = Counter()
        for item in items:
            error_type = str(item.get("latest_error_type", "") or "").strip()
            if error_type:
                error_type_counts[error_type] += 1
        return {
            "meta": {
                "registry_version": REGISTRY_VERSION,
                "updated_at": utc_now_iso(),
                "total_bad_videos": len(items),
                "error_type_counts": dict(sorted(error_type_counts.items())),
            },
            "videos": items,
        }

    def _ensure_parent_dirs(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _locked_entries(self) -> tuple[Any, dict[str, dict[str, Any]]]:
        self._ensure_parent_dirs()
        lock_file = self.lock_path.open("a+", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            payload = self._load_payload_unlocked()
            entries = self._entries_from_payload(payload)
            return lock_file, entries
        except Exception:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            raise

    def _release_lock(self, lock_file: Any) -> None:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()

    def _write_entries_unlocked(self, entries: dict[str, dict[str, Any]]) -> None:
        self._ensure_parent_dirs()
        payload = self._payload_from_entries(entries)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            prefix=f".{self.path.stem}_",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, self.path)
        stat = self.path.stat()
        self._cache_mtime_ns = stat.st_mtime_ns
        self._cache_entries = {key: dict(value) for key, value in entries.items()}

    def _refresh_cache_if_needed(self) -> None:
        if not self.path.exists():
            self._cache_entries = {}
            self._cache_mtime_ns = None
            return
        stat = self.path.stat()
        if self._cache_mtime_ns == stat.st_mtime_ns:
            return
        payload = self._load_payload_unlocked()
        self._cache_entries = self._entries_from_payload(payload)
        self._cache_mtime_ns = stat.st_mtime_ns

    def get_entry(self, video_path: object) -> Optional[dict[str, Any]]:
        normalized = normalize_video_path(video_path)
        if not normalized:
            return None
        self._refresh_cache_if_needed()
        entry = self._cache_entries.get(normalized)
        return dict(entry) if entry else None

    def has_entry(self, video_path: object) -> bool:
        return self.get_entry(video_path) is not None

    def record_failure(
        self,
        *,
        video_path: object,
        error_type: str,
        error_message: str,
        source: str,
    ) -> dict[str, Any]:
        normalized = normalize_video_path(video_path)
        if not normalized:
            raise ValueError("video_path must be non-empty")
        timestamp = utc_now_iso()
        lock_file, entries = self._locked_entries()
        try:
            entry = dict(entries.get(normalized) or {})
            sources = set(entry.get("sources") or [])
            if source:
                sources.add(str(source).strip())
            entry.update(
                {
                    "video_path": normalized,
                    "sources": sorted(src for src in sources if src),
                    "first_seen": entry.get("first_seen") or timestamp,
                    "last_seen": timestamp,
                    "latest_error_type": str(error_type or entry.get("latest_error_type") or "VideoReadFailure"),
                    "latest_error_message": str(error_message or entry.get("latest_error_message") or "").strip(),
                }
            )
            entries[normalized] = entry
            self._write_entries_unlocked(entries)
            return dict(entry)
        finally:
            self._release_lock(lock_file)

    def merge_failure_records(self, records: Iterable[dict[str, Any]], source: Optional[str] = None) -> dict[str, int]:
        added = 0
        updated = 0
        for record in records:
            if not isinstance(record, dict):
                continue
            video_path = record.get("video_path", "")
            normalized = normalize_video_path(video_path)
            if not normalized:
                continue
            error_type = str(record.get("error_type", "") or "VideoReadFailure")
            error_message = str(record.get("error_message", "") or "")
            if not is_registry_eligible_video_failure(error_type, error_message):
                continue
            existing = self.get_entry(normalized)
            merged_entry = self.record_failure(
                video_path=normalized,
                error_type=error_type,
                error_message=error_message,
                source=source or str(record.get("source", "") or ""),
            )
            if existing is None:
                added += 1
            elif merged_entry != existing:
                updated += 1
        return {
            "added": added,
            "updated": updated,
            "total": len(self.list_entries()),
        }

    def list_entries(self) -> list[dict[str, Any]]:
        self._refresh_cache_if_needed()
        return [dict(self._cache_entries[path]) for path in sorted(self._cache_entries)]

    def prune_ineligible_entries(self) -> dict[str, int]:
        lock_file, entries = self._locked_entries()
        try:
            kept_entries = {
                path: entry
                for path, entry in entries.items()
                if is_registry_eligible_video_failure(
                    entry.get("latest_error_type", ""),
                    entry.get("latest_error_message", ""),
                )
            }
            removed = len(entries) - len(kept_entries)
            if removed > 0:
                self._write_entries_unlocked(kept_entries)
            else:
                self._cache_entries = {key: dict(value) for key, value in kept_entries.items()}
            return {
                "removed": removed,
                "kept": len(kept_entries),
                "total": len(kept_entries),
            }
        finally:
            self._release_lock(lock_file)
