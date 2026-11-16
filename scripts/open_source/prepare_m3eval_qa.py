#!/usr/bin/env python3
"""Prepare the minimal non-N-back QA tree used by the public InternVL runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASKS = {
    "memory_interference": {"families": {"memory_interference"}},
    "split_screen": {"families": {"split_screen_dynamic_swap", "source_memory"}},
    "interleaved": {"families": {"interleaved_storyline", "source_memory", "false_source_attribution"}},
}

PROFILE_PATH = Path(__file__).resolve().parent / "config/non_nback.json"


def paper_rows(data_root: Path, stage: str) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Prepare pinned release prompts, requiring the exact questions and labels."""
    profile_bytes = PROFILE_PATH.read_bytes()
    profile = json.loads(profile_bytes)
    source = {}
    for task in TASKS:
        path = find_task_file(data_root, task)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != profile["question_files_sha256"][task]:
            raise ValueError(f"Question file differs from the paper profile: {path}")
        source[task] = load_rows(path)

    output: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    seen: set[str] = set()
    allowed_overrides = {"question", "pre_prompt_version", "hint_mode", "segment_count"}
    for record in profile["records"]:
        task = record["task"]
        row = dict(source[task][record["release_index"]])
        sample_id = f"{task}:{row['question_signature']}"
        if sample_id != record["sample_id"] or sample_id in seen:
            raise ValueError(f"Invalid or duplicate sample identity: {sample_id}")
        seen.add(sample_id)
        if not set(record["overrides"]).issubset(allowed_overrides):
            raise ValueError(f"Profile may only override prompt fields: {sample_id}")
        if stage != "all" and not record["stage"].startswith(stage + "_"):
            continue
        row.update(record["overrides"])
        row["_reproduction_id"] = sample_id
        row["_reproduction_stage"] = record["stage"]
        row["_expected_prompt_sha256"] = record["prompt_sha256"]
        output[task].append(row)
    if len(seen) != 739:
        raise ValueError(f"Paper profile must contain 739 unique questions, found {len(seen)}")
    return output, hashlib.sha256(profile_bytes).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        candidates = [value for value in payload.values() if isinstance(value, list)]
        if len(candidates) != 1:
            raise ValueError(f"Cannot find a single row list in {path}")
        rows = candidates[0]
    else:
        raise ValueError(f"Unsupported JSON shape in {path}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Rows must be objects in {path}")
    return rows


def find_task_file(data_root: Path, task: str) -> Path:
    preferred = data_root / "questions" / "non_nback" / f"{task}.json"
    if preferred.is_file():
        return preferred
    matches = sorted(data_root.rglob(f"{task}.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {task}.json below {data_root}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-qa-root", type=Path, required=True)
    parser.add_argument("--limit-per-task", type=int, default=0)
    parser.add_argument("--profile", choices=("release", "paper-2026"), default="release")
    parser.add_argument("--stage", choices=("all", "main", "supplement"), default="all")
    parser.add_argument("--task", choices=("m3eval", "m3eval_memory_interference", "m3eval_split_screen", "m3eval_interleaved", "m3eval_nback"), default="m3eval")
    args = parser.parse_args()
    if args.limit_per_task < 0:
        raise SystemExit("--limit-per-task must be non-negative")

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_qa_root.expanduser().resolve()
    summary: dict[str, Any] = {"data_root": str(data_root), "profile": args.profile, "stage": args.stage, "task": args.task, "counts": {}, "full_counts": {}}
    restored = None
    if args.profile == "paper-2026":
        restored, summary["profile_sha256"] = paper_rows(data_root, args.stage)
    elif args.stage != "all":
        raise ValueError("--stage requires --profile paper-2026")

    for task, spec in TASKS.items():
        if restored is not None:
            selected = restored[task]
        else:
            rows = load_rows(find_task_file(data_root, task))
            selected = [row for row in rows if row.get("task_family") in spec["families"]]
        if args.task not in ("m3eval", f"m3eval_{task}"):
            selected = []
        if args.limit_per_task:
            output_rows = selected[: args.limit_per_task]
        else:
            output_rows = selected
        task_dir = output_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"{task}_qa.json").write_text(json.dumps(output_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["full_counts"][task] = len(selected)
        summary["counts"][task] = len(output_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "prepared_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
