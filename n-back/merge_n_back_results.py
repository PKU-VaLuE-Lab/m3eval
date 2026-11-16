#!/usr/bin/env python3
"""Merge multiple n-back results.json files into one deduplicated results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple n-back results.json files")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for raw_path in args.inputs:
        path = Path(raw_path).resolve()
        for row in load_rows(path):
            task_id = str(row.get("task_id", "")).strip()
            if task_id and task_id in seen_task_ids:
                continue
            if task_id:
                seen_task_ids.add(task_id)
            merged.append(row)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] merged_results={output_path}")
    print(f"[DONE] merged_count={len(merged)}")


if __name__ == "__main__":
    main()
