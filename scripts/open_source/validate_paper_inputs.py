#!/usr/bin/env python3
"""Check actual renderer output and video availability before loading the model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lmms_eval.tasks.m3eval.utils import v4_doc_to_text, v4_doc_to_visual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-root", type=Path, required=True)
    args = parser.parse_args()
    seen: set[str] = set()
    videos: set[str] = set()
    for path in sorted(args.qa_root.glob("*/*_qa.json")):
        for row in json.loads(path.read_text()):
            sample_id = row["_reproduction_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate sample identity: {sample_id}")
            seen.add(sample_id)
            digest = hashlib.sha256(v4_doc_to_text(row).encode()).hexdigest()
            if digest != row["_expected_prompt_sha256"]:
                raise ValueError(f"Actual prompt differs from the release configuration: {sample_id}")
            videos.update(v4_doc_to_visual(row))
    if not seen:
        raise ValueError("No prepared questions found")
    print(json.dumps({"verified_prompts": len(seen), "available_videos": len(videos)}))


if __name__ == "__main__":
    main()
