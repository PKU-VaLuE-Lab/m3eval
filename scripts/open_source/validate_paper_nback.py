#!/usr/bin/env python3
"""Verify the N-back question source and full selection against the archived run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from summarize_paper_run import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--tests-per-split", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    profile = json.loads((Path(__file__).resolve().parent / "config/nback.json").read_text())
    for split in ("action", "scene"):
        source = root / "n-back" / f"nback_test_{split}.json"
        if hashlib.sha256(source.read_bytes()).hexdigest() != profile["source_files_sha256"][split]:
            raise ValueError(f"N-back question source changed: {source}")
        if args.tests_per_split == 32:
            selected = json.loads((args.selection_root / split / f"{split}_full.json").read_text())
            if digest(selected) != profile["full_selection_sha256"][split]:
                raise ValueError(f"N-back selected clips or labels changed: {split}")
    print("Verified N-back sources" + (" and all 64 historical sequences" if args.tests_per_split == 32 else " (smoke subset)"))


if __name__ == "__main__":
    main()
