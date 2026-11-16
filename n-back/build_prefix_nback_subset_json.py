#!/usr/bin/env python3
"""Build reduced prefix N-back JSON files for multi-GPU split/shard runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from nback_eval_utils import (
    build_balanced_test_plan,
    build_random_test_plan,
    resolve_nback_video_path,
)


def collect_eligible_test_indices(data: list[dict[str, Any]], *, data_root: Path, min_video_file_size_bytes: int) -> dict[int, list[int]]:
    eligible_test_indices: dict[int, list[int]] = {}
    for domain_idx, domain_data in enumerate(data):
        domain_eligible: list[int] = []
        for test_idx, test in enumerate(domain_data.get("tests", [])):
            is_valid = True
            for video_path in test.get("video_path_sequence", []):
                resolved_path = resolve_nback_video_path(video_path, data_root=data_root)
                if not resolved_path.exists():
                    is_valid = False
                    break
                if resolved_path.stat().st_size < min_video_file_size_bytes:
                    is_valid = False
                    break
            if is_valid:
                domain_eligible.append(test_idx)
        eligible_test_indices[domain_idx] = domain_eligible
    return eligible_test_indices


def flatten_selected_pairs(data: list[dict[str, Any]], selection_plan: dict[str, Any]) -> list[tuple[int, int]]:
    selected_pairs = selection_plan.get("selected_ordered_tests") or []
    if selected_pairs:
        return [(int(domain_idx), int(test_idx)) for domain_idx, test_idx in selected_pairs]

    flattened: list[tuple[int, int]] = []
    for domain_idx in range(len(data)):
        for test_idx in selection_plan["selected_test_indices"].get(domain_idx, []):
            flattened.append((domain_idx, int(test_idx)))
    return flattened


def build_subset_dataset(data: list[dict[str, Any]], selected_pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    selected_map: dict[int, list[int]] = {}
    for domain_idx, test_idx in selected_pairs:
        selected_map.setdefault(domain_idx, []).append(test_idx)

    subset_data: list[dict[str, Any]] = []
    for domain_idx, domain_data in enumerate(data):
        chosen = selected_map.get(domain_idx, [])
        if not chosen:
            continue
        tests = [domain_data["tests"][test_idx] for test_idx in chosen]
        subset_data.append(
            {
                **domain_data,
                "tests": tests,
            }
        )
    return subset_data


def load_excluded_pairs(manifest_paths: list[str]) -> set[tuple[int, int]]:
    excluded_pairs: set[tuple[int, int]] = set()
    for raw_path in manifest_paths:
        manifest_path = Path(raw_path).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"exclude selection manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in payload.get("selected_pairs", []) or []:
            if not isinstance(item, dict):
                continue
            try:
                excluded_pairs.add((int(item["domain_index"]), int(item["test_index"])))
            except (KeyError, TypeError, ValueError):
                continue
    return excluded_pairs


def load_selected_pairs_manifest(manifest_path: str | None) -> list[tuple[int, int]]:
    if not manifest_path:
        return []
    resolved = Path(manifest_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"selected_pairs manifest not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    selected_pairs: list[tuple[int, int]] = []
    for item in payload.get("selected_pairs", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            selected_pairs.append((int(item["domain_index"]), int(item["test_index"])))
        except (KeyError, TypeError, ValueError):
            continue
    return selected_pairs


def validate_selected_pairs(
    selected_pairs: list[tuple[int, int]],
    *,
    data: list[dict[str, Any]],
    eligible_test_indices: dict[int, list[int]],
    excluded_pairs: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    validated: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for domain_idx, test_idx in selected_pairs:
        pair = (int(domain_idx), int(test_idx))
        if pair in seen:
            continue
        seen.add(pair)
        if pair in excluded_pairs:
            raise ValueError(f"explicit selected pair excluded by exclude manifest: {pair}")
        if domain_idx < 0 or domain_idx >= len(data):
            raise ValueError(f"explicit selected pair has invalid domain_index: {pair}")
        tests = data[domain_idx].get("tests", [])
        if test_idx < 0 or test_idx >= len(tests):
            raise ValueError(f"explicit selected pair has invalid test_index: {pair}")
        if test_idx not in eligible_test_indices.get(domain_idx, []):
            raise ValueError(f"explicit selected pair is not eligible after video validation: {pair}")
        validated.append(pair)
    return validated


def filter_eligible_test_indices(
    eligible_test_indices: dict[int, list[int]],
    excluded_pairs: set[tuple[int, int]],
) -> dict[int, list[int]]:
    if not excluded_pairs:
        return eligible_test_indices

    filtered: dict[int, list[int]] = {}
    for domain_idx, test_indices in eligible_test_indices.items():
        filtered[domain_idx] = [test_idx for test_idx in test_indices if (int(domain_idx), int(test_idx)) not in excluded_pairs]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reduced prefix N-back subset/shard JSON")
    parser.add_argument("--source_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--selection_manifest_out", default="")
    parser.add_argument("--tests_per_split", type=int, required=True)
    parser.add_argument("--selection_mode", choices=["balanced", "random"], default="balanced")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--min_video_file_size_bytes", type=int, default=10000)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--exclude_selection_manifest", action="append", default=[])
    parser.add_argument("--selected_pairs_manifest", default="")
    args = parser.parse_args()

    source_json = Path(args.source_json).resolve()
    output_json = Path(args.output_json).resolve()
    selection_manifest_out = Path(args.selection_manifest_out).resolve() if args.selection_manifest_out else None
    data_root = Path(args.data_root).resolve()

    data = json.loads(source_json.read_text(encoding="utf-8"))
    questions_per_test = len(data[0]["tests"][0]["n_back_evaluation"])
    target_qa_count = args.tests_per_split * questions_per_test

    eligible_test_indices = collect_eligible_test_indices(
        data,
        data_root=data_root,
        min_video_file_size_bytes=args.min_video_file_size_bytes,
    )
    excluded_pairs = load_excluded_pairs(args.exclude_selection_manifest)
    eligible_test_indices = filter_eligible_test_indices(eligible_test_indices, excluded_pairs)
    explicit_selected_pairs = load_selected_pairs_manifest(args.selected_pairs_manifest)
    selection_plan: dict[str, Any] | None = None
    selection_mode = args.selection_mode

    if explicit_selected_pairs:
        selection_mode = "explicit"
        selected_pairs = validate_selected_pairs(
            explicit_selected_pairs,
            data=data,
            eligible_test_indices=eligible_test_indices,
            excluded_pairs=excluded_pairs,
        )
    else:
        if args.selection_mode == "balanced":
            selection_plan = build_balanced_test_plan(
                data,
                target_qa_count=target_qa_count,
                questions_per_test=questions_per_test,
                eligible_test_indices=eligible_test_indices,
            )
        else:
            selection_plan = build_random_test_plan(
                data,
                target_qa_count=target_qa_count,
                questions_per_test=questions_per_test,
                eligible_test_indices=eligible_test_indices,
                random_seed=args.random_seed,
            )
        selected_pairs = flatten_selected_pairs(data, selection_plan)

    selected_pairs_before_shard = list(selected_pairs)
    if args.shard_count > 1:
        selected_pairs = [pair for index, pair in enumerate(selected_pairs) if index % args.shard_count == args.shard_index]

    subset_data = build_subset_dataset(data, selected_pairs)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(subset_data, indent=2, ensure_ascii=False), encoding="utf-8")

    if selection_manifest_out:
        payload = {
            "source_json": str(source_json),
            "output_json": str(output_json),
            "selection_mode": selection_mode,
            "random_seed": args.random_seed if selection_mode == "random" else None,
            "tests_per_split": args.tests_per_split,
            "target_qa_count": target_qa_count,
            "selected_pair_count_before_shard": len(selected_pairs_before_shard),
            "selected_pair_count_after_shard": len(selected_pairs),
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "excluded_selection_manifests": [str(Path(path).resolve()) for path in args.exclude_selection_manifest],
            "excluded_pair_count": len(excluded_pairs),
            "selected_pairs_manifest": str(Path(args.selected_pairs_manifest).resolve()) if args.selected_pairs_manifest else "",
            "selected_pairs": [{"domain_index": domain_idx, "test_index": test_idx} for domain_idx, test_idx in selected_pairs],
        }
        selection_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        selection_manifest_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] subset_json={output_json}")
    if selection_manifest_out:
        print(f"[DONE] subset_manifest={selection_manifest_out}")


if __name__ == "__main__":
    main()
