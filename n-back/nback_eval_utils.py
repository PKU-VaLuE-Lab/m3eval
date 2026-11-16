from __future__ import annotations

import random
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_NBACK_DATA_ROOT = PROJECT_ROOT / "data/share/hf_nback_1_incremental_2026-04-11/precut_full_20pairs_per_dataset_seg10"


YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
LEADING_YES_NO_PATTERN = re.compile(
    r"^\s*(?:answer\s*[:\-]\s*|the answer is\s+|final answer\s*[:\-]\s*)?(yes|no)\b",
    re.IGNORECASE,
)
LABELED_N_BACK_LINE_PATTERN = re.compile(
    r"^\s*N\s*[-–]?\s*(\d+)\s*[:：\-]\s*(yes|no)\b",
    re.IGNORECASE,
)


def normalize_n_back_key(n_back_key: str) -> int:
    match = re.search(r"N-(\d+)_back", str(n_back_key or ""))
    if not match:
        return 0
    return int(match.group(1))


def resolve_nback_video_path(video_path: str, *, data_root: str | Path | None = None) -> Path:
    """Map dataset-relative video paths to the real local files in this repo."""
    candidate = Path(video_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    repo_relative = (PROJECT_ROOT / video_path).resolve()
    if repo_relative.exists():
        return repo_relative

    root = Path(data_root).resolve() if data_root is not None else DEFAULT_NBACK_DATA_ROOT.resolve()
    dataset_prefix = "data/precut_full_20pairs_per_dataset_seg10/"
    if video_path.startswith(dataset_prefix):
        remapped = (root / video_path[len(dataset_prefix) :]).resolve()
        if remapped.exists():
            return remapped

    return (root / video_path).resolve()


def parse_yes_no_response(text: Any) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return {
            "raw_text": raw_text,
            "parsed_answer": "",
            "resolved": False,
            "format_compliant": False,
            "parse_status": "empty",
        }

    lowered = raw_text.lower().strip()
    bare = re.sub(r"^[\"'`\(\[]+", "", lowered)
    bare = re.sub(r"[\"'`\)\]\.\!\?,:;]+$", "", bare)
    bare = bare.strip()
    if bare in {"yes", "no"} and len(bare.split()) == 1:
        return {
            "raw_text": raw_text,
            "parsed_answer": bare,
            "resolved": True,
            "format_compliant": True,
            "parse_status": f"exact_{bare}",
        }

    leading = LEADING_YES_NO_PATTERN.search(lowered)
    if leading:
        parsed = leading.group(1).lower()
        return {
            "raw_text": raw_text,
            "parsed_answer": parsed,
            "resolved": True,
            "format_compliant": False,
            "parse_status": f"leading_{parsed}",
        }

    mentions = [item.lower() for item in YES_NO_PATTERN.findall(lowered)]
    if not mentions:
        return {
            "raw_text": raw_text,
            "parsed_answer": "",
            "resolved": False,
            "format_compliant": False,
            "parse_status": "no_yes_no_token",
        }

    unique_mentions = sorted(set(mentions))
    if len(unique_mentions) == 1:
        parsed = unique_mentions[0]
        return {
            "raw_text": raw_text,
            "parsed_answer": parsed,
            "resolved": True,
            "format_compliant": False,
            "parse_status": f"single_polarity_{parsed}",
        }

    return {
        "raw_text": raw_text,
        "parsed_answer": "",
        "resolved": False,
        "format_compliant": False,
        "parse_status": "ambiguous_yes_no",
    }


def parse_batched_n_back_response(text: Any, expected_n_values: list[int]) -> dict[int, dict[str, Any]]:
    raw_text = str(text or "").strip()
    parsed_map: dict[int, dict[str, Any]] = {}
    if not expected_n_values:
        return parsed_map

    non_empty_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    for line in non_empty_lines:
        labeled = LABELED_N_BACK_LINE_PATTERN.match(line)
        if not labeled:
            continue
        n_value = int(labeled.group(1))
        if n_value not in expected_n_values or n_value in parsed_map:
            continue
        answer = labeled.group(2).lower()
        parsed_map[n_value] = {
            "raw_text": line,
            "parsed_answer": answer,
            "resolved": True,
            "format_compliant": True,
            "parse_status": f"batched_labeled_{answer}",
        }

    if len(parsed_map) == len(expected_n_values):
        return parsed_map

    unresolved_n_values = [n_value for n_value in expected_n_values if n_value not in parsed_map]
    sequential_candidates = []
    for line in non_empty_lines:
        line_parse = parse_yes_no_response(line)
        if not line_parse["resolved"]:
            continue
        if LABELED_N_BACK_LINE_PATTERN.match(line):
            continue
        sequential_candidates.append(line_parse)

    if len(sequential_candidates) >= len(unresolved_n_values):
        for n_value, payload in zip(unresolved_n_values, sequential_candidates[: len(unresolved_n_values)]):
            parsed_map[n_value] = {
                **payload,
                "format_compliant": False,
                "parse_status": f"batched_sequential_{payload['parsed_answer']}",
            }
        return parsed_map

    bare_mentions = [item.lower() for item in YES_NO_PATTERN.findall(raw_text)]
    if len(bare_mentions) >= len(unresolved_n_values):
        for n_value, answer in zip(unresolved_n_values, bare_mentions[: len(unresolved_n_values)]):
            parsed_map[n_value] = {
                "raw_text": raw_text,
                "parsed_answer": answer,
                "resolved": True,
                "format_compliant": False,
                "parse_status": f"batched_global_{answer}",
            }

    return parsed_map


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def safe_percent(numerator: int, denominator: int) -> float | None:
    rate = safe_rate(numerator, denominator)
    if rate is None:
        return None
    return rate * 100.0


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def d_prime(hit_count: int, same_count: int, false_alarm_count: int, different_count: int) -> float | None:
    if same_count <= 0 or different_count <= 0:
        return None
    hit_rate = (hit_count + 0.5) / (same_count + 1.0)
    false_alarm_rate = (false_alarm_count + 0.5) / (different_count + 1.0)
    normal = NormalDist()
    return normal.inv_cdf(hit_rate) - normal.inv_cdf(false_alarm_rate)


def evenly_spaced_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= 0 or target_count <= 0:
        return []
    if target_count >= total_count:
        return list(range(total_count))

    indices: list[int] = []
    for index in range(target_count):
        candidate = int((index + 0.5) * total_count / target_count)
        if candidate >= total_count:
            candidate = total_count - 1
        while candidate in indices and candidate + 1 < total_count:
            candidate += 1
        while candidate in indices and candidate - 1 >= 0:
            candidate -= 1
        if candidate not in indices:
            indices.append(candidate)
    return sorted(indices)


def _allocate_balanced_slots(available_counts: list[int], target_count: int) -> list[int]:
    selected_counts = [0 for _ in available_counts]
    remaining = min(sum(available_counts), target_count)
    while remaining > 0:
        eligible_domains = [domain_idx for domain_idx, available_count in enumerate(available_counts) if selected_counts[domain_idx] < available_count]
        if not eligible_domains:
            break
        if remaining >= len(eligible_domains):
            pick_domains = eligible_domains
        else:
            pick_domains = [eligible_domains[idx] for idx in evenly_spaced_indices(len(eligible_domains), remaining)]
        for domain_idx in pick_domains:
            selected_counts[domain_idx] += 1
            remaining -= 1
            if remaining <= 0:
                break
    return selected_counts


def extract_test_order_from_selection_manifest(selection_manifest: dict[str, Any] | None) -> list[tuple[int, int]]:
    if not isinstance(selection_manifest, dict):
        return []

    ordered_pairs: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for item in selection_manifest.get("ordered_valid_tests", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            pair = (int(item["domain_index"]), int(item["test_index"]))
        except Exception:
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        ordered_pairs.append(pair)

    if ordered_pairs:
        return ordered_pairs

    for item in selection_manifest.get("selected_tests", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            domain_index = int(item["domain_index"])
        except Exception:
            continue
        for test_index in item.get("selected_test_indices", []) or []:
            try:
                pair = (domain_index, int(test_index))
            except Exception:
                continue
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            ordered_pairs.append(pair)

    return ordered_pairs


def build_balanced_test_plan(
    data: list[dict[str, Any]],
    *,
    target_qa_count: int,
    questions_per_test: int,
    eligible_test_indices: dict[int, list[int]] | None = None,
) -> dict[str, Any]:
    total_domains = len(data)
    if eligible_test_indices is None:
        eligible_test_indices = {domain_idx: list(range(len(domain_data.get("tests", [])))) for domain_idx, domain_data in enumerate(data)}

    total_tests = sum(len(eligible_test_indices.get(domain_idx, [])) for domain_idx in range(total_domains))
    total_qas = total_tests * questions_per_test
    if target_qa_count <= 0 or target_qa_count >= total_qas:
        return {
            "target_qa_count": target_qa_count,
            "selected_test_count": total_tests,
            "selected_qa_count": total_qas,
            "questions_per_test": questions_per_test,
            "selection_strategy": "all_tests",
            "selected_test_indices": {domain_idx: list(eligible_test_indices.get(domain_idx, [])) for domain_idx in range(total_domains)},
        }

    selected_test_count = min(total_tests, target_qa_count // questions_per_test)
    selected_tests_per_domain = _allocate_balanced_slots(
        [len(eligible_test_indices.get(domain_idx, [])) for domain_idx in range(total_domains)],
        selected_test_count,
    )

    selected_test_indices: dict[int, list[int]] = {}
    for domain_idx in range(total_domains):
        available_indices = list(eligible_test_indices.get(domain_idx, []))
        take_count = selected_tests_per_domain[domain_idx]
        relative_indices = evenly_spaced_indices(len(available_indices), take_count)
        selected_test_indices[domain_idx] = [available_indices[idx] for idx in relative_indices]

    return {
        "target_qa_count": target_qa_count,
        "selected_test_count": sum(len(indices) for indices in selected_test_indices.values()),
        "selected_qa_count": sum(len(indices) for indices in selected_test_indices.values()) * questions_per_test,
        "questions_per_test": questions_per_test,
        "selection_strategy": "balanced_by_valid_domain_group",
        "selected_test_indices": selected_test_indices,
    }


def build_random_test_plan(
    data: list[dict[str, Any]],
    *,
    target_qa_count: int,
    questions_per_test: int,
    eligible_test_indices: dict[int, list[int]] | None = None,
    random_seed: int = 0,
    locked_prefix_tests: list[tuple[int, int]] | None = None,
    precomputed_test_order: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    total_domains = len(data)
    if eligible_test_indices is None:
        eligible_test_indices = {domain_idx: list(range(len(domain_data.get("tests", [])))) for domain_idx, domain_data in enumerate(data)}

    all_eligible_tests: list[tuple[int, int]] = []
    for domain_idx in range(total_domains):
        for test_idx in eligible_test_indices.get(domain_idx, []):
            all_eligible_tests.append((domain_idx, test_idx))

    eligible_set = set(all_eligible_tests)
    rng = random.Random(random_seed)

    ordered_valid_tests: list[tuple[int, int]] = []
    seen_tests: set[tuple[int, int]] = set()

    def _append_unique(candidate_pairs: list[tuple[int, int]]) -> None:
        for pair in candidate_pairs:
            if pair not in eligible_set or pair in seen_tests:
                continue
            seen_tests.add(pair)
            ordered_valid_tests.append(pair)

    strategy_suffix = "shuffled_full_order"
    if precomputed_test_order:
        _append_unique(precomputed_test_order)
        strategy_suffix = "reused_precomputed_order"
    elif locked_prefix_tests:
        _append_unique(locked_prefix_tests)
        strategy_suffix = f"locked_prefix_{len(ordered_valid_tests)}"

    remaining_tests = [pair for pair in all_eligible_tests if pair not in seen_tests]
    rng.shuffle(remaining_tests)
    ordered_valid_tests.extend(remaining_tests)

    total_tests = len(all_eligible_tests)
    total_qas = total_tests * questions_per_test
    selection_strategy = f"random_valid_tests_seed_{random_seed}_{strategy_suffix}"
    if target_qa_count <= 0 or target_qa_count >= total_qas:
        selected_test_indices: dict[int, list[int]] = {domain_idx: [] for domain_idx in range(total_domains)}
        for domain_idx, test_idx in ordered_valid_tests:
            selected_test_indices[domain_idx].append(test_idx)
        return {
            "target_qa_count": target_qa_count,
            "selected_test_count": total_tests,
            "selected_qa_count": total_qas,
            "questions_per_test": questions_per_test,
            "selection_strategy": selection_strategy if ordered_valid_tests else "all_tests",
            "selected_test_indices": selected_test_indices,
            "ordered_valid_tests": ordered_valid_tests,
            "selected_ordered_tests": ordered_valid_tests,
        }

    selected_test_count = min(total_tests, target_qa_count // questions_per_test)
    chosen_tests = ordered_valid_tests[:selected_test_count]

    selected_test_indices: dict[int, list[int]] = {domain_idx: [] for domain_idx in range(total_domains)}
    for domain_idx, test_idx in chosen_tests:
        selected_test_indices[domain_idx].append(test_idx)

    return {
        "target_qa_count": target_qa_count,
        "selected_test_count": selected_test_count,
        "selected_qa_count": selected_test_count * questions_per_test,
        "questions_per_test": questions_per_test,
        "selection_strategy": selection_strategy,
        "selected_test_indices": selected_test_indices,
        "ordered_valid_tests": ordered_valid_tests,
        "selected_ordered_tests": chosen_tests,
    }
