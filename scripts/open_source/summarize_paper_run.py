#!/usr/bin/env python3
"""Score paper groups using validated sample indices and N-back video sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    """Hash a canonical JSON value for an order-independent file representation."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def metric_group(row: dict[str, Any]) -> str:
    """Return the paper metric group, without relying on question text or bare id."""
    family = row["task_family"]
    if family == "memory_interference":
        return f"mi/{row['interference_direction']}"
    if family == "split_screen_dynamic_swap":
        return f"divided/swap{int(row['swap_count'])}/{row['storyline_error_type']}"
    if family == "interleaved_storyline":
        return f"interleaved/{row['storyline_error_type']}"
    if family == "source_memory":
        return f"source/{row['task_type']}"
    if family == "false_source_attribution":
        return "false_discrimination"
    raise ValueError(f"Unknown family: {family}")


def score_non_nback(stage_root: Path) -> list[dict[str, Any]]:
    """Pair each response with its frozen QA index and verify prompt and target."""
    scored = []
    for qa_path in sorted((stage_root / "qa_root").glob("*/*_qa.json")):
        task = qa_path.parent.name
        rows = json.loads(qa_path.read_text())
        if not rows:
            continue
        matches = list((stage_root / "eval").rglob(f"*_samples_v4_runtime_{task}.jsonl"))
        if len(matches) != 1:
            raise ValueError(f"Expected one sample file for {task}, found {len(matches)}")
        samples = [json.loads(line) for line in matches[0].read_text().splitlines()]
        indices = [s["doc_id"] for s in samples]
        if sorted(indices) != list(range(len(rows))):
            raise ValueError(f"Missing, repeated or invalid doc_id in {matches[0]}")
        for sample in samples:
            row = rows[sample["doc_id"]]
            prompt_sha = hashlib.sha256(sample["input"].encode()).hexdigest()
            if prompt_sha != row["_expected_prompt_sha256"] or sample["target"] != row["answer"]:
                raise ValueError(f"Response input or target mismatch: {task}/{sample['doc_id']}")
            if sample.get("v4_skipped_due_to_bad_video"):
                raise ValueError(f"Skipped video: {row['_reproduction_id']}")
            prediction = sample["v4_parsed_answer"]
            scored.append(
                {
                    "sample_id": row["_reproduction_id"],
                    "group": metric_group(row),
                    "target": row["answer"],
                    "prediction": prediction,
                    "correct": bool(sample["v4_answer_resolution_success"]) and prediction == row["answer"],
                    "prompt_sha256": prompt_sha,
                    "resolved": bool(sample["v4_answer_resolution_success"]),
                    "intrusion": row.get("option_role_by_letter", {}).get(prediction) == "intrusion",
                }
            )
    return scored


def score_nback(nback_root: Path) -> list[dict[str, Any]]:
    """Use each shard's sequence identity; domain_index is only local to a shard."""
    output = []
    seen: set[str] = set()
    model_root = nback_root / "model_runs/internvl3_5_8b"
    for split in ("action", "scene"):
        matched_results = {}
        for result_path in sorted((model_root / "shards").glob(f"{split}_shard*/results/results.json")):
            shard_name = result_path.parents[1].name
            selection = json.loads((nback_root / "shared_selection" / split / "shards" / f"{shard_name}.json").read_text())
            results = json.loads(result_path.read_text())
            for result in results:
                domain = selection[result["domain_index"]]
                test = domain["tests"][result["test_index"]]
                if domain["domain"] != result["domain"] or test["variation_id"] != result["variation_id"]:
                    raise ValueError(f"N-back shard mapping mismatch: {result['task_id']}")
                clips = test["video_path_sequence"]
                labels = test[result["label_key"]]
                k, n = result["clip_count"], result["n_value"]
                target = int(labels[k - 1] == labels[k - n - 1])
                if target != result["ground_truth"] or labels[k - 1] != result["target_label"]:
                    raise ValueError(f"N-back label mismatch: {result['task_id']}")
                sequence_hash = digest({"clips": clips, "labels": labels})
                sample_id = f"nback:{split}:{sequence_hash}:K{k}:N{n}"
                if sample_id in seen or result["task_id"] in matched_results:
                    raise ValueError(f"Duplicate N-back response: {sample_id}")
                seen.add(sample_id)
                matched_results[result["task_id"]] = (result["predicted_value"], result["ground_truth"], result["batch_question"])
                prediction = result["predicted_value"]
                output.append(
                    {
                        "sample_id": sample_id,
                        "group": f"nback/{split}",
                        "target": target,
                        "prediction": prediction,
                        "correct": bool(result["resolved_prediction"]) and prediction == target,
                        "resolved": bool(result["resolved_prediction"]),
                        "K": k,
                        "N": n,
                        "prompt_sha256": digest({"clips": clips[:k], "question": result["question"], "batch_question": result["batch_question"]}),
                    }
                )
        final = json.loads((model_root / "final" / split / "results.json").read_text())
        merged_results = {r["task_id"]: (r["predicted_value"], r["ground_truth"], r["batch_question"]) for r in final}
        if matched_results != merged_results or len(final) != len(matched_results):
            raise ValueError(f"Merged N-back results differ from the shards: {split}")
    return output


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report all questions in the denominator and separately report parse failures."""
    groups: dict[str, Any] = {}
    seen: set[str] = set()
    for row in rows:
        if row["sample_id"] in seen:
            raise ValueError(f"Duplicate sample identity: {row['sample_id']}")
        seen.add(row["sample_id"])
        bucket = groups.setdefault(row["group"], {"correct": 0, "total": 0, "unresolved": 0})
        bucket["correct"] += int(row["correct"])
        bucket["total"] += 1
        bucket["unresolved"] += int(not row["resolved"])
        if row["group"].startswith("mi/"):
            bucket["intrusions"] = bucket.get("intrusions", 0) + int(row["intrusion"])
    for bucket in groups.values():
        bucket["accuracy_percent"] = 100 * bucket["correct"] / bucket["total"]
        if "intrusions" in bucket:
            bucket["intrusion_percent"] = 100 * bucket["intrusions"] / bucket["total"]
    return groups


def write_paper_tables(report: dict[str, Any], output_root: Path) -> None:
    """Write the released paper metrics from identity-validated predictions."""
    groups = report["groups"]

    def value(group: str, metric: str = "accuracy_percent") -> float | None:
        return groups.get(group, {}).get(metric)

    def delta(a: float | None, b: float | None) -> float | None:
        return None if a is None or b is None else a - b

    divided = {"spatial_source_grounding": value("source/split_screen")}
    for name, error in (("source_identification", "source_confusion"), ("order_understanding", "order_disruption"), ("content_retention", "content_forgetting")):
        no_swap = value(f"divided/swap0/{error}")
        swap = value(f"divided/swap10/{error}")
        divided.update({f"no_swap_{name}": no_swap, f"swap_{name}": swap, f"swap_{name}_delta": delta(swap, no_swap)})
    memory = {}
    for label, metric in (("accuracy", "accuracy_percent"), ("intrusion_rate", "intrusion_percent")):
        proactive, retroactive = value("mi/proactive", metric), value("mi/retroactive", metric)
        memory.update({f"proactive_{label}": proactive, f"retroactive_{label}": retroactive, f"{label}_delta_proactive_minus_retroactive": delta(proactive, retroactive)})
    interleaved = {
        "source_identification": value("interleaved/source_confusion"),
        "order_understanding": value("interleaved/order_disruption"),
        "content_retention": value("interleaved/content_forgetting"),
        "temporal_source_grounding": value("source/interleaved"),
        "false_memory_discrimination": value("false_discrimination"),
    }
    nback_rows = [r for r in report["samples"] if r["group"].startswith("nback/")]

    def accuracy(rows: list[dict[str, Any]]) -> float | None:
        return 100 * sum(r["correct"] for r in rows) / len(rows) if rows else None

    nback = {"overall": accuracy(nback_rows), "action": value("nback/action"), "scene": value("nback/scene")}
    nback.update({f"n{n}": accuracy([r for r in nback_rows if r["N"] == n]) for n in range(1, 5)})
    breakdown = []
    for split in ("action", "scene"):
        for k in (2, 3, 4, 5, 6, 8, 10, 12):
            for n in range(1, min(k - 1, 4) + 1):
                rows = [r for r in nback_rows if r["group"] == f"nback/{split}" and r["K"] == k and r["N"] == n]
                breakdown.append({"split": split, "K": k, "N": n, "correct": sum(r["correct"] for r in rows), "total": len(rows), "accuracy_percent": accuracy(rows)})
    tables = {"table_1_divided_attention": divided, "table_2_memory_interference": memory, "table_3_interleaved_events": interleaved, "nback_summary": nback}
    payload = {"total_questions": report["total_questions"], "tables": tables, "nback_k_n_breakdown": breakdown}
    (output_root / "m3eval_paper_tables.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = ["# M³Eval paper tables", "", f"Evaluated questions: {report['total_questions']}. Values are percentages; N/A means this run did not cover that metric.", ""]
    for name, metrics in tables.items():
        text.extend([f"## {name}", "", "| Metric | Value (%) |", "|---|---:|"])
        for name, result in metrics.items():
            formatted = "N/A" if result is None else f"{result:.2f}"
            text.append(f"| {name} | {formatted} |")
        text.append("")
    (output_root / "m3eval_paper_tables.md").write_text("\n".join(text) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()
    records = []
    for stage in ("main", "supplement"):
        if (args.run_root / stage).is_dir():
            records.extend(score_non_nback(args.run_root / stage))
    if (args.run_root / "nback").is_dir():
        records.extend(score_nback(args.run_root / "nback"))
    if not records or (args.require_full and len(records) != 2403):
        raise ValueError(f"Unexpected number of scored questions: {len(records)}")
    profile_path = args.run_root / "run_profile.json"
    run_profile = json.loads(profile_path.read_text()) if profile_path.is_file() else {}
    report = {"total_questions": len(records), "groups": aggregate(records), "samples": records, "run_profile": run_profile}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_paper_tables(report, args.output.parent)
    print(json.dumps({"total_questions": len(records), "groups": report["groups"]}, indent=2))


if __name__ == "__main__":
    main()
