#!/usr/bin/env python3
"""Analyze n-back action/scene evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from nback_eval_utils import (
    d_prime,
    parse_yes_no_response,
    round_or_none,
    safe_percent,
    safe_rate,
)


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Any) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    positive_rows = [row for row in rows if int(row["ground_truth"]) == 1]
    negative_rows = [row for row in rows if int(row["ground_truth"]) == 0]
    resolved_rows = [row for row in rows if row["resolved_prediction"]]
    format_rows = [row for row in rows if row["format_compliant"]]
    correct_rows = [row for row in rows if row["is_correct"]]

    hit_count = sum(1 for row in positive_rows if row["parsed_answer"] == "yes")
    false_alarm_count = sum(1 for row in negative_rows if row["parsed_answer"] == "yes")

    accuracy = safe_rate(len(correct_rows), total)
    hit_rate = safe_rate(hit_count, len(positive_rows))
    false_alarm_rate = safe_rate(false_alarm_count, len(negative_rows))
    d_prime_value = d_prime(hit_count, len(positive_rows), false_alarm_count, len(negative_rows))

    return {
        "count": total,
        "correct_count": len(correct_rows),
        "accuracy": round_or_none(accuracy, 6),
        "accuracy_pct": round_or_none(safe_percent(len(correct_rows), total), 2),
        "resolved_count": len(resolved_rows),
        "resolved_rate": round_or_none(safe_rate(len(resolved_rows), total), 6),
        "resolved_rate_pct": round_or_none(safe_percent(len(resolved_rows), total), 2),
        "format_compliance_count": len(format_rows),
        "format_compliance_rate": round_or_none(safe_rate(len(format_rows), total), 6),
        "format_compliance_rate_pct": round_or_none(safe_percent(len(format_rows), total), 2),
        "same_count": len(positive_rows),
        "different_count": len(negative_rows),
        "hit_count": hit_count,
        "false_alarm_count": false_alarm_count,
        "hit_rate": round_or_none(hit_rate, 6),
        "hit_rate_pct": round_or_none(safe_percent(hit_count, len(positive_rows)), 2),
        "false_alarm_rate": round_or_none(false_alarm_rate, 6),
        "false_alarm_rate_pct": round_or_none(safe_percent(false_alarm_count, len(negative_rows)), 2),
        "d_prime": round_or_none(d_prime_value, 2),
    }


def _hydrate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hydrated: list[dict[str, Any]] = []
    for row in results:
        parse_payload = {
            "parsed_answer": row.get("parsed_answer"),
            "resolved": row.get("resolved_prediction"),
            "format_compliant": row.get("format_compliant"),
            "parse_status": row.get("parse_status"),
        }
        if parse_payload["parsed_answer"] is None or parse_payload["resolved"] is None:
            parse_payload = parse_yes_no_response(row.get("model_response", ""))

        parsed_answer = str(parse_payload["parsed_answer"] or "").lower().strip()
        predicted_value = row.get("predicted_value")
        if predicted_value is None:
            predicted_value = 1 if parsed_answer == "yes" else 0 if parsed_answer == "no" else None

        is_correct = row.get("is_correct")
        if is_correct is None:
            is_correct = predicted_value is not None and int(predicted_value) == int(row.get("ground_truth", 0))

        hydrated.append(
            {
                **row,
                "parsed_answer": parsed_answer,
                "predicted_value": predicted_value,
                "resolved_prediction": bool(parse_payload["resolved"]),
                "format_compliant": bool(parse_payload["format_compliant"]),
                "parse_status": str(parse_payload["parse_status"] or ""),
                "is_correct": bool(is_correct),
            }
        )
    return hydrated


def analyze(results_path: str | Path, *, output_dir: str | Path | None = None) -> dict[str, Any]:
    results_path = Path(results_path).resolve()
    results = _hydrate_rows(_load_json(results_path))
    if not results:
        raise ValueError(f"Empty results file: {results_path}")

    run_dir = results_path.parent
    if output_dir is None:
        output_dir = run_dir / "analysis"
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = str(results[0].get("model_name", "")).strip() or run_dir.name
    label_key = str(results[0].get("label_key", "")).strip()
    attribute = "action" if label_key == "action_sequence" else "scene"
    selection_manifest_path = run_dir / "selection_manifest.json"
    selection_manifest = _load_json(selection_manifest_path) if selection_manifest_path.exists() else {}

    parse_status_counts = Counter(str(row.get("parse_status", "")) for row in results)
    overall = _metric_block(results)

    n_values = sorted({int(row.get("n_value", 0) or 0) for row in results})
    by_n_value = []
    for n_value in n_values:
        rows = [row for row in results if int(row.get("n_value", 0) or 0) == n_value]
        by_n_value.append({"n_value": n_value, **_metric_block(rows)})

    clip_count_values = sorted({int(row.get("clip_count", 0) or 0) for row in results if int(row.get("clip_count", 0) or 0) > 0})
    by_clip_count = []
    for clip_count in clip_count_values:
        rows = [row for row in results if int(row.get("clip_count", 0) or 0) == clip_count]
        by_clip_count.append({"clip_count": clip_count, **_metric_block(rows)})

    by_clip_count_and_n_value = []
    for clip_count in clip_count_values:
        for n_value in n_values:
            rows = [row for row in results if int(row.get("clip_count", 0) or 0) == clip_count and int(row.get("n_value", 0) or 0) == n_value]
            if not rows:
                continue
            by_clip_count_and_n_value.append({"clip_count": clip_count, "n_value": n_value, **_metric_block(rows)})

    domain_groups = sorted({(int(row.get("domain_index", -1)), str(row.get("domain", ""))) for row in results})
    by_domain_group = []
    for domain_index, domain_name in domain_groups:
        rows = [row for row in results if int(row.get("domain_index", -1)) == domain_index and str(row.get("domain", "")) == domain_name]
        by_domain_group.append({"domain_index": domain_index, "domain": domain_name, **_metric_block(rows)})

    paper_cells = {}
    for entry in by_n_value:
        paper_cells[f"N={entry['n_value']}"] = {
            "Acc": entry["accuracy_pct"],
            "HR": entry["hit_rate_pct"],
            "FAR": entry["false_alarm_rate_pct"],
            "d_prime": entry["d_prime"],
            "count": entry["count"],
        }

    summary = {
        "results_path": str(results_path),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "attribute": attribute,
        "label_key": label_key,
        "total_questions": len(results),
        "selection_manifest_path": str(selection_manifest_path) if selection_manifest_path.exists() else "",
        "selection_manifest": selection_manifest,
        "overall": overall,
        "by_n_value": by_n_value,
        "by_clip_count": by_clip_count,
        "by_clip_count_and_n_value": by_clip_count_and_n_value,
        "by_domain_group": by_domain_group,
        "parse_status_counts": dict(parse_status_counts),
        "paper_cells": paper_cells,
    }

    json_path = _write_json(output_dir / "summary.json", summary)

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "model_name",
                "domain_index",
                "domain",
                "test_index",
                "variation_id",
                "clip_count",
                "evaluation_mode",
                "current_clip_index",
                "previous_clip_index",
                "n_back_type",
                "n_value",
                "label_key",
                "target_label",
                "ground_truth",
                "model_response",
                "parsed_answer",
                "predicted_value",
                "resolved_prediction",
                "format_compliant",
                "parse_status",
                "is_correct",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})

    by_clip_csv_path = output_dir / "by_clip_count.csv"
    with by_clip_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "clip_count",
            "count",
            "correct_count",
            "accuracy",
            "accuracy_pct",
            "resolved_count",
            "resolved_rate",
            "resolved_rate_pct",
            "format_compliance_count",
            "format_compliance_rate",
            "format_compliance_rate_pct",
            "same_count",
            "different_count",
            "hit_count",
            "false_alarm_count",
            "hit_rate",
            "hit_rate_pct",
            "false_alarm_rate",
            "false_alarm_rate_pct",
            "d_prime",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_clip_count:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    by_clip_n_csv_path = output_dir / "by_clip_count_and_n.csv"
    with by_clip_n_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "clip_count",
            "n_value",
            "count",
            "correct_count",
            "accuracy",
            "accuracy_pct",
            "resolved_count",
            "resolved_rate",
            "resolved_rate_pct",
            "format_compliance_count",
            "format_compliance_rate",
            "format_compliance_rate_pct",
            "same_count",
            "different_count",
            "hit_count",
            "false_alarm_count",
            "hit_rate",
            "hit_rate_pct",
            "false_alarm_rate",
            "false_alarm_rate_pct",
            "d_prime",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_clip_count_and_n_value:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    md_lines = [
        f"# N-back Summary: {model_name} / {attribute}",
        "",
        f"- Results: `{results_path}`",
        f"- Total questions: `{len(results)}`",
        f"- Accuracy: `{overall['correct_count']}/{overall['count']}` = `{overall['accuracy_pct']}`",
        f"- Resolved prediction rate: `{overall['resolved_rate_pct']}`",
        f"- Format compliance rate: `{overall['format_compliance_rate_pct']}`",
        "",
        "## By N",
        "",
        "| N | Count | Acc | HR | FAR | d' |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entry in by_n_value:
        md_lines.append(f"| {entry['n_value']} | {entry['count']} | {entry['accuracy_pct']} | " f"{entry['hit_rate_pct']} | {entry['false_alarm_rate_pct']} | {entry['d_prime']} |")
    if len(by_clip_count) > 1:
        md_lines.extend(
            [
                "",
                "## By Clip Count",
                "",
                "| K | Count | Acc | HR | FAR | d' |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for entry in by_clip_count:
            md_lines.append(f"| {entry['clip_count']} | {entry['count']} | {entry['accuracy_pct']} | " f"{entry['hit_rate_pct']} | {entry['false_alarm_rate_pct']} | {entry['d_prime']} |")
    md_lines.extend(
        [
            "",
            "## Parse Status",
            "",
        ]
    )
    for key, value in sorted(parse_status_counts.items()):
        md_lines.append(f"- `{key}`: `{value}`")

    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[DONE] summary_json={json_path}")
    print(f"[DONE] summary_md={md_path}")
    print(f"[DONE] predictions_csv={csv_path}")
    print(f"[DONE] by_clip_count_csv={by_clip_csv_path}")
    print(f"[DONE] by_clip_count_and_n_csv={by_clip_n_csv_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze n-back evaluation outputs")
    parser.add_argument("--results", required=True, help="Path to results.json produced by test_N_back.py")
    parser.add_argument("--output_dir", default=None, help="Optional analysis output directory")
    args = parser.parse_args()
    analyze(args.results, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
