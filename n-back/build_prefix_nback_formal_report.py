#!/usr/bin/env python3
"""Build combined CSV/Markdown artifacts for prefix N-back formal runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

MODEL_ORDER = [
    "InternVL3.5-8B",
]

ATTRIBUTE_ORDER = [
    ("scene", "Scene"),
    ("action", "Action"),
]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_model_name(name: str) -> str:
    mapping = {
        "InternVL3_5-8B": "InternVL3.5-8B",
        "OpenGVLab/InternVL3_5-8B": "InternVL3.5-8B",
    }
    return mapping.get(str(name).strip(), str(name).strip())


def _format_cell(value: Any) -> str:
    if value in {None, ""}:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _sort_model_key(name: str) -> tuple[int, str]:
    if name in MODEL_ORDER:
        return (MODEL_ORDER.index(name), name)
    return (len(MODEL_ORDER), name)


def build_report(summary_paths: list[str | Path], *, output_dir: str | Path) -> dict[str, Any]:
    summaries = [_load_json(path) for path in summary_paths]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    by_clip_rows: list[dict[str, Any]] = []
    by_clip_n_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for summary_path, summary in zip(summary_paths, summaries):
        model_name = _canonical_model_name(summary.get("model_name", ""))
        attribute = str(summary.get("attribute", "")).strip().lower()
        if not model_name or attribute not in {"scene", "action"}:
            continue

        source_rows.append(
            {
                "summary_path": str(Path(summary_path).resolve()),
                "model_name": model_name,
                "attribute": attribute,
                "total_questions": int(summary.get("total_questions", 0) or 0),
            }
        )

        for row in summary.get("by_clip_count", []):
            by_clip_rows.append(
                {
                    "model_name": model_name,
                    "attribute": attribute,
                    "clip_count": int(row.get("clip_count", 0) or 0),
                    "count": int(row.get("count", 0) or 0),
                    "correct_count": int(row.get("correct_count", 0) or 0),
                    "accuracy_pct": row.get("accuracy_pct"),
                    "hit_rate_pct": row.get("hit_rate_pct"),
                    "false_alarm_rate_pct": row.get("false_alarm_rate_pct"),
                    "d_prime": row.get("d_prime"),
                    "resolved_rate_pct": row.get("resolved_rate_pct"),
                    "format_compliance_rate_pct": row.get("format_compliance_rate_pct"),
                }
            )

        for row in summary.get("by_clip_count_and_n_value", []):
            by_clip_n_rows.append(
                {
                    "model_name": model_name,
                    "attribute": attribute,
                    "clip_count": int(row.get("clip_count", 0) or 0),
                    "n_value": int(row.get("n_value", 0) or 0),
                    "count": int(row.get("count", 0) or 0),
                    "correct_count": int(row.get("correct_count", 0) or 0),
                    "accuracy_pct": row.get("accuracy_pct"),
                    "hit_rate_pct": row.get("hit_rate_pct"),
                    "false_alarm_rate_pct": row.get("false_alarm_rate_pct"),
                    "d_prime": row.get("d_prime"),
                    "resolved_rate_pct": row.get("resolved_rate_pct"),
                    "format_compliance_rate_pct": row.get("format_compliance_rate_pct"),
                }
            )

    by_clip_rows.sort(key=lambda row: (row["attribute"], _sort_model_key(row["model_name"]), row["clip_count"]))
    by_clip_n_rows.sort(key=lambda row: (row["attribute"], _sort_model_key(row["model_name"]), row["n_value"], row["clip_count"]))
    source_rows.sort(key=lambda row: (row["attribute"], _sort_model_key(row["model_name"])))

    by_clip_csv = output_dir / "combined_by_clip_count.csv"
    with by_clip_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_name",
            "attribute",
            "clip_count",
            "count",
            "correct_count",
            "accuracy_pct",
            "hit_rate_pct",
            "false_alarm_rate_pct",
            "d_prime",
            "resolved_rate_pct",
            "format_compliance_rate_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_clip_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    by_clip_n_csv = output_dir / "combined_by_clip_count_and_n.csv"
    with by_clip_n_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_name",
            "attribute",
            "clip_count",
            "n_value",
            "count",
            "correct_count",
            "accuracy_pct",
            "hit_rate_pct",
            "false_alarm_rate_pct",
            "d_prime",
            "resolved_rate_pct",
            "format_compliance_rate_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in by_clip_n_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    clip_counts = sorted({int(row["clip_count"]) for row in by_clip_rows})
    payload: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for row in by_clip_rows:
        payload.setdefault(row["attribute"], {}).setdefault(row["model_name"], {})[int(row["clip_count"])] = row

    md_lines = [
        "# Prefix N-back Formal Report",
        "",
        f"- Summary files: `{len(source_rows)}`",
        f"- Combined CSV: `{by_clip_csv}`",
        f"- Combined CSV (K x N): `{by_clip_n_csv}`",
        "",
    ]

    for attribute_key, attribute_title in ATTRIBUTE_ORDER:
        rows_by_model = payload.get(attribute_key, {})
        if not rows_by_model:
            continue

        md_lines.extend(
            [
                f"## {attribute_title}",
                "",
                "### Accuracy by Clip Count",
                "",
                "| Model | " + " | ".join(f"K={clip_count}" for clip_count in clip_counts) + " |",
                "| --- | " + " | ".join("---:" for _ in clip_counts) + " |",
            ]
        )
        for model_name in sorted(rows_by_model, key=_sort_model_key):
            cells = [_format_cell(rows_by_model[model_name].get(clip_count, {}).get("accuracy_pct")) for clip_count in clip_counts]
            md_lines.append(f"| {model_name} | " + " | ".join(cells) + " |")

        md_lines.extend(
            [
                "",
                "### False Alarm Rate by Clip Count",
                "",
                "| Model | " + " | ".join(f"K={clip_count}" for clip_count in clip_counts) + " |",
                "| --- | " + " | ".join("---:" for _ in clip_counts) + " |",
            ]
        )
        for model_name in sorted(rows_by_model, key=_sort_model_key):
            cells = [_format_cell(rows_by_model[model_name].get(clip_count, {}).get("false_alarm_rate_pct")) for clip_count in clip_counts]
            md_lines.append(f"| {model_name} | " + " | ".join(cells) + " |")

        if 2 in clip_counts and 12 in clip_counts:
            md_lines.extend(["", "### K=2 -> K=12 Change", ""])
            for model_name in sorted(rows_by_model, key=_sort_model_key):
                row_k2 = rows_by_model[model_name].get(2, {})
                row_k12 = rows_by_model[model_name].get(12, {})
                acc2 = row_k2.get("accuracy_pct")
                acc12 = row_k12.get("accuracy_pct")
                far2 = row_k2.get("false_alarm_rate_pct")
                far12 = row_k12.get("false_alarm_rate_pct")
                acc_delta = None if acc2 in {None, ""} or acc12 in {None, ""} else round(float(acc12) - float(acc2), 2)
                far_delta = None if far2 in {None, ""} or far12 in {None, ""} else round(float(far12) - float(far2), 2)
                md_lines.append(f"- `{model_name}`: Acc `{_format_cell(acc2)} -> {_format_cell(acc12)}` " f"(`{_format_cell(acc_delta)}`), FAR `{_format_cell(far2)} -> {_format_cell(far12)}` " f"(`{_format_cell(far_delta)}`)")
        md_lines.append("")

    analysis_md = output_dir / "analysis.md"
    analysis_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")

    report_payload = {
        "summary_count": len(source_rows),
        "summary_sources": source_rows,
        "clip_counts": clip_counts,
        "by_clip_count_csv": str(by_clip_csv),
        "by_clip_count_and_n_csv": str(by_clip_n_csv),
        "analysis_md": str(analysis_md),
    }
    report_json = output_dir / "report.json"
    report_json.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] by_clip_count_csv={by_clip_csv}")
    print(f"[DONE] by_clip_count_and_n_csv={by_clip_n_csv}")
    print(f"[DONE] analysis_md={analysis_md}")
    print(f"[DONE] report_json={report_json}")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined prefix N-back formal report")
    parser.add_argument("--summary", nargs="+", required=True, help="One or more summary.json files")
    parser.add_argument("--output_dir", required=True, help="Output directory for combined artifacts")
    args = parser.parse_args()
    build_report(args.summary, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
