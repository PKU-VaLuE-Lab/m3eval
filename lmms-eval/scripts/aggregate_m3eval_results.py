#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

TASKS = {
    "memory": "m3eval_memory_interference",
    "split": "m3eval_split_screen",
    "interleaved": "m3eval_interleaved",
    "nback": "m3eval_nback",
}

ALIASES = {
    "m3eval_divided_attention_accuracy": ["m3eval_split_screen_score"],
    "m3eval_divided_attention_spatial_source_grounding_accuracy": ["m3eval_split_screen_source_memory_spatial_score"],
    "m3eval_divided_attention_no_swap_source_identification_accuracy": ["m3eval_split_screen_swap0_source_confusion_score"],
    "m3eval_divided_attention_no_swap_order_understanding_accuracy": ["m3eval_split_screen_swap0_order_disruption_score"],
    "m3eval_divided_attention_no_swap_content_retention_accuracy": ["m3eval_split_screen_swap0_content_forgetting_score"],
    "m3eval_divided_attention_swap_source_identification_accuracy": ["m3eval_split_screen_swap10_source_confusion_score"],
    "m3eval_divided_attention_swap_order_understanding_accuracy": ["m3eval_split_screen_swap10_order_disruption_score"],
    "m3eval_divided_attention_swap_content_retention_accuracy": ["m3eval_split_screen_swap10_content_forgetting_score"],
    "m3eval_memory_interference_accuracy": ["m3eval_memory_interference_score"],
    "m3eval_memory_interference_proactive_accuracy": ["m3eval_memory_interference_proactive_score"],
    "m3eval_memory_interference_retroactive_accuracy": ["m3eval_memory_interference_retroactive_score"],
    "m3eval_interleaved_events_accuracy": ["m3eval_interleaved_score"],
    "m3eval_interleaved_events_temporal_source_grounding_accuracy": ["m3eval_interleaved_source_memory_narrative_score"],
    "m3eval_interleaved_events_false_memory_discrimination_accuracy": ["m3eval_interleaved_false_memory_narrative_score"],
    "m3eval_interleaved_events_source_identification_accuracy": ["m3eval_interleaved_source_confusion_score"],
    "m3eval_interleaved_events_order_understanding_accuracy": ["m3eval_interleaved_order_disruption_score"],
    "m3eval_interleaved_events_content_retention_accuracy": ["m3eval_interleaved_content_forgetting_score"],
    "m3eval_nback_accuracy": ["m3eval_nback_score"],
    "m3eval_nback_action_accuracy": ["m3eval_nback_action_score"],
    "m3eval_nback_scene_accuracy": ["m3eval_nback_scene_score"],
    "m3eval_nback_n1_accuracy": ["m3eval_nback_n1_score"],
    "m3eval_nback_n2_accuracy": ["m3eval_nback_n2_score"],
    "m3eval_nback_n3_accuracy": ["m3eval_nback_n3_score"],
    "m3eval_nback_n4_accuracy": ["m3eval_nback_n4_score"],
}


def resolve_result_file(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.rglob("*_results.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *_results.json found under {path}")
    return candidates[0]


def metric_value(results: dict[str, Any], task: str, metric: str) -> float | None:
    task_results = results.get("results", {}).get(task, {})
    for name in [metric, *ALIASES.get(metric, [])]:
        for key in (f"{name},none", name):
            value = task_results.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def pct(value: float | None) -> float | None:
    return None if value is None else round(value * 100.0, 4)


def delta(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else round((a - b) * 100.0, 4)


def fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def samples_from_jsonl(result_file: Path, task: str) -> list[dict[str, Any]]:
    pattern = f"*_samples_{task}.jsonl"
    samples = []
    for path in result_file.parent.glob(pattern):
        with path.open("r", encoding="utf-8") as f:
            samples.extend(json.loads(line) for line in f if line.strip())
    return samples


def nback_breakdown(results: dict[str, Any], result_file: Path) -> dict[str, Any]:
    samples = results.get("samples", {}).get(TASKS["nback"], [])
    if not samples:
        samples = samples_from_jsonl(result_file, TASKS["nback"])

    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        doc = sample.get("doc") or {}
        score = sample.get("m3eval_nback_accuracy")
        if score is None:
            score = sample.get("m3eval_nback_score")
        if not isinstance(score, bool):
            continue
        value = 1.0 if score else 0.0
        n_value = doc.get("n_value")
        clip_count = doc.get("clip_count") or doc.get("k_value")
        attribute = doc.get("attribute")
        if n_value is not None:
            grouped[f"N={n_value}"].append(value)
        if clip_count is not None:
            grouped[f"K={clip_count}"].append(value)
        if n_value is not None and clip_count is not None:
            grouped[f"K={clip_count},N={n_value}"].append(value)
        if attribute and n_value is not None:
            grouped[f"{attribute},N={n_value}"].append(value)

    return {key: pct(sum(values) / len(values)) for key, values in sorted(grouped.items()) if values}


def build_tables(results: dict[str, Any], result_file: Path) -> dict[str, Any]:
    split = TASKS["split"]
    memory = TASKS["memory"]
    interleaved = TASKS["interleaved"]
    nback = TASKS["nback"]

    da_spatial_source = metric_value(results, split, "m3eval_divided_attention_spatial_source_grounding_accuracy")
    da_no_source = metric_value(results, split, "m3eval_divided_attention_no_swap_source_identification_accuracy")
    da_no_order = metric_value(results, split, "m3eval_divided_attention_no_swap_order_understanding_accuracy")
    da_no_content = metric_value(results, split, "m3eval_divided_attention_no_swap_content_retention_accuracy")
    da_swap_source = metric_value(results, split, "m3eval_divided_attention_swap_source_identification_accuracy")
    da_swap_order = metric_value(results, split, "m3eval_divided_attention_swap_order_understanding_accuracy")
    da_swap_content = metric_value(results, split, "m3eval_divided_attention_swap_content_retention_accuracy")

    mi_pro_acc = metric_value(results, memory, "m3eval_memory_interference_proactive_accuracy")
    mi_ret_acc = metric_value(results, memory, "m3eval_memory_interference_retroactive_accuracy")
    mi_pro_intr = metric_value(results, memory, "m3eval_memory_interference_proactive_intrusion_rate")
    mi_ret_intr = metric_value(results, memory, "m3eval_memory_interference_retroactive_intrusion_rate")

    tables = {
        "table_1_divided_attention": {
            "no_swap_source_identification": pct(da_no_source),
            "no_swap_order_understanding": pct(da_no_order),
            "no_swap_content_retention": pct(da_no_content),
            "swap_source_identification": pct(da_swap_source),
            "swap_source_identification_delta": delta(da_swap_source, da_no_source),
            "swap_order_understanding": pct(da_swap_order),
            "swap_order_understanding_delta": delta(da_swap_order, da_no_order),
            "swap_content_retention": pct(da_swap_content),
            "swap_content_retention_delta": delta(da_swap_content, da_no_content),
            "spatial_source_grounding": pct(da_spatial_source),
        },
        "table_2_memory_interference": {
            "proactive_accuracy": pct(mi_pro_acc),
            "retroactive_accuracy": pct(mi_ret_acc),
            "accuracy_delta_proactive_minus_retroactive": delta(mi_pro_acc, mi_ret_acc),
            "proactive_intrusion_rate": pct(mi_pro_intr),
            "retroactive_intrusion_rate": pct(mi_ret_intr),
            "intrusion_delta_proactive_minus_retroactive": delta(mi_pro_intr, mi_ret_intr),
        },
        "table_3_interleaved_events": {
            "source_identification": pct(metric_value(results, interleaved, "m3eval_interleaved_events_source_identification_accuracy")),
            "order_understanding": pct(metric_value(results, interleaved, "m3eval_interleaved_events_order_understanding_accuracy")),
            "content_retention": pct(metric_value(results, interleaved, "m3eval_interleaved_events_content_retention_accuracy")),
            "false_memory_discrimination": pct(metric_value(results, interleaved, "m3eval_interleaved_events_false_memory_discrimination_accuracy")),
            "temporal_source_grounding": pct(metric_value(results, interleaved, "m3eval_interleaved_events_temporal_source_grounding_accuracy")),
        },
        "nback_summary": {
            "overall": pct(metric_value(results, nback, "m3eval_nback_accuracy")),
            "action": pct(metric_value(results, nback, "m3eval_nback_action_accuracy")),
            "scene": pct(metric_value(results, nback, "m3eval_nback_scene_accuracy")),
            "n1": pct(metric_value(results, nback, "m3eval_nback_n1_accuracy")),
            "n2": pct(metric_value(results, nback, "m3eval_nback_n2_accuracy")),
            "n3": pct(metric_value(results, nback, "m3eval_nback_n3_accuracy")),
            "n4": pct(metric_value(results, nback, "m3eval_nback_n4_accuracy")),
        },
    }
    breakdown = nback_breakdown(results, result_file)
    if breakdown:
        tables["nback_k_n_breakdown"] = breakdown
    return tables


def markdown_table(title: str, values: dict[str, Any]) -> str:
    headers = list(values.keys())
    row = [fmt(values[key]) if isinstance(values[key], (int, float)) or values[key] is None else str(values[key]) for key in headers]
    return "\n".join([
        f"### {title}",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(row) + " |",
        "",
    ])


def write_outputs(result_file: Path, payload: dict[str, Any], output_dir: Path | None) -> tuple[Path, Path]:
    out_dir = output_dir or result_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "m3eval_paper_tables.json"
    md_path = out_dir / "m3eval_paper_tables.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    parts = [f"# M3Eval Paper Tables\n", f"Source results: `{payload['source_results']}`\n"]
    for key, values in payload["tables"].items():
        parts.append(markdown_table(key, values))
    md_path.write_text("\n".join(parts), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate M3Eval lmms-eval results into paper-facing tables.")
    parser.add_argument("result_path", type=Path, help="A *_results.json file or a directory containing one.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for m3eval_paper_tables.json/.md. Defaults to the result file directory.")
    args = parser.parse_args()

    result_file = resolve_result_file(args.result_path)
    results = json.loads(result_file.read_text(encoding="utf-8"))
    model = results.get("model_name") or results.get("config", {}).get("model") or result_file.parent.name
    payload = {
        "model": model,
        "source_results": str(result_file),
        "unit": "percent",
        "tables": build_tables(results, result_file),
    }
    json_path, md_path = write_outputs(result_file, payload, args.output_dir)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
