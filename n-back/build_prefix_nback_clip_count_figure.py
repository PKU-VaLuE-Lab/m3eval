#!/usr/bin/env python3
"""Build the prefix N-back clip-count sweep figure as a raster image."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

MODEL_ORDER = ["InternVL3.5-8B"]

MODEL_STYLES = {
    "InternVL3.5-8B": {"color": "#ff7f0e", "marker": "P", "linestyle": "-", "linewidth": 2.0, "markersize": 5.8, "zorder": 4},
}

ATTRIBUTE_ORDER = [
    ("scene", "Scene"),
    ("action", "Action"),
]

METRICS = [
    {"key": "accuracy_pct", "title": "Accuracy (%) ↑", "ylim": (20, 100), "yticks": [20, 40, 60, 80, 100]},
    {"key": "hit_rate_pct", "title": "Hit Rate (%) ↑", "ylim": (0, 100), "yticks": [0, 25, 50, 75, 100]},
    {"key": "false_alarm_rate_pct", "title": "False Alarm Rate (%) ↓", "ylim": (0, 100), "yticks": [0, 25, 50, 75, 100]},
    {"key": "d_prime", "title": "d' ↑", "ylim": (-1.0, 2.5), "yticks": [-1, 0, 1, 2]},
]

DPRIME_BREAK = 2.0
DPRIME_COMPRESS = 0.22
DPRIME_TICK_LABELS = [-1, 0, 1, 2, 3, 4]


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _canonical_model_name(name: str) -> str:
    mapping = {
        "InternVL3_5-8B": "InternVL3.5-8B",
        "OpenGVLab/InternVL3_5-8B": "InternVL3.5-8B",
    }
    return mapping.get(str(name).strip(), str(name).strip())


def _random_cells(clip_counts: list[int]) -> dict[int, dict[str, float]]:
    return {
        clip_count: {
            "accuracy_pct": 50.0,
            "hit_rate_pct": 50.0,
            "false_alarm_rate_pct": 50.0,
            "d_prime": 0.0,
        }
        for clip_count in clip_counts
    }


def _collect_payload(summary_paths: list[str | Path]) -> tuple[dict[str, dict[str, dict[int, dict[str, Any]]]], list[int]]:
    payload: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    clip_counts: set[int] = set()
    for summary_path in summary_paths:
        summary = _load_json(summary_path)
        model_name = _canonical_model_name(summary.get("model_name", ""))
        attribute = str(summary.get("attribute", "")).strip().lower()
        if not model_name or attribute not in {"scene", "action"}:
            continue
        for row in summary.get("by_clip_count", []):
            clip_count = int(row.get("clip_count", 0) or 0)
            if clip_count <= 0:
                continue
            clip_counts.add(clip_count)
            payload.setdefault(attribute, {}).setdefault(model_name, {})[clip_count] = dict(row)

    sorted_clip_counts = sorted(clip_counts)
    for attribute, _ in ATTRIBUTE_ORDER:
        payload.setdefault(attribute, {})["Random"] = _random_cells(sorted_clip_counts)
    return payload, sorted_clip_counts


def _finite_xy(x_values: list[int], y_values: list[float]) -> tuple[list[int], list[float]]:
    finite_pairs = [(x_value, y_value) for x_value, y_value in zip(x_values, y_values) if y_value is not None and not math.isnan(y_value)]
    return [x for x, _ in finite_pairs], [y for _, y in finite_pairs]


def _transform_dprime(value: float) -> float:
    if value <= DPRIME_BREAK:
        return value
    return DPRIME_BREAK + (value - DPRIME_BREAK) * DPRIME_COMPRESS


def _transform_metric_value(metric_key: str, value: float) -> float:
    if metric_key == "d_prime":
        return _transform_dprime(value)
    return value


def _axis_spec(metric: dict[str, Any]) -> tuple[tuple[float, float], list[float], list[str]]:
    if metric["key"] != "d_prime":
        return metric["ylim"], [float(tick) for tick in metric["yticks"]], [str(tick) for tick in metric["yticks"]]
    tick_positions = [_transform_dprime(float(tick)) for tick in DPRIME_TICK_LABELS]
    ylim = (-1.0, _transform_dprime(4.0) + 0.08)
    return ylim, tick_positions, [str(tick) for tick in DPRIME_TICK_LABELS]


def _draw_y_axis_break(ax: Any, *, y_value: float) -> None:
    ymin, ymax = ax.get_ylim()
    if ymax <= ymin:
        return
    ypos = (y_value - ymin) / (ymax - ymin)
    delta = 0.015
    kwargs = {"transform": ax.transAxes, "color": "#444444", "clip_on": False, "linewidth": 1.0}
    ax.plot((-delta, +delta), (ypos - delta, ypos + delta), **kwargs)
    ax.plot((-delta, +delta), (ypos + delta, ypos + 3 * delta), **kwargs)
    ax.plot((1 - delta, 1 + delta), (ypos - delta, ypos + delta), **kwargs)
    ax.plot((1 - delta, 1 + delta), (ypos + delta, ypos + 3 * delta), **kwargs)


def build_figure(summary_paths: list[str | Path], *, output_path: str | Path, dpi: int = 300) -> Path:
    payload, clip_counts = _collect_payload(summary_paths)
    if not clip_counts:
        raise ValueError("No by_clip_count rows found in the provided summaries")

    plt.style.use("default")
    fig, axes = plt.subplots(2, 4, figsize=(16.4, 6.8), constrained_layout=False)

    for row_idx, (attribute, row_title) in enumerate(ATTRIBUTE_ORDER):
        attribute_payload = payload.get(attribute, {})
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            for model_name in MODEL_ORDER:
                model_rows = attribute_payload.get(model_name, {})
                y_values = []
                for clip_count in clip_counts:
                    metric_value = model_rows.get(clip_count, {}).get(metric["key"])
                    y_values.append(_transform_metric_value(metric["key"], float(metric_value)) if metric_value not in {None, ""} else float("nan"))
                if all(value != value for value in y_values):
                    continue
                style = MODEL_STYLES[model_name]
                plot_x, plot_y = _finite_xy(clip_counts, y_values)
                if not plot_x:
                    continue
                ax.plot(
                    plot_x,
                    plot_y,
                    label=model_name,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=style["markersize"],
                    zorder=style["zorder"],
                    clip_on=False,
                )

            ax.set_title(metric["title"], fontsize=15, pad=6)
            ax.set_xlabel("Clip Count K", fontsize=13)
            ax.set_xlim(min(clip_counts) - 0.2, max(clip_counts) + 0.2)
            ax.set_xticks(clip_counts)
            ylim, yticks, yticklabels = _axis_spec(metric)
            ax.set_ylim(*ylim)
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.tick_params(axis="both", labelsize=11)
            ax.set_axisbelow(True)
            ax.grid(True, axis="both", color="#d9d9d9", alpha=0.45, linewidth=0.9, zorder=0)
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_color("#444444")
                spine.set_linewidth(1.0)
                spine.set_zorder(0.2)
            if metric["key"] == "d_prime":
                _draw_y_axis_break(ax, y_value=_transform_dprime(DPRIME_BREAK))

            if col_idx == 0:
                ax.text(
                    -0.34,
                    1.06,
                    row_title,
                    transform=ax.transAxes,
                    fontsize=17,
                    fontweight="bold",
                    va="center",
                )

    legend_items: dict[str, Any] = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_items.setdefault(label, handle)
    legend = fig.legend(
        list(legend_items.values()),
        list(legend_items.keys()),
        loc="lower center",
        ncol=min(8, max(1, len(legend_items))),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        fontsize=12,
        columnspacing=1.5,
        handlelength=2.5,
        handletextpad=0.6,
    )
    legend.get_frame().set_edgecolor("#d0d0d0")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_facecolor("white")

    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.18, wspace=0.24, hspace=0.32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prefix N-back clip-count figure as PNG")
    parser.add_argument("--summary", nargs="+", required=True, help="Summary JSON files from analyze_n_back_results.py")
    parser.add_argument("--output", default="paper/overleaf/Figs/nback_prefix_clip_count_curves.png", help="Output image path")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI")
    args = parser.parse_args()
    output_path = build_figure(args.summary, output_path=args.output, dpi=args.dpi)
    print(f"[DONE] wrote_figure={output_path}")


if __name__ == "__main__":
    main()
