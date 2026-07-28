#!/usr/bin/env python3
"""Plot nano-vLLM benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RATE_OUTPUT = RESULTS / "nano-vllm-rate-2.png"
THROUGHPUT_OUTPUT = RESULTS / "nano-vllm-max-throughput.png"

BLUE = "#1F77B4"
ORANGE = "#FF7F0E"
SERIES = (
    ("nano-vLLM", "nano-vllm", BLUE),
    ("nano-vLLM-v1", "nano-vllm-v1", ORANGE),
)


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def style_axis(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#A0A0A0")
    ax.spines["bottom"].set_color("#A0A0A0")
    ax.tick_params(colors="#555555")


def save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(path)


def plot_rate_limited(data: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    percentiles = ["p50", "p90", "p99"]
    x = list(range(len(percentiles)))
    width = 0.34
    metric_panels = (
        (axes[0], "ttft_ms", "TTFT", "Time to first token (ms)"),
        (axes[1], "tpot_ms", "TPOT", "Time per output token (ms)"),
        (axes[2], "itl_ms", "ITL", "Inter-token latency (ms)"),
    )

    for ax, metric, title, ylabel in metric_panels:
        for index, (label, key, color) in enumerate(SERIES):
            offset = (index - 0.5) * width
            positions = [value + offset for value in x]
            values = [
                data[key]["rate-2"][metric][percentile]
                for percentile in percentiles
            ]
            ax.bar(
                positions,
                values,
                width=width,
                label=label,
                color=color,
            )
        ax.set_title(f"{title} · smaller is better ↓")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, percentiles)
        ax.set_ylim(bottom=0)
        style_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.88), w_pad=2.2)
    save(fig, RATE_OUTPUT)


def plot_max_throughput(data: dict[str, dict[str, Any]]) -> None:
    labels = [item[0] for item in SERIES]
    colors = [item[2] for item in SERIES]
    values = [
        data[key]["saturated"]["output_tokens_per_second"]
        for _, key, _ in SERIES
    ]

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    bars = ax.bar(labels, values, width=0.56, color=colors)
    ax.set_title("Maximum output throughput · higher is better ↑")
    ax.set_ylabel("Output tokens/s")
    ax.set_ylim(0, max(values) * 1.18)
    style_axis(ax)

    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            color="#333333",
            fontsize=10,
        )

    fig.tight_layout()
    save(fig, THROUGHPUT_OUTPUT)


def main() -> None:
    data = {
        "nano-vllm": {
            "rate-2": load("nano-vllm-rate-2.json"),
            "saturated": load("nano-vllm-rate-inf.json"),
        },
        "nano-vllm-v1": {
            "rate-2": load("nano-vllm-v1-rate-2.json"),
            "saturated": load("nano-vllm-v1-rate-inf.json"),
        },
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    plot_rate_limited(data)
    plot_max_throughput(data)


if __name__ == "__main__":
    main()
