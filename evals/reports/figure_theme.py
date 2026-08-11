"""
Shared chart theme and metric primitives for eval figures.

Holds what every eval figure needs regardless of which module draws it: the
palette, axes styling, SVG writing, and the metric spec (margin normalisation,
grader-score collection, unit-aware formatting).

The palette is the validated data-viz default, checked with the dataviz skill's
scripts/validate_palette.js in both modes:

    light  #2a78d6,#eb6834,#1baf7a  → all checks pass
                                      (WARN: aqua contrast 2.74 vs surface,
                                       relieved by direct labels on aqua marks)
    dark   #3987e5,#d95926,#199e70  → all checks pass, all ≥3:1

Categorical hues are assigned in fixed order and never cycled — a fourth series
folds into small multiples rather than inventing a hue. Status colors are
reserved for pass/fail state and never reused as a series color.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from src.agents.state import EvalReport

# ── Categorical (fixed order, never cycled) ───────────────────────────────────
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
CATEGORICAL = (BLUE, ORANGE, AQUA)

# ── Status (reserved — never a series color) ──────────────────────────────────
GOOD = "#1baf7a"
WARNING = "#d99a1b"
CRITICAL = "#d64545"

# ── Ink & surface ─────────────────────────────────────────────────────────────
INK = "#1c1b1a"  # primary text
SECONDARY = "#4a4744"  # labels, axis titles
MUTED = "#7d7973"  # tick labels, de-emphasised history
AXIS = "#b8b4ae"  # baselines, threshold rules
GRID = "#e6e3de"  # solid hairline grid — never dashed
SURFACE = "#ffffff"  # chart surface; also the 2px gap/ring color

OUT_DIR = Path("evals/reports/output/figures")


def apply_theme(ax: Axes, *, grid_axis: str = "y") -> None:
    """
    Style one axes: recessive grid and spines, ink-token text.

    grid_axis: "y" (default), "x", "both", or "none".
    """
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)

    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)

    if grid_axis == "none":
        ax.grid(False)
    else:
        # Solid hairline behind the marks — never dashed, never in front.
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=1, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    """Write an SVG under OUT_DIR (optionally a subdir) and close the figure."""
    out = OUT_DIR / subdir if subdir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


# ── Metric spec ───────────────────────────────────────────────────────────────

# Thresholds mirror evals/metrics/constants.py tier3 defaults; the report carries
# per-grader thresholds, which take precedence when present.
MAX_MARGIN = 1.0  # clamp normalised margin so one outlier can't flatten the rest

# Metrics whose values are percentages, so _fmt suffixes them rather than
# printing three decimals of a number that is already scaled 0–100.
PERCENT_METRICS = frozenset({"SMAPE", "DirectionalAccuracy", "Coverage80", "IntervalWidth"})


def margin(value: float, threshold: float, lower_is_better: bool) -> float:
    """
    Signed, normalised distance from the pass threshold. Positive is always
    better regardless of the metric's direction, so incommensurable metrics
    (MASE ~0.8, Coverage ~79%) become comparable on one axis. Zero is the gate.
    """
    if threshold == 0 or not np.isfinite(value) or not np.isfinite(threshold):
        return float("nan")
    raw = (threshold - value) / abs(threshold)
    if not lower_is_better:
        raw = -raw
    return float(np.clip(raw, -MAX_MARGIN, MAX_MARGIN))


def collect_metrics(report: EvalReport) -> list[dict]:
    """
    Pull metric name/value/threshold/direction from the report's grader scores,
    averaging across series. Falls back to the report's scalar fields when no
    per-series scores are present.
    """
    by_name: dict[str, dict] = {}
    for scores in report.series_scores.values():
        for s in scores:
            slot = by_name.setdefault(
                s.grader_name,
                {
                    "name": s.grader_name,
                    "values": [],
                    "threshold": s.threshold,
                    "lower_is_better": s.lower_is_better,
                },
            )
            slot["values"].append(s.metric_value)

    metrics = []
    for slot in by_name.values():
        value = float(np.mean(slot["values"])) if slot["values"] else float("nan")
        metrics.append(
            {
                "name": slot["name"],
                "value": value,
                "threshold": slot["threshold"],
                "lower_is_better": slot["lower_is_better"],
                "margin": margin(value, slot["threshold"], slot["lower_is_better"]),
            }
        )
    return metrics


def fmt_metric(name: str, value: float) -> str:
    """Format a metric value with the unit its name implies."""
    if not np.isfinite(value):
        return "n/a"
    if name in PERCENT_METRICS:
        return f"{value:.1f}%"
    return f"{value:.3f}"


__all__ = [
    "AQUA",
    "AXIS",
    "BLUE",
    "CATEGORICAL",
    "CRITICAL",
    "GOOD",
    "GRID",
    "INK",
    "MAX_MARGIN",
    "MUTED",
    "ORANGE",
    "OUT_DIR",
    "PERCENT_METRICS",
    "SECONDARY",
    "SURFACE",
    "WARNING",
    "apply_theme",
    "collect_metrics",
    "fmt_metric",
    "margin",
    "save_figure",
]
