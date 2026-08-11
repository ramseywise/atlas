"""
Atlas eval figures — the cross-sectional and cross-cycle views.

Where evals/reports/eval_figures.py answers "how did this cycle do", these two
answer "who and when":

    fig_series_heatmap  — which series is dragging the cycle's mean?
    fig_cycle_history   — how volatile is the model across cycles?

Both take the whole population (all series, or all cycles) rather than a single
forecast, which is why they live apart from the single-cycle diagnostics. Shared
theme and metric primitives come from evals/reports/figure_theme.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from evals.reports.figure_theme import (
    AXIS,
    BLUE,
    CRITICAL,
    INK,
    MAX_MARGIN,
    MUTED,
    SECONDARY,
    SURFACE,
    apply_theme,
    fmt_metric,
    margin,
    save_figure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.agents.state import EvalReport

__all__ = ["fig_cycle_history", "fig_series_heatmap"]


# ── Series × metric heatmap ───────────────────────────────────────────────────


def fig_series_heatmap(report: EvalReport, *, subdir: str = "") -> Path:
    """
    Series × metric grid of margin-to-threshold. Exposes compensating averages:
    the harness reports one scalar mean per metric, so a series at 95% coverage
    and one at 63% average to a healthy-looking 79%.

    Diverging blue↔red on a neutral midpoint — the midpoint is the pass gate, so
    "nothing" genuinely means "exactly at threshold".
    """
    series_ids = sorted(report.series_scores)
    if not series_ids:
        msg = "fig_series_heatmap requires at least one scored series"
        raise ValueError(msg)

    # Drift is cycle-level — identical in every row, so it would be a column of
    # the same value. It belongs on the margin chart, not here.
    metric_names: list[str] = []
    for sid in series_ids:
        for s in report.series_scores[sid]:
            if s.grader_name != "DriftDetection" and s.grader_name not in metric_names:
                metric_names.append(s.grader_name)

    grid = np.full((len(series_ids), len(metric_names)), np.nan)
    raw = np.full_like(grid, np.nan)
    for i, sid in enumerate(series_ids):
        for s in report.series_scores[sid]:
            if s.grader_name not in metric_names:
                continue
            j = metric_names.index(s.grader_name)
            grid[i, j] = margin(s.metric_value, s.threshold, s.lower_is_better)
            raw[i, j] = s.metric_value

    # Diverging: red (failing) → neutral (at gate) → blue (passing).
    cmap = LinearSegmentedColormap.from_list("atlas_div", [CRITICAL, "#f0efec", BLUE], N=256)
    # Width budget: cells + y-labels + colorbar. The colorbar lives outside the
    # axes, so it must be paid for here — tight bbox cannot reclaim an overflow.
    fig, ax = plt.subplots(figsize=(1.75 * len(metric_names) + 5.0, 0.72 * len(series_ids) + 2.6))
    apply_theme(ax, grid_axis="none")
    im = ax.imshow(grid, cmap=cmap, vmin=-MAX_MARGIN, vmax=MAX_MARGIN, aspect="auto")

    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=9, color=INK, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(series_ids)))
    ax.set_yticklabels(series_ids, fontsize=9, color=INK)

    # 2px surface gap between cells — the spacer, not a border on each mark.
    ax.set_xticks(np.arange(len(metric_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(series_ids) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    # Every cell carries its real value — the color is the scan layer, the text
    # is the read layer, so nothing is gated behind hue.
    for i in range(len(series_ids)):
        for j in range(len(metric_names)):
            if not np.isfinite(raw[i, j]):
                continue
            shade = grid[i, j]
            txt_color = "#ffffff" if abs(shade) > 0.55 else INK
            ax.text(
                j,
                i,
                fmt_metric(metric_names[j], raw[i, j]),
                ha="center",
                va="center",
                fontsize=8.5,
                color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Margin vs threshold", color=SECONDARY, fontsize=9)
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.outline.set_visible(False)

    ax.set_title(
        f"Per-series margin — {report.cycle_id}",
        fontsize=12,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    fig.tight_layout()
    return save_figure(fig, f"series_heatmap_{report.cycle_id}", subdir)


# ── Cycle history small multiples ─────────────────────────────────────────────


def fig_cycle_history(reports: list[EvalReport], *, subdir: str = "") -> Path:
    """
    Small multiples: one panel per metric across eval cycles, each with its own
    y-scale and its own threshold line.

    This is the volatility view. A single snapshot showed PASS while the drift
    log held a cycle at directional accuracy 32% — worse than a coin flip. Small
    multiples put every cycle on screen at once, so spread is visible without
    interaction.
    """
    if not reports:
        msg = "fig_cycle_history requires at least one EvalReport"
        raise ValueError(msg)

    specs = [
        ("MASE", [r.overall_mase for r in reports], 1.0, True),
        ("SMAPE %", [r.overall_smape for r in reports], 15.0, True),
        ("Directional %", [r.directional_accuracy for r in reports], 55.0, False),
        ("Coverage 80 %", [r.coverage_80 for r in reports], 75.0, False),
        ("Interval width %", [r.interval_width for r in reports], 40.0, True),
        ("Drift ratio", [r.drift_ratio for r in reports], 1.2, True),
    ]
    # Drop metrics with no finite data (e.g. width on pre-upgrade reports).
    specs = [s for s in specs if np.any(np.isfinite(np.asarray(s[1], dtype=float)))]

    ncols = 3
    nrows = (len(specs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.7 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    x = np.arange(len(reports))
    labels = [str(r.forecast_date) for r in reports]

    for ax, (name, values, threshold, lower_better) in zip(axes_flat, specs, strict=False):
        apply_theme(ax)
        vals = np.asarray(values, dtype=float)
        ax.plot(x, vals, color=BLUE, linewidth=2, zorder=5)
        # Points that fail their gate wear the status color — direction-aware.
        failing = vals > threshold if lower_better else vals < threshold
        ax.scatter(x, vals, s=32, color=BLUE, zorder=6, edgecolors=SURFACE, linewidths=2)
        if np.any(failing):
            ax.scatter(
                x[failing],
                vals[failing],
                s=42,
                color=CRITICAL,
                zorder=7,
                edgecolors=SURFACE,
                linewidths=2,
            )
        ax.axhline(threshold, color=AXIS, linewidth=1.2, zorder=3)
        # Autoscale sees only the data, so a gate outside the data range lands on
        # the frame and reads as a border. Fit the axis to data ∪ threshold, then
        # pad, so the gate is always a visible line inside the panel.
        span = [*vals.tolist(), threshold]
        lo_v, hi_v = min(span), max(span)
        pad = max((hi_v - lo_v) * 0.18, abs(hi_v) * 0.02, 1e-6)
        ax.set_ylim(lo_v - pad, hi_v + pad)
        ax.set_title(name, fontsize=10, color=INK, fontweight="bold", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, color=MUTED, rotation=45, ha="right")
        ax.annotate(
            f"{vals[-1]:.2f}",
            (x[-1], vals[-1]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color=SECONDARY,
        )

    for ax in axes_flat[len(specs) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Eval history — {len(reports)} cycles",
        fontsize=12,
        color=INK,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    fig.tight_layout(h_pad=2.2, w_pad=2.0, rect=(0, 0, 1, 0.97))
    return save_figure(fig, "cycle_history", subdir)
