"""
Atlas eval figures — the single-cycle metric diagnostics.

Replaces the two original figures (`fig_grader_pass_rates`, `fig_forecast_grid`
in evals/reports/figures.py), which shared a single linear y-axis across
incommensurable metrics and never plotted holdout actuals over the forecast
window. Each function here answers one diagnostic question about one cycle:

    fig_margin_to_threshold  — is each metric passing, and by how much?
    fig_forecast_panels      — where in the window was the model wrong?
    fig_error_by_horizon     — does accuracy decay with steps-ahead?

The population views — per-series breakdown and cross-cycle history — live in
evals/reports/history_figures.py. Shared theme and metric primitives (margin
normalisation, grader-score collection, formatting) are in figure_theme.py.

Palette is the validated data-viz default (3 categorical slots, blue/orange/aqua),
checked with scripts/validate_palette.js in both light and dark modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.reports.figure_theme import (
    AXIS,
    BLUE,
    CRITICAL,
    GOOD,
    GRID,
    INK,
    MAX_MARGIN,
    MUTED,
    ORANGE,
    SECONDARY,
    SURFACE,
    apply_theme,
    collect_metrics,
    fmt_metric,
    save_figure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.agents.state import EvalReport, ForecastResult

BAR_HEIGHT = 0.62
LABEL_PAD = 0.04
MIN_HISTORY_STEPS = 20  # floor on trailing history drawn beside a forecast window


# ── 1. Margin to threshold ────────────────────────────────────────────────────


def fig_margin_to_threshold(report: EvalReport, *, subdir: str = "") -> Path:
    """
    Diverging bars: signed distance from each metric's pass threshold, normalised
    so all metrics share one axis and positive always means "better than gate".

    This replaces the original shared-linear-axis bar chart, where MASE (~0.8)
    and drift (~1.05) were invisible slivers beside percentages, and a dashed
    line connected thresholds across unrelated metrics.
    """
    metrics = collect_metrics(report)
    if not metrics:
        metrics = []
    # Worst margin first — the metric closest to (or past) failing leads.
    metrics.sort(key=lambda m: (np.isnan(m["margin"]), m["margin"]))

    labels = [m["name"] for m in metrics]
    margins = [0.0 if np.isnan(m["margin"]) else m["margin"] for m in metrics]
    colors = [GOOD if m >= 0 else CRITICAL for m in margins]

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(metrics) + 1.9))
    apply_theme(ax)
    y = np.arange(len(labels))

    ax.barh(y, margins, height=BAR_HEIGHT, color=colors, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1.4, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Margin vs threshold  (0 = gate · right = passing)", color=SECONDARY)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}" if v else "gate")
    ax.set_xlim(-MAX_MARGIN * 1.35, MAX_MARGIN * 1.35)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.yaxis.grid(False)

    # Direct-label every bar with the real value + threshold: the normalised axis
    # answers "is it passing", the label answers "what is it".
    #
    # Passing bars label outside the bar end; failing bars label *inside* in white.
    # A failing metric can clamp to -100%, where an outside label would overprint
    # the y-axis tick text.
    for yi, m, margin in zip(y, metrics, margins, strict=False):
        text = (
            f"{fmt_metric(m['name'], m['value'])}  "
            f"({'≤' if m['lower_is_better'] else '≥'} {fmt_metric(m['name'], m['threshold'])})"
        )
        if margin >= 0:
            ax.text(
                margin + LABEL_PAD,
                yi,
                text,
                va="center",
                ha="left",
                fontsize=9,
                color=SECONDARY,
            )
        else:
            ax.text(
                margin + LABEL_PAD,
                yi,
                text,
                va="center",
                ha="left",
                fontsize=9,
                color=SURFACE,
                fontweight="bold",
                zorder=5,
            )

    status = "PASS" if report.all_passed else "FAIL"
    ax.set_title(
        f"Margin to threshold — {report.cycle_id}  [{status}]",
        fontsize=12,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    return save_figure(fig, f"margin_to_threshold_{report.cycle_id}", subdir)


# ── 2. Forecast panels ────────────────────────────────────────────────────────


def fig_forecast_panels(
    results: list[ForecastResult],
    actuals_map: dict[str, np.ndarray] | None = None,
    history_map: dict[str, np.ndarray] | None = None,
    *,
    dates_map: dict[str, list] | None = None,
    subdir: str = "",
) -> Path:
    """
    One row per series: history, then the forecast window with holdout actuals
    overlaid and miss regions shaded.

    The original grid drew only pre-cutoff history, so the eval scored against
    actuals the reader could never see. It also clipped its right column via
    figsize/tight_layout and used bare step indices on the x-axis. Here each
    series gets a full-width row, actuals continue through the window, and the
    band between forecast and actual is shaded where the actual falls outside
    the 80% PI — those shaded regions are exactly the coverage misses.
    """
    if not results:
        msg = "fig_forecast_panels requires at least one ForecastResult"
        raise ValueError(msg)

    actuals_map = actuals_map or {}
    history_map = history_map or {}
    dates_map = dates_map or {}
    n = len(results)
    # History is context, not the subject. Showing all of it (often 90+ steps)
    # squeezes the forecast window — the part being judged — into a sliver.
    hist_tail = max(2 * max(len(r.point_forecast) for r in results), MIN_HISTORY_STEPS)

    fig, axes = plt.subplots(n, 1, figsize=(11, 2.9 * n), squeeze=False)
    axes_flat = axes.flatten()

    for ax, r in zip(axes_flat, results, strict=False):
        apply_theme(ax)
        hist = np.asarray(history_map.get(r.series_id, []), dtype=float)[-hist_tail:]
        act = np.asarray(actuals_map.get(r.series_id, []), dtype=float)
        pt = np.asarray(r.point_forecast, dtype=float)
        lo = np.asarray(r.lower_80, dtype=float)
        hi = np.asarray(r.upper_80, dtype=float)

        x_fc = np.arange(len(pt))
        if len(hist):
            x_hist = np.arange(-len(hist), 0)
            ax.plot(x_hist, hist, color=MUTED, linewidth=1.6, zorder=3)
            # Join history to the forecast so the line reads continuous.
            ax.plot([x_hist[-1], 0], [hist[-1], pt[0]], color=MUTED, linewidth=1.6, zorder=3)
            ax.axvline(-0.5, color=AXIS, linewidth=1.2, linestyle="-", zorder=2)

        ax.fill_between(x_fc, lo, hi, color=BLUE, alpha=0.12, zorder=2, label="80% PI", linewidth=0)
        ax.plot(x_fc, pt, color=BLUE, linewidth=2, zorder=5, label="Forecast")

        if len(act):
            m = min(len(act), len(pt))
            xa, aa = x_fc[:m], act[:m]
            # Shade only where the actual escaped the interval — the misses.
            outside = (aa < lo[:m]) | (aa > hi[:m])
            ax.fill_between(
                xa,
                aa,
                np.clip(aa, lo[:m], hi[:m]),
                where=outside,
                color=CRITICAL,
                alpha=0.22,
                zorder=4,
                linewidth=0,
                interpolate=True,
            )
            ax.plot(xa, aa, color=ORANGE, linewidth=2, zorder=6, label="Actual")
            n_miss = int(np.sum(outside))
            if n_miss:
                ax.scatter(
                    xa[outside],
                    aa[outside],
                    s=34,
                    color=CRITICAL,
                    zorder=7,
                    edgecolors=SURFACE,
                    linewidths=2,
                )
            cov = 100.0 * (1 - n_miss / m)
            ax.text(
                0.995,
                0.06,
                f"{n_miss} outside 80% PI · coverage {cov:.0f}%",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                color=SECONDARY,
            )

        ax.set_title(
            f"{r.series_id}  ·  {r.model_used.value}  ·  horizon {r.horizon.value}",
            fontsize=10,
            color=INK,
            fontweight="bold",
            loc="left",
            pad=8,
        )
    # One x-label and one legend for the figure — the axis means the same thing
    # in every panel, and repeating the identity mapping n times is noise.
    axes_flat[-1].set_xlabel("Steps ahead  (negative = history)", color=SECONDARY, fontsize=9)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=8,
            frameon=False,
            loc="upper right",
            ncol=3,
            bbox_to_anchor=(1.0, 1.0),
            labelcolor=SECONDARY,
        )

    fig.tight_layout(h_pad=2.4, rect=(0, 0, 1, 0.98))
    return save_figure(fig, "forecast_panels", subdir)


# ── 3. Error by horizon step ──────────────────────────────────────────────────


def fig_error_by_horizon(report: EvalReport, *, subdir: str = "") -> Path:
    """
    Per-horizon-step decay for the two metrics that degrade with distance:
    scaled error (MASE) and directional accuracy. Two stacked panels, never a
    dual axis — the units don't share a scale.

    This is the view that exposes mean-reversion: a model that flattens toward
    the series mean shows MASE climbing and directional accuracy sliding toward
    50% as steps-ahead grows.
    """
    per_step: dict[str, list[list[float]]] = {}
    for scores in report.series_scores.values():
        for s in scores:
            if s.per_step:
                per_step.setdefault(s.grader_name, []).append(list(s.per_step))

    def _mean_curve(name: str) -> np.ndarray:
        runs = per_step.get(name, [])
        if not runs:
            return np.array([])
        width = min(len(r) for r in runs)
        return np.mean([r[:width] for r in runs], axis=0)

    mase_curve = _mean_curve("MASE")
    dir_curve = _mean_curve("DirectionalAccuracy")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
    for ax in (ax1, ax2):
        apply_theme(ax)

    # Directional accuracy is a diff, so its curve is one step shorter than MASE.
    # Both panels answer "how bad at step k", so they must share an x-extent or
    # step 10 in the top panel sits above step 11 in the bottom one.
    n_steps = max(mase_curve.size, dir_curve.size)
    if n_steps:
        ax2.set_xlim(0.4, n_steps + 0.6)

    if mase_curve.size:
        x = np.arange(1, len(mase_curve) + 1)
        ax1.plot(x, mase_curve, color=BLUE, linewidth=2, zorder=5)
        ax1.scatter(x, mase_curve, s=30, color=BLUE, zorder=6, edgecolors=SURFACE, linewidths=2)
        ax1.axhline(1.0, color=AXIS, linewidth=1.2, zorder=3)
        # Reference labels anchor inside the axes — an outward label at the last
        # x value falls outside the data area and gets clipped.
        ax1.text(
            x[0],
            1.0,
            "naïve baseline",
            va="top",
            ha="left",
            fontsize=8,
            color=MUTED,
        )
        # Label only the endpoint — the value that says whether it decayed.
        # Offset below the mark: the curve descends into its own label when the
        # endpoint is anchored above.
        ax1.annotate(
            f"{mase_curve[-1]:.2f}",
            (x[-1], mase_curve[-1]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color=SECONDARY,
        )
        ax1.margins(y=0.18)
    ax1.set_ylabel("Scaled error (MASE)", color=SECONDARY, fontsize=9)
    ax1.set_title("Error by horizon step", fontsize=12, color=INK, fontweight="bold", loc="left")

    if dir_curve.size:
        x = np.arange(1, len(dir_curve) + 1)
        ax2.plot(x, dir_curve, color=ORANGE, linewidth=2, zorder=5)
        ax2.scatter(x, dir_curve, s=30, color=ORANGE, zorder=6, edgecolors=SURFACE, linewidths=2)
        ax2.axhline(50.0, color=AXIS, linewidth=1.2, zorder=3)
        ax2.text(x[0], 50.0, "coin flip", va="bottom", ha="left", fontsize=8, color=MUTED)
        ax2.set_ylim(0, 118)  # headroom so a 100% endpoint label clears the frame
        ax2.annotate(
            f"{dir_curve[-1]:.0f}%",
            (x[-1], dir_curve[-1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=SECONDARY,
        )
    ax2.set_ylabel("Directional accuracy (%)", color=SECONDARY, fontsize=9)
    ax2.set_xlabel("Steps ahead", color=SECONDARY, fontsize=9)

    fig.tight_layout(h_pad=2.0)
    return save_figure(fig, f"error_by_horizon_{report.cycle_id}", subdir)


__all__ = [
    "fig_error_by_horizon",
    "fig_forecast_panels",
    "fig_margin_to_threshold",
]
