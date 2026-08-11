"""
Atlas eval figures — the metric-diagnostic chart set.

Replaces the two original figures (`fig_grader_pass_rates`, `fig_forecast_grid`
in evals/reports/figures.py), which shared a single linear y-axis across
incommensurable metrics and never plotted holdout actuals over the forecast
window. Each function here answers one diagnostic question:

    fig_margin_to_threshold  — is each metric passing, and by how much?
    fig_forecast_panels      — where in the window was the model wrong?
    fig_error_by_horizon     — does accuracy decay with steps-ahead?
    fig_series_heatmap       — which series is dragging the mean?
    fig_cycle_history        — how volatile is the model across cycles?

Palette is the validated data-viz default (3 categorical slots, blue/orange/aqua),
checked with scripts/validate_palette.js in both light and dark modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from evals.reports.figure_theme import (
    AQUA,
    AXIS,
    BLUE,
    CRITICAL,
    GOOD,
    GRID,
    INK,
    MUTED,
    ORANGE,
    SECONDARY,
    SURFACE,
    apply_theme,
    save_figure,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.agents.state import EvalReport, ForecastResult

# ── Metric spec ───────────────────────────────────────────────────────────────

# (attribute, display label, threshold attr on report or literal, lower_is_better)
# Thresholds mirror evals/metrics/constants.py tier3 defaults; the report carries
# per-grader thresholds, which take precedence when present.
BAR_HEIGHT = 0.62
LABEL_PAD = 0.04
MAX_MARGIN = 1.0  # clamp normalised margin so one outlier can't flatten the rest
MIN_HISTORY_STEPS = 20  # floor on trailing history drawn beside a forecast window


def _margin(value: float, threshold: float, lower_is_better: bool) -> float:
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


def _collect_metrics(report: EvalReport) -> list[dict]:
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
                "margin": _margin(value, slot["threshold"], slot["lower_is_better"]),
            }
        )
    return metrics


def _fmt(name: str, value: float) -> str:
    """Format a metric value with the unit its name implies."""
    if not np.isfinite(value):
        return "n/a"
    if name in {"SMAPE", "DirectionalAccuracy", "Coverage80", "IntervalWidth"}:
        return f"{value:.1f}%"
    return f"{value:.3f}"


# ── 1. Margin to threshold ────────────────────────────────────────────────────


def fig_margin_to_threshold(report: EvalReport, *, subdir: str = "") -> Path:
    """
    Diverging bars: signed distance from each metric's pass threshold, normalised
    so all metrics share one axis and positive always means "better than gate".

    This replaces the original shared-linear-axis bar chart, where MASE (~0.8)
    and drift (~1.05) were invisible slivers beside percentages, and a dashed
    line connected thresholds across unrelated metrics.
    """
    metrics = _collect_metrics(report)
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
            f"{_fmt(m['name'], m['value'])}  "
            f"({'≤' if m['lower_is_better'] else '≥'} {_fmt(m['name'], m['threshold'])})"
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


# ── 4. Series × metric heatmap ────────────────────────────────────────────────


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
            grid[i, j] = _margin(s.metric_value, s.threshold, s.lower_is_better)
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
                _fmt(metric_names[j], raw[i, j]),
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


# ── 5. Cycle history small multiples ──────────────────────────────────────────


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


__all__ = [
    "AQUA",
    "fig_cycle_history",
    "fig_error_by_horizon",
    "fig_forecast_panels",
    "fig_margin_to_threshold",
    "fig_series_heatmap",
]
