"""Atlas eval figures — matplotlib → SVG for embedding in HTML reports.

Usage:
    from evals.reports.figures import fig_forecast, fig_grader_pass_rates, fig_segments_scatter

Each function returns a Path to the written SVG and accepts the relevant
data types from src/agents/state.py and core/segmentation/evaluation.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.agents.state import EvalReport, ForecastResult
    from core.segmentation.evaluation import SegmentEvalReport

# ── Palette (mirrors web/tailwind dark theme) ─────────────────────────────────
NAVY   = "#1e3a5f"
TEAL   = "#028090"
AMBER  = "#f59e0b"
GREEN  = "#059669"
RED    = "#dc2626"
PURPLE = "#7c3aed"
MID    = "#64748b"
SLATE  = "#e2e8f0"

PALETTE = [TEAL, NAVY, AMBER, GREEN, RED, PURPLE, MID]

OUT_DIR = Path("evals/reports/output/figures")


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     SLATE,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.grid.axis":     "y",
        "grid.color":         SLATE,
        "grid.linewidth":     0.8,
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.labelcolor":    MID,
        "xtick.color":        MID,
        "ytick.color":        MID,
        "text.color":         NAVY,
        "axes.titlecolor":    NAVY,
        "axes.titlesize":     13,
        "axes.titleweight":   "bold",
    })


def _save(fig: plt.Figure, name: str, subdir: str = "") -> Path:
    out = OUT_DIR / subdir if subdir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# ── Forecast ──────────────────────────────────────────────────────────────────


def fig_forecast(
    result: ForecastResult,
    actuals: np.ndarray | None = None,
    *,
    subdir: str = "",
) -> Path:
    """Actuals (if available) + point forecast + 80% PI for one series."""
    n_forecast = result.forecast_steps
    x_forecast = np.arange(n_forecast)

    fig, ax = plt.subplots(figsize=(10, 4))
    _style()

    if actuals is not None and len(actuals) > 0:
        x_hist = np.arange(-len(actuals), 0)
        ax.plot(x_hist, actuals, color=NAVY, linewidth=1.8, label="Actuals")
        ax.axvline(-0.5, color=SLATE, linewidth=1, linestyle="--")

    lo = np.array(result.lower_80)
    hi = np.array(result.upper_80)
    pt = np.array(result.point_forecast)

    ax.fill_between(x_forecast, lo, hi, color=TEAL, alpha=0.18, label="80% PI")
    ax.plot(x_forecast, pt, color=TEAL, linewidth=2, marker="o", markersize=3, label="Forecast")

    ax.set_xlabel("Steps")
    ax.set_title(
        f"{result.series_id} · {result.category.value} · {result.model_used.value}\n"
        f"{result.forecast_date}  horizon={result.horizon.value}",
        fontsize=11,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, f"forecast_{result.series_id}", subdir)


def fig_forecast_grid(
    results: list[ForecastResult],
    actuals_map: dict[str, np.ndarray] | None = None,
    *,
    ncols: int = 2,
    subdir: str = "",
) -> Path:
    """Grid of forecast panels, one per series."""
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()
    am = actuals_map or {}

    for ax, r in zip(axes_flat, results):
        actuals = am.get(r.series_id)
        x_forecast = np.arange(r.forecast_steps)
        if actuals is not None and len(actuals):
            x_hist = np.arange(-len(actuals), 0)
            ax.plot(x_hist, actuals, color=NAVY, linewidth=1.5)
            ax.axvline(-0.5, color=SLATE, linewidth=0.8, linestyle="--")
        lo, hi, pt = np.array(r.lower_80), np.array(r.upper_80), np.array(r.point_forecast)
        ax.fill_between(x_forecast, lo, hi, color=TEAL, alpha=0.18)
        ax.plot(x_forecast, pt, color=TEAL, linewidth=1.8, marker="o", markersize=2)
        ax.set_title(f"{r.series_id}", fontsize=9, color=NAVY, fontweight="bold")
        ax.tick_params(labelsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Forecast Grid", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "forecast_grid", subdir)


# ── Eval graders ─────────────────────────────────────────────────────────────


def fig_grader_pass_rates(report: EvalReport, *, subdir: str = "") -> Path:
    """Pass/fail bar for each grader metric in an EvalReport."""
    metrics = [
        ("MASE",        report.overall_mase,         1.0,   True),   # lower better
        ("SMAPE %",     report.overall_smape,         15.0,  True),
        ("Directional", report.directional_accuracy,  55.0,  False),  # higher better
        ("Coverage 80", report.coverage_80,           75.0,  False),
        ("Drift ratio", report.drift_ratio,            1.2,   True),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    thresholds = [m[2] for m in metrics]
    lower_better = [m[3] for m in metrics]

    colors = [
        GREEN if (v <= t if lb else v >= t) else RED
        for v, t, lb in zip(values, thresholds, lower_better)
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4))
    _style()
    ax.yaxis.grid(True, color=SLATE, zorder=0)
    ax.xaxis.grid(False)

    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)
    ax.plot(x, thresholds, "o--", color=NAVY, linewidth=1.5, markersize=5, zorder=4, label="Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8, framealpha=0.9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * ax.get_ylim()[1],
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold",
        )

    status = "PASS" if report.all_passed else "FAIL"
    status_color = GREEN if report.all_passed else RED
    ax.set_title(
        f"Eval Report — {report.cycle_id}  [{status}]\n{report.forecast_date}",
        fontsize=11, color=status_color,
    )
    fig.tight_layout()
    return _save(fig, f"grader_pass_rates_{report.cycle_id}", subdir)


def fig_eval_history(
    reports: list[EvalReport],
    *,
    subdir: str = "",
) -> Path:
    """MASE + SMAPE over eval cycles — shows trend and drift."""
    dates = [r.forecast_date for r in reports]
    mase   = [r.overall_mase for r in reports]
    smape  = [r.overall_smape for r in reports]
    drift  = [r.drift_ratio for r in reports]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    _style()

    ax1.plot(dates, mase, color=TEAL, linewidth=2, marker="o", markersize=4, label="MASE")
    ax1.axhline(1.0, color=RED, linewidth=1.2, linestyle="--", alpha=0.7, label="threshold 1.0")
    ax1.plot(dates, drift, color=AMBER, linewidth=1.5, linestyle=":", marker="s", markersize=3, label="Drift ratio")
    ax1.axhline(1.2, color=AMBER, linewidth=1, linestyle="--", alpha=0.5, label="drift threshold 1.2")
    ax1.set_ylabel("MASE / Drift")
    ax1.legend(fontsize=8, framealpha=0.9)
    ax1.set_title("Eval History — MASE & Drift", fontsize=11)

    ax2.plot(dates, smape, color=PURPLE, linewidth=2, marker="o", markersize=4, label="SMAPE %")
    ax2.axhline(15.0, color=RED, linewidth=1.2, linestyle="--", alpha=0.7, label="threshold 15%")
    ax2.set_ylabel("SMAPE %")
    ax2.set_xlabel("Forecast date")
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    return _save(fig, "eval_history", subdir)


# ── Segmentation ─────────────────────────────────────────────────────────────


def fig_segments_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    segment_names: dict[int, str] | None = None,
    *,
    title: str = "Customer Segments",
    subdir: str = "",
) -> Path:
    """2-D scatter (PCA/UMAP reduced) coloured by cluster label."""
    unique = sorted(set(labels))
    colors_map = {lbl: PALETTE[i % len(PALETTE)] for i, lbl in enumerate(l for l in unique if l >= 0)}
    names = segment_names or {}

    fig, ax = plt.subplots(figsize=(8, 6))
    _style()
    ax.grid(axis="both")

    for lbl in unique:
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=SLATE, s=18, alpha=0.4, label="noise")
        else:
            color = colors_map[lbl]
            name = names.get(lbl, f"Seg {lbl}")
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, s=30, alpha=0.75, label=name)
            cx, cy = X_2d[mask, 0].mean(), X_2d[mask, 1].mean()
            ax.text(cx, cy, name, fontsize=8, color=color, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return _save(fig, "segments_scatter", subdir)


def fig_segment_eval(report: SegmentEvalReport, *, subdir: str = "") -> Path:
    """Silhouette / CH / DB metrics vs thresholds for one segmentation run."""
    from core.segmentation.evaluation import THRESHOLDS as SEG_THR

    metrics = [
        ("Silhouette",        report.silhouette,          SEG_THR["silhouette_min"],     False),
        ("Davies-Bouldin",    report.davies_bouldin,      SEG_THR["davies_bouldin_max"], True),
        ("Calinski-Harabasz", report.calinski_harabasz,   None,                          False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _style()

    # ── Left: metric bars ────────────────────────────────────────────────────
    ax = axes[0]
    ax.yaxis.grid(True, color=SLATE, zorder=0)
    ax.xaxis.grid(False)
    plotted = [(lbl, val, thr, lb) for lbl, val, thr, lb in metrics if not np.isnan(val)]
    labels = [m[0] for m in plotted]
    values = [m[1] for m in plotted]
    thrs   = [m[2] for m in plotted]
    lbs    = [m[3] for m in plotted]
    colors = [
        (GREEN if (v <= t if lb else v >= t) else RED) if t is not None else TEAL
        for v, t, lb in zip(values, thrs, lbs)
    ]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.5, zorder=3)
    for i, (t, lb) in enumerate(zip(thrs, lbs)):
        if t is not None:
            ax.plot([i - 0.3, i + 0.3], [t, t], color=NAVY, linewidth=2, linestyle="--", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9, color=NAVY, fontweight="bold",
        )
    status = "PASS" if report.passed else "FAIL"
    ax.set_title(
        f"{report.algorithm}  k={report.n_clusters}  noise={report.n_noise}  [{status}]",
        fontsize=10, color=GREEN if report.passed else RED,
    )

    # ── Right: cluster sizes ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.yaxis.grid(True, color=SLATE, zorder=0)
    ax2.xaxis.grid(False)
    seg_ids = list(report.cluster_sizes.keys())
    sizes = [report.cluster_sizes[k] for k in seg_ids]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(seg_ids))]
    ax2.bar(range(len(seg_ids)), sizes, color=bar_colors, width=0.6, zorder=3)
    ax2.axhline(SEG_THR["min_cluster_size"], color=RED, linewidth=1.2, linestyle="--",
                label=f"min size={SEG_THR['min_cluster_size']}")
    ax2.set_xticks(range(len(seg_ids)))
    ax2.set_xticklabels([f"Seg {k}" for k in seg_ids], fontsize=9)
    ax2.set_ylabel("Customers")
    ax2.set_title("Cluster Sizes", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)

    fig.suptitle("Segmentation Eval", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return _save(fig, f"segment_eval_{report.algorithm.lower()}", subdir)


def fig_segment_sizes_bar(
    cluster_sizes: dict[int, int],
    segment_names: dict[int, str] | None = None,
    *,
    title: str = "Segment Sizes",
    subdir: str = "",
) -> Path:
    """Horizontal bar of customer counts per segment — good for dashboards."""
    names = segment_names or {}
    ids = sorted(cluster_sizes.keys())
    labels = [names.get(k, f"Seg {k}") for k in ids]
    sizes  = [cluster_sizes[k] for k in ids]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(ids))]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(ids))))
    _style()
    ax.grid(axis="x")
    ax.barh(labels, sizes, color=colors, height=0.55, zorder=3)
    for i, v in enumerate(sizes):
        ax.text(v + 0.3, i, str(v), va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlabel("Customers")
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, "segment_sizes", subdir)
