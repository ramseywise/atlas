"""Export key eval framework figures as SVGs for embedding in docs, PPT, and Excalidraw.

Usage:
    uv run python -m evals.reports.utils.figures          # all figures
    uv run python -m evals.reports.utils.figures --fig mrr_comparison cohen_d

Output: evals/reports/figures/*.svg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless — no display required
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from evals.metrics.base import THRESHOLDS

# ---------------------------------------------------------------------------
# Palette — mirrors eval_framework_tabs.html CSS variables
# ---------------------------------------------------------------------------
NAVY = "#1e3a5f"
TEAL = "#028090"
AMBER = "#f59e0b"
GREEN = "#059669"
RED = "#dc2626"
PURPLE = "#7c3aed"
MID = "#64748b"
OFFWHITE = "#f8fafc"
SLATE = "#e2e8f0"

PALETTE = [TEAL, NAVY, AMBER, GREEN, RED, PURPLE, MID]

OUT_DIR = Path("evals/reports/output/figures/shared")


def set_figures_source(source: str | None) -> Path:
    """Set SVG output dir to evals/reports/output/figures/{source} (or shared)."""
    global OUT_DIR
    from evals.reports.utils.layout import FIGURES_ROOT

    OUT_DIR = FIGURES_ROOT / (source or "shared")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": SLATE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "grid.color": SLATE,
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.labelcolor": MID,
            "xtick.color": MID,
            "ytick.color": MID,
            "text.color": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path


# ---------------------------------------------------------------------------
# Figure 1 — Cohen's d grader discrimination ranking
# ---------------------------------------------------------------------------


def fig_cohen_d(data: dict | None = None) -> Path:
    """Cohen's d for every grader in golden quality JSON — highlights VA golden default gates."""
    rows, n_liked, n_disliked, vtag = _golden_grader_calibration_stats()
    if not rows:
        raise RuntimeError(
            "No golden calibration data — run: make golden-report && make golden-quality-v2"
        )

    labels = [f"★ {r['label']}" if r.get("is_default") else r["label"] for r in rows]
    values = [r["d"] for r in rows]
    colors = [GREEN if v > 0.1 else TEAL if v > 0 else AMBER if v > -0.05 else RED for v in values]
    edges = [NAVY if r.get("is_default") else "none" for r in rows]
    linewidths = [2.2 if r.get("is_default") else 0 for r in rows]

    xmax = max(0.35, max(values) + 0.08)
    xmin = min(-0.14, min(values) - 0.08)

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.42 * len(labels))))
    _style()
    bars = ax.barh(
        labels,
        values,
        color=colors,
        height=0.55,
        zorder=3,
        edgecolor=edges,
        linewidth=linewidths,
    )
    ax.axvline(0, color=MID, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axvline(
        0.2, color=TEAL, linewidth=1.0, linestyle=":", alpha=0.6, label="small effect (d=0.2)"
    )
    ax.set_xlabel("Cohen's d  (positive = grader scores liked > disliked)")
    n_default = sum(1 for r in rows if r.get("is_default"))
    ax.set_title(
        f"Eval Graders — Cohen's d vs User Sentiment\n"
        f"(VA golden {vtag}, n={n_liked + n_disliked:,}: {n_liked} liked / {n_disliked} disliked · "
        f"{len(rows)} graders · ★ = default gate when d > 0.05 ({n_default})",
        fontsize=10,
    )
    ax.set_xlim(xmin, xmax)
    ax.invert_yaxis()

    for bar, val in zip(bars, values, strict=False):
        sign = "+" if val > 0 else ""
        ax.text(
            val + (0.006 if val >= 0 else -0.006),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
            color=NAVY,
            fontweight="bold",
        )

    legend = [
        mpatches.Patch(
            facecolor=GREEN, edgecolor=NAVY, linewidth=2, label="★ Default gate (d > 0.1)"
        ),
        mpatches.Patch(color=GREEN, label="Strong discriminator (d > 0.1)"),
        mpatches.Patch(color=TEAL, label="Weak positive"),
        mpatches.Patch(color=AMBER, label="Near-noise (|d| < 0.05)"),
        mpatches.Patch(color=RED, label="Anti-correlated — not for A/B"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, "cohen_d")


# ---------------------------------------------------------------------------
# Figure 1b — Threshold calibration: liked vs disliked mean per grader
# ---------------------------------------------------------------------------


def fig_threshold_viz() -> Path:
    """Paired bars: liked vs disliked mean score and pass rate per grader (VA golden archive)."""
    rows, n_liked, n_disliked, vtag = _golden_grader_calibration_stats()
    if not rows:
        raise RuntimeError(
            "No golden calibration data — run: make golden-report && make golden-quality-v2"
        )

    labels = [r["label"] for r in rows]
    tholds = [r["threshold"] for r in rows]
    liked_m = [r["liked_mean"] for r in rows]
    dislike_m = [r["disliked_mean"] for r in rows]
    liked_p = [r["liked_pass_pct"] for r in rows]
    dislike_p = [r["disliked_pass_pct"] for r in rows]

    x = np.arange(len(labels))
    width = 0.3

    fig_w = max(14, len(labels) * 1.15)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, max(5, len(labels) * 0.35)))
    _style()

    # ── Left: mean scores ──────────────────────────────────────────────────
    ax = axes[0]
    ax.yaxis.grid(True, color=SLATE, zorder=0)
    ax.xaxis.grid(False)

    ax.bar(x - width / 2, liked_m, width, color=GREEN, alpha=0.8, zorder=3, label="Liked mean")
    ax.bar(x + width / 2, dislike_m, width, color=RED, alpha=0.8, zorder=3, label="Disliked mean")

    for i, (thr, _lm, _dm) in enumerate(zip(tholds, liked_m, dislike_m, strict=False)):
        ax.plot(
            [i - width, i + width], [thr, thr], color=NAVY, linewidth=2, linestyle="--", zorder=4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Mean Score: Liked vs Disliked\n(dashed = calibration threshold)")
    ax.legend(fontsize=9, framealpha=0.9)

    # ── Right: pass rates ──────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.yaxis.grid(True, color=SLATE, zorder=0)
    ax2.xaxis.grid(False)

    ax2.bar(x - width / 2, liked_p, width, color=GREEN, alpha=0.8, zorder=3, label="Liked pass %")
    ax2.bar(
        x + width / 2, dislike_p, width, color=RED, alpha=0.8, zorder=3, label="Disliked pass %"
    )

    for i, (lp, dp) in enumerate(zip(liked_p, dislike_p, strict=False)):
        gap = lp - dp
        color = GREEN if gap > 5 else TEAL if gap > 1 else AMBER
        ax2.annotate(
            f"Δ{gap:+.0f}",
            xy=(i, max(lp, dp) + 1.5),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax2.set_ylabel("Pass Rate (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title(
        "Pass Rate at Threshold: Liked vs Disliked\n(Δ = separation at current threshold)"
    )
    ax2.legend(fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"Grader Threshold Calibration — VA Golden {vtag} "
        f"(n={n_liked + n_disliked:,}: {n_liked} liked / {n_disliked} disliked)",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "threshold_viz")


# ---------------------------------------------------------------------------
# Data helper — VA golden set (quality scores + sentiment labels)
# ---------------------------------------------------------------------------


def _load_golden_quality_path() -> Path | None:
    """Prefer v2 (kb_url_map passages); fall back to v1."""
    from evals.reports.paths import (
        DEFAULT_GOLDEN_QUALITY_V1,
        DEFAULT_GOLDEN_QUALITY_V2,
    )

    if DEFAULT_GOLDEN_QUALITY_V2.exists():
        return DEFAULT_GOLDEN_QUALITY_V2
    if DEFAULT_GOLDEN_QUALITY_V1.exists():
        return DEFAULT_GOLDEN_QUALITY_V1
    return None


def _load_golden_quality() -> dict:
    """Load grader_summary + metadata from golden quality JSON."""
    path = _load_golden_quality_path()
    if not path:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_path"] = str(path)
    return data


def _load_golden_quality_merged() -> tuple[dict[str, dict], str]:
    """Merge v1 + v2 grader_results per task (v2 wins on overlap). For full Cohen's d chart."""
    from evals.reports.paths import (
        DEFAULT_GOLDEN_QUALITY_V1,
        DEFAULT_GOLDEN_QUALITY_V2,
    )

    v1_path = DEFAULT_GOLDEN_QUALITY_V1
    v2_path = DEFAULT_GOLDEN_QUALITY_V2
    merged: dict[str, dict] = {}
    if v1_path.exists():
        for qr in json.loads(v1_path.read_text(encoding="utf-8")).get("query_results", []):
            merged[qr["task_id"]] = dict(qr.get("grader_results", {}))
    tag = "v1"
    if v2_path.exists():
        for qr in json.loads(v2_path.read_text(encoding="utf-8")).get("query_results", []):
            tid = qr["task_id"]
            merged.setdefault(tid, {}).update(qr.get("grader_results", {}))
        tag = "v2+v1" if v1_path.exists() else "v2"
    return merged, tag


def _golden_heuristic_retrieval() -> dict:
    """Layer-1 citation proxy from golden_all_responses_stats.json."""
    stats = _load_golden_stats()
    rp = stats.get("retrieval_proxy", {})
    n_rated = stats.get("sentiment", {}).get("n_liked", 0) + stats.get("sentiment", {}).get(
        "n_disliked", 0
    )
    return {
        "n_total": stats.get("n_total", 0),
        "n_rated": n_rated,
        "precision": rp.get("precision", 0) * 100,
        "recall": rp.get("recall", 0) * 100,
        "f1": rp.get("f1", 0) * 100,
    }


# BKH full-corpus heuristic baseline (all_suite, n=69,198 rated n=1,145)
_BKH_HEURISTIC = {
    "n_total": 69198,
    "n_rated": 1145,
    "precision": 30.4,
    "recall": 78.5,
    "f1": 43.8,
}


def _load_golden_data() -> list[dict]:
    """Join grader scores with sentiment labels from the VA golden archive."""
    quality_path = _load_golden_quality_path()
    from evals.reports.paths import resolve_golden_responses_path

    responses_path = resolve_golden_responses_path()
    if not quality_path or not responses_path.exists():
        return []

    with open(quality_path) as f:
        quality = {r["task_id"]: r["grader_results"] for r in json.load(f)["query_results"]}

    ratings: dict[str, str] = {}
    with open(responses_path) as f:
        for line in f:
            r = json.loads(line)
            v = r.get("rating")
            if v == 1.0 or v == 1:
                ratings[r["task_id"]] = "liked"
            elif v == 0.0 or v == 0 or v == "dislike":
                ratings[r["task_id"]] = "disliked"

    records = []
    for tid, gr in quality.items():
        sent = ratings.get(tid)
        if sent is None:
            continue
        rec: dict = {"task_id": tid, "sentiment": sent}
        for grader, res in gr.items():
            if isinstance(res, dict):
                rec[grader + "_score"] = res.get("score")
        records.append(rec)
    return records


def _cohens_d(liked: list[float], disliked: list[float]) -> float:
    a, b = np.asarray(liked, float), np.asarray(disliked, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# Display labels for grader keys (all keys in golden_all_quality*.json).
_GRADER_LABELS: dict[str, str] = {
    "grounding": "Grounding (custom)",
    "ragas_context_precision": "RAGAS Ctx Precision",
    "ragas_faithfulness": "RAGAS Faithfulness",
    "answer_relevancy": "Answer Relevancy (custom)",
    "completeness": "Completeness (custom)",
    "escalation": "Escalation (VA)",
    "deepeval_answer_relevancy": "Answer Relevancy (DeepEval)",
    "deepeval_completeness": "Completeness (DeepEval)",
    "deepeval_escalation": "DeepEval Escalation",
    "source_match": "Source Match",
    "boundary_adherence": "Boundary Adherence",
    "f1_correctness": "F1 Correctness",
}

# make golden-quality-v2 / VA_GOLDEN_QUALITY_GRADERS (expanded from combined_*).
_VA_GOLDEN_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "grounding",
        "ragas_context_precision",
        "ragas_faithfulness",
        "deepeval_completeness",
    }
)

# Fixed 6-panel KDE/box layout (2×3) — subset for readability.
_CORE_GRADER_KEYS: list[str] = [
    "completeness",
    "deepeval_completeness",
    "ragas_context_precision",
    "grounding",
    "deepeval_answer_relevancy",
    "ragas_faithfulness",
]

# All LLM graders for Cohen's d + threshold_viz (every key in merged quality JSON).
_ALL_CALIBRATION_GRADER_ORDER: list[str] = [
    "ragas_context_precision",
    "ragas_faithfulness",
    "grounding",
    "completeness",
    "answer_relevancy",
    "deepeval_completeness",
    "deepeval_answer_relevancy",
    "escalation",
    "deepeval_escalation",
    "source_match",
    "boundary_adherence",
    "f1_correctness",
]

_CALIBRATION_DISPLAY_ORDER: list[str] = list(_ALL_CALIBRATION_GRADER_ORDER)

# Pass thresholds for golden-deck figure panels — overrides production gates where viz differs.
_CALIBRATION_THRESHOLD_OVERRIDES: dict[str, float] = {
    "ragas_context_precision": 0.40,
    "ragas_faithfulness": 0.60,
    "completeness": 0.90,
    "deepeval_completeness": 0.90,
    "source_match": 0.75,
}

_CALIBRATION_GRADER_KEYS = (
    "grounding",
    "ragas_context_precision",
    "ragas_faithfulness",
    "completeness",
    "answer_relevancy",
    "escalation",
    "deepeval_answer_relevancy",
    "deepeval_completeness",
    "deepeval_escalation",
    "source_match",
)

_CALIBRATION_THRESHOLDS: dict[str, float] = {
    key: _CALIBRATION_THRESHOLD_OVERRIDES.get(key, THRESHOLDS[key])
    for key in _CALIBRATION_GRADER_KEYS
    if key in THRESHOLDS or key in _CALIBRATION_THRESHOLD_OVERRIDES
}

# Fixed 2×3 KDE/box layout — never auto-pick escalation / flat VA graders.
# Row-major: row1 Completeness ours|DeepEval|RAGAS ctx prec; row2 Grounding|Relevancy|RAGAS faith
_KDE_PANEL_LAYOUT: list[tuple[str, float, float, float]] = [
    # (grader_key, threshold, xlim_min, xlim_max)
    ("completeness", 0.90, 0.0, 1.0),
    ("deepeval_completeness", 0.90, 0.0, 1.0),
    ("ragas_context_precision", 0.40, 0.0, 1.0),
    ("grounding", 0.60, 0.0, 1.0),
    ("answer_relevancy", 0.75, 0.0, 1.0),
    ("deepeval_answer_relevancy", 0.75, 0.0, 1.0),
    ("ragas_faithfulness", 0.60, 0.2, 1.0),  # faith x-axis zoomed past left-tail noise
]

# Layer 1 heuristic signals — mirrors eval_stats_metrics() gates (evals/metrics/stats.py).
_HEURISTIC_KDE_PANEL_LAYOUT: list[tuple[str, str, float, float, float]] = [
    # (score_key sans _score, label, threshold, xlim_min, xlim_max) — threshold 0 = no line
    ("has_source", "Has source (proxy ŷ)", 0.5, 0.0, 1.0),
    ("source_count", "Sources cited (count)", 1.0, 0.0, 3.5),
    ("bkh_va_overlap", "BKH↔VA slug overlap", 0.5, 0.0, 1.0),
    ("known_response", "Known response (1−unknown)", THRESHOLDS["known_response_rate"], 0.0, 1.0),
    ("weighted_resolution", "Resolution score", THRESHOLDS["weighted_resolution_score"], 0.0, 1.0),
    ("response_words", "Response length (words)", 0.0, 0.0, 200.0),
]

# v2 — corpus-adjusted ŷ + kb_url_map expanded overlap (matches retrieval proxy v2).
_HEURISTIC_KDE_PANEL_LAYOUT_V2: list[tuple[str, str, float, float, float]] = [
    ("has_source_v2", "Has source (adj ŷ)", 0.5, 0.0, 1.0),
    ("source_count", "Sources cited (count)", 1.0, 0.0, 3.5),
    ("bkh_va_overlap_v2", "BKH↔VA expanded overlap", 0.5, 0.0, 1.0),
    ("known_response", "Known response (1−unknown)", THRESHOLDS["known_response_rate"], 0.0, 1.0),
    ("weighted_resolution", "Resolution score", THRESHOLDS["weighted_resolution_score"], 0.0, 1.0),
    ("response_words", "Response length (words)", 0.0, 0.0, 200.0),
]

_RESOLUTION_COMPONENT_SCORES = {
    "resolved": 1.0,
    "resolved_with_friction": 0.4,
    "unresolved": 0.0,
}


def _grader_label(key: str) -> str:
    return _GRADER_LABELS.get(key, key.replace("_", " ").title())


def _grader_threshold(key: str) -> float:
    return _CALIBRATION_THRESHOLDS.get(key, THRESHOLDS.get(key, 0.75))


def _distribution_panel_specs() -> list[tuple[str, str, float, float, float]]:
    """Fixed panel list for KDE + box plots (2×3). Skips keys absent from merged quality JSON."""
    quality_by_task, _ = _load_golden_quality_merged()
    available = set()
    for gr in quality_by_task.values():
        available.update(gr.keys())
    layout_keys = list(_CORE_GRADER_KEYS)
    specs: list[tuple[str, str, float, float, float]] = []
    by_key = {p[0]: p for p in _KDE_PANEL_LAYOUT}
    for key in layout_keys:
        if key not in available:
            continue
        if key in by_key:
            thr, xmin, xmax = by_key[key][1], by_key[key][2], by_key[key][3]
        else:
            thr, xmin, xmax = _grader_threshold(key), 0.0, 1.0
        specs.append((key, _grader_label(key), thr, xmin, xmax))
    return specs


def _discover_grader_keys_from_quality() -> list[str]:
    """All grader keys in merged v1+v2 golden quality JSON."""
    quality_by_task, _ = _load_golden_quality_merged()
    keys: set[str] = set()
    for gr in quality_by_task.values():
        keys.update(gr.keys())
    return sorted(keys)


def _all_calibration_grader_keys() -> list[str]:
    """Ordered list of every grader present in merged golden quality (for Cohen's d bar chart)."""
    discovered = set(_discover_grader_keys_from_quality())
    ordered = [k for k in _ALL_CALIBRATION_GRADER_ORDER if k in discovered]
    for key in sorted(discovered - set(ordered)):
        ordered.append(key)
    return ordered


def _golden_rated_records_merged() -> tuple[list[dict], str]:
    """Rated turns with grader scores from merged v1+v2 quality."""
    quality_by_task, vtag = _load_golden_quality_merged()
    if not quality_by_task:
        return [], "none"
    from evals.reports.paths import resolve_golden_responses_path

    responses_path = resolve_golden_responses_path()
    ratings: dict[str, str] = {}
    if responses_path.exists():
        with open(responses_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                v = r.get("rating")
                if v == 1.0 or v == 1:
                    ratings[r["task_id"]] = "liked"
                elif v == 0.0 or v == 0 or v == "dislike":
                    ratings[r["task_id"]] = "disliked"
    records: list[dict] = []
    for tid, gr in quality_by_task.items():
        sent = ratings.get(tid)
        if sent is None:
            continue
        rec: dict = {"task_id": tid, "sentiment": sent}
        for grader, res in gr.items():
            if isinstance(res, dict) and res.get("score") is not None:
                rec[grader + "_score"] = res["score"]
        records.append(rec)
    return records, vtag


def _golden_rated_records_calibrated() -> tuple[list[dict], str]:
    """Rated turns with corpus-calibrated LLM scores (v2 KDE / pass-rate parity)."""
    from evals.metrics.calibration.pass_overrides import (
        build_calibration_index,
        calibrated_score,
    )

    quality_by_task, vtag = _load_golden_quality_merged()
    records, _ = _golden_rated_records_merged()
    if not records:
        return [], vtag

    cal = build_calibration_index()
    out: list[dict] = []
    for rec in records:
        tid = rec["task_id"]
        gr = quality_by_task.get(tid, {})
        ctx = cal.get(tid)
        new_rec: dict = {"task_id": tid, "sentiment": rec["sentiment"]}
        for grader in _CORE_GRADER_KEYS:
            score = calibrated_score(
                grader,
                gr.get(grader),
                _grader_threshold(grader),
                ctx,
            )
            if score is not None:
                new_rec[grader + "_score"] = score
        out.append(new_rec)
    return out, f"{vtag}+cal"


def _grader_kde_cohens_d(records: list[dict], grader_key: str) -> float | None:
    """Cohen's d for one grader score column on rated golden turns."""
    score_key = grader_key + "_score"
    l_scores = [
        r[score_key] for r in records if r["sentiment"] == "liked" and r.get(score_key) is not None
    ]
    d_scores = [
        r[score_key]
        for r in records
        if r["sentiment"] == "disliked" and r.get(score_key) is not None
    ]
    if len(l_scores) < 2 or len(d_scores) < 2:
        return None
    return _cohens_d(l_scores, d_scores)


def _golden_heuristic_rated_records() -> list[dict]:
    """Rated VA golden turns with Layer 1 per-turn scores (no LLM graders)."""
    from evals.metrics.comparison.url_overlap import (
        DEFAULT_BKH_ALL,
        compute_overlap_records,
    )
    from evals.pipelines.datasets import load_jsonl
    from evals.reports.paths import resolve_golden_responses_path

    path = resolve_golden_responses_path()
    if not path.exists():
        return []
    tasks = load_jsonl(path)
    overlap: dict[str, float] = {}
    if DEFAULT_BKH_ALL.exists():
        for row in compute_overlap_records(tasks, DEFAULT_BKH_ALL):
            overlap[row["task_id"]] = 1.0 if row["slug_overlap"] else 0.0

    records: list[dict] = []
    for t in tasks:
        if t.rating in (1.0, 1):
            sent = "liked"
        elif t.rating in (0.0, 0) or t.rating == "dislike":
            sent = "disliked"
        else:
            continue
        urls = t.expected_urls or []
        nsrc = len(urls)
        meta = t.metadata or {}
        co = meta.get("conv_outcome")
        res = _RESOLUTION_COMPONENT_SCORES.get(co) if co else None
        rt = meta.get("response_type")
        records.append(
            {
                "task_id": t.task_id,
                "sentiment": sent,
                "has_source_score": 1.0 if nsrc > 0 else 0.0,
                "source_count_score": float(nsrc),
                "response_words_score": float(len((t.response or "").split())),
                "known_response_score": 0.0 if rt == "unknown" else 1.0,
                "weighted_resolution_score": res,
                "bkh_va_overlap_score": overlap.get(t.task_id),
            }
        )
    return records


def _golden_heuristic_rated_records_v2() -> list[dict]:
    """Rated VA golden turns with v2-adjusted Layer 1 retrieval scores."""
    from evals.metrics.calibration.pass_overrides import (
        build_calibration_index,
        effective_has_source,
    )
    from evals.metrics.comparison.url_overlap import (
        DEFAULT_BKH_ALL,
        compute_overlap_records,
        same_source_adjusted,
    )
    from evals.pipelines.datasets import load_jsonl
    from evals.reports.paths import resolve_golden_responses_path

    records = _golden_heuristic_rated_records()
    if not records:
        return []

    cal = build_calibration_index()
    overlap_by_id: dict[str, dict] = {}
    if DEFAULT_BKH_ALL.exists():
        for row in compute_overlap_records(
            load_jsonl(resolve_golden_responses_path()),
            DEFAULT_BKH_ALL,
        ):
            overlap_by_id[row["task_id"]] = row

    tasks = {t.task_id: t for t in load_jsonl(resolve_golden_responses_path())}

    for rec in records:
        tid = rec["task_id"]
        task = tasks.get(tid)
        ov = overlap_by_id.get(tid)
        liked = rec["sentiment"] == "liked"
        nsrc = len((task.expected_urls if task else None) or [])
        ctx = cal.get(tid)
        rec["has_source_v2_score"] = float(
            effective_has_source(nsrc > 0, liked, ctx),
        )
        rec["bkh_va_overlap_v2_score"] = float(same_source_adjusted(ov)) if ov else None
    return records


def _heuristic_panel_specs(
    records: list[dict],
    layout: list[tuple[str, str, float, float, float]] | None = None,
) -> list[tuple[str, str, float, float, float]]:
    """Panel list for heuristic KDE — dynamic xmax on response length."""
    specs: list[tuple[str, str, float, float, float]] = []
    rw_scores = [
        r["response_words_score"] for r in records if r.get("response_words_score") is not None
    ]
    rw_max = max(120.0, float(np.percentile(rw_scores, 98)) * 1.05) if rw_scores else 200.0
    panel_layout = layout or _HEURISTIC_KDE_PANEL_LAYOUT
    for key, label, thr, xmin, xmax in panel_layout:
        if key == "response_words":
            xmax = rw_max
        specs.append((key, label, thr, xmin, xmax))
    return specs


def _heuristic_kde_cohens_d(
    records: list[dict],
    key: str,
) -> float | None:
    """Cohen's d for one heuristic score column (key sans _score suffix)."""
    score_key = key + "_score"
    l_scores = [
        r[score_key] for r in records if r["sentiment"] == "liked" and r.get(score_key) is not None
    ]
    d_scores = [
        r[score_key]
        for r in records
        if r["sentiment"] == "disliked" and r.get(score_key) is not None
    ]
    if len(l_scores) < 2 or len(d_scores) < 2:
        return None
    return _cohens_d(l_scores, d_scores)


def _golden_grader_calibration_stats() -> tuple[list[dict], int, int, str]:
    """Per-grader liked/disliked stats for all graders in merged golden quality JSON."""
    records, vtag = _golden_rated_records_merged()
    if not records:
        return [], 0, 0, "none"

    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")

    rows: list[dict] = []
    for key in _all_calibration_grader_keys():
        score_key = key + "_score"
        l_scores = [
            r[score_key]
            for r in records
            if r["sentiment"] == "liked" and r.get(score_key) is not None
        ]
        d_scores = [
            r[score_key]
            for r in records
            if r["sentiment"] == "disliked" and r.get(score_key) is not None
        ]
        if len(l_scores) < 2 or len(d_scores) < 2:
            continue
        threshold = _grader_threshold(key)
        d_val = _cohens_d(l_scores, d_scores)
        rows.append(
            {
                "key": key,
                "label": _grader_label(key),
                "threshold": threshold,
                "d": d_val,
                "liked_mean": float(np.mean(l_scores)),
                "disliked_mean": float(np.mean(d_scores)),
                "liked_pass_pct": round(100 * np.mean([s >= threshold for s in l_scores])),
                "disliked_pass_pct": round(100 * np.mean([s >= threshold for s in d_scores])),
                "is_default": key in _VA_GOLDEN_DEFAULT_KEYS and d_val > 0.05,
                "n_liked": len(l_scores),
                "n_disliked": len(d_scores),
            }
        )

    rows.sort(key=lambda r: r["d"], reverse=True)
    return rows, n_liked, n_disliked, vtag


def _golden_calibration_graders_for_plots(max_panels: int = 6) -> list[tuple[str, str, float]]:
    """Fixed layout for distribution plots (ignores max_panels — always up to 6)."""
    return [(k, lbl, thr) for k, lbl, thr, _, _ in _distribution_panel_specs()[:max_panels]]


def _plot_score_distribution_kde(
    ax,
    records: list[dict],
    key: str,
    label: str,
    threshold: float,
    x_min: float,
    x_max: float,
) -> None:
    """Single KDE panel: liked vs disliked + threshold."""
    from scipy.stats import gaussian_kde

    score_key = key + "_score"
    l_scores = [
        r[score_key] for r in records if r["sentiment"] == "liked" and r.get(score_key) is not None
    ]
    d_scores = [
        r[score_key]
        for r in records
        if r["sentiment"] == "disliked" and r.get(score_key) is not None
    ]
    x_grid = np.linspace(x_min, x_max, 300)

    if x_min > 0:
        ax.axvspan(0, x_min, alpha=0.12, color=SLATE, zorder=0, label="left tail omitted")

    def _plot_kde(
        scores: list[float],
        color: str,
        linestyle: str,
        fill_alpha: float,
        legend: str,
    ) -> None:
        if len(scores) < 3 or len(set(scores)) < 2:
            return
        kde = gaussian_kde(scores, bw_method=0.15)
        y = kde(x_grid)
        ax.plot(x_grid, y, color=color, linewidth=2, linestyle=linestyle, label=legend)
        ax.fill_between(x_grid, y, alpha=fill_alpha, color=color)

    _plot_kde(l_scores, TEAL, "-", 0.20, "liked")
    _plot_kde(d_scores, RED, "--", 0.12, "disliked")

    if threshold > 0:
        ax.axvline(
            threshold,
            color=NAVY,
            linewidth=1.8,
            linestyle="--",
            alpha=0.85,
            label=f"thr={threshold:g}",
        )
        band = 0.08 if x_max <= 1.5 else max(2.0, (x_max - x_min) * 0.03)
        ax.axvspan(
            max(x_min, threshold - band),
            min(x_max, threshold + band),
            alpha=0.10,
            color=AMBER,
            zorder=0,
        )
    if l_scores:
        ax.axvline(np.mean(l_scores), color=TEAL, linewidth=1, linestyle=":", alpha=0.7)
    if d_scores:
        ax.axvline(np.mean(d_scores), color=RED, linewidth=1, linestyle=":", alpha=0.7)

    ax.set_title(label, fontsize=10, color=NAVY, fontweight="bold")
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_xlim(x_min, x_max)
    ax.legend(fontsize=7, framealpha=0.8, loc="upper left")


# ---------------------------------------------------------------------------
# Figure 1c — Box plots: score by sentiment per grader
# ---------------------------------------------------------------------------


def fig_box_plots() -> Path:
    """Box plots showing score distributions for liked vs disliked per grader.

    Data-driven: loads VA staging golden set scores and sentiment labels.
    Source: data/datasets/va_staging/va_staging_responses/ + golden traces notebook §8b.
    """
    records, _ = _golden_rated_records_merged()
    if not records:
        raise RuntimeError("No golden data — run grading first: make va-grade-golden-all")

    panels = _distribution_panel_specs()
    if not panels:
        raise RuntimeError("No graders in quality JSON for distribution plots")

    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (key, label, threshold, _, _) in zip(axes_flat, panels, strict=False):
        l_scores = [
            r[key + "_score"]
            for r in records
            if r["sentiment"] == "liked" and r.get(key + "_score") is not None
        ]
        d_scores = [
            r[key + "_score"]
            for r in records
            if r["sentiment"] == "disliked" and r.get(key + "_score") is not None
        ]

        ax.boxplot(
            [l_scores],
            positions=[0],
            widths=0.4,
            patch_artist=True,
            boxprops={"facecolor": TEAL + "55", "edgecolor": TEAL, "linewidth": 1.5},
            medianprops={"color": NAVY, "linewidth": 2},
            whiskerprops={"color": TEAL, "linewidth": 1.5},
            capprops={"color": TEAL, "linewidth": 1.5},
            flierprops={"marker": "o", "color": TEAL, "alpha": 0.25, "markersize": 3},
        )
        ax.boxplot(
            [d_scores],
            positions=[1],
            widths=0.4,
            patch_artist=True,
            boxprops={"facecolor": "none", "edgecolor": RED, "linewidth": 1.5, "hatch": "///"},
            medianprops={"color": RED, "linewidth": 2},
            whiskerprops={"color": RED, "linewidth": 1.5},
            capprops={"color": RED, "linewidth": 1.5},
            flierprops={"marker": "o", "color": RED, "alpha": 0.25, "markersize": 3},
        )
        ax.axhline(
            threshold,
            color=NAVY,
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            label=f"thr={threshold}",
        )
        ax.set_title(label, fontsize=9, color=NAVY, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["liked", "disliked"], fontsize=9)
        ax.set_ylim(-0.05, 1.12)
        if ax is axes_flat[0]:
            ax.set_ylabel("score")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    legend_patches = [
        mpatches.Patch(facecolor=TEAL + "55", edgecolor=TEAL, label="Liked"),
        mpatches.Patch(facecolor="none", edgecolor=RED, hatch="///", label="Disliked"),
    ]
    fig.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(1.0, 1.0),
    )
    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Grader Score Distribution — VA Golden  ({n_liked} liked / {n_disliked} disliked)\n"
        "Row1: Completeness (custom | DeepEval) | RAGAS ctx prec (thr=0.4)  ·  "
        "Row2: Grounding (custom) | DeepEval relevancy | RAGAS faith (thr=0.6)  |  dashed = threshold",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "box_plots")


# ---------------------------------------------------------------------------
# Figure 1d — KDE threshold calibration plots
# ---------------------------------------------------------------------------


def fig_kde_thresholds() -> Path:
    """KDE plots — fixed 2×3: Completeness pair + RAGAS col3; Grounding bottom-left + relevancy + RAGAS faith."""
    records, _ = _golden_rated_records_merged()
    if not records:
        raise RuntimeError("No golden data — run grading first: make va-grade-golden-all")

    panels = _distribution_panel_specs()
    if not panels:
        raise RuntimeError("No graders in quality JSON for KDE plots")

    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (key, label, threshold, x_min, x_max) in zip(axes_flat, panels, strict=False):
        _plot_score_distribution_kde(ax, records, key, label, threshold, x_min, x_max)

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Threshold Calibration — VA Golden  ({n_liked} liked / {n_disliked} disliked)\n"
        "Row1: Completeness (custom | DeepEval, thr=0.9) | RAGAS ctx prec (thr=0.4)  ·  "
        "Row2: Grounding (custom, thr=0.6) | DeepEval relevancy | RAGAS faith (thr=0.6, x from 0.2)",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "kde_thresholds")


def fig_kde_thresholds_v2() -> Path:
    """KDE plots with corpus-calibrated LLM scores (same 2×3 layout as v1)."""
    raw_records, _ = _golden_rated_records_merged()
    records, vtag = _golden_rated_records_calibrated()
    if not records:
        raise RuntimeError("No golden data — run grading first: make va-grade-golden-all")

    panels = _distribution_panel_specs()
    if not panels:
        raise RuntimeError("No graders in quality JSON for KDE plots")

    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")
    n_uplift = sum(
        1
        for raw, cal in zip(raw_records, records, strict=False)
        for key in _CORE_GRADER_KEYS
        if raw.get(key + "_score") is not None
        and cal.get(key + "_score") is not None
        and cal[key + "_score"] > raw[key + "_score"]
    )

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (key, label, threshold, x_min, x_max) in zip(axes_flat, panels, strict=False):
        _plot_score_distribution_kde(ax, records, key, label, threshold, x_min, x_max)
        d_raw = _grader_kde_cohens_d(raw_records, key)
        d_cal = _grader_kde_cohens_d(records, key)
        title = label
        if d_raw is not None and d_cal is not None and abs(d_cal - d_raw) >= 0.01:
            title = f"{label}\nd: {d_raw:.2f} → {d_cal:.2f}"
        ax.set_title(title, fontsize=9, color=NAVY, fontweight="bold")

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Threshold Calibration v2 — VA Golden ({vtag}, {n_liked} liked / {n_disliked} disliked)\n"
        f"Corpus-calibrated scores (uplift to threshold when aligned) · {n_uplift} score uplifts  ·  "
        "same graders/thresholds as v1",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "kde_thresholds_v2")


def fig_kde_heuristic_thresholds() -> Path:
    """KDE for Layer 1 heuristic signals — same 2×3 layout as LLM calibration."""
    records = _golden_heuristic_rated_records()
    if not records:
        raise RuntimeError(
            "No rated VA staging responses — need va_staging_all_responses.jsonl (make golden-report)"
        )

    panels = _heuristic_panel_specs(records)
    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")
    n_overlap = sum(1 for r in records if r.get("bkh_va_overlap_score") is not None)

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (key, label, threshold, x_min, x_max) in zip(axes_flat, panels, strict=False):
        _plot_score_distribution_kde(ax, records, key, label, threshold, x_min, x_max)
        title = label
        if key == "bkh_va_overlap":
            title = f"{label}\n(n={n_overlap} BKH-paired)"
        elif key == "weighted_resolution":
            n_labeled = sum(1 for r in records if r.get("weighted_resolution_score") is not None)
            title = f"{label}\n(n={n_labeled} with conv_outcome)"
        ax.set_title(title, fontsize=9, color=NAVY, fontweight="bold")

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Heuristic Threshold Calibration — VA Staging  ({n_liked} liked / {n_disliked} disliked)\n"
        "Layer 1 (pre-LLM): retrieval proxy · BKH↔VA overlap · resolution · response shape  |  "
        "dashed = suite gate from evals/metrics/_constants.py",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "kde_heuristic_thresholds")


def fig_kde_heuristic_thresholds_v2() -> Path:
    """KDE for Layer 1 heuristics with retrieval v2 adjustments (adj ŷ + expanded overlap)."""
    records = _golden_heuristic_rated_records_v2()
    if not records:
        raise RuntimeError(
            "No rated VA staging responses — need va_staging_all_responses.jsonl (make golden-report)"
        )

    panels = _heuristic_panel_specs(records, _HEURISTIC_KDE_PANEL_LAYOUT_V2)
    n_liked = sum(1 for r in records if r["sentiment"] == "liked")
    n_disliked = sum(1 for r in records if r["sentiment"] == "disliked")
    n_overlap = sum(1 for r in records if r.get("bkh_va_overlap_v2_score") is not None)

    d_hs_v1 = _heuristic_kde_cohens_d(
        _golden_heuristic_rated_records(),
        "has_source",
    )
    d_hs_v2 = _heuristic_kde_cohens_d(records, "has_source_v2")
    d_ov_v1 = _heuristic_kde_cohens_d(
        _golden_heuristic_rated_records(),
        "bkh_va_overlap",
    )
    d_ov_v2 = _heuristic_kde_cohens_d(records, "bkh_va_overlap_v2")

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    _style()
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (key, label, threshold, x_min, x_max) in zip(axes_flat, panels, strict=False):
        _plot_score_distribution_kde(ax, records, key, label, threshold, x_min, x_max)
        title = label
        if key == "bkh_va_overlap_v2":
            title = f"{label}\n(n={n_overlap} BKH-paired)"
        elif key == "weighted_resolution":
            n_labeled = sum(1 for r in records if r.get("weighted_resolution_score") is not None)
            title = f"{label}\n(n={n_labeled} with conv_outcome)"
        elif key == "has_source_v2" and d_hs_v1 is not None and d_hs_v2 is not None:
            title = f"{label}\nd: {d_hs_v1:.2f} → {d_hs_v2:.2f}"
        elif key == "bkh_va_overlap_v2" and d_ov_v1 is not None and d_ov_v2 is not None:
            title = f"{label}\nd: {d_ov_v1:.2f} → {d_ov_v2:.2f}"
        ax.set_title(title, fontsize=9, color=NAVY, fontweight="bold")

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Heuristic Threshold Calibration v2 — VA Golden  ({n_liked} liked / {n_disliked} disliked)\n"
        "Adj ŷ = effective_has_source (corpus promote/demote) · overlap = kb_url_map expanded  |  "
        "other panels unchanged from v1",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, "kde_heuristic_thresholds_v2")


# ---------------------------------------------------------------------------
# Figure 2 — MRR by ablation configuration
# ---------------------------------------------------------------------------


def fig_mrr_comparison(configs: dict | None = None) -> Path:
    """Horizontal bar chart of MRR per ablation config."""
    if configs:
        items = sorted(
            configs.items(), key=lambda kv: kv[1].get("aggregate", {}).get("mrr", 0), reverse=True
        )
        labels = [v[1].get("display_label", k) for k, v in items]
        values = [v[1].get("aggregate", {}).get("mrr", 0) for k, v in items]
    else:
        labels = [
            "adk_flash_thinking1024",
            "lg_multi_query",
            "adk_thinking1024",
            "adk_flash",
            "lg_crag",
            "lg_no_crag",
            "lg_llm_planner",
            "va_staging  (prod ref)",
            "lg_crag_thinking1024",
            "adk_baseline",
        ]
        values = [0.656, 0.594, 0.583, 0.563, 0.547, 0.542, 0.516, 0.500, 0.484, 0.418]

    colors = [
        GREEN
        if v > 0.57
        else TEAL
        if v > 0.52
        else AMBER
        if v > 0.49
        else MID
        if v == 0.500
        else RED
        for v in values
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    _style()
    bars = ax.barh(labels, values, color=colors, height=0.6, zorder=3)
    ax.axvline(
        0.500,
        color=NAVY,
        linewidth=1.5,
        linestyle="--",
        alpha=0.6,
        label="Production ref (va_staging) 0.500",
    )
    ax.set_xlabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Ablation Study — MRR by Configuration  (n = 44 tasks)")
    ax.set_xlim(0.28, 0.74)
    ax.invert_yaxis()

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            color=NAVY,
            fontweight="bold",
        )

    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    return _save(fig, "mrr_comparison")


# ---------------------------------------------------------------------------
# Figure 3 — Feature flag ΔMRR impact
# ---------------------------------------------------------------------------


def fig_feature_impact() -> Path:
    labels = [
        "Thinking budget (hc_adk)",
        "Flash model (hc_adk)",
        "Multi-query (hc_lg)",
        "CRAG alone (hc_lg)",
        "LLM planner (hc_lg)",
        "CRAG + Thinking (hc_lg)",
    ]
    deltas = [+0.165, +0.145, +0.076, +0.005, -0.031, -0.063]
    colors = [GREEN if d > 0.05 else TEAL if d > 0 else AMBER if d > -0.04 else RED for d in deltas]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    _style()
    bars = ax.barh(labels, deltas, color=colors, height=0.55, zorder=3)
    ax.axvline(0, color=MID, linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("ΔMRR vs same-agent baseline")
    ax.set_title("Feature Flag Impact on MRR")
    ax.set_xlim(-0.12, 0.22)
    ax.invert_yaxis()

    for bar, val in zip(bars, deltas, strict=False):
        sign = "+" if val >= 0 else ""
        ax.text(
            val + (0.004 if val >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=10,
            color=NAVY,
            fontweight="bold",
        )

    legend = [
        mpatches.Patch(color=GREEN, label="Strong positive"),
        mpatches.Patch(color=RED, label="Negative — avoid in this combo"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, "feature_impact")


# ---------------------------------------------------------------------------
# Figure 4 — BKH grader pass rates vs threshold
# ---------------------------------------------------------------------------


def fig_bkh_pass_rates(stats: dict | None = None) -> Path:
    """BKH calibration sample LLM pass rates (n=50). Full-corpus heuristic is separate."""
    cal_path = Path("data/datasets/bkh/quality_results/calibration_quality_v3.json")
    gs: dict = {}
    n = 50
    if cal_path.exists():
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        gs = data.get("grader_summary", {})
        n = data.get("n_queries", 50)
    elif stats:
        gr = stats.get("stats", {}).get("bkh_regression_main", {}).get("grader_results", {})
        gs = {k: {"pass_rate": v.get("pass_rate", 0)} for k, v in gr.items()}

    from evals.metrics.calibration.grader_scope import BKH_LAYER2_CHART_METRICS

    metric_keys = list(BKH_LAYER2_CHART_METRICS)
    if not gs:
        gs = {
            "answer_relevancy": {"pass_rate": 0.72},
            "completeness": {"pass_rate": 0.68},
            "escalation": {"pass_rate": 0.80},
        }
    bh = _BKH_HEURISTIC
    fig, _ = _pass_rate_chart(
        gs,
        metric_keys,
        title=f"BKH Calibration — LLM Pass Rates (n={n})",
        subtitle=(
            f"Full corpus heuristic (n={bh['n_total']:,}): "
            f"P={bh['precision']:.1f}% R={bh['recall']:.1f}% F1={bh['f1']:.1f}% — see heuristic_llm_compare"
        ),
    )
    return _save(fig, "bkh_pass_rates")


# ---------------------------------------------------------------------------
# Figure 5 — VA staging pass rates vs threshold
# ---------------------------------------------------------------------------


def _load_bkh_llm_pass_summary() -> tuple[dict, int]:
    cal_path = Path("data/datasets/bkh/quality_results/calibration_quality_v3.json")
    if not cal_path.exists():
        return {}, 50
    data = json.loads(cal_path.read_text(encoding="utf-8"))
    gs = data.get("grader_summary", {})
    n = data.get("n_queries", 50)
    return gs, n


def _bkh_layer2_pass_rate(bkh_gs: dict, key: str) -> float | None:
    """BKH cal pass rate % — None when grader was not run on BKH cal (e.g. RAGAS)."""
    pr = bkh_gs.get(key, {}).get("pass_rate")
    if pr is None:
        return None
    return round(float(pr) * 100)


def _pass_rate_chart(
    grader_summary: dict,
    metric_keys: list[tuple[str, str, float]],
    *,
    title: str,
    subtitle: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Shared pass-rate bar chart. metric_keys: (grader_key, label, threshold_pct)."""
    labels = [m[1] for m in metric_keys]
    thresholds = [m[2] for m in metric_keys]
    rates = [round(grader_summary.get(m[0], {}).get("pass_rate", 0) * 100) for m in metric_keys]
    colors = [GREEN if r >= t else RED for r, t in zip(rates, thresholds, strict=False)]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), 4.5))
    _style()
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.55, zorder=3)
    ax.plot(
        x, thresholds, "o--", color=NAVY, linewidth=1.8, markersize=5, zorder=4, label="Threshold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, framealpha=0.9, loc="upper right")
    for bar, val in zip(bars, rates, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1.2,
            f"{val}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color=NAVY,
            fontweight="bold",
        )
    if subtitle:
        fig.suptitle(
            f"{title}\n{subtitle}",
            fontsize=10,
            color=NAVY,
            fontweight="bold",
            y=0.98,
        )
    else:
        fig.suptitle(title, fontsize=11, color=NAVY, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.78, bottom=0.14)
    return fig, ax


# Golden LLM pass-rate metrics (see pipelines/golden/pass_overrides.py)
import contextlib  # noqa: E402

from evals.metrics.calibration.pass_overrides import (  # noqa: E402
    GOLDEN_LLM_METRICS_V1,
    build_calibration_index,
    compute_llm_pass_summary,
    compute_va_retrieval_proxy_calibrated,
    load_golden_quality_v1_merged_source_match,
)


def _load_golden_quality_v1() -> dict:
    return load_golden_quality_v1_merged_source_match()


def _golden_llm_pass_rates_figure(
    quality: dict,
    metric_keys: list[tuple[str, str, float]],
    *,
    version: str,
    out_name: str,
    calibrated: bool = False,
    cal_index: dict | None = None,
) -> Path:
    gs = compute_llm_pass_summary(
        quality,
        metric_keys,
        calibrated=calibrated,
        cal_index=cal_index,
    )
    h = _golden_heuristic_retrieval()
    n_tasks = quality.get("n_queries", len(quality.get("query_results", [])))
    kind = "LLM Pass Rates (corpus-calibrated)" if calibrated else "LLM Pass Rates (raw)"
    uplift_note = ""
    if calibrated:
        ups = sum(gs[k].get("n_calibrated_up", 0) for k, _, _ in metric_keys)
        uplift_note = f" · {ups} corpus uplift passes"
    fig, _ = _pass_rate_chart(
        gs,
        metric_keys,
        title=f"VA Golden — {kind} ({version}, n={n_tasks})",
        subtitle=(
            f"Layer 2 LLM graders · {len(metric_keys)} metrics · {h['n_rated']:,} rated · "
            f"citation proxy P={h['precision']:.0f}%{uplift_note}"
        ),
    )
    return _save(fig, out_name)


def fig_golden_pass_rates_v1(quality: dict | None = None) -> Path:
    """Raw LLM pass rates — golden_all_quality.json, 7 graders (3 custom + 2 DeepEval + 2 RAGAS)."""
    q = quality or _load_golden_quality_v1()
    if not q:
        raise FileNotFoundError("golden_all_quality.json (v1) not found")
    return _golden_llm_pass_rates_figure(
        q,
        GOLDEN_LLM_METRICS_V1,
        version="v1",
        out_name="golden_pass_rates_v1",
        calibrated=False,
    )


def fig_golden_pass_rates_v2(quality: dict | None = None) -> Path:
    """BKH | VA v1 | VA v2 grouped pass rates — same 7 graders, v2 corpus-calibrated."""
    return fig_golden_pass_rates_compare(quality)


def fig_golden_pass_rates_compare(quality: dict | None = None) -> Path:
    """Grouped pass-rate bars: BKH cal vs VA v1 raw vs VA v2 corpus-calibrated."""
    q = quality or _load_golden_quality_v1()
    if not q:
        raise FileNotFoundError("golden_all_quality.json (v1) not found")

    _, layer2, bkh_n, n_va = _benchmark_cohort_data()
    cal_idx = build_calibration_index()
    va_gs_v2 = compute_llm_pass_summary(
        q,
        GOLDEN_LLM_METRICS_V1,
        calibrated=True,
        cal_index=cal_idx,
    )
    ups = sum(va_gs_v2[k].get("n_calibrated_up", 0) for k, _, _ in GOLDEN_LLM_METRICS_V1)
    h = _golden_heuristic_retrieval()

    fig, ax = plt.subplots(figsize=(max(10, len(GOLDEN_LLM_METRICS_V1) * 1.35), 5.2))
    _style()
    _draw_pass_rate_cohort_panel(
        ax,
        layer2,
        GOLDEN_LLM_METRICS_V1,
        bkh_n=bkh_n,
        n_va=n_va,
    )
    fig.suptitle(
        f"VA Golden — LLM Pass Rates: BKH vs v1 vs v2 (n={n_va})\n"
        f"Layer 2 · {len(GOLDEN_LLM_METRICS_V1)} graders · {h['n_rated']:,} rated · "
        f"citation proxy P={h['precision']:.0f}% · {ups} corpus uplift passes",
        fontsize=10,
        color=NAVY,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.82, bottom=0.16)
    return _save(fig, "golden_pass_rates_compare")


def fig_golden_pass_rates(quality: dict | None = None) -> Path:
    """Default golden pass rates chart = BKH | v1 | v2 compare."""
    return fig_golden_pass_rates_compare(quality)


def fig_va_pass_rates(va_stats: dict | None = None) -> Path:
    """Export compare (BKH | v1 | v2) + v1-only; va_pass_rates.svg → compare."""
    fig_golden_pass_rates_v1()
    fig_golden_pass_rates_compare()
    src = OUT_DIR / "golden_pass_rates_compare.svg"
    dst = OUT_DIR / "va_pass_rates.svg"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"))
    legacy = OUT_DIR / "golden_pass_rates.svg"
    if src.exists():
        legacy.write_text(src.read_text(encoding="utf-8"))
    v1_src = OUT_DIR / "golden_pass_rates_v1.svg"
    v2_src = OUT_DIR / "golden_pass_rates_compare.svg"
    for copy_src, copy_dst in [
        (v1_src, OUT_DIR / "va_pass_rates_v1.svg"),
        (v2_src, OUT_DIR / "va_pass_rates_v2.svg"),
        (v2_src, OUT_DIR / "golden_pass_rates_v2.svg"),
    ]:
        if copy_src.exists():
            copy_dst.write_text(copy_src.read_text(encoding="utf-8"))
    return dst


# ---------------------------------------------------------------------------
# BKH vs VA benchmark — horizontal cohort bars (Layer 1 top, Layer 2 bottom)
# ---------------------------------------------------------------------------

_COHORT_COLORS = {
    "BKH": NAVY,
    "VA v1": TEAL,
    "VA v2": PURPLE,
}
_COHORT_ORDER = ["BKH", "VA v1", "VA v2"]
_COHORT_DISPLAY = {
    "BKH": "BKH baseline",
    "VA v1": "VA staging (raw)",
    "VA v2": "VA staging (calibrated)",
}
_LAYER1_METRICS: list[tuple[str, str, float | None]] = [
    ("satisfaction", "Satisfaction rate", THRESHOLDS["satisfaction_rate"] * 100),
    ("weighted_resolution", "Weighted resolution", THRESHOLDS["weighted_resolution_score"] * 100),
    ("precision", "Retrieval precision (proxy)", THRESHOLDS["retrieval_precision"] * 100),
    ("recall", "Proxy retrieval recall", THRESHOLDS["proxy_retrieval_recall"] * 100),
    ("f1", "F1 (conservative)", None),
]

_BKH_LLM_METRICS: list[tuple[str, str, float]] = [
    ("completeness", "Completeness", 70),
    ("answer_relevancy", "Answer Relevancy", 75),
    ("grounding", "Grounding", 60),
    ("deepeval_completeness", "DeepEval Complete.", 70),
    ("deepeval_answer_relevancy", "DeepEval Relevancy", 75),
    ("ragas_context_precision", "RAGAS Ctx Prec.", 50),
    ("ragas_faithfulness", "RAGAS Faith.", 50),
]


def _load_bkh_stats_dict() -> dict:
    from evals.reports.paths import bkh_stats_path

    path = bkh_stats_path()
    if path.exists():
        return _extract_file_stats(json.loads(path.read_text(encoding="utf-8")))
    legacy = Path("data/datasets/bkh/stats/all_stats.json")
    if legacy.exists():
        return _extract_file_stats(json.loads(legacy.read_text(encoding="utf-8")))
    return {}


def _bkh_heuristic_live() -> dict:
    stats = _load_bkh_stats_dict()
    if stats:
        h = _golden_heuristic_retrieval_from_stats(stats)
        h["n_total"] = stats.get("n_total", _BKH_HEURISTIC["n_total"])
        return h
    return dict(_BKH_HEURISTIC)


def _golden_heuristic_retrieval_from_stats(stats: dict, rp: dict | None = None) -> dict:
    """Layer-1 percentages from a stats dict; optional retrieval_proxy override (v2)."""
    rp = rp or stats.get("retrieval_proxy", {})
    n_rated = stats.get("sentiment", {}).get("n_liked", 0) + stats.get("sentiment", {}).get(
        "n_disliked", 0
    )
    s = stats.get("sentiment", {})
    sat = 100 * s.get("n_liked", 0) / n_rated if n_rated else None
    co = stats.get("conv_outcome_turn_breakdown", {})
    resolved = co.get("resolved", 0)
    friction = co.get("resolved_with_friction", 0)
    unresolved = co.get("unresolved", 0)
    denom = resolved + friction + unresolved
    wrs = 100 * (resolved + 0.4 * friction) / denom if denom else None
    prec = rp.get("precision")
    rec = rp.get("recall")
    f1 = rp.get("f1")
    return {
        "n_rated": n_rated,
        "satisfaction": sat,
        "weighted_resolution": wrs,
        "precision": prec * 100 if prec is not None else None,
        "recall": rec * 100 if rec is not None else None,
        "f1": f1 * 100 if f1 is not None else None,
    }


def _va_retrieval_v2_metrics() -> dict:
    from evals.pipelines.datasets import load_jsonl
    from evals.reports.paths import resolve_golden_responses_path

    tasks = load_jsonl(resolve_golden_responses_path())
    cal_index = build_calibration_index()
    rp = compute_va_retrieval_proxy_calibrated(tasks, cal_index)
    return {
        "precision": (rp.get("precision") or 0) * 100,
        "recall": (rp.get("recall") or 0) * 100,
        "f1": (rp.get("f1") or 0) * 100,
        "confusion": rp.get("confusion", {}),
    }


def _benchmark_cohort_data() -> tuple[
    dict[str, dict[str, float | None]], dict[str, dict[str, float]], int, int
]:
    """Layer-1 + Layer-2 values for BKH, VA v1, VA v2."""
    bkh_stats = _load_bkh_stats_dict()
    va_stats = _load_golden_stats()
    bkh_h = (
        _golden_heuristic_retrieval_from_stats(bkh_stats) if bkh_stats else _bkh_heuristic_live()
    )
    va_h_v1 = _golden_heuristic_retrieval_from_stats(va_stats)
    va_h_v2_extra = _va_retrieval_v2_metrics()
    va_h_v2 = dict(va_h_v1)
    va_h_v2.update(
        {
            "precision": va_h_v2_extra["precision"],
            "recall": va_h_v2_extra["recall"],
            "f1": va_h_v2_extra["f1"],
        }
    )

    layer1: dict[str, dict[str, float | None]] = {
        "BKH": {k: bkh_h.get(k) for k, _, _ in _LAYER1_METRICS},
        "VA v1": {k: va_h_v1.get(k) for k, _, _ in _LAYER1_METRICS},
        "VA v2": {k: va_h_v2.get(k) for k, _, _ in _LAYER1_METRICS},
    }

    q_v1 = _load_golden_quality_v1()
    cal_idx = build_calibration_index()
    va_gs_v1 = compute_llm_pass_summary(q_v1, GOLDEN_LLM_METRICS_V1, calibrated=False)
    va_gs_v2 = compute_llm_pass_summary(
        q_v1,
        GOLDEN_LLM_METRICS_V1,
        calibrated=True,
        cal_index=cal_idx,
    )
    bkh_gs, bkh_n = _load_bkh_llm_pass_summary()

    from evals.metrics.calibration.grader_scope import (
        BKH_CALIBRATION_LLM_KEYS,
        COMPARISON_LAYER2_METRICS,
    )

    layer2: dict[str, dict[str, float | None]] = {"BKH": {}, "VA v1": {}, "VA v2": {}}
    layer2_keys = {k for k, _, _ in GOLDEN_LLM_METRICS_V1} | {
        k for k, _, _ in COMPARISON_LAYER2_METRICS
    }
    for key in layer2_keys:
        if key in BKH_CALIBRATION_LLM_KEYS:
            layer2["BKH"][key] = _bkh_layer2_pass_rate(bkh_gs, key)
        else:
            layer2["BKH"][key] = None  # grounding/RAGAS — VA staging only
        v1 = va_gs_v1.get(key, {}).get("pass_rate")
        v2 = va_gs_v2.get(key, {}).get("pass_rate")
        layer2["VA v1"][key] = round(float(v1) * 100) if v1 is not None else None
        layer2["VA v2"][key] = round(float(v2) * 100) if v2 is not None else None

    n_va = q_v1.get("n_queries", len(q_v1.get("query_results", [])))
    return layer1, layer2, bkh_n, n_va


def benchmark_layer2_cohorts() -> tuple[dict[str, dict[str, float | None]], int, int]:
    """Layer-2 pass rates (0–100) for BKH / VA v1 / VA v2 plus sample sizes."""
    _, layer2, bkh_n, n_va = _benchmark_cohort_data()
    return layer2, bkh_n, n_va


def fig_ab_side_by_side() -> Path:
    """Demo §3 style: vertical grouped bars — BKH baseline vs VA calibrated (primary gates)."""
    from evals.metrics.calibration.grader_scope import COMPARISON_LAYER2_METRICS

    _, layer2, bkh_n, n_va = _benchmark_cohort_data()
    metrics = list(COMPARISON_LAYER2_METRICS)
    labels = [lbl for _, lbl, _ in metrics]
    thresholds = [thr for _, _, thr in metrics]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _style()

    series = [
        ("BKH", -width / 2, NAVY, f"BKH baseline (n={bkh_n})"),
        ("VA v2", width / 2, TEAL, f"VA staging calibrated (n={n_va})"),
    ]
    for cohort, offset, color, legend in series:
        vals = []
        for key, _, _thr in metrics:
            v = layer2.get(cohort, {}).get(key)
            vals.append(v if v is not None else 0)
        bars = ax.bar(x + offset, vals, width, label=legend, color=color, alpha=0.82, zorder=3)
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=NAVY,
                    fontweight="bold",
                )

    ax.plot(
        x,
        thresholds,
        "o--",
        color=AMBER,
        linewidth=1.8,
        markersize=5,
        zorder=4,
        label="Threshold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass rate (%)")
    ax.set_title(
        "Primary LLM gates — BKH calibration vs VA staging",
        color=NAVY,
        fontweight="bold",
        fontsize=11,
        pad=10,
    )
    ax.yaxis.grid(True, color=SLATE)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    return _save(fig, "ab_side_by_side")


def _draw_pass_rate_cohort_panel(
    ax,
    cohort_data: dict[str, dict[str, float | None]],
    metrics: list[tuple[str, str, float]],
    *,
    bkh_n: int,
    n_va: int,
) -> None:
    """Grouped pass-rate bars per grader: BKH | VA v1 | VA v2 (mirrors retrieval score panel)."""
    labels = [lbl for _, lbl, _ in metrics]
    thresholds = [thr for _, _, thr in metrics]
    x = np.arange(len(labels))
    n_series = len(_COHORT_ORDER)
    w = 0.72 / n_series
    offset = (n_series - 1) / 2
    legend_seen: set[str] = set()

    for si, cohort in enumerate(_COHORT_ORDER):
        for mi, (key, _, _) in enumerate(metrics):
            v = cohort_data.get(cohort, {}).get(key)
            if v is None:
                continue
            bx = x[mi] + (si - offset) * w
            show_label = cohort not in legend_seen
            bar = ax.bar(
                bx,
                v,
                w,
                color=_COHORT_COLORS[cohort],
                alpha=0.88,
                zorder=3,
                label=cohort if show_label else None,
            )
            if show_label:
                legend_seen.add(cohort)
            ax.text(
                bar[0].get_x() + bar[0].get_width() / 2,
                max(v, 0) + 1.5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=NAVY,
                fontweight="bold",
            )

    ax.plot(
        x,
        thresholds,
        "o--",
        color=AMBER,
        linewidth=1.8,
        markersize=4,
        zorder=4,
        label="Threshold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title(
        f"Layer 2 — LLM grader pass rates "
        f"(BKH cal n={bkh_n} · VA n={n_va} · v1 raw · v2 corpus-calibrated)",
        color=NAVY,
        fontweight="bold",
        fontsize=10,
    )
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)


def _horizontal_cohort_panel(
    ax,
    metrics: list[tuple[str, str, float | None]],
    cohort_data: dict[str, dict[str, float | None]],
    *,
    panel_title: str,
    show_thresholds: bool = True,
) -> None:
    """Grouped horizontal bars: one row per metric, three cohort bars."""
    n_metrics = len(metrics)
    n_cohorts = len(_COHORT_ORDER)
    bar_h = 0.22
    group_h = bar_h * n_cohorts + 0.12
    y_centers = np.arange(n_metrics) * group_h

    for mi, (key, _label, thr) in enumerate(metrics):
        y0 = y_centers[mi]
        for ci, cohort in enumerate(_COHORT_ORDER):
            val = cohort_data.get(cohort, {}).get(key)
            if val is None:
                continue
            y = y0 + (ci - (n_cohorts - 1) / 2) * bar_h
            color = _COHORT_COLORS[cohort]
            ax.barh(y, val, height=bar_h * 0.88, color=color, alpha=0.88, zorder=3)
            ax.text(
                val + 1.2,
                y,
                f"{val:.0f}%",
                va="center",
                ha="left",
                fontsize=7.5,
                color=NAVY,
                fontweight="bold",
            )
            if show_thresholds and thr is not None:
                ax.plot(
                    [thr, thr],
                    [y - bar_h * 0.44, y + bar_h * 0.44],
                    color=AMBER,
                    linewidth=1.2,
                    linestyle="--",
                    zorder=2,
                    alpha=0.85,
                )

    ax.set_yticks(y_centers)
    ax.set_yticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Score (%)")
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=SLATE, zorder=0)
    ax.set_title(panel_title, fontweight="bold", color=NAVY, fontsize=10, loc="left", pad=8)


def _golden_benchmark_figure(
    *,
    include_layer1: bool = True,
    include_layer2: bool = True,
    out_name: str,
    suptitle: str,
) -> Path:
    layer1, layer2, bkh_n, n_va = _benchmark_cohort_data()
    bkh_h = _bkh_heuristic_live()
    n_panels = int(include_layer1) + int(include_layer2)
    fig_h = 3.2 + n_panels * 3.8
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, fig_h))
    _style()
    if n_panels == 1:
        axes = [axes]

    ai = 0
    if include_layer1:
        va_rated = _load_golden_stats().get("sentiment", {})
        va_n = va_rated.get("n_liked", 0) + va_rated.get("n_disliked", 0)
        _horizontal_cohort_panel(
            axes[ai],
            _LAYER1_METRICS,
            layer1,
            panel_title="",
            show_thresholds=True,
        )
        axes[ai].set_title(
            f"Layer 1 — Heuristic baseline "
            f"(BKH n={bkh_h.get('n_rated', 0):,} rated · VA n={va_n:,} rated · P/R/F1 v2 = BKH URL overlap)",
            fontweight="bold",
            color=NAVY,
            fontsize=10,
            loc="left",
            pad=8,
        )
        ai += 1

    if include_layer2:
        from evals.metrics.calibration.grader_scope import COMPARISON_LAYER2_METRICS

        llm_metrics = list(COMPARISON_LAYER2_METRICS)
        _horizontal_cohort_panel(
            axes[ai],
            llm_metrics,
            layer2,
            panel_title="",
            show_thresholds=True,
        )
        axes[ai].set_title(
            f"Layer 2 — primary LLM gates (BKH cal n={bkh_n} · VA n={n_va} · grounding N/A on BKH)",
            fontweight="bold",
            color=NAVY,
            fontsize=10,
            loc="left",
            pad=8,
        )

    handles = [
        mpatches.Patch(color=_COHORT_COLORS[c], label=_COHORT_DISPLAY.get(c, c))
        for c in _COHORT_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(suptitle, fontsize=12, color=NAVY, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.45)
    return _save(fig, out_name)


def fig_golden_perf_compare() -> Path:
    """PM A/B chart: Layer 1 heuristics + primary Layer 2 gates, BKH | VA v1 | VA v2."""
    return _golden_benchmark_figure(
        include_layer1=True,
        include_layer2=True,
        out_name="golden_perf_compare",
        suptitle="BKH vs VA staging — A/B (primary gates; BKH grounding not scored)",
    )


def fig_golden_perf_heuristic() -> Path:
    """Layer 1 only — satisfaction, weighted resolution, citation-proxy P/R/F1."""
    return _golden_benchmark_figure(
        include_layer1=True,
        include_layer2=False,
        out_name="golden_perf_heuristic",
        suptitle="Layer 1 — Heuristic Baseline: BKH vs VA v1 (strict) vs VA v2 (corpus-adjusted)",
    )


def fig_golden_perf_llm() -> Path:
    """Layer 2 only — LLM pass rates, BKH calibration vs VA v1 raw vs VA v2 calibrated."""
    return _golden_benchmark_figure(
        include_layer1=False,
        include_layer2=True,
        out_name="golden_perf_llm",
        suptitle="Layer 2 — LLM Grader Pass Rates: BKH vs VA v1 vs VA v2",
    )


def fig_golden_benchmark() -> Path:
    """Export all three matching horizontal benchmark figures."""
    fig_golden_perf_compare()
    fig_golden_perf_heuristic()
    fig_golden_perf_llm()
    return OUT_DIR / "golden_perf_compare.svg"


def fig_heuristic_llm_compare() -> Path:
    """Compare Layer-1 citation proxy (P/R/F1) vs Layer-2 LLM pass rates — BKH vs Golden."""
    gh = _golden_heuristic_retrieval()
    bh = _BKH_HEURISTIC
    q = _load_golden_quality()
    q.get("grader_summary", {})
    n_golden_llm = q.get("n_queries", 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _style()

    # ── Left: Heuristic citation proxy P / R / F1 ──
    ax = axes[0]
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    w = 0.35
    bkh_vals = [bh["precision"], bh["recall"], bh["f1"]]
    g_vals = [gh["precision"], gh["recall"], gh["f1"]]
    ax.bar(
        x - w / 2,
        bkh_vals,
        w,
        label=f"BKH (n={bh['n_rated']:,} rated)",
        color=NAVY,
        alpha=0.85,
        zorder=3,
    )
    ax.bar(
        x + w / 2,
        g_vals,
        w,
        label=f"VA Golden (n={gh['n_rated']:,} rated)",
        color=TEAL,
        alpha=0.85,
        zorder=3,
    )
    ax.axhline(75, color=AMBER, linestyle="--", linewidth=1, alpha=0.7, label="P threshold 75%")
    ax.axhline(65, color=AMBER, linestyle=":", linewidth=1, alpha=0.5, label="R threshold 65%")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.set_title(
        "Layer 1 — Citation Proxy (has_sources → liked)\nPre-LLM · full corpus",
        fontweight="bold",
        color=NAVY,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.yaxis.grid(True, color=SLATE)

    # ── Right: LLM pass rates v2 (corpus-calibrated, 7+1 graders) ──
    ax2 = axes[1]
    q_v1 = _load_golden_quality_v1()
    cal_idx = build_calibration_index()
    compute_llm_pass_summary(q_v1, GOLDEN_LLM_METRICS_V1, calibrated=False)
    gs_cal = compute_llm_pass_summary(
        q_v1, GOLDEN_LLM_METRICS_V1, calibrated=True, cal_index=cal_idx
    )
    llm_labels = [m[1].replace(" ", "\n") for m in GOLDEN_LLM_METRICS_V1]
    llm_keys = [(m[0], m[2]) for m in GOLDEN_LLM_METRICS_V1]
    g_rates = [round(gs_cal.get(k, {}).get("pass_rate", 0) * 100) for k, _ in llm_keys]
    tholds = [t for _, t in llm_keys]
    colors = [GREEN if r >= t else RED for r, t in zip(g_rates, tholds, strict=False)]
    x2 = np.arange(len(llm_labels))
    ax2.bar(x2, g_rates, color=colors, width=0.5, zorder=3)
    ax2.plot(x2, tholds, "o--", color=NAVY, linewidth=1.5, markersize=5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(llm_labels, fontsize=7)
    ax2.set_ylabel("Pass Rate (%)")
    ax2.set_ylim(0, 105)
    n_golden_llm = q_v1.get("n_queries", n_golden_llm)
    ax2.set_title(
        f"Layer 2 — LLM Pass Rates v2 calibrated (n={n_golden_llm})\n"
        f"3 custom + 2 DeepEval + 2 RAGAS + source_match · see va_pass_rates_v1 raw",
        fontweight="bold",
        color=NAVY,
        fontsize=9,
    )
    ax2.yaxis.grid(True, color=SLATE)
    for i, val in enumerate(g_rates):
        ax2.text(i, val + 1.5, f"{val}%", ha="center", fontsize=8, fontweight="bold", color=NAVY)

    fig.suptitle(
        "Heuristic Retrieval Proxy vs LLM Graders — Apples-to-oranges by design",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, "heuristic_llm_compare")


# ---------------------------------------------------------------------------
# Figure 6 — Statistical power curve
# ---------------------------------------------------------------------------


def fig_power_curve() -> Path:
    n_vals = [20, 44, 100, 200, 500]
    mde = [0.18, 0.12, 0.08, 0.057, 0.036]
    labels = ["n=20", "n=44\n(current)", "n=100", "n=200", "n=500\n(merge gate)"]
    colors = [RED, AMBER, AMBER, TEAL, GREEN]

    fig, ax = plt.subplots(figsize=(7.5, 4))
    _style()
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)

    bars = ax.bar(range(len(n_vals)), mde, color=colors, width=0.55, zorder=3)
    ax.axhline(0.02, color=GREEN, linewidth=1.8, linestyle="--", label="Merge gate: ΔMRR ≥ 0.02")
    ax.set_xticks(range(len(n_vals)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Min Detectable ΔMRR  (80% power, α = 0.05)")
    ax.set_title("A/B Test Statistical Power — Required Sample Size")
    ax.legend(fontsize=9, framealpha=0.9)

    for bar, val in zip(bars, mde, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.003,
            f"±{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=NAVY,
            fontweight="bold",
        )

    fig.tight_layout()
    return _save(fig, "power_curve")


# ---------------------------------------------------------------------------
# Golden dataset stats loader
# ---------------------------------------------------------------------------


def _golden_stats_path() -> Path:
    from evals.reports.paths import va_staging_all_responses_stats_path

    return va_staging_all_responses_stats_path()


def _load_golden_stats() -> dict:
    stats_path = _golden_stats_path()
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Golden stats not found: {stats_path}\n"
            "Run: make golden-report  (or eval_stats on golden_all_responses.jsonl)"
        )
    data = json.loads(stats_path.read_text())
    file_stats = data.get("stats", {})
    if not file_stats:
        raise ValueError(f"No stats in {stats_path}")
    return next(iter(file_stats.values()))


def _extract_file_stats(data: dict | None) -> dict:
    """Accept full export JSON or a single file's stats dict."""
    if not data:
        return {}
    if "n_total" in data:
        return data
    file_stats = list(data.get("stats", {}).values())
    return file_stats[0] if file_stats else {}


# ---------------------------------------------------------------------------
# Figure 7 — BKH corpus overview (grouped bars: key stats at a glance)
# ---------------------------------------------------------------------------


def fig_bkh_overview() -> Path:
    """Pie / donut for response type distribution."""
    labels = ["Has Sources", "Unknown\n(A_retrieval)", "Escalation", "Clarification", "Interrupted"]
    sizes = [74.2, 17.3, 5.0, 1.9, 1.7]
    colors_pie = [TEAL, AMBER, RED, PURPLE, MID]
    explode = [0, 0.06, 0, 0, 0]  # pull out the "unknown" slice

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _style()

    # Left: response type donut
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors_pie,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color(NAVY)
    ax.set_title("Response Type Distribution\n(n = 69,198 turns)", pad=12)

    # Right: failure taxonomy donut
    ax2 = axes[1]
    ft_labels = [
        "Unrated",
        "A_retrieval\n(no KB ans.)",
        "B_language\n(lang mismatch)",
        "E_grounding\n(cited+disliked)",
        "C_friction",
        "no_failure\n(liked)",
    ]
    ft_sizes = [79.6, 15.1, 4.2, 0.6, 0.2, 0.3]
    ft_colors = [SLATE, AMBER, PURPLE, RED, "#f97316", GREEN]
    wedges2, texts2, autotexts2 = ax2.pie(
        ft_sizes,
        labels=ft_labels,
        colors=ft_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts2:
        t.set_fontsize(8.5)
        t.set_color(NAVY)
    ax2.set_title("Failure Mode Taxonomy\n(share of all turns)", pad=12)

    fig.suptitle(
        "BKH Dataset — 69,198 turns · 30,557 conversations · 1.7% rated",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, "bkh_overview")


# ---------------------------------------------------------------------------
# Figure 8 — Prompt version evolution (v2 → v3 Answer Relevancy)
# ---------------------------------------------------------------------------


def fig_prompt_evolution() -> Path:
    categories = ["Liked (v2)", "Disliked (v2)", "Liked (v3)", "Disliked (v3)"]
    values = [0.600, 0.680, 0.885, 0.670]
    colors = [GREEN, RED, GREEN, RED]
    hatches = ["", "", "//", "//"]

    fig, ax = plt.subplots(figsize=(7, 4))
    _style()
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)

    bars = ax.bar(
        categories,
        values,
        color=colors,
        hatch=hatches,
        width=0.55,
        zorder=3,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Answer Relevancy Score")
    ax.set_title("Grader Prompt Evolution — v2 → v3\nAnswer Relevancy (n = 50 calibration tasks)")
    ax.axhline(0.75, color=NAVY, linewidth=1.5, linestyle="--", alpha=0.6, label="Threshold 75%")

    legend = [
        mpatches.Patch(color=GREEN, label="Liked turns"),
        mpatches.Patch(color=RED, label="Disliked turns"),
        mpatches.Patch(facecolor="white", edgecolor=MID, hatch="//", label="v3 (CoT prompt)"),
    ]
    ax.legend(handles=legend, fontsize=9, framealpha=0.9)

    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
            fontweight="bold",
        )

    fig.tight_layout()
    return _save(fig, "prompt_evolution")


# ---------------------------------------------------------------------------
# Figure 9 — Topic sentiment comparison (liked vs disliked top 10)
# ---------------------------------------------------------------------------


def fig_topic_sentiment(
    bkh_stats: dict | None = None,
    *,
    fig_name: str = "topic_sentiment",
    title: str | None = None,
) -> Path:
    """Side-by-side top-10 liked and top-10 disliked topics with has_sources% overlay."""
    cats: dict = _extract_file_stats(bkh_stats).get("categories", {})

    _excl = {"unspecified"}

    # Build top-10 disliked (by dislike_rate, min 5 turns)
    dislike_items = sorted(
        [(t, v) for t, v in cats.items() if v.get("count", 0) >= 5 and t not in _excl],
        key=lambda x: x[1].get("dislike_rate", 0),
        reverse=True,
    )[:10]

    # Build top-10 liked (by like_rate, min 5 turns)
    like_items = sorted(
        [
            (t, v)
            for t, v in cats.items()
            if v.get("count", 0) >= 5 and "like_rate" in v and t not in _excl
        ],
        key=lambda x: x[1].get("like_rate", 0),
        reverse=True,
    )[:10]

    # Fallback static data if stats missing
    if not dislike_items:
        dislike_items = [
            (
                "Managing Accounting Entries",
                {"dislike_rate": 0.035, "sources_rate": 0.64, "count": 200},
            ),
            (
                "Human Support Chat Assistance",
                {"dislike_rate": 0.034, "sources_rate": 0.54, "count": 932},
            ),
            (
                "Account Usage and Differences",
                {"dislike_rate": 0.033, "sources_rate": 0.34, "count": 61},
            ),
            (
                "Invoice Sending and Management Issues",
                {"dislike_rate": 0.032, "sources_rate": 0.63, "count": 281},
            ),
            ("Bank Connection Issues", {"dislike_rate": 0.030, "sources_rate": 0.62, "count": 168}),
            (
                "Accounting and Payment Integration",
                {"dislike_rate": 0.029, "sources_rate": 0.70, "count": 244},
            ),
            (
                "Creating and Managing Account Plans",
                {"dislike_rate": 0.028, "sources_rate": 0.73, "count": 143},
            ),
            (
                "Bank Reconciliation Issues",
                {"dislike_rate": 0.023, "sources_rate": 0.76, "count": 87},
            ),
            (
                "Tax Reporting of Financial Results",
                {"dislike_rate": 0.019, "sources_rate": 0.76, "count": 53},
            ),
            (
                "Invoice Payment and Reconciliation",
                {"dislike_rate": 0.018, "sources_rate": 0.75, "count": 57},
            ),
        ]

    def _short(label: str, n: int = 32) -> str:
        return label if len(label) <= n else label[: n - 1] + "…"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _style()

    # ---- Left: disliked topics ----
    ax = axes[0]
    d_labels = [_short(t) for t, _ in reversed(dislike_items)]
    d_rates = [v.get("dislike_rate", 0) * 100 for _, v in reversed(dislike_items)]
    d_src = [v.get("sources_rate", 0) * 100 for _, v in reversed(dislike_items)]
    y = np.arange(len(d_labels))

    ax.barh(y, d_rates, height=0.5, color=RED, alpha=0.75, zorder=3, label="Dislike rate")
    ax.plot(d_src, y, "o", color=NAVY, markersize=6, zorder=4, label="Has sources %")
    ax.set_yticks(y)
    ax.set_yticklabels(d_labels, fontsize=9)
    ax.set_xlabel("Dislike rate (%)")
    ax.set_title("Top 10 Disliked Topics", color=NAVY, fontweight="bold")
    ax.set_xlim(0, max(d_rates) * 1.5 if d_rates else 10)
    for i, (rate, _src) in enumerate(zip(d_rates, d_src, strict=False)):
        ax.text(
            rate + 0.05, i, f"{rate:.1f}%", va="center", fontsize=8, color=RED, fontweight="bold"
        )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # ---- Right: liked topics ----
    ax2 = axes[1]
    if like_items:
        l_labels = [_short(t) for t, _ in reversed(like_items)]
        l_rates = [v.get("like_rate", 0) * 100 for _, v in reversed(like_items)]
        l_src = [v.get("sources_rate", 0) * 100 for _, v in reversed(like_items)]
        y2 = np.arange(len(l_labels))
        ax2.barh(y2, l_rates, height=0.5, color=GREEN, alpha=0.75, zorder=3, label="Like rate")
        ax2.plot(l_src, y2, "o", color=NAVY, markersize=6, zorder=4, label="Has sources %")
        ax2.set_yticks(y2)
        ax2.set_yticklabels(l_labels, fontsize=9)
        ax2.set_xlabel("Like rate (%)")
        ax2.set_title("Top 10 Liked Topics", color=NAVY, fontweight="bold")
        ax2.set_xlim(0, max(l_rates) * 1.5 if l_rates else 10)
        for i, (rate, _src) in enumerate(zip(l_rates, l_src, strict=False)):
            ax2.text(
                rate + 0.05,
                i,
                f"{rate:.1f}%",
                va="center",
                fontsize=8,
                color=GREEN,
                fontweight="bold",
            )
        ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)
    else:
        ax2.text(
            0.5,
            0.5,
            "Run eval_stats to\ngenerate liked-topic data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            color=MID,
            fontsize=11,
        )
        ax2.set_title("Top 10 Liked Topics", color=NAVY, fontweight="bold")

    fig.suptitle(
        title or "Topic Sentiment — Top 10 Liked vs Disliked  (dots = has_sources %)",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, fig_name)


# ---------------------------------------------------------------------------
# Figure 10 — Retrieval proxy (pseudo-F1, confusion matrix)
# ---------------------------------------------------------------------------

_RETRIEVAL_SCORE_METRICS: list[tuple[str, str]] = [
    ("satisfaction", "Archive like %"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]


def _build_cohort_scores(
    stats: dict,
    rp: dict | None = None,
    *,
    use_full_recall: bool = False,
) -> dict[str, float | None]:
    """Layer-1 score row (%): satisfaction, resolution, citation-proxy P/R/F1."""
    h = _golden_heuristic_retrieval_from_stats(stats, rp)
    if rp and use_full_recall:
        if rp.get("recall_full") is not None:
            h["recall"] = float(rp["recall_full"]) * 100
        if rp.get("f1_full") is not None:
            h["f1"] = float(rp["f1_full"]) * 100
    return h


def _scores_from_adjusted_proxy(adj: dict) -> dict[str, float | None]:
    """Layer-1 scores from coverage-gap-adjusted overlap CM + north-star exclusions."""
    out: dict[str, float | None] = {}
    if adj.get("satisfaction") is not None:
        out["satisfaction"] = float(adj["satisfaction"])
    if adj.get("precision") is not None:
        out["precision"] = float(adj["precision"]) * 100
    # Adj. CM has low FN — recall_full (TP/(TP+FN)) is meaningful post-coverage fix
    if adj.get("recall_full") is not None:
        out["recall"] = float(adj["recall_full"]) * 100
    if adj.get("f1_full") is not None:
        out["f1"] = float(adj["f1_full"]) * 100
    return out


def _draw_retrieval_metrics_panel(
    ax,
    cohort: dict[str, float | None],
    *,
    cohort_label: str = "VA",
    bkh: dict[str, float | None] | None = None,
    alt: dict[str, float | None] | None = None,
    alt_label: str = "Cov. adj.",
    y_hat_line: str = "",
    dlr_note: str = "",
    alt_footnote: str = "",
) -> None:
    """Grouped score bars: satisfaction + resolution + P/R/F1; optional BKH + cov-adj."""
    keys = [k for k, _ in _RETRIEVAL_SCORE_METRICS]
    labels = [lbl for _, lbl in _RETRIEVAL_SCORE_METRICS]
    x = np.arange(len(labels))
    n_series = 1 + int(bkh is not None) + int(alt is not None)
    w = 0.72 / max(n_series, 1)
    offset = (n_series - 1) / 2

    def _val(scores: dict[str, float | None] | None, key: str) -> float:
        if not scores:
            return 0.0
        v = scores.get(key)
        return float(v) if v is not None else 0.0

    si = 0
    if bkh is not None:
        bx = x + (si - offset) * w
        vals = [_val(bkh, k) for k in keys]
        bars = ax.bar(bx, vals, w, color=NAVY, alpha=0.88, zorder=3, label="BKH")
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=NAVY,
                    fontweight="bold",
                )
        si += 1

    cx = x + (si - offset) * w
    cvals = [_val(cohort, k) for k in keys]
    bars = ax.bar(cx, cvals, w, color=TEAL, alpha=0.88, zorder=3, label=cohort_label)
    for bar, v in zip(bars, cvals, strict=False):
        if v > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 1.5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color=NAVY,
                fontweight="bold",
            )
    si += 1

    if alt is not None:
        ax2_vals = [_val(alt, k) for k in keys]
        ax_x = x + (si - offset) * w
        bars = ax.bar(ax_x, ax2_vals, w, color=PURPLE, alpha=0.88, zorder=3, label=alt_label)
        for bar, v in zip(bars, ax2_vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=NAVY,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        f"Layer 1 Scores — satisfaction + citation proxy P/R/F1\n{y_hat_line}",
        color=NAVY,
        fontweight="bold",
        fontsize=10,
    )
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    if alt_footnote:
        ax.text(
            0.5,
            -0.20,
            alt_footnote,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=7.5,
            color=MID,
            style="italic",
        )
    elif dlr_note:
        ax.text(
            0.5,
            -0.16,
            f"⚠ Eval dislike:like={dlr_note} (natural ~2:1) — precision biased low vs production",
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=7.5,
            color=MID,
            style="italic",
        )


_V3_SCENARIO_ORDER = ["bkh", "va v2", "reclass verified", "HITL max"]
_V3_SCENARIO_COLORS = {
    "bkh": NAVY,
    "va v2": TEAL,
    "reclass verified": PURPLE,
    "HITL max": AMBER,
}


def _draw_retrieval_scenario_compare_panel(
    ax,
    scenarios: dict[str, dict[str, Any]],
    *,
    metrics: list[tuple[str, str]] | None = None,
    footnote: str = "",
) -> None:
    """Grouped score bars: BKH | VA v2 | reclass adjusted | HITL upper bound."""
    metric_list = metrics or _RETRIEVAL_SCORE_METRICS
    keys = [k for k, _ in metric_list]
    labels = [lbl for _, lbl in metric_list]
    x = np.arange(len(labels))
    order = [k for k in _V3_SCENARIO_ORDER if k in scenarios]
    n_series = len(order)
    w = 0.72 / max(n_series, 1)
    offset = (n_series - 1) / 2

    for si, key in enumerate(order):
        sc = scenarios[key]
        scores = sc.get("scores") or {}
        bx = x + (si - offset) * w
        label = sc.get("label", key)
        color = _V3_SCENARIO_COLORS.get(key, TEAL)
        vals = [float(scores.get(k) or 0) for k in keys]
        bars = ax.bar(bx, vals, w, color=color, alpha=0.88, zorder=3, label=label)
        for bar, v in zip(bars, vals, strict=False):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.5,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=NAVY,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Layer 1 Scores — BKH | VA v2 | reclass | HITL (Recall = TP/(TP+FN))",
        color=NAVY,
        fontweight="bold",
        fontsize=10,
    )
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    if footnote:
        ax.text(
            0.5,
            -0.22,
            footnote,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=7.5,
            color=MID,
            style="italic",
        )


def fig_retrieval_proxy(
    bkh_stats: dict | None = None,
    *,
    title_suffix: str = "",
    out_name: str | None = None,
    version_note: str = "",
    y_hat_subtitle: str | None = None,
    cell_subs: tuple[str, str, str, str] | None = None,
    alt_proxy: dict | None = None,
    compare_bkh: bool = False,
    cohort_label: str = "BKH",
    cohort_scores: dict[str, float | None] | None = None,
) -> Path:
    """Confusion matrix + P/R/F1 bars for the citation-as-retrieval proxy."""
    tp, fp, tn, fn = 193, 442, 83, 427
    prec, recall, f1 = 0.304, 0.785, 0.438
    n_total = 69198
    dlr_note = "3.1:1"

    file_stats = _extract_file_stats(bkh_stats)
    if file_stats:
        n_total = file_stats.get("n_total", n_total)
        s = file_stats.get("sentiment", {})
        dlr = s.get("dislike_like_ratio")
        if dlr:
            dlr_note = f"{dlr:.1f}:1"
        rp = file_stats.get("retrieval_proxy", {})
        cm = rp.get("confusion", {})
        tp = cm.get("tp", tp)
        fp = cm.get("fp", fp)
        tn = cm.get("tn", tn)
        fn = cm.get("fn", fn)
        prec = rp.get("precision", prec)
        recall = rp.get("recall", recall)
        f1 = rp.get("f1", f1)

    rp_data = file_stats.get("retrieval_proxy", {}) if file_stats else {}
    if cohort_scores is None and file_stats:
        cohort_scores = _build_cohort_scores(
            file_stats,
            rp_data,
            use_full_recall=False,
        )
    bkh_scores = _build_cohort_scores(_load_bkh_stats_dict()) if compare_bkh else None
    alt_scores = _scores_from_adjusted_proxy(alt_proxy) if alt_proxy else None
    alt_footnote = ""
    if alt_proxy:
        excl = alt_proxy.get("fn_excluded_missing_cite", 0)
        fn_adj = alt_proxy.get("fn", alt_proxy.get("confusion", {}).get("fn", 0))
        alt_footnote = (
            f"Purple: excl. {excl} missing_cite from satisfaction denom (+~6pp); "
            f"CM FN→{fn_adj} (coverage_gap only). P unchanged; R/F1 use TP/(TP+FN)"
        )

    y_hat_line = y_hat_subtitle or "(ŷ = has_source, y = liked)"
    subs = cell_subs or (
        "liked + sourced",
        "liked + no source",
        "disliked + sourced",
        "disliked + no source",
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1.35]})
    _style()

    # ---- Left: confusion matrix ----
    ax = axes[0]
    ax.set_xlim(-0.35, 2.15)
    ax.set_ylim(-0.05, 2.35)
    ax.axis("off")

    cell_data = [
        # (x, y, label, sub, bg, fg)
        (0, 1, f"TP\n{tp:,}", subs[0], "#d4edda", "#155724"),
        (1, 1, f"TN\n{tn:,}", subs[1], "#e8f4ea", "#4a7c59"),
        (0, 0, f"FP\n{fp:,}", subs[2], "#f8d7da", "#721c24"),
        (1, 0, f"FN\n{fn:,}", subs[3], "#fff3e0", "#a04000"),
    ]
    for x, y, label, sub, bg, fg in cell_data:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=bg, zorder=2))
        ax.text(
            x + 0.5,
            y + 0.64,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color=fg,
            fontweight="bold",
            zorder=3,
        )
        ax.text(x + 0.5, y + 0.28, sub, ha="center", va="center", fontsize=7, color=fg, zorder=3)

    # Column / row headers (extra margin avoids overlap with title)
    ax.text(0.5, 2.22, "has source  (ŷ=1)", ha="center", va="bottom", fontsize=8.5, color=MID)
    ax.text(1.5, 2.22, "no source  (ŷ=0)", ha="center", va="bottom", fontsize=8.5, color=MID)
    ax.text(-0.12, 1.5, "liked\n(y=1)", ha="right", va="center", fontsize=8.5, color=MID)
    ax.text(-0.12, 0.5, "disliked\n(y=0)", ha="right", va="center", fontsize=8.5, color=MID)
    ax.set_title(
        "Citation Proxy — Confusion Matrix",
        color=NAVY,
        fontweight="bold",
        pad=28,
        fontsize=11,
    )
    ax.text(
        0.5,
        2.05,
        "(rated turns only, unrated excluded)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MID,
        transform=ax.transData,
    )

    # ---- Right: Layer 1 scores (satisfaction, resolution, P/R/F1) ----
    _draw_retrieval_metrics_panel(
        axes[1],
        cohort_scores or {},
        cohort_label=cohort_label,
        bkh=bkh_scores,
        alt=alt_scores,
        alt_label="Cov. adj.",
        y_hat_line=y_hat_line,
        dlr_note=dlr_note if not alt_footnote else "",
        alt_footnote=alt_footnote,
    )

    suffix = title_suffix or "BKH"
    title = f"{suffix} Citation-Based Retrieval Proxy  (pre-grader, n={n_total:,} turns)"
    if version_note:
        title = f"{title}  ·  {version_note}"
    fig.suptitle(title, fontsize=12, color=NAVY, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.86, bottom=0.22 if alt_proxy else 0.16, wspace=0.38)
    if out_name is None:
        out_name = "retrieval_proxy" if not title_suffix else "golden_retrieval_proxy"
    return _save(fig, out_name)


# ---------------------------------------------------------------------------
# Golden dataset — VA agent response archive (mirrors BKH Tab 02 figures)
# ---------------------------------------------------------------------------

_RT_LABELS = {
    "has_sources": "Has Sources",
    "unknown": "Unknown\n(A_retrieval)",
    "escalation": "Escalation",
    "clarification": "Clarification",
    "interrupted": "Interrupted",
}

_FT_LABELS = {
    "n/a": "Unrated",
    "A_retrieval": "A_retrieval\n(no KB ans.)",
    "B_language": "B_language\n(lang mismatch)",
    "E_grounding": "E_grounding\n(cited+disliked)",
    "C_friction": "C_friction",
    "no_failure": "no_failure\n(liked)",
    "unclassified": "unclassified",
    "D_scope": "D_scope",
}

_FT_COLORS = {
    "no_failure": GREEN,
    "n/a": SLATE,
    "unclassified": AMBER,
    "A_retrieval": AMBER,
    "B_language": PURPLE,
    "E_grounding": RED,
    "C_friction": "#f97316",
    "D_scope": MID,
}

_LANG_DISPLAY = {
    "da": "Danish",
    "en": "English",
    "de": "German",
    "unknown": "Unknown",
    "nl": "Dutch",
}


def _golden_donut(
    sizes: list[float],
    labels: list[str],
    colors: list[str],
    fig_name: str,
    *,
    explode: list[float] | None = None,
) -> Path:
    """Single donut — percentages in legend only (avoids wedge label overlap)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _style()
    if explode is None:
        explode = [0.0] * len(sizes)
    legend_labels = [f"{lab} — {sz:.1f}%" for lab, sz in zip(labels, sizes, strict=False)]
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        explode=explode,
        startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
    )
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8.5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.02, right=0.62, top=0.96, bottom=0.06)
    return _save(fig, fig_name)


def fig_golden_response_type(
    golden_stats: dict | None = None,
    *,
    out_name: str = "va_staging_response_type",
) -> Path:
    stats = golden_stats or _load_golden_stats()
    stats["n_total"]
    rt = stats.get("response_types", {})
    labels, sizes, colors, explode = [], [], [], []
    palette = [TEAL, AMBER, RED, PURPLE, MID]
    for i, key in enumerate(
        ("has_sources", "unknown", "escalation", "clarification", "interrupted")
    ):
        if key not in rt:
            continue
        labels.append(_RT_LABELS.get(key, key).replace("\n", " "))
        sizes.append(rt[key]["rate"] * 100)
        colors.append(palette[i % len(palette)])
        explode.append(0.04 if key == "unknown" else 0.0)
    return _golden_donut(sizes, labels, colors, out_name, explode=explode)


def fig_golden_failure_mode(
    golden_stats: dict | None = None,
    *,
    out_name: str = "va_staging_failure_mode",
    ft_order: list[str] | None = None,
) -> Path:
    stats = golden_stats or _load_golden_stats()
    stats["n_total"]
    ft = stats.get("failure_types", {})
    if ft_order is None:
        n_rated_pct = stats.get("sentiment", {}).get("rating_coverage_rate", 0)
        if n_rated_pct and n_rated_pct < 0.1:
            ft_order = [
                "n/a",
                "unknown",
                "A_retrieval",
                "B_language",
                "E_grounding",
                "C_friction",
                "no_failure",
                "unclassified",
                "D_scope",
            ]
        else:
            ft_order = [
                "unclassified",
                "no_failure",
                "E_grounding",
                "n/a",
                "A_retrieval",
                "B_language",
                "C_friction",
                "D_scope",
                "unknown",
            ]
    labels, sizes, colors = [], [], []
    for key in ft_order:
        if key not in ft:
            continue
        short = _FT_LABELS.get(key, key).replace("\n", " ")
        labels.append(short)
        sizes.append(ft[key]["rate"] * 100)
        colors.append(_FT_COLORS.get(key, MID))
    return _golden_donut(sizes, labels, colors, out_name)


def fig_golden_source_count(
    golden_stats: dict | None = None,
    *,
    out_name: str = "va_staging_source_count",
) -> Path:
    """Bar chart: turn counts by sources cited (matches eval_framework_report §1)."""
    stats = golden_stats or _load_golden_stats()
    stats["n_total"]
    dist = stats.get("retrieval_proxy", {}).get("source_count_distribution", {})
    if not dist:
        raise ValueError("source_count_distribution missing from stats")

    keys = sorted(dist, key=lambda x: int(x))
    counts = [dist[k] for k in keys]
    bar_colors = [
        AMBER if k == "0" else TEAL if k == "1" else NAVY if k == "2" else PURPLE for k in keys
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _style()
    plt.rcParams["axes.grid.axis"] = "y"
    ax.yaxis.grid(True, color=SLATE)
    ax.xaxis.grid(False)
    ax.bar(keys, counts, color=bar_colors, width=0.55, zorder=3)
    ax.set_xlabel("Sources cited per response")
    ax.set_ylabel("Turns")
    ymax = max(counts) * 1.12 if counts else 10
    ax.set_ylim(0, ymax)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.94, bottom=0.20)
    return _save(fig, out_name)


def fig_golden_language(
    golden_stats: dict | None = None,
    *,
    out_name: str = "va_staging_language",
) -> Path:
    """Query language donut (eval_framework_report §1)."""
    stats = golden_stats or _load_golden_stats()
    n = stats["n_total"]
    lang = stats.get("language_breakdown", {})
    if not lang:
        raise ValueError("language_breakdown missing from stats")

    order = ["da", "unknown", "en", "de", "nl"]
    labels, sizes, colors = [], [], []
    lang_colors = {"da": NAVY, "en": TEAL, "de": PURPLE, "unknown": AMBER, "nl": MID}
    for key in order:
        if key not in lang:
            continue
        labels.append(_LANG_DISPLAY.get(key, key.title()))
        sizes.append(lang[key] / n * 100)
        colors.append(lang_colors.get(key, MID))
    for key in sorted(lang, key=lambda k: -lang[k]):
        if key in order:
            continue
        labels.append(_LANG_DISPLAY.get(key, key.title()))
        sizes.append(lang[key] / n * 100)
        colors.append(MID)

    return _golden_donut(sizes, labels, colors, out_name)


def fig_bkh_response_type(bkh_stats: dict | None = None) -> Path:
    return fig_golden_response_type(
        bkh_stats or _load_bkh_stats_dict(),
        out_name="bkh_response_type",
    )


def fig_bkh_failure_mode(bkh_stats: dict | None = None) -> Path:
    return fig_golden_failure_mode(
        bkh_stats or _load_bkh_stats_dict(),
        out_name="bkh_failure_mode",
    )


def fig_bkh_source_count(bkh_stats: dict | None = None) -> Path:
    return fig_golden_source_count(
        bkh_stats or _load_bkh_stats_dict(),
        out_name="bkh_source_count",
    )


def fig_bkh_language(bkh_stats: dict | None = None) -> Path:
    return fig_golden_language(
        bkh_stats or _load_bkh_stats_dict(),
        out_name="bkh_language",
    )


def _golden_stats_with_retrieval(rp: dict) -> dict:
    """Golden stats dict with retrieval_proxy replaced (for v2 calibrated CM)."""
    stats = dict(_load_golden_stats())
    stats["retrieval_proxy"] = {
        "confusion": rp["confusion"],
        "precision": rp["precision"],
        "recall": rp["recall"],
        "f1": rp["f1"],
        "recall_full": rp.get("recall_full"),
        "f1_full": rp.get("f1_full"),
    }
    return stats


def fig_va_retrieval_proxy_v1() -> Path:
    """VA golden citation proxy v1 — same template as golden_retrieval_proxy (strict ŷ)."""
    stats = _load_golden_stats()
    return fig_retrieval_proxy(
        stats,
        title_suffix="VA Golden",
        out_name="va_retrieval_proxy_v1",
        version_note="v1 · strict ŷ=has_source",
        compare_bkh=True,
        cohort_label="VA v1",
        cohort_scores=_build_cohort_scores(stats),
    )


def fig_va_retrieval_proxy_v2() -> Path:
    """VA golden citation proxy v2 — golden_retrieval_proxy template + corpus-adjusted ŷ."""
    from evals.pipelines.datasets import load_jsonl
    from evals.reports.paths import resolve_golden_responses_path

    tasks = load_jsonl(resolve_golden_responses_path())
    rp = compute_va_retrieval_proxy_calibrated(tasks, build_calibration_index())
    stats = _golden_stats_with_retrieval(rp)
    va_scores = _build_cohort_scores(_load_golden_stats(), rp, use_full_recall=True)
    bd = rp.get("breakdown", {})
    edge = bd.get("edge_no_overlap", 0)
    verify = bd.get("verify_cited_no_match", 0)
    cov = rp.get("coverage_gap_adjusted", {})
    return fig_retrieval_proxy(
        stats,
        title_suffix="VA Golden",
        out_name="va_retrieval_proxy_v2",
        version_note=(
            f"v2 · BKH URL overlap · CM n={rp.get('n_in_cm', 0)} "
            f"(+{edge} liked edge / +{verify} verify outside CM)"
        ),
        y_hat_subtitle="(Recall = TP/(TP+FN) · ŷ = BKH URL match · 98 missing_cite removed from FN in purple)",
        cell_subs=(
            "liked + BKH URL match",
            "liked, neither cited",
            "disliked + BKH URL match",
            "disliked, VA no cite",
        ),
        alt_proxy=cov,
        compare_bkh=True,
        cohort_label="VA v2",
        cohort_scores=va_scores,
    )


def fig_va_retrieval_proxy_v3() -> Path:
    """VA golden retrieval v3 — BKH | VA v2 | reclass adjusted | HITL upper bound."""
    from evals.metrics.comparison.url_overlap import compute_v3_retrieval_scenarios
    from evals.pipelines.datasets import load_jsonl
    from evals.reports.paths import resolve_golden_responses_path

    tasks = load_jsonl(resolve_golden_responses_path())
    v3 = compute_v3_retrieval_scenarios(tasks)
    bkh_stats = _load_bkh_stats_dict()
    bkh_scores = _build_cohort_scores(
        bkh_stats,
        bkh_stats.get("retrieval_proxy"),
        use_full_recall=True,
    )
    scenarios = {
        "bkh": {"scores": bkh_scores, "label": "BKH"},
        **v3["scenarios"],
    }
    cm_v2 = scenarios["va v2"]["cm"]
    cm_re = scenarios["reclass verified"]["cm"]
    tp_v2 = cm_v2["tp"]
    tp_re = cm_re["tp"]
    tp_hi = scenarios["HITL max"]["cm"]["tp"]
    n_hitl_pool = v3["n_hitl_va_may_be_right"]
    v3.get("n_hitl_in_cm", 0)
    v3.get("n_edge_in_cm", 0)
    like_rates = v3.get("like_rates", {})

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"width_ratios": [1, 1.45]})
    _style()

    ax = axes[0]
    ax.set_xlim(-0.35, 2.15)
    ax.set_ylim(-0.05, 2.35)
    ax.axis("off")
    tp, fp, tn, fn = cm_re["tp"], cm_re["fp"], cm_re["tn"], cm_re["fn"]
    cell_data = [
        (0, 1, f"TP\n{tp:,}", "liked + overlap match", "#d4edda", "#155724"),
        (1, 1, f"TN\n{tn:,}", "liked, neither cited", "#e8f4ea", "#4a7c59"),
        (0, 0, f"FP\n{fp:,}", "disliked + overlap/capability", "#f8d7da", "#721c24"),
        (1, 0, f"FN\n{fn:,}", "disliked, VA no cite", "#fff3e0", "#a04000"),
    ]
    for x, y, label, sub, bg, fg in cell_data:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, color=bg, zorder=2))
        ax.text(
            x + 0.5,
            y + 0.64,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color=fg,
            fontweight="bold",
            zorder=3,
        )
        ax.text(x + 0.5, y + 0.28, sub, ha="center", va="center", fontsize=7, color=fg, zorder=3)
    ax.text(0.5, 2.22, "retrieval match  (ŷ=1)", ha="center", va="bottom", fontsize=8.5, color=MID)
    ax.text(1.5, 2.22, "no match  (ŷ=0)", ha="center", va="bottom", fontsize=8.5, color=MID)
    ax.text(-0.12, 1.5, "liked\n(y=1)", ha="right", va="center", fontsize=8.5, color=MID)
    ax.text(-0.12, 0.5, "disliked\n(y=0)", ha="right", va="center", fontsize=8.5, color=MID)
    excl = scenarios["reclass verified"].get("fn_excluded_missing_cite", 0)
    ax.set_title(
        f"Reclass adjusted CM\n(+{v3['promoted_regression']} promoted · {excl} missing_cite excl.)",
        color=NAVY,
        fontweight="bold",
        pad=28,
        fontsize=11,
    )
    ax.text(
        0.5,
        -0.08,
        f"TP progression: {tp_v2} v2 → {tp_re} reclass → {tp_hi} HITL max",
        ha="center",
        va="top",
        fontsize=8,
        color=NAVY,
        transform=ax.transAxes,
        fontweight="bold",
    )

    excl = scenarios["reclass verified"].get("fn_excluded_missing_cite", 0)
    n_rated = like_rates.get("n_rated", 0)
    n_liked = like_rates.get("n_liked", 0)
    like_v2 = like_rates.get("like_rate")
    hitl_like = like_rates.get("hitl_upper_bound")
    footnote = (
        f"Archive like % = liked/rated on human labels (n={n_rated}); "
        f"v2/reclass {like_v2:.0f}% ({n_liked} liked); "
        f"HITL upper bound {hitl_like:.0f}% = ({n_liked}+HITL credits)/{n_rated} · "
        f"Not CM-derived (CM like % ≈ precision when TN small) · "
        f"Recall: reclass R uses overlap TP ({tp_v2}) + cov-adj FN"
        if like_v2 is not None and hitl_like is not None
        else "Navy=BKH · Teal=VA v2 · Purple=reclass · Amber=HITL upper bound"
    )
    _draw_retrieval_scenario_compare_panel(axes[1], scenarios, footnote=footnote)

    n_va = len(tasks)
    fig.suptitle(
        f"VA Golden — Retrieval Proxy v3: BKH vs staged VA verification (n={n_va} turns)\n"
        f"Regression pool {v3['regression_pool']} · {n_hitl_pool} potential TP pending HITL",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.84, bottom=0.14, wspace=0.28)
    return _save(fig, "va_retrieval_proxy_v3")


def fig_va_retrieval_proxy() -> Path:
    """Export v1 + v2 + v3; va_retrieval_proxy.svg aliases v2."""
    fig_va_retrieval_proxy_v1()
    fig_va_retrieval_proxy_v2()
    fig_va_retrieval_proxy_v3()
    src = OUT_DIR / "va_retrieval_proxy_v2.svg"
    dst = OUT_DIR / "va_retrieval_proxy.svg"
    if src.exists():
        dst.write_text(src.read_text(encoding="utf-8"))
    return dst


def fig_golden_retrieval_proxy(golden_stats: dict | None = None) -> Path:
    stats = golden_stats or _load_golden_stats()
    return fig_retrieval_proxy(
        stats,
        title_suffix="VA Staging",
        compare_bkh=True,
        cohort_label="VA",
        cohort_scores=_build_cohort_scores(stats),
        out_name="va_staging_retrieval_proxy",
    )


# ---------------------------------------------------------------------------
# Staging aliases (output/figures/va — canonical doc names)
# ---------------------------------------------------------------------------


def fig_staging_pass_rates(quality: dict | None = None) -> Path:
    """VA staging calibrated pass rates (v1 raw vs URL-map calibrated)."""
    src = fig_golden_pass_rates_compare(quality)
    dest = OUT_DIR / "staging_pass_rates.svg"
    dest.write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  ✓ {dest}")
    return dest


def fig_staging_retrieval_proxy() -> Path:
    """VA staging retrieval proxy (overlap + kb_url_map, not raw v2 grounding)."""
    src = fig_va_retrieval_proxy_v2()
    dest = OUT_DIR / "staging_retrieval_proxy.svg"
    dest.write_text(Path(src).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  ✓ {dest}")
    return dest


# ---------------------------------------------------------------------------
# Registry and CLI
# ---------------------------------------------------------------------------

ALL_FIGURES: dict[str, callable] = {
    # Methods / calibration
    "cohen_d": fig_cohen_d,
    "threshold_viz": fig_threshold_viz,
    "box_plots": fig_box_plots,
    "kde_thresholds": fig_kde_thresholds,
    "kde_thresholds_v2": fig_kde_thresholds_v2,
    "kde_heuristic_thresholds": fig_kde_heuristic_thresholds,
    "kde_heuristic_thresholds_v2": fig_kde_heuristic_thresholds_v2,
    # Retrieval
    "mrr_comparison": fig_mrr_comparison,
    "feature_impact": fig_feature_impact,
    "retrieval_proxy": fig_retrieval_proxy,
    # Quality pass rates
    "bkh_pass_rates": fig_bkh_pass_rates,
    "va_pass_rates": fig_va_pass_rates,
    "golden_pass_rates": fig_golden_pass_rates,
    "golden_pass_rates_v1": fig_golden_pass_rates_v1,
    "golden_pass_rates_v2": fig_golden_pass_rates_v2,
    "golden_pass_rates_compare": fig_golden_pass_rates_compare,
    "golden_perf_compare": fig_golden_perf_compare,
    "golden_perf_heuristic": fig_golden_perf_heuristic,
    "golden_perf_llm": fig_golden_perf_llm,
    "golden_benchmark": fig_golden_benchmark,
    "heuristic_llm_compare": fig_heuristic_llm_compare,
    # Statistical power
    "power_curve": fig_power_curve,
    # Dataset overview
    "bkh_overview": fig_bkh_overview,
    "topic_sentiment": fig_topic_sentiment,
    # VA staging responses archive (mirrors BKH Tab 02)
    "va_staging_response_type": fig_golden_response_type,
    "va_staging_failure_mode": fig_golden_failure_mode,
    "va_staging_source_count": fig_golden_source_count,
    "va_staging_language": fig_golden_language,
    "va_staging_retrieval_proxy": fig_golden_retrieval_proxy,
    "golden_response_type": fig_golden_response_type,
    "golden_failure_mode": fig_golden_failure_mode,
    "golden_source_count": fig_golden_source_count,
    "golden_language": fig_golden_language,
    "golden_retrieval_proxy": fig_golden_retrieval_proxy,
    "va_retrieval_proxy": fig_va_retrieval_proxy,
    "va_retrieval_proxy_v1": fig_va_retrieval_proxy_v1,
    "va_retrieval_proxy_v2": fig_va_retrieval_proxy_v2,
    "va_retrieval_proxy_v3": fig_va_retrieval_proxy_v3,
    "staging_pass_rates": fig_staging_pass_rates,
    "staging_retrieval_proxy": fig_staging_retrieval_proxy,
    # Archival — kept for reference, not included in default demo report
    "prompt_evolution": fig_prompt_evolution,
}


def _load_live_data() -> dict:
    """Load live data from JSON reports where available."""
    data: dict = {}

    ablation_dir = Path("data/datasets/support-agents/ablation")
    if ablation_dir.exists():
        configs: dict = {}
        for p in ablation_dir.glob("*.json"):
            try:
                raw = json.loads(p.read_text())
                configs[p.stem] = raw
            except Exception:
                pass
        if configs:
            data["ablation_configs"] = configs

    bkh_all_stats = Path("data/datasets/bkh/stats/all_stats.json")
    if bkh_all_stats.exists():
        with contextlib.suppress(Exception):
            data["bkh_all_stats"] = json.loads(bkh_all_stats.read_text())

    bkh_quality = Path("data/datasets/bkh/quality_results/graded_train.json")
    if bkh_quality.exists():
        with contextlib.suppress(Exception):
            data["bkh_quality"] = json.loads(bkh_quality.read_text())

    gq = _load_golden_quality()
    if gq:
        data["golden_quality"] = gq

    return data


_ARCHIVAL_FIGURES = {"prompt_evolution"}


def export_all(fig_names: list[str] | None = None) -> None:
    _style()
    data = _load_live_data()

    # Default run excludes archival figures; pass --fig prompt_evolution explicitly to export them
    default_figs = [k for k in ALL_FIGURES if k not in _ARCHIVAL_FIGURES]
    figures_to_run = fig_names or default_figs
    bkh_stats = data.get("bkh_all_stats")

    _DATA_FIGS = {"topic_sentiment", "retrieval_proxy"}
    _GOLDEN_FIGS = {
        "golden_response_type",
        "golden_failure_mode",
        "golden_source_count",
        "golden_language",
        "golden_retrieval_proxy",
    }
    golden_stats = None
    if any(n in _GOLDEN_FIGS for n in figures_to_run):
        try:
            golden_stats = _load_golden_stats()
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ⚠ golden stats: {exc}")

    print(f"Exporting {len(figures_to_run)} figure(s) → {OUT_DIR}/")
    for name in figures_to_run:
        fn = ALL_FIGURES.get(name)
        if fn is None:
            print(f"  ⚠ unknown figure: {name}")
            continue
        try:
            if name in _GOLDEN_FIGS:
                if golden_stats is None:
                    print(f"  ✗ {name}: golden stats unavailable")
                    continue
                fn(golden_stats)
            elif name in _DATA_FIGS:
                fn(bkh_stats)
            elif (
                name == "golden_pass_rates_v1"
                or name == "golden_pass_rates_v2"
                or name in ("golden_pass_rates", "va_pass_rates")
            ):
                fn()
            else:
                fn()
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")

    print("\nDone. Open evals/reports/figures/ to view SVGs.")
    print("Embed into the doc: run the eval-docs skill → 'refresh eval doc figures'")
    print("Use in PPT: drag SVG files into slide or run pptx skill")
    print("Use in Excalidraw: File → Import → select SVG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export eval framework figures as SVGs")
    parser.add_argument(
        "--source",
        default="shared",
        help="Subfolder under evals/reports/figures/ (bkh, va, golden, shared, …)",
    )
    parser.add_argument(
        "--fig",
        nargs="+",
        choices=list(ALL_FIGURES.keys()),
        help="Specific figures to export (default: all)",
    )
    args = parser.parse_args()
    set_figures_source(args.source)
    export_all(args.fig)
