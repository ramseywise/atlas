"""Dataset profile narrative + reclassification blocks for stats HTML."""

from __future__ import annotations

import html as _html
from typing import Any


def normalize_stats_record(stats: dict) -> dict:
    """Flatten per-file stats dict for profile cards (BKH + VA staging)."""
    if "n_total" not in stats and stats.get("stats"):
        stats = next(iter(stats["stats"].values()), stats)

    s = stats.get("sentiment") or {}
    n_liked = int(s.get("n_liked", stats.get("n_liked", 0)))
    n_disliked = int(s.get("n_disliked", stats.get("n_disliked", 0)))
    n_unrated = int(s.get("n_unrated", stats.get("n_unrated", 0)))
    n_total = int(stats.get("n_total", n_liked + n_disliked + n_unrated))
    n_rated = n_liked + n_disliked

    rated_pct = float(s.get("rating_coverage_rate", 0)) * 100
    if not s.get("rating_coverage_rate") and n_total:
        rated_pct = 100.0 * n_rated / n_total

    rt = stats.get("response_types") or {}
    unknown_pct = 17.3
    if rt.get("unknown", {}).get("rate"):
        unknown_pct = round(100 * float(rt["unknown"]["rate"]), 1)
    has_src_pct = round(100 * (1 - rt.get("unknown", {}).get("rate", 0.173)), 1) if rt else 74.2
    if rt.get("has_sources", {}).get("rate"):
        has_src_pct = round(100 * float(rt["has_sources"]["rate"]), 1)

    return {
        **stats,
        "n_total": n_total,
        "n_liked": n_liked,
        "n_disliked": n_disliked,
        "n_unrated": n_unrated,
        "n_rated": n_rated,
        "n_conversations": stats.get("n_unique_convs", stats.get("n_conversations", 0)),
        "n_users": stats.get("n_unique_users", stats.get("n_users", 0)),
        "rated_pct": rated_pct,
        "unknown_pct": unknown_pct,
        "has_sources_pct": has_src_pct,
        "dislike_like_ratio": s.get("dislike_like_ratio", stats.get("dislike_like_ratio")),
    }


def _fmt_num(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{int(n):,}"


def dataset_profile_section(
    stats: dict,
    *,
    corpus: str,
    section_title: str,
    lead: str,
    links: dict[str, str] | None = None,
) -> str:
    """Unified Section 01 profile — identical narrative for BKH and VA staging."""
    s = normalize_stats_record(stats)
    n_total = int(s["n_total"])
    n_convs = int(s.get("n_conversations") or 0)
    n_users = int(s.get("n_users") or 0)
    n_rated = int(s["n_rated"])
    rated_pct = float(s["rated_pct"])
    dl = s.get("dislike_like_ratio")
    dl_str = f"{dl:.1f}:1" if dl is not None else "—"

    links = links or {}
    suite = links.get("suite", "")
    suite_a = (
        f'<a href="{_html.escape(suite)}">{_html.escape(suite.split("/")[-1])}</a>'
        if suite
        else "eval suite"
    )

    from evals.reports.html.pm_narratives import (
        pm_insight_cards_grid,
        profile_tldr,
    )

    tldr = profile_tldr(
        stats,
        corpus=corpus,
        suite_link=links.get("suite", ""),
        calibration_link=links.get("calibration", ""),
    )
    insights = pm_insight_cards_grid(stats)

    return f"""
{tldr}
<div class="section-header">
  <div class="num">Dataset profile</div>
  <h2>{_html.escape(section_title)}</h2>
  <p class="lead">{lead}</p>
</div>
<div class="grid-4">
  <div class="stat-card"><div class="num">{_fmt_num(n_total)}</div><div class="label">Total turns</div>
    <div class="sub">all eval records</div></div>
  <div class="stat-card"><div class="num">{_fmt_num(n_convs)}</div><div class="label">Conversations</div>
    <div class="sub">unique threads</div></div>
  <div class="stat-card"><div class="num">{_fmt_num(n_users)}</div><div class="label">Unique users</div>
    <div class="sub">{"—" if not n_users else "distinct accounts"}</div></div>
  <div class="stat-card"><div class="num" style="color:var(--amber)">{rated_pct:.1f}%</div>
    <div class="label">Rated turns</div>
    <div class="sub">{_fmt_num(n_rated)} rated · {dl_str} dislike:like</div></div>
</div>
{insights}
<p class="muted" style="font-size:13px;margin:0 0 8px">
  Charts: sources cited · response type · failure taxonomy · language.
  Cited-but-disliked % is of <b>rated</b> turns only; A/B are % of full corpus.
  Pass-rate gates → <b>{suite_a}</b>.
</p>
"""


def reclassification_overlap_section(staging: Any) -> str:
    """VA-only: overlap reclassification counts (not raw failure_type)."""
    if staging is None:
        return ""
    ov = getattr(staging, "overlap_summary", None) or {}
    rc = ov.get("reclassification") or {}
    if not rc:
        return ""

    n_paired = ov.get("n_paired", "—")
    ov.get("expanded_overlap_total", "—")
    exp_rate = ov.get("expanded_overlap_rate")
    exp_pct = f"{100 * float(exp_rate):.1f}%" if exp_rate is not None else "—"

    rows = [
        (
            "grounded_regression",
            rc.get("promoted_to_regression", 0),
            "Promoted from edge_case / verify_grounding",
        ),
        (
            "capability_test",
            rc.get("promoted_to_capability_test", 0),
            "High composite + grounding threshold",
        ),
        ("kb_indexing_gap", rc.get("n_kb_indexing_gap", 0), "Legacy KB URL not in KB map"),
        (
            "hitl_va_may_be_right",
            rc.get("n_hitl_va_may_be_right", 0),
            "Overlap match; human review candidate",
        ),
        ("hitl_cs_disagree", rc.get("n_hitl_cs_disagree", 0), "CS disliked but VA may be correct"),
    ]
    body = ""
    for label, count, note in rows:
        body += (
            f"<tr><td><code>{_html.escape(label)}</code></td>"
            f"<td class='num'>{count}</td><td>{_html.escape(note)}</td></tr>\n"
        )

    return f"""
<details class="methodology-fold" style="margin-bottom:20px">
  <summary>URL-map reclassification (overlap pool · {n_paired} paired · {exp_pct} expanded match)</summary>
  <p class="muted" style="font-size:13px;margin:10px 0">
    Raw A–E labels drive charts above; rows below feed suite/regression only.
  </p>
  <table class="data-table">
    <thead><tr><th>Reclassified slice</th><th class="num">n</th><th>Rule (summary)</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</details>
"""
