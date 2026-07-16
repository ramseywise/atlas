"""PM / design-facing narrative blocks (distilled from eval_framework_report)."""

from __future__ import annotations

import html as _html
from typing import Any


def _fmt_pct(rate: float) -> str:
    return f"{100 * float(rate):.1f}%"


def _failure_rate(stats: dict, key: str) -> tuple[float, int] | None:
    ft = stats.get("failure_types") or {}
    row = ft.get(key)
    if not row:
        return None
    return float(row.get("rate", 0)), int(row.get("count", 0))


def _insight_card(
    value: str,
    label: str,
    sub: str,
    *,
    accent: str = "var(--navy)",
) -> str:
    return (
        f'<div class="stat-card insight-card" style="border-top:3px solid {accent}">'
        f'<div class="num" style="color:{accent}">{_html.escape(value)}</div>'
        f'<div class="label">{_html.escape(label)}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def profile_tldr(
    stats: dict,
    *,
    corpus: str,
    suite_link: str = "",
    calibration_link: str = "",
) -> str:
    """One-line callout above dataset profile (BKH + VA stats pages)."""
    from evals.reports.html.stats_profile import normalize_stats_record

    s = normalize_stats_record(stats)
    n = int(s.get("n_total", 0))
    rated = float(s.get("rated_pct", 0))
    corpus_label = "BKH production" if corpus == "bkh" else "VA staging archive"
    suite = (
        f'<a href="{_html.escape(suite_link)}">suite</a>'
        if suite_link
        else "suite"
    )
    cal = (
        f' · <a href="{_html.escape(calibration_link)}">calibration</a>'
        if calibration_link
        else ""
    )
    return (
        f'<div class="callout teal" style="margin-bottom:16px">'
        f"<strong>{_html.escape(corpus_label)}</strong> — {_fmt_num(n)} turns · "
        f"{rated:.1f}% rated. Profile charts + heuristic tables below; "
        f"LLM pass gates on {suite}{cal}.</div>"
    )


def _fmt_num(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{int(n):,}"


def pm_insight_cards_grid(stats: dict) -> str:
    """Four product insights under the scale stat cards — A/B/E + citation mix."""
    from evals.reports.html.stats_profile import normalize_stats_record

    s = normalize_stats_record(stats)
    a = _failure_rate(stats, "A_retrieval")
    b = _failure_rate(stats, "B_language")
    e = _failure_rate(stats, "E_grounding")

    a_pct = f"~{_fmt_pct(a[0])}" if a else "—"
    a_n = f"{a[1]:,}" if a else "—"
    b_pct = f"~{_fmt_pct(b[0])}" if b else "—"
    b_n = f"{b[1]:,}" if b else "—"
    n_rated = int(s.get("n_rated", 0))
    e_count = e[1] if e else 0
    if n_rated and e_count:
        e_pct = f"{100 * e_count / n_rated:.1f}%"
        e_sub = f"{e_count:,} of {n_rated:,} rated · disliked + cited"
    else:
        e_pct = "—"
        e_sub = "E_grounding — no rated turns"
    has_src = float(s.get("has_sources_pct", 0))
    unknown = float(s.get("unknown_pct", 0))

    cards = [
        _insight_card(
            f"{has_src:.1f}%",
            "Cites ≥1 source",
            f"{unknown:.1f}% unknown (no cite) · all turns",
            accent="var(--navy)",
        ),
        _insight_card(
            a_pct,
            "No KB answer",
            f"{a_n} turns · % of all corpus",
            accent="var(--red)",
        ),
        _insight_card(
            b_pct,
            "Wrong language",
            f"{b_n} turns · % of all corpus",
            accent="var(--amber)",
        ),
        _insight_card(
            e_pct,
            "Cited but disliked",
            e_sub,
            accent="var(--teal)",
        ),
    ]
    return (
        '<p class="sub-hdr" style="margin-top:4px">Product insights</p>'
        f'<div class="grid-4">{"".join(cards)}</div>'
    )


def pm_comparison_story(*, staging: Any | None = None) -> str:
    """Headline for bkh_va_comparison — A/B two scores, not aspirational gates."""
    ov = getattr(staging, "overlap_summary", None) or {} if staging else {}
    exp = ov.get("expanded_overlap_total", "—")
    exp_rate = ov.get("expanded_overlap_rate")
    exp_pct = f"{100 * float(exp_rate):.1f}%" if exp_rate is not None else "—"

    return f"""
<div class="section-header">
  <div class="num">A/B comparison</div>
  <h2>BKH production baseline vs VA staging (URL-map calibrated)</h2>
  <p class="lead">Two comparable scores — not pass/fail theater. BKH = n≈69k heuristics + n=50 LLM calibration.
  VA = full staging archive with <b>URL-map reclassification</b> on judge pass rates.</p>
</div>
<div class="callout amber"><strong>Do not compare BKH grounding to VA.</strong>
  Grounding / RAGAS graders were <em>not</em> run on BKH calibration — they apply to VA staging only.
  Any BKH grounding bar is omitted or N/A by design.</div>
<div class="callout teal"><strong>Why VA can look &ldquo;better&rdquo;:</strong> help.shine.co vs billy.dk URL overlap is an
  infrastructure/domain problem ({exp} paired turns expanded-match {exp_pct} after <code>kb_url_map</code>).
  Reclassification explains dislikes that are really domain mismatch — see overlap table on
  <a href="va/va_suite.html">va_suite.html</a>.</div>
<div class="narrative">
  <p><strong>Chart to read:</strong> three-way pass rates (BKH cal · VA v1 raw · VA v2 calibrated) — primary gates only.
  Full methodology (Cohen&rsquo;s d, Billypedia flags, KDE): <a href="calibration.html">calibration.html</a>.
  Per-corpus detail: <a href="../bkh/bkh_suite.html">bkh_suite</a> · <a href="va_suite.html">va_suite</a>.</p>
</div>
"""


def comparison_sample_blurb(*, n_paired: int | None = None) -> str:
    """Short sample-size note for comparison (details in calibration.html)."""
    paired = f"{n_paired:,} paired task_ids" if n_paired else "597 paired task_ids"
    return (
        '<p class="muted" style="font-size:13px;margin-bottom:16px">'
        f"Layer 1 heuristics = {paired} (BKH production vs VA staging). "
        "Layer 2 BKH LLM = stratified calibration subset (not all 597 graded) · "
        "VA LLM = full staging archive (URL-map calibrated). "
        '<a href="calibration.html">calibration.html</a> has sample tables.</p>'
    )
