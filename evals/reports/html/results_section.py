"""Suite summary blocks — BKH cal narrative; VA summary + BKH↔VA A/B chart."""

from __future__ import annotations

from evals.metrics.suite import SuiteReport
from evals.reports.html.ab_comparison import (
    ab_delta_table_html,
    ab_highlight_cards,
)
from evals.reports.html.pm_narratives import comparison_sample_blurb
from evals.reports.utils.embed import chart_card
from evals.reports.utils.figures import (
    fig_ab_side_by_side,
    fig_golden_perf_compare,
    set_figures_source,
)
from evals.reports.utils.layout import FIGURES_ROOT, report_href


def _judge_metric(report: SuiteReport, key: str):
    for m in report.judge_metric_results:
        if m.metric_name == key:
            return m
    return None


def _highlights(report: SuiteReport, *, va_only: bool = False) -> dict[str, dict]:
    keys = (
        "answer_relevancy",
        "completeness",
        "escalation",
    )
    if va_only:
        keys = (*keys, "grounding", "ragas_faithfulness")
    out: dict[str, dict] = {}
    for k in keys:
        m = _judge_metric(report, k)
        if m and m.n_graded:
            out[k] = {
                "value": m.value,
                "passed": m.passed,
                "threshold": m.threshold,
                "n": m.n_graded,
            }
    return out


def _va_ab_chart_section(report: SuiteReport, *, export_figure: bool = True) -> str:
    """BKH baseline vs VA calibrated — demo §3 layout on va_suite footer."""
    if export_figure:
        set_figures_source("shared")
        fig_ab_side_by_side()
        fig_golden_perf_compare()

    n_paired = 597
    try:
        from evals.reports.html.cohort_compare_report import _paired_n

        n_paired = _paired_n() or n_paired
    except Exception:
        pass

    main_chart = chart_card(
        "Primary gates — pass rate comparison",
        f"Grouped bars: BKH cal subset vs VA full archive (Layer 1 n≈{n_paired:,} paired).",
        "ab_side_by_side",
        FIGURES_ROOT / "shared",
        tall=True,
    )
    detail_chart = (
        '<details class="methodology-fold" style="margin-top:12px">'
        "<summary>Full A/B — Layer 1 heuristics + all cohorts (BKH · VA raw · VA calibrated)</summary>"
        + chart_card(
            "Layer 1 + Layer 2 cohort breakdown",
            "Three-way comparison including VA v1 raw scores.",
            "golden_perf_compare",
            FIGURES_ROOT / "shared",
            wide=True,
            tall=True,
        )
        + "</details>"
    )
    highlights = ab_highlight_cards(report)
    delta = ab_delta_table_html()

    return f"""
<div class="section-block" id="bk-va-ab">
  <div class="section-header">
    <div class="num">Results</div>
    <h2>BKH baseline vs VA staging</h2>
    <p class="lead">597 paired task_ids — Layer 1 heuristics on full cohort; Layer 2 where LLM-graded.</p>
  </div>
  {comparison_sample_blurb(n_paired=n_paired)}
  <div class="callout amber">
    <strong>Grounding caveat.</strong> BKH calibration omits grounding / RAGAS (VA staging pipeline only).
    Do not read a zero BKH grounding bar as a product regression.
  </div>
  {highlights}
  <div class="grid-2" style="margin-bottom:0">
    {main_chart}
    {delta}
  </div>
  {detail_chart}
  <p class="muted" style="margin-top:14px">
    <a href="{report_href("va", "cohort_compare")}">Full cohort compare</a> ·
    <a href="{report_href("va", "bkh_suite")}">BKH suite</a> ·
    <a href="{report_href("va", "bkh_stats")}">BKH stats</a> ·
    <a href="{report_href("va", "calibration")}">calibration methods</a>
  </p>
</div>
"""


def suite_results_section(
    report: SuiteReport,
    *,
    is_va: bool,
    export_figure: bool = True,
) -> str:
    """Footer on suite HTML — VA includes A/B chart; BKH is calibration summary only."""
    if is_va:
        narrative = _narrative_va(report)
        ab = _va_ab_chart_section(report, export_figure=export_figure)
        links = (
            '<p class="muted" style="margin-top:8px">'
            f'<a href="{report_href("va", "va_stats")}">corpus stats</a> · '
            f'<a href="{report_href("va", "calibration")}">calibration methods</a></p>'
        )
        return f"{narrative}\n{ab}\n{links}"

    narrative = _narrative_bkh(report)
    links = (
        '<p class="muted" style="margin-top:8px">'
        f'<a href="{report_href("bkh", "bkh_stats")}">corpus stats (597 paired)</a> · '
        f'<a href="{report_href("bkh", "cohort_compare")}">BKH↔VA compare</a> · '
        f'<a href="{report_href("bkh", "calibration")}">methods</a> · '
        f'<a href="{report_href("bkh", "va_suite")}">VA suite</a></p>'
    )
    return f"{narrative}\n{links}"


def _narrative_bkh(report: SuiteReport) -> str:
    h = _highlights(report, va_only=False)
    n_llm = max((m.n_graded for m in report.judge_metric_results), default=0)
    n_heur = int((report.heuristic_stats or {}).get("n_total", 0) or 0)
    ar = h.get("answer_relevancy", {})
    comp = h.get("completeness", {})

    return f"""
<div class="section-header">
  <div class="num">Summary</div>
  <h2>BKH — Layer 1 on {n_heur:,} paired turns · Layer 2 on n={n_llm} cal</h2>
  <p class="lead">Heuristic gates use the full golden cohort; LLM judges run on the stratified calibration subset only.
  Side-by-side A/B on <a href="{report_href("bkh", "cohort_compare")}">cohort compare</a>.</p>
</div>
<div class="callout amber"><strong>Grounding not scored on BKH.</strong> Pass-rate tables exclude grounding/RAGAS
  (VA staging pipeline only).</div>
<div class="narrative">
  <p><strong>Answer relevancy</strong> {ar.get('value', 0):.0%} · <strong>Completeness</strong> {comp.get('value', 0):.0%}
  on n={n_llm} LLM-graded turns.</p>
</div>
"""


def _narrative_va(report: SuiteReport) -> str:
    h = _highlights(report, va_only=True)
    n = max((m.n_graded for m in report.judge_metric_results), default=0)

    bullets = []
    if h.get("completeness", {}).get("passed"):
        bullets.append(f"<strong>Completeness</strong> {h['completeness']['value']:.0%} — clears gate on rated archive.")
    if h.get("escalation", {}).get("passed"):
        bullets.append(f"<strong>Escalation</strong> {h['escalation']['value']:.0%} — handoff behavior strong.")
    if h.get("answer_relevancy") and not h["answer_relevancy"].get("passed"):
        bullets.append(
            f"<strong>Answer relevancy</strong> {h['answer_relevancy']['value']:.0%} — still the main product gap."
        )

    body = "".join(f"<p>{b}</p>" for b in bullets) or "<p>See calibrated judge table above.</p>"

    return f"""
<div class="section-header">
  <div class="num">Summary</div>
  <h2>VA staging — calibrated LLM gates</h2>
  <p class="lead">Layer 1 → Layer 2 tables above are authoritative. n≈{n:,} graded.</p>
</div>
<div class="narrative">{body}</div>
"""
