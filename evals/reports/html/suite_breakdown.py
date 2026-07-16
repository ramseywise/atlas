"""Per-eval_set / slice breakdown — card layout aligned across BKH and VA suites."""

from __future__ import annotations

import html as _html

from evals.metrics._constants import METRIC_FRIENDLY_NAMES, TIER_THRESHOLDS
from evals.metrics.suite import SuiteReport
from evals.reports.html.eval_set_group import (
    _GROUP_ORDER,
    group_display_name,
)
from evals.metrics.calibration.grader_scope import COMPARISON_LAYER2_METRICS
from evals.reports.utils.layout import report_href


def breakdown_grader_columns(is_va: bool, *, all_graders: set[str]) -> tuple[list[str], list[str]]:
    """(primary columns, tracking columns) — same order on BKH and VA where applicable."""
    primary_keys = [k for k, _, _ in COMPARISON_LAYER2_METRICS]
    if not is_va:
        primary_keys = [k for k in primary_keys if k != "grounding"]

    primary = [k for k in primary_keys if k in all_graders]
    tracking_order = [
        "deepeval_answer_relevancy",
        "deepeval_completeness",
        "deepeval_escalation",
        "ragas_context_precision",
        "ragas_faithfulness",
    ]
    tracking = [k for k in tracking_order if k in all_graders and k not in primary]
    extra = sorted(all_graders - set(primary) - set(tracking) - {"grounding"})
    return primary, tracking + extra


def _pass_cell(pass_rate: float, *, threshold: float, n: int) -> str:
    if n <= 0:
        return '<td class="num muted">—</td>'
    passed = pass_rate >= threshold if threshold > 0 else None
    if passed is True:
        cls, pill = "pass", "PASS"
    elif passed is False:
        cls, pill = "fail", "FAIL"
    else:
        cls, pill = "warn", "INFO"
    return (
        f'<td class="num {cls}"><span class="verdict-pill {cls}" style="font-size:10px;padding:2px 6px">'
        f"{pill}</span> {pass_rate:.0%}</td>"
    )


def _slice_card(
    group: str,
    graders: dict,
    *,
    primary_cols: list[str],
    tracking_cols: list[str],
    is_va: bool,
    cal_sample: bool = False,
) -> str:
    n_total = sum(g.get("n", 0) for g in graders.values()) // max(len(graders), 1)
    title = group_display_name(group, cal_sample=cal_sample)

    def _rows(cols: list[str]) -> str:
        rows = []
        for key in cols:
            gd = graders.get(key)
            if not gd:
                continue
            label = METRIC_FRIENDLY_NAMES.get(key, key.replace("_", " "))
            thr = TIER_THRESHOLDS.get(key, 75) / 100.0 if key in TIER_THRESHOLDS else 0.75
            if key in {k for k, _, t in COMPARISON_LAYER2_METRICS}:
                thr = next((t / 100.0 for k, _, t in COMPARISON_LAYER2_METRICS if k == key), thr)
            rows.append(
                f"<tr><td>{_html.escape(label)}</td>"
                f"{_pass_cell(gd['pass_rate'], threshold=thr, n=gd.get('n', 0))}"
                f'<td class="num">{gd.get("avg_score", 0):.2f}</td>'
                f'<td class="num muted">{gd.get("n", 0):,}</td></tr>'
            )
        return "".join(rows)

    primary_body = _rows(primary_cols)
    track_body = _rows(tracking_cols)

    track_block = ""
    if tracking_cols and track_body:
        track_block = (
            f'<details class="methodology-fold" style="margin-top:8px">'
            f"<summary>DeepEval / RAGAS ({len(tracking_cols)})</summary>"
            f'<table class="data-table slice-metrics" style="margin-top:6px">'
            f"<thead><tr><th>Metric</th><th>Pass</th><th>Avg</th><th>n</th></tr></thead>"
            f"<tbody>{track_body}</tbody></table></details>"
        )

    grounding_note = ""
    if not is_va and group in ("stratified_liked", "stratified_disliked", "regression", "unspecified"):
        grounding_note = (
            '<p class="muted" style="font-size:11px;margin:6px 0 0">'
            "Grounding / RAGAS: N/A on BKH cal — see VA suite.</p>"
        )

    return f"""
<div class="card slice-card">
  <h3>{_html.escape(title)}</h3>
  <p class="chart-note">~{n_total:,} graded queries in this slice</p>
  <table class="data-table slice-metrics">
    <thead><tr><th>Metric</th><th>Pass</th><th>Avg score</th><th>n</th></tr></thead>
    <tbody>{primary_body or '<tr><td colspan="4" class="muted">No primary graders</td></tr>'}</tbody>
  </table>
  {track_block}
  {grounding_note}
</div>
"""


def _slice_grid_html(
    breakdown: dict,
    *,
    primary_cols: list[str],
    tracking_cols: list[str],
    is_va: bool,
    cal_sample: bool,
) -> str:
    ordered = [g for g in _GROUP_ORDER if g in breakdown] + [
        g for g in breakdown if g not in _GROUP_ORDER
    ]
    cards = [
        _slice_card(
            group,
            breakdown[group],
            primary_cols=primary_cols,
            tracking_cols=tracking_cols,
            is_va=is_va,
            cal_sample=cal_sample,
        )
        for group in ordered
    ]
    return f'<div class="grid-2 slice-grid">{"".join(cards)}</div>'


def suite_eval_set_breakdown_html(report: SuiteReport) -> str:
    """Card grid per slice — replaces wide HTML table."""
    breakdown = report.heuristic_stats.get("group_breakdown")
    if not breakdown:
        return ""

    is_va = bool(
        report.heuristic_stats.get("staging_calibration")
        or report.heuristic_stats.get("va_stats")
    )
    cal_sample = bool(report.heuristic_stats.get("is_calibration_sample"))
    if not cal_sample and report.heuristic_stats.get("slice_grouping") == "sentiment":
        cal_sample = True
    all_graders: set[str] = set()
    for graders in breakdown.values():
        all_graders.update(graders.keys())

    primary_cols, tracking_cols = breakdown_grader_columns(is_va, all_graders=all_graders)

    slice_grid = _slice_grid_html(
        breakdown,
        primary_cols=primary_cols,
        tracking_cols=tracking_cols,
        is_va=is_va,
        cal_sample=cal_sample,
    )

    scenario_block = ""
    scenario_bd = report.heuristic_stats.get("group_breakdown_scenario")
    if scenario_bd and cal_sample:
        scenario_block = (
            '<details class="methodology-fold" style="margin-top:16px">'
            "<summary>By scenario (regression / capability) — same n=50 cal sample</summary>"
            + _slice_grid_html(
                scenario_bd,
                primary_cols=primary_cols,
                tracking_cols=tracking_cols,
                is_va=is_va,
                cal_sample=False,
            )
            + "</details>"
        )

    cal_note = ""
    if report.heuristic_stats.get("staging_calibration", {}).get("url_map_adjusted"):
        cal_note = (
            "<div class='callout green' style='margin-bottom:14px'>"
            "<b>URL-map calibrated passes</b> — counts reflect reclassification + kb_url_map "
            "(not raw LLM). Compare cohort totals on "
            f'<a href="{report_href("va", "cohort_compare")}">cohort compare</a>.</div>'
        )
    elif cal_sample:
        cal_note = (
            "<div class='callout amber' style='margin-bottom:14px'>"
            "<b>BKH n=50 stratified cal</b> — cards grouped by <b>liked / disliked</b> (25+25). "
            "Scenario split (regression vs capability) is in the fold below. "
            "Not URL-reclassified — that applies to VA staging only.</div>"
        )
    elif not is_va:
        cal_note = (
            "<div class='callout amber' style='margin-bottom:14px'>"
            "<b>BKH corpus</b> — slices use <code>eval_set</code> / <code>source</code> when present. "
            "Not the same population as VA full archive.</div>"
        )

    other_href = report_href("va" if not is_va else "bkh", "va_suite" if not is_va else "bkh_suite")
    ab_href = report_href("va", "cohort_compare")
    return f"""
<div class="section-block">
  <div class="section-header" style="margin-bottom:12px">
    <div class="num">Slices</div>
    <h2>Pass rates by eval slice</h2>
    <p class="lead">Same primary columns on BKH and VA — DeepEval/RAGAS folded per card.
    Cross-corpus gates: <a href="{ab_href}">A/B comparison</a> ·
    <a href="{other_href}">other suite</a>.</p>
  </div>
  {cal_note}
  {slice_grid}
  {scenario_block}
</div>
"""
