"""Standalone BKH↔VA A/B report on the 597 paired golden cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.reports.html.ab_comparison import ab_delta_table_html, ab_highlight_cards
from evals.reports.paths import (
    bkh_calibration_quality_path,
    bkh_compare_qa_path,
    bkh_stats_path,
    ensure_bkh_stats,
    va_staging_all_quality_v1_path,
    va_staging_all_responses_path,
    va_staging_all_responses_stats_path,
)
from evals.reports.html.pm_narratives import comparison_sample_blurb
from evals.reports.html.suite_corpus_compare import suite_corpus_compare_html
from evals.reports.utils._rebuild import assemble_suite_report
from evals.reports.utils.embed import chart_card
from evals.reports.utils.figures import fig_ab_side_by_side, fig_golden_perf_compare, set_figures_source
from evals.reports.utils.layout import FIGURES_ROOT, ReportLayout, report_href
from evals.reports.utils.theme import REPORT_THEME_CSS, default_meta_line, report_doc_header


def cohort_compare_html_path() -> Path:
    return ReportLayout("va").html_dir / "cohort_compare.html"


def _paired_n() -> int:
    stats_path = bkh_stats_path()
    if not stats_path.exists():
        return 0
    data = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = data.get("stats") or {}
    blob = next(iter(stats.values()), {}) if stats else data
    return int(blob.get("n_total", 0) or 0)


def build(*, staging: Any = None) -> Path:
    """Publish merged BKH↔VA compare page (Layer 1 on full paired cohort)."""
    ensure_bkh_stats(force=True)
    layout = ReportLayout("va")
    layout.ensure_dirs()
    out = cohort_compare_html_path()

    va_quality = va_staging_all_quality_v1_path()
    va_stats = va_staging_all_responses_stats_path()
    va_responses = va_staging_all_responses_path()
    bkh_quality = bkh_calibration_quality_path()
    bkh_stats = bkh_stats_path()
    bkh_source = bkh_compare_qa_path()

    va_suite, _ = assemble_suite_report(
        graded_path=va_quality if va_quality.exists() else None,
        stats_json_path=va_stats if va_stats.exists() else None,
        source_path=va_responses if va_responses.exists() else None,
        staging=staging,
    )
    assemble_suite_report(
        graded_path=bkh_quality if bkh_quality.exists() else None,
        stats_json_path=bkh_stats if bkh_stats.exists() else None,
        source_path=bkh_source if bkh_source.exists() else None,
    )

    n_paired = _paired_n()
    compare_tables = suite_corpus_compare_html(va_suite)
    highlights = ab_highlight_cards(va_suite)

    set_figures_source("shared")
    fig_ab_side_by_side()
    fig_golden_perf_compare()
    main_chart = chart_card(
        "Primary gates — pass rate comparison",
        f"Grouped bars — BKH cal subset vs VA full archive (Layer 1 n≈{n_paired:,} paired).",
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

    header = report_doc_header(
        title="BKH ↔ VA — Cohort A/B Compare",
        meta=default_meta_line(),
        summary=(
            f"{n_paired:,} paired task_ids — same queries, BKH production baseline vs VA staging. "
            "Layer 1 heuristics on the full cohort; Layer 2 LLM gates where graded "
            "(BKH = calibration subset only)."
        ),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>BKH ↔ VA Cohort Compare — Eval Report</title>
  <style>{REPORT_THEME_CSS}</style>
</head>
<body>
{header}
<div class="page">
  <div class="callout teal">
    <strong>Paired golden cohort (n={n_paired:,}).</strong> Corpus profiles:
    <a href="{report_href("va", "bkh_stats")}">BKH stats</a> ·
    <a href="{report_href("va", "va_stats")}">VA stats</a> ·
    per-agent pass gates:
    <a href="{report_href("va", "bkh_suite")}">BKH suite</a> ·
    <a href="{report_href("va", "va_suite")}">VA suite</a>.
  </div>

  {comparison_sample_blurb(n_paired=n_paired or None)}

  {compare_tables}

  <div class="section-block" id="bk-va-ab">
    <div class="section-header">
      <div class="num">Results</div>
      <h2>Primary LLM gates — pass rate A/B</h2>
      <p class="lead">Retrieval + LLM pass rates: BKH production baseline vs VA staging on matched task_ids.</p>
    </div>
    <div class="callout amber">
      <strong>Grounding caveat.</strong> BKH calibration omits grounding / RAGAS (VA staging only).
    </div>
    {highlights}
    <div class="grid-2" style="margin-bottom:0">
      {main_chart}
      {ab_delta_table_html()}
    </div>
    {detail_chart}
  </div>

  <p class="footer-note">Regenerate: <code>make reports-validate</code></p>
</div>
</body>
</html>"""

    out.write_text(html, encoding="utf-8")
    return out
