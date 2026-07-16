"""Eval methods report — Layer 1 heuristics + Layer 2 LLM calibration (VA staging rated turns)."""

from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path

from evals.reports.html.narratives import eval_methods_report_body
from evals.reports.utils.embed import chart_card
from evals.reports.utils.figures import (
    _golden_grader_calibration_stats,
    fig_box_plots,
    fig_cohen_d,
    fig_kde_heuristic_thresholds,
    fig_kde_heuristic_thresholds_v2,
    fig_kde_thresholds,
    fig_threshold_viz,
    set_figures_source,
)
from evals.reports.utils.layout import FIGURES_ROOT, ReportLayout, report_href
from evals.reports.utils.theme import (
    REPORT_THEME_CSS,
    default_meta_line,
    report_doc_header,
    report_footer_note,
)

OUT_PATH = ReportLayout.calibration_html()
FIG_DIR = FIGURES_ROOT / "calibration"


def export_calibration_figures() -> None:
    set_figures_source("calibration")
    fig_kde_heuristic_thresholds()
    fig_kde_heuristic_thresholds_v2()
    fig_cohen_d()
    fig_threshold_viz()
    fig_box_plots()
    fig_kde_thresholds()


def build(output_path: Path = OUT_PATH, *, export_figures: bool = True) -> Path:
    if export_figures:
        export_calibration_figures()

    from evals.reports.paths import va_staging_all_quality_v1_path

    quality = va_staging_all_quality_v1_path()
    n_note = ""
    if quality.exists():
        import json

        d = json.loads(quality.read_text())
        n = d.get("n_queries", len(d.get("query_results", [])))
        n_note = f" Quality JSON: <code>va_staging_all_quality.json</code> (n={n} graded)."

    rows, n_liked, n_disliked, vtag = _golden_grader_calibration_stats()
    body_html = eval_methods_report_body(
        rows,
        n_liked=n_liked,
        n_disliked=n_disliked,
        vtag=vtag,
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    layer1_charts = (
        '<p class="sub-hdr">Layer 1 — heuristic threshold KDE (v1 strict ŷ)</p>'
        + chart_card(
            "Heuristic score distributions — citation & resolution proxies",
            "Dashed = production gate from evals/metrics · teal = liked · red = disliked · "
            "orange band = ±0.10 around threshold",
            "kde_heuristic_thresholds",
            FIG_DIR,
            wide=True,
        )
        + chart_card(
            "Heuristic KDE v2 — URL-map adjusted ŷ + expanded BKH↔VA overlap",
            "Same panels after human_validated_map + kb_url_map — compare d on has_source / overlap titles",
            "kde_heuristic_thresholds_v2",
            FIG_DIR,
            wide=True,
        )
    )

    layer2_charts = (
        '<p class="sub-hdr">Layer 2 — LLM grader calibration (rated VA staging turns)</p>'
        + chart_card(
            "Grader discrimination — Cohen's d vs user sentiment",
            "All graders in merged quality JSON · positive d = liked higher · ★ = default VA gate",
            "cohen_d",
            FIG_DIR,
            wide=True,
        )
        + chart_card(
            "Pass threshold calibration — mean scores & pass rates",
            "Liked/disliked means (dashed = report-panel threshold) · Δ = pass-rate gap at that threshold",
            "threshold_viz",
            FIG_DIR,
            wide=True,
        )
        + chart_card(
            "Score distributions by sentiment — box plots (core 6-panel set)",
            "Fixed panel set for readability · dashed = calibration threshold per grader",
            "box_plots",
            FIG_DIR,
            wide=True,
        )
        + chart_card(
            "KDE threshold calibration — LLM score distributions",
            "Kernel density per grader · dashed threshold · orange noise band",
            "kde_thresholds",
            FIG_DIR,
            wide=True,
        )
    )

    header = report_doc_header(
        title="Eval Methods — Heuristics & LLM Judges",
        meta=default_meta_line(),
        summary=(
            "How we measure support-agent quality in two layers: free structural heuristics (Layer 1), "
            "then LLM-as-judge graders (Layer 2) calibrated against human like/dislike on VA staging."
            + n_note
            + " Corpus profiles: bkh_stats · va_stats · va_suite (A/B chart)."
        ),
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Eval Methods — Layer 1 &amp; Layer 2</title>
  <style>{REPORT_THEME_CSS}</style>
</head>
<body>
{header}
<div class="page">
  <div class="callout teal"><strong>Cost order:</strong> Run <code>make eval-stats</code> / <code>reports-validate</code>
    (Layer 1) before <code>eval-quality</code> (Layer 2). Cap LLM runs with <code>LIMIT=20</code> on first passes.
    Links: <a href="{report_href("va", "bkh_stats")}">BKH stats</a> ·
    <a href="{report_href("va", "va_stats")}">VA staging stats</a> ·
    <a href="{report_href("va", "cohort_compare")}">Cohort compare — BKH↔VA A/B</a>.
    Generated {_html.escape(ts)}.</div>

  {body_html}

  {layer1_charts}

  {layer2_charts}

  {report_footer_note()}
</div>
</body>
</html>
"""

    ReportLayout("va").ensure_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  Eval methods report: {output_path}")
    return output_path


def main() -> None:
    build()


if __name__ == "__main__":
    main()
