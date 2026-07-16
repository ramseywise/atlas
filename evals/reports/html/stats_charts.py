"""Export profile figures and embed them into canonical stats HTML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.reports.html.narratives import (
    bkh_dataset_section,
    va_staging_dataset_section,
)
from evals.reports.html.stats_profile import reclassification_overlap_section
from evals.reports.paths import va_staging_all_responses_stats_path
from evals.reports.utils.embed import (
    chart_card,
    inject_charts_section,
    inject_narrative_section,
)
from evals.reports.utils.figures import (
    fig_bkh_failure_mode,
    fig_bkh_language,
    fig_bkh_response_type,
    fig_bkh_source_count,
    fig_golden_failure_mode,
    fig_golden_language,
    fig_golden_response_type,
    fig_golden_source_count,
    set_figures_source,
)
from evals.reports.utils.layout import ReportLayout

LANGUAGE_CHART_NOTE = (
    "Query language via langdetect — often confuses Danish with Norwegian/Swedish "
    "or marks short Danish as unknown; directional only"
)

LANGUAGE_DETECT_DISCLAIMER = """
<p class="muted" style="font-size:12px;margin-top:4px;margin-bottom:20px">
  <code>langdetect</code> on queries — rough mix only (often mislabels Danish); B_language uses routing metadata.
</p>
"""


def _stats_from_json(json_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    stats_map = data.get("stats") or {}
    if not stats_map:
        raise ValueError(f"No stats in {json_path}")
    return next(iter(stats_map.values()))


def _profile_charts_section(n: int, fig_dir: Path, *, prefix: str) -> str:
    """Four charts in 2×2 grid — eval_framework_report.html §1 (no retrieval proxy here)."""
    return (
        '<p class="sub-hdr">Dataset profile — charts</p>\n'
        '<div class="grid-2">\n'
        + chart_card(
            "Source count distribution",
            "KB sources cited per response — 0-source turns are shortest answers",
            f"{prefix}_source_count",
            fig_dir,
        )
        + chart_card(
            "Response type distribution",
            f"How the agent responded across {n:,} turns",
            f"{prefix}_response_type",
            fig_dir,
        )
        + '</div>\n<div class="grid-2">\n'
        + chart_card(
            "Failure mode taxonomy",
            "Share of turns by failure category (raw eval_stats labels)",
            f"{prefix}_failure_mode",
            fig_dir,
        )
        + chart_card(
            "Language breakdown",
            LANGUAGE_CHART_NOTE,
            f"{prefix}_language",
            fig_dir,
        )
        + "</div>\n"
        + LANGUAGE_DETECT_DISCLAIMER
    )


def _export_corpus_figures(stats: dict, *, va: bool) -> None:
    if va:
        fig_golden_response_type(stats)
        fig_golden_failure_mode(stats)
        fig_golden_source_count(stats)
        fig_golden_language(stats)
    else:
        fig_bkh_response_type(stats)
        fig_bkh_failure_mode(stats)
        fig_bkh_source_count(stats)
        fig_bkh_language(stats)


def enrich_va_stats(
    html_path: Path,
    stats_json: Path | None = None,
    staging: Any = None,
) -> None:
    """Section 01-style header + charts at top of va_stats.html (before tables)."""
    layout = ReportLayout("va")
    if stats_json is None:
        stats_json = va_staging_all_responses_stats_path()
    stats = _stats_from_json(stats_json)

    narrative = va_staging_dataset_section(stats)
    if staging is None:
        try:
            from evals.metrics.calibration.staging import get_va_staging

            staging = get_va_staging(auto_prepare=False)
        except Exception:
            staging = None
    narrative += reclassification_overlap_section(staging)

    inject_narrative_section(html_path, narrative)

    set_figures_source("va")
    _export_corpus_figures(stats, va=True)

    n = stats.get("n_total", 0)
    inject_charts_section(
        html_path,
        _profile_charts_section(n, layout.figures_dir, prefix="va_staging"),
        at_top=True,
    )

    from evals.reports.html.va_stats_retrieval import (
        replace_va_detail_retrieval_proxy,
    )

    if replace_va_detail_retrieval_proxy(html_path, stats, staging):
        print("  VA stats detail tables: overlap-adjusted retrieval proxy")
    print("  VA stats narrative + charts embedded (top)")


def enrich_bkh_stats(html_path: Path) -> None:
    """Section 01 narrative + four charts at top of bkh_stats.html (before tables)."""
    from evals.reports.paths import (
        bkh_stats_path,
        ensure_bkh_stats,
    )

    layout = ReportLayout("bkh")
    ensure_bkh_stats(force=True)
    stats_json = bkh_stats_path()
    stats = _stats_from_json(stats_json) if stats_json.exists() else {}

    inject_narrative_section(html_path, bkh_dataset_section(stats))

    set_figures_source("bkh")
    if stats:
        _export_corpus_figures(stats, va=False)

    section = _profile_charts_section(stats.get("n_total", 0), layout.figures_dir, prefix="bkh")
    inject_charts_section(html_path, section, at_top=True)
    print("  BKH stats narrative + charts embedded (top)")
