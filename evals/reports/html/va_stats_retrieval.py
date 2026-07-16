"""Replace raw citation-proxy block on va_stats.html with overlap-adjusted CM."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evals.pipelines.datasets import load_jsonl
from evals.reports.paths import va_staging_all_responses_path
from evals.metrics.comparison.url_overlap import compute_overlap_retrieval_proxy
from evals.reports.utils._sections import retrieval_proxy_block_html


def _overlap_stats(stats: dict, staging: Any) -> dict | None:
    if staging is None:
        return None
    va_path = getattr(staging, "va_responses_path", None) or va_staging_all_responses_path()
    if not va_path.exists():
        return None
    tasks = load_jsonl(va_path)
    records = getattr(staging, "overlap_records", None) or []
    if not records:
        return None
    overlap_rp = compute_overlap_retrieval_proxy(tasks, records)
    return {**stats, "retrieval_proxy_overlap": overlap_rp}


def replace_va_detail_retrieval_proxy(html_path: Path, stats: dict, staging: Any) -> bool:
    """Swap raw has_source CM in detail tables for BKH↔VA overlap-adjusted CM."""
    merged = _overlap_stats(stats, staging)
    if not merged:
        return False

    overlap_html = retrieval_proxy_block_html(merged, variant="overlap")
    if not overlap_html:
        return False

    preamble = (
        '<p class="muted" style="font-size:.85em;margin:0 0 10px">'
        "<b>Canonical retrieval proxy for VA staging</b> — overlap-adjusted (not raw ŷ=has_source). "
        "Layer-1 pass-rate table: <a href=\"../va/va_suite.html\">va_suite.html</a>.</p>"
    )
    new_block = preamble + overlap_html

    html = html_path.read_text(encoding="utf-8")
    pattern = (
        r"<h3>Retrieval proxy[^<]*(?:<[^>]+>[^<]*)*</h3>"
        r"[\s\S]*?"
        r"<!-- end-retrieval-proxy -->"
    )
    if not re.search(pattern, html):
        return False

    html = re.sub(pattern, new_block, html, count=1)
    html_path.write_text(html, encoding="utf-8")
    return True
