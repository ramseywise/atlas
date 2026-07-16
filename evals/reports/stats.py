"""HTML and JSON report generation for eval pipeline results.

Importable by any eval runner. Pass `title` and `ft_order` to customise per-dataset.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path

from evals.reports.utils._sections import (
    _section,
    _url_coverage_section,
)
from evals.reports.utils.embed import CHART_MARKER, NARRATIVE_MARKER
from evals.reports.utils.theme import (
    REPORT_THEME_CSS,
    default_meta_line,
    report_doc_header,
    report_footer_note,
)


def collect_qa_samples(
    tasks: list, sentiments: list[str], n_per_sentiment: int = 3
) -> dict[str, list[dict]]:
    """Sample n QA pairs per sentiment (liked/disliked), diverse across failure types."""
    buckets: dict[str, list] = {"liked": [], "disliked": [], "unrated": []}
    for task, s in zip(tasks, sentiments, strict=False):
        meta = task.metadata or {}
        raw_ft = meta.get("failure_type")
        if raw_ft and raw_ft != "none":
            ft = raw_ft
        elif s == "liked":
            ft = "no_failure"
        elif s == "unrated":
            ft = "n/a"
        else:
            ft = "unclassified"
        q = task.query.strip() if isinstance(task.query, str) else ""
        r = task.response.strip() if isinstance(task.response, str) else ""
        if not q or not r:
            continue
        buckets[s].append(
            {
                "query": q,
                "response": r,
                "sentiment": s,
                "ft": ft,
                "urls": task.expected_urls or [],
            }
        )

    def _diverse_sample(records: list[dict], n: int) -> list[dict]:
        seen_fts: set = set()
        result: list[dict] = []
        for rec in records:
            if rec["ft"] not in seen_fts:
                seen_fts.add(rec["ft"])
                result.append(rec)
                if len(result) >= n:
                    return result
        for rec in records:
            if rec not in result:
                result.append(rec)
                if len(result) >= n:
                    return result
        return result

    return {s: _diverse_sample(recs, n_per_sentiment) for s, recs in buckets.items() if recs}


def export_stats(
    all_stats: dict,
    paths: list[Path],
    output_path: str | Path,
    url_coverage: dict | None = None,
    qa_samples_by_file: dict | None = None,
    title: str = "Eval Stats",
    ft_order: list[str] | None = None,
    metrics: list | None = None,
) -> Path:
    """Dump stats to JSON. Stores everything needed to re-render HTML via render_report_from_json()."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_data = [
        {
            "metric_name": m.metric_name,
            "value": m.value,
            "threshold": m.threshold,
            "passed": m.passed,
            "n_graded": m.n_graded,
            "breakdown": m.breakdown,
        }
        for m in (metrics or [])
    ]
    out.write_text(
        json.dumps(
            {
                "exported_at": datetime.now().isoformat(),
                "title": title,
                "ft_order": ft_order or ["none"],
                "files": [p.name for p in paths],
                "stats": all_stats,
                "metrics": metrics_data,
                "url_coverage": url_coverage or {},
                "qa_samples": qa_samples_by_file or {},
            },
            indent=2,
        )
    )
    return out


def render_report_from_json(json_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Re-render an HTML report from a previously exported JSON — no eval re-run needed."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    html_path = output_path or Path(json_path).with_suffix(".html")
    paths = [Path(f) for f in data.get("files", [])]
    return export_stats_html(
        all_stats=data["stats"],
        paths=paths,
        output_path=html_path,
        qa_samples_by_file=data.get("qa_samples") or {},
        url_coverage=data.get("url_coverage") or {},
        title=data.get("title", "Eval Stats"),
        ft_order=data.get("ft_order"),
    )


_DETAIL_CSS = """
.section {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 20px 24px;
  margin-bottom: 24px;
}
.section > h2 {
  font-size: 16px;
  font-weight: 700;
  color: var(--navy);
  margin: 0 0 12px;
  border-bottom: 1px solid var(--slate);
  padding-bottom: 8px;
}
.section h3 { font-size: 13px; color: var(--navy); margin: 16px 0 6px; }
.section table { border-collapse: collapse; width: 100%; margin-bottom: 12px; font-size: 13px; }
.section th {
  text-align: left;
  font-size: 11px;
  color: var(--mid);
  border-bottom: 1px solid var(--slate);
  padding: 6px 8px;
}
.section td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--slate);
  font-variant-numeric: tabular-nums;
}
.section .num { text-align: right; }
@page { margin: 1.5cm; size: A4; }
@media print {
  body { background: #fff; }
  .section, .qa-card { page-break-inside: avoid; break-inside: avoid; }
  h2, h3 { page-break-after: avoid; break-after: avoid; }
  tr { page-break-inside: avoid; break-inside: avoid; }
}
"""


def export_stats_html(
    all_stats: dict,
    paths: list[Path],
    output_path: str | Path,
    qa_samples_by_file: dict[str, dict] | None = None,
    url_coverage: dict | None = None,
    title: str = "Eval Stats",
    ft_order: list[str] | None = None,
    pdf: bool = False,
) -> Path:
    """Write a self-contained HTML stats report — no external dependencies.

    Args:
        all_stats: mapping of label → stats dict (from compute_stats).
        paths: source file paths (used for the subtitle).
        output_path: where to write the HTML file.
        qa_samples_by_file: optional mapping of label → collect_qa_samples output.
        url_coverage: optional output of compute_url_coverage.
        title: page and h1 title.
        ft_order: preferred ordering for failure types in QA sample cards.
            Unknown types are appended after. Defaults to ["none"].
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    resolved_ft_order = ft_order or ["none"]
    file_note = ", ".join(p.name for p in paths[:6])
    meta = default_meta_line(files=[file_note] if file_note else None)
    is_va = "va" in title.lower() or "staging" in title.lower()
    summary = (
        "VA staging corpus profile — TLDR and insight cards at top; pass-rate gates on va_suite."
        if is_va
        else "BKH corpus profile — TLDR and insight cards at top; pass-rate gates on bkh_suite."
    )

    qa_samples_by_file = qa_samples_by_file or {}
    sections_html = ""
    for key, stats in all_stats.items():
        qa = qa_samples_by_file.get(key)
        sections_html += _section(key, stats, qa, resolved_ft_order)

    coverage_html = _url_coverage_section(url_coverage or {})
    header = report_doc_header(title=title, meta=meta, summary=summary)

    html = (
        f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">\n'
        f"<title>{_html.escape(title)} — {ts}</title>\n"
        f"<style>{REPORT_THEME_CSS}{_DETAIL_CSS}</style></head><body>\n"
        f"{header}\n"
        f'<div class="page">\n'
        f"{NARRATIVE_MARKER}\n{CHART_MARKER}\n"
        f'<p class="sub-hdr">Detail tables (eval_stats)</p>\n'
        f"{sections_html}\n"
        f"{coverage_html}\n"
        f"{report_footer_note()}\n"
        f"</div></body></html>"
    )

    out.write_text(html, encoding="utf-8")
    if pdf:
        from evals.reports.utils._pdf import html_to_pdf

        pdf_path = html_to_pdf(out)
        print(f"PDF: {pdf_path}")
    return out
