"""Shared VIR-193 report theme (eval_framework_report.html)."""

from __future__ import annotations

import html as _html
from datetime import datetime

REPORT_THEME_CSS = """
:root {
  --navy: #1e3a5f;
  --teal: #028090;
  --amber: #f59e0b;
  --green: #059669;
  --red: #dc2626;
  --purple: #7c3aed;
  --offwhite: #f8fafc;
  --slate: #e2e8f0;
  --mid: #64748b;
  --dark: #1e293b;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f1f5f9;
  color: var(--dark);
  font-size: 15px;
  line-height: 1.6;
}
.doc-header {
  background: var(--navy);
  color: white;
  padding: 40px 0 36px;
  margin-bottom: 0;
}
.doc-header .inner { max-width: 1100px; margin: 0 auto; padding: 0 32px; }
.doc-header h1 { font-size: 26px; font-weight: 700; margin-bottom: 8px; }
.doc-header .meta { color: #93c5fd; font-size: 13px; margin-bottom: 12px; }
.doc-header .summary { color: #bfdbfe; font-size: 14px; max-width: 720px; line-height: 1.55; }
.page { max-width: 1100px; margin: 0 auto; padding: 28px 32px 48px; }
.section-block {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 20px 24px;
  margin-bottom: 24px;
}
.section-header {
  border-left: 5px solid var(--teal);
  padding: 12px 0 12px 18px;
  margin-bottom: 20px;
}
.section-header .num {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: var(--teal);
  font-weight: 600;
}
.section-header h2 { font-size: 20px; font-weight: 700; color: var(--navy); margin: 4px 0 6px; }
.section-header .lead { color: var(--mid); font-size: 14px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }
@media (max-width: 900px) {
  .grid-2, .grid-3 { grid-template-columns: 1fr; }
  .grid-4 { grid-template-columns: repeat(2, 1fr); }
}
.stat-card {
  background: white;
  border-radius: 8px;
  padding: 18px;
  text-align: center;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.stat-card .num { font-size: 30px; font-weight: 800; color: var(--navy); }
.stat-card .label {
  font-size: 11px;
  color: var(--mid);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.stat-card .sub { font-size: 12px; color: var(--teal); margin-top: 2px; }
.insight-card .sub { font-size: 11px; line-height: 1.35; color: var(--mid); }
.card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 20px;
  margin-bottom: 20px;
}
.card h3 {
  font-size: 12px;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.card .chart-note { font-size: 12px; color: var(--mid); margin-bottom: 12px; }
.card.wide .fig-embed svg { max-height: none; }
.fig-embed svg { width: 100%; height: auto; display: block; max-height: 280px; }
.narrative {
  background: white;
  border-radius: 8px;
  padding: 20px 24px;
  margin-bottom: 20px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
}
.narrative p { margin-bottom: 10px; font-size: 14px; }
.narrative p:last-child { margin-bottom: 0; }
.narrative strong { color: var(--navy); }
.callout {
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 18px;
  font-size: 13px;
  line-height: 1.45;
}
.callout.amber { background: #fef3c7; border-left: 4px solid var(--amber); }
.callout.teal { background: #e0f4f7; border-left: 4px solid var(--teal); }
.callout.red { background: #fee2e2; border-left: 4px solid var(--red); }
.callout.green { background: #d1fae5; border-left: 4px solid var(--green); }
.callout strong { font-weight: 700; }
.data-table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }
.data-table th {
  background: var(--navy);
  color: white;
  padding: 8px 10px;
  text-align: left;
  font-weight: 600;
  font-size: 11px;
}
.data-table td { padding: 8px 10px; border-bottom: 1px solid var(--slate); }
.data-table tr:hover td { background: var(--offwhite); }
.data-table .num { text-align: right; font-variant-numeric: tabular-nums; }
.pass { color: var(--green); font-weight: 700; }
.fail { color: var(--red); font-weight: 700; }
.warn { color: var(--amber); font-weight: 700; }
.sub-hdr {
  font-size: 12px;
  font-weight: 700;
  color: var(--navy);
  margin: 24px 0 12px;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 4px;
}
.chip.liked { background: #d1fae5; color: var(--green); }
.chip.disliked { background: #fee2e2; color: var(--red); }
.chip.unrated { background: #e2e8f0; color: #475569; }
.muted { color: var(--mid); font-size: 13px; }
.detail-hdr { font-size: 13px; font-weight: 700; color: var(--navy); margin: 0 0 8px; }
.detail-table th { background: #f1f5f9; color: var(--navy); font-size: 11px; }
.detail-table td { font-size: 13px; padding: 6px 8px; border-bottom: 1px solid var(--slate); }
.detail-table .num { text-align: right; }
.qa-card {
  background: var(--offwhite);
  border: 1px solid var(--slate);
  border-radius: 6px;
  padding: 12px 14px;
  margin-bottom: 10px;
}
.qa-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
.qa-q { font-weight: 600; margin-bottom: 4px; font-size: 14px; }
.qa-r { color: var(--dark); white-space: pre-wrap; font-size: 13px; }
.footer-note { color: var(--mid); font-size: 12px; margin-top: 28px; }
.chart-wrap { min-height: 320px; }
.chart-wrap .fig-embed svg { max-height: 420px; }
"""

SUITE_METRICS_CSS = """
.verdict-banner {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 16px 20px;
  margin-bottom: 24px;
}
.verdict-pill {
  display: inline-block;
  padding: 6px 16px;
  border-radius: 6px;
  font-weight: 700;
  font-size: 13px;
  letter-spacing: .04em;
}
.verdict-pill.pass { background: #d1fae5; color: var(--green); }
.verdict-pill.fail { background: #fee2e2; color: var(--red); }
.verdict-pill.warn { background: #fef3c7; color: #92400e; }
.verdict-line { color: var(--mid); font-size: 14px; }
.layer-panel {
  background: white;
  border-radius: 8px;
  box-shadow: 0 1px 4px rgba(0,0,0,.08);
  padding: 20px 24px;
  margin-bottom: 24px;
}
.layer-panel h2 {
  font-size: 16px;
  font-weight: 700;
  color: var(--navy);
  margin: 0 0 14px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--slate);
}
.metric-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.metric-table th {
  background: var(--navy);
  color: white;
  padding: 9px 12px;
  text-align: left;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.metric-table td { padding: 9px 12px; border-bottom: 1px solid var(--slate); }
.metric-table tr:hover td { background: var(--offwhite); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 700;
}
.b-pass { background: #d1fae5; color: var(--green); }
.b-fail { background: #fee2e2; color: var(--red); }
.b-wip { background: #ede9fe; color: var(--purple); }
.b-info { background: #e0f4f7; color: var(--teal); }
.b-warn { background: #fef3c7; color: #92400e; }
.pre-llm-banner {
  background: #e0f4f7;
  border-left: 4px solid var(--teal);
  border-radius: 6px;
  padding: 12px 16px;
  font-size: 13px;
  color: var(--dark);
  margin-bottom: 14px;
  line-height: 1.45;
}
.caveat { font-size: 12px; color: var(--mid); font-style: italic; margin-top: 4px; max-width: 520px; }
.inversion-warn { color: var(--red); font-weight: 600; font-size: 12px; }
.suite-heur-grid .stat-card .stat-label {
  font-size: 11px;
  color: var(--mid);
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-bottom: 4px;
}
.suite-heur-grid .stat-card .stat-value { font-size: 26px; font-weight: 800; color: var(--navy); }
.suite-heur-grid .stat-card .stat-sub { font-size: 12px; color: var(--teal); margin-top: 2px; }
.ft-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.ft-table th { background: #f1f5f9; color: var(--navy); padding: 8px 10px; font-size: 11px; }
.ft-table td { padding: 8px 10px; border-bottom: 1px solid var(--slate); }
.methodology-fold summary { cursor: pointer; font-weight: 600; color: var(--mid); font-size: 14px; }
.methodology-fold table { margin-top: 10px; }
.gate-card { text-align: center; padding: 16px 14px; }
.gate-card .gate-layer {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--teal);
  font-weight: 700;
  margin-bottom: 6px;
}
.gate-grid .stat-card .num { font-size: 26px; }
.slice-card h3 { font-size: 14px; color: var(--navy); margin-bottom: 4px; }
.slice-metrics { font-size: 12px; }
.slice-metrics td, .slice-metrics th { padding: 6px 8px; }
.slice-grid { align-items: start; }
.table-scroll { overflow-x: auto; margin: 12px 0; }
.no-metrics {
  background: #fef3c7;
  border-left: 4px solid var(--amber);
  border-radius: 6px;
  padding: 14px 16px;
  color: #78350f;
  font-size: 14px;
}
"""


def report_doc_header(
    *,
    title: str,
    meta: str,
    summary: str,
) -> str:
    return f"""
<div class="doc-header">
  <div class="inner">
    <div class="meta">{_html.escape(meta)}</div>
    <h1>{_html.escape(title)}</h1>
    <p class="summary">{summary}</p>
  </div>
</div>
"""


def report_footer_note(text: str = "Regenerate: make reports-validate") -> str:
    return f'<p class="footer-note">{_html.escape(text)}</p>'


def default_meta_line(*, files: list[str] | None = None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    files_note = ""
    if files:
        files_note = " · " + ", ".join(files[:6])
    return f"Generated {ts}{files_note}"
