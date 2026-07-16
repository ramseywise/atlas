"""Build golden_all_responses_eval.html — VA golden dataset profile (BKH Tab 02 layout).

Mirrors eval_framework_tabs.html Tab 02:
  Row 1: Response type | Failure mode
  Row 2: Source count  | Language
  Full width: Citation-based retrieval proxy

Topic sentiment omitted (<15% labeled topics in golden vs BKH).

Usage:
    make golden-profile-report
"""

from __future__ import annotations

import argparse
import html as _html
import json
import re
from datetime import datetime
from pathlib import Path

from evals.reports.utils.figures import (
    _extract_file_stats,
    _load_golden_stats,
    fig_golden_failure_mode,
    fig_golden_language,
    fig_golden_response_type,
    fig_golden_retrieval_proxy,
    fig_golden_source_count,
)
from evals.reports.utils.layout import FIGURES_ROOT, ReportLayout

OUT_PATH = ReportLayout("va").html_dir / "va_profile.html"
FIG_DIR = FIGURES_ROOT / "va"

_TAB_CSS = """
:root{--navy:#1e3a5f;--teal:#028090;--amber:#f59e0b;--green:#059669;--red:#dc2626;--mid:#64748b;--dark:#1e293b;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f1f5f9;color:var(--dark);font-size:15px;line-height:1.6;}
.page{max-width:1000px;margin:0 auto;padding:28px 32px 48px;}
.sec-hdr{border-left:5px solid var(--teal);padding:16px 0 16px 20px;margin-bottom:24px;}
.sec-hdr .num{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--teal);font-weight:600;margin-bottom:3px;}
.sec-hdr h2{font-size:20px;font-weight:700;color:var(--navy);margin-bottom:4px;}
.sec-hdr .lead{color:var(--mid);font-size:14px;}
.card{background:white;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:20px;}
.card h3{font-size:12px;font-weight:700;color:var(--navy);margin-bottom:3px;text-transform:uppercase;letter-spacing:.05em;}
.chart-note{font-size:12px;color:var(--mid);margin-bottom:14px;}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}
.g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:16px;}
.stat{background:white;border-radius:8px;padding:18px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08);}
.stat .n{font-size:30px;font-weight:800;color:var(--navy);}
.stat .lbl{font-size:11px;color:var(--mid);margin-top:3px;text-transform:uppercase;letter-spacing:.05em;}
.stat .sub{font-size:11px;color:var(--teal);margin-top:2px;}
.cue{border-radius:6px;padding:13px 16px;margin-bottom:16px;font-size:13px;line-height:1.45;}
.cue.amber{background:#fef3c7;border-left:4px solid var(--amber);}
.cue.teal{background:#e0f4f7;border-left:4px solid var(--teal);}
.cue.red{background:#fee2e2;border-left:4px solid var(--red);}
.fig-embed svg{width:100%;height:auto;max-height:220px;display:block;}
.card.wide .fig-embed svg{max-height:340px;}
.fig-source{font-size:11px;color:var(--mid);margin-top:8px;}
.sub-hdr{font-size:13px;font-weight:700;color:var(--navy);margin:28px 0 10px;text-transform:uppercase;letter-spacing:.05em;}
"""


def _read_svg(name: str) -> str:
    path = FIG_DIR / f"{name}.svg"
    if not path.exists():
        return f'<p style="color:var(--mid);font-size:12px">Missing {name}.svg</p>'
    raw = path.read_text(encoding="utf-8")
    raw = re.sub(r"<\?xml[^?]*\?>", "", raw).strip()
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw).strip()
    raw = re.sub(r'(<svg[^>]*)\s+width="[^"]*"', r"\1", raw, count=1)
    raw = re.sub(r'(<svg[^>]*)\s+height="[^"]*"', r"\1", raw, count=1)
    return raw


def _chart_card(title: str, note: str, svg_name: str) -> str:
    return (
        f'<div class="card">'
        f"<h3>{_html.escape(title)}</h3>"
        f'<p class="chart-note">{_html.escape(note)}</p>'
        f'<div class="fig-embed">{_read_svg(svg_name)}</div>'
        f"</div>"
    )


def build(
    stats: dict,
    output_path: Path = OUT_PATH,
    *,
    export_figures: bool = True,
) -> Path:
    from evals.reports.utils.figures import set_figures_source

    if export_figures:
        set_figures_source("va")
        fig_golden_response_type(stats)
        fig_golden_failure_mode(stats)
        fig_golden_source_count(stats)
        fig_golden_language(stats)
        fig_golden_retrieval_proxy(stats)

    n = stats["n_total"]
    s = stats.get("sentiment", {})
    n_convs = stats.get("n_unique_convs", 0)
    n_users = stats.get("n_unique_users", 0)
    cov = s.get("rating_coverage_rate", 0) * 100
    dlr = s.get("dislike_like_ratio")
    dlr_s = f"{dlr:.1f}:1" if dlr else "—"
    unknown_rate = stats.get("response_types", {}).get("unknown", {}).get("rate", 0) * 100
    rp = stats.get("retrieval_proxy", {})
    prec = rp.get("precision", 0) * 100
    recall = rp.get("recall", 0) * 100
    rp.get("f1", 0) * 100
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    avg_q = stats.get("qa_stats", {}).get("avg_query_words", 0)
    e_grounding_pct = stats.get("failure_types", {}).get("E_grounding", {}).get("rate", 0) * 100

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VA Golden Dataset — Profile (BKH Tab 02 layout)</title>
  <style>{_TAB_CSS}</style>
</head>
<body>
<div class="page">

  <div class="sec-hdr">
    <div class="num">VA Golden · mirrors BKH Tab 02</div>
    <h2>VA Agent Sample — What We Have and Its Limitations</h2>
    <p class="lead">{n:,} turns of rated VA production responses (golden_all_responses.jsonl).
      {cov:.1f}% rating coverage vs BKH 1.7%. Same charts as BookKeeper Hero baseline — compare shape side-by-side.</p>
  </div>

  <div class="g3">
    <div class="stat"><div class="n">{n:,}</div><div class="lbl">Total turns</div><div class="sub">{n_convs:,} unique conversations</div></div>
    <div class="stat"><div class="n">{n_users:,}</div><div class="lbl">Unique users</div><div class="sub">{avg_q:.1f} words avg query</div></div>
    <div class="stat"><div class="n" style="color:var(--green)">{cov:.1f}%</div><div class="lbl">Turns rated</div><div class="sub">{s.get('n_liked', 0)} liked · {s.get('n_disliked', 0)} disliked</div></div>
  </div>

  <div class="g2">
    {_chart_card("Response Type Distribution", f"How the agent responded across all {n:,} turns", "golden_response_type")}
    {_chart_card("Failure Mode Taxonomy", "Share of turns by failure category (91.6% rated — not BKH-style unrated-dominated)", "golden_failure_mode")}
  </div>

  <div class="cue amber"><strong>⚠ The unknown problem ({unknown_rate:.1f}% of turns):</strong>
    Turns where the VA answered but could not cite KB content — same signal as BKH unknown bucket (17.3%).
    Highest-priority hallucination risk. BKH: 11,976 turns; golden: {int(unknown_rate / 100 * n):,} turns.</div>

  <div class="g2">
    {_chart_card("Source Count Distribution", "Number of KB sources cited per response (max: 3)", "golden_source_count")}
    {_chart_card("Language Breakdown", "langdetect on query text — Danish vs Scandinavian often misclassified; directional only", "golden_language")}
  </div>

  <div class="cue red"><strong>⚠ VA golden already uses Shine URLs — E_grounding is not kb_url_map:</strong>
    This sample cites <code>help.shine.co</code> (live VA). <code>data/kb_url_map.json</code> fixes Billy↔Shine slug match for
    retrieval MRR and <code>SourceMatchGrader</code> on BKH-era keys — it does not change the failure taxonomy.
    <strong>E_grounding ({e_grounding_pct:.1f}%)</strong> means disliked + had sources (heuristic), not URL mismatch or LLM GroundingGrader.</div>

  <p class="sub-hdr">Citation-Based Retrieval Proxy — Pseudo-F1</p>
  <p class="chart-note" style="margin:-6px 0 12px">Treats &quot;has sources&quot; as a binary quality predictor.
    Precision {prec:.1f}% is biased low (eval sample is {dlr_s} dislike:like vs natural ~2:1). Recall {recall:.1f}% is more stable.
    Pre-grader signal — run <code>make va-grade-golden-all</code> for LLM grounding eval.</p>

  <div class="card wide">
    <div class="fig-embed">{_read_svg("golden_retrieval_proxy")}</div>
    <p class="fig-source">Source: evals/reports/output/figures/va/golden_retrieval_proxy.svg · Regenerate: <code>make reports-validate</code></p>
  </div>

  <p class="fig-source" style="margin-top:20px">Tables &amp; QA samples:
    <a href="../va/va_stats.html">va_stats.html</a>
    · Generated {_html.escape(ts)}</p>

</div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Written: {output_path} ({len(html):,} bytes)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--no-export-figures", action="store_true")
    args = parser.parse_args()

    if args.stats_json:
        data = json.loads(args.stats_json.read_text(encoding="utf-8"))
        stats = _extract_file_stats(data)
    else:
        stats = _load_golden_stats()

    build(stats, args.output, export_figures=not args.no_export_figures)


if __name__ == "__main__":
    main()
