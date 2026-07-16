"""Atlas HTML report builder — assembles SVG panels into a static HTML page.

Usage:
    from evals.reports.report import build_eval_report, build_segment_report

    # Forecast + eval cycle
    build_eval_report(eval_report, forecast_results, actuals_map)

    # Segmentation run
    build_segment_report(seg_eval, X_2d, labels, segment_names)

Output: evals/reports/output/{eval|segment}/*.html
"""

from __future__ import annotations

import html as _html
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from evals.reports.utils.embed import read_svg as _read_svg_by_name
from evals.reports.utils.theme import REPORT_THEME_CSS

if TYPE_CHECKING:
    from core.segmentation.evaluation import (
        SegmentEvalReport,  # type: ignore[import-untyped]
    )
    from src.agents.state import EvalReport, ForecastResult

OUTPUT_ROOT = Path("evals/reports/output")

_CSS = (
    REPORT_THEME_CSS
    + """
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;}
.pass{background:#d1fae5;color:var(--green);}
.fail{background:#fee2e2;color:var(--red);}
.stat{background:white;border-radius:8px;padding:16px 20px;box-shadow:0 1px 4px rgba(0,0,0,.08);text-align:center;}
.stat .n{font-size:28px;font-weight:800;color:var(--navy);}
.stat .lbl{font-size:11px;color:var(--mid);margin-top:2px;text-transform:uppercase;letter-spacing:.05em;}
.stat .sub{font-size:11px;color:var(--teal);margin-top:2px;}
.fig svg{width:100%;height:auto;display:block;}
.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.ts{font-size:11px;color:var(--mid);margin-top:24px;}
"""
)


def _read_svg(path: Path) -> str:
    """Read SVG from an absolute path — thin wrapper around embed.read_svg."""
    return _read_svg_by_name(path.parent, path.stem)


def _card(content: str) -> str:
    return f'<div class="card"><div class="fig-embed">{content}</div></div>'


def _stat(value: str, label: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{_html.escape(sub)}</div>' if sub else ""
    return (
        f'<div class="stat">'
        f'<div class="n">{_html.escape(value)}</div>'
        f'<div class="lbl">{_html.escape(label)}</div>'
        f"{sub_html}"
        f"</div>"
    )


def _page(title: str, body: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_html.escape(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">
{body}
<p class="ts">Generated {_html.escape(ts)}</p>
</div>
</body>
</html>"""


def build_eval_report(
    report: EvalReport,
    forecast_results: list[ForecastResult] | None = None,
    actuals_map: dict[str, np.ndarray] | None = None,
    *,
    output_path: Path | None = None,
) -> Path:
    """HTML eval report: grader pass rates + forecast grid + summary stats."""
    from evals.reports.figures import (
        fig_forecast_grid,
        fig_grader_pass_rates,
    )

    subdir = f"eval/{report.cycle_id}"
    out_dir = OUTPUT_ROOT / "eval" / report.cycle_id
    out_dir.mkdir(parents=True, exist_ok=True)

    grader_svg = fig_grader_pass_rates(report, subdir=subdir)
    forecast_svg = None
    if forecast_results:
        forecast_svg = fig_forecast_grid(forecast_results, actuals_map, subdir=subdir)

    status = "PASS" if report.all_passed else "FAIL"
    badge_cls = "pass" if report.all_passed else "fail"
    n_series = len(report.series_scores)
    n_passed = sum(1 for scores in report.series_scores.values() if all(s.passed for s in scores))

    stats_row = "".join(
        [
            _stat(f"{report.overall_mase:.3f}", "MASE", "< 1.0 to pass"),
            _stat(f"{report.overall_smape:.1f}%", "SMAPE", "< 15% to pass"),
            _stat(f"{report.directional_accuracy:.1f}%", "Directional", "> 55% to pass"),
            _stat(f"{report.coverage_80:.1f}%", "80% PI Coverage", "≥ 75% to pass"),
            _stat(f"{report.drift_ratio:.3f}", "Drift ratio", "< 1.2 warning"),
        ]
    )

    forecast_section = ""
    if forecast_svg:
        forecast_section = f"""
<h2>Forecast — {n_series} series</h2>
{_card(_read_svg(forecast_svg))}
"""

    summary_section = ""
    if report.summary:
        summary_section = f'<div class="card"><p>{_html.escape(report.summary)}</p></div>'

    body = f"""
<h1>Atlas Eval — {_html.escape(report.cycle_id)}
  <span class="badge {badge_cls}" style="margin-left:12px">{status}</span>
</h1>
<p class="lead">{_html.escape(str(report.forecast_date))} · {n_passed}/{n_series} series fully passing</p>

<h2>Grader Metrics</h2>
<div class="grid3">{stats_row}</div>
{_card(_read_svg(grader_svg))}

{forecast_section}
{summary_section}
"""

    path = output_path or out_dir / "eval_report.html"
    path.write_text(_page(f"Atlas Eval — {report.cycle_id}", body), encoding="utf-8")
    return path


def build_segment_report(
    seg_eval: SegmentEvalReport,
    X_2d: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    segment_names: dict[int, str] | None = None,
    *,
    output_path: Path | None = None,
) -> Path:
    """HTML segmentation report: scatter + eval metrics + cluster sizes."""
    from evals.reports.figures import (
        fig_segment_eval,
        fig_segment_sizes_bar,
        fig_segments_scatter,
    )

    run_id = f"{seg_eval.algorithm.lower()}_k{seg_eval.n_clusters}"
    subdir = f"segment/{run_id}"
    out_dir = OUTPUT_ROOT / "segment" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_svg = fig_segment_eval(seg_eval, subdir=subdir)
    sizes_svg = fig_segment_sizes_bar(
        seg_eval.cluster_sizes,
        segment_names,
        subdir=subdir,
    )
    scatter_svg = None
    if X_2d is not None and labels is not None:
        scatter_svg = fig_segments_scatter(
            X_2d,
            labels,
            segment_names,
            subdir=subdir,
        )

    status = "PASS" if seg_eval.passed else "FAIL"
    badge_cls = "pass" if seg_eval.passed else "fail"
    n_customers = sum(seg_eval.cluster_sizes.values())

    stats_row = "".join(
        [
            _stat(str(seg_eval.n_clusters), "Clusters"),
            _stat(str(n_customers), "Customers", f"{seg_eval.n_noise} noise"),
            _stat(
                f"{seg_eval.silhouette:.3f}" if not np.isnan(seg_eval.silhouette) else "—",
                "Silhouette",
                "≥ 0.25 to pass",
            ),
            _stat(
                f"{seg_eval.davies_bouldin:.3f}" if not np.isnan(seg_eval.davies_bouldin) else "—",
                "Davies-Bouldin",
                "≤ 1.5 to pass",
            ),
        ]
    )

    scatter_section = ""
    if scatter_svg:
        scatter_section = f"""
<h2>Cluster Scatter (2-D projection)</h2>
{_card(_read_svg(scatter_svg))}
"""

    names_section = ""
    if segment_names:
        rows = "".join(
            f"<tr><td>Seg {k}</td><td>{_html.escape(v)}</td>"
            f"<td>{seg_eval.cluster_sizes.get(k, '—')}</td></tr>"
            for k, v in sorted(segment_names.items())
        )
        names_section = f"""
<h2>Segment Labels</h2>
<div class="card">
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="border-bottom:2px solid var(--navy)">
  <th style="text-align:left;padding:6px">ID</th>
  <th style="text-align:left;padding:6px">Name</th>
  <th style="text-align:right;padding:6px">Customers</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
"""

    body = f"""
<h1>Atlas Segmentation — {_html.escape(seg_eval.algorithm)}
  <span class="badge {badge_cls}" style="margin-left:12px">{status}</span>
</h1>
<p class="lead">k={seg_eval.n_clusters} · {n_customers} customers · {seg_eval.n_noise} noise points</p>

<h2>Quality Metrics</h2>
<div class="grid3">{stats_row}</div>
{_card(_read_svg(eval_svg))}

<h2>Cluster Sizes</h2>
{_card(_read_svg(sizes_svg))}

{scatter_section}
{names_section}
"""

    path = output_path or out_dir / "segment_report.html"
    path.write_text(_page(f"Atlas Segments — {seg_eval.algorithm}", body), encoding="utf-8")
    return path
