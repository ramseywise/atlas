"""BKH ↔ VA side-by-side metric table — embedded on both suite pages."""

from __future__ import annotations

import html as _html
import json

from evals.metrics.suite import SuiteReport
from evals.metrics.calibration.grader_scope import COMPARISON_LAYER2_METRICS
from evals.reports.paths import (
    bkh_stats_path,
    va_staging_all_responses_stats_path,
)
from evals.reports.utils.layout import report_href


def _load_stats_blob(corpus: str) -> dict:
    if corpus == "va":
        p = va_staging_all_responses_stats_path()
    else:
        p = bkh_stats_path()
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    stats = data.get("stats") or {}
    return next(iter(stats.values()), {}) if stats else data


def _heuristic_compare_rows() -> list[tuple[str, float | None, float | None, int, int]]:
    bkh = _load_stats_blob("bkh")
    va = _load_stats_blob("va")

    def _sat(s: dict) -> tuple[float | None, int]:
        sent = s.get("sentiment") or {}
        n_l, n_d = sent.get("n_liked", 0), sent.get("n_disliked", 0)
        return ((n_l / (n_l + n_d)) if (n_l + n_d) else None, n_l + n_d)

    def _rp(s: dict, key: str) -> tuple[float | None, int]:
        rp = s.get("retrieval_proxy") or {}
        conf = rp.get("confusion") or {}
        if key == "f1":
            return rp.get("f1_full"), conf.get("tp", 0) + conf.get("fn", 0)
        if key == "precision":
            return rp.get("precision"), conf.get("tp", 0) + conf.get("fp", 0)
        if key == "recall":
            return rp.get("recall"), conf.get("tp", 0) + conf.get("tn", 0)
        return None, 0

    rows: list[tuple[str, float | None, float | None, int, int]] = []
    b_sat, b_sat_n = _sat(bkh)
    v_sat, v_sat_n = _sat(va)
    rows.append(("Satisfaction (rated turns)", b_sat, v_sat, b_sat_n, v_sat_n))

    for label, key in (
        ("Retrieval F1 (proxy)", "f1"),
        ("Retrieval precision (proxy)", "precision"),
        ("Retrieval recall — conservative (proxy)", "recall"),
    ):
        bv, bn = _rp(bkh, key)
        vv, vn = _rp(va, key)
        rows.append((label, bv, vv, bn, vn))
    return rows


def suite_corpus_compare_html(report: SuiteReport) -> str:
    """Primary metrics: BKH baseline vs VA staging (same rows on both suite pages)."""
    from evals.reports.utils.figures import benchmark_layer2_cohorts

    layer2, bkh_n, va_n = benchmark_layer2_cohorts()
    bool(report.heuristic_stats.get("staging_calibration") or report.heuristic_stats.get("va_stats"))

    h_rows = _heuristic_compare_rows()

    body = (
        '<p class="sub-hdr" style="margin-top:0">Layer 1 — corpus heuristics (full archive)</p>'
        '<table class="data-table"><thead><tr>'
        "<th>Metric</th><th class=\"num\">BKH</th><th class=\"num\">VA staging</th>"
        "<th class=\"num\">Δ</th><th class=\"num\">n (BKH / VA)</th></tr></thead><tbody>"
    )
    for label, bkh_v, va_v, bn, vn in h_rows:
        bkh_txt = f"{bkh_v:.1%}" if bkh_v is not None else "—"
        va_txt = f"{va_v:.1%}" if va_v is not None else "—"
        delta = (
            f"{(va_v - bkh_v):+.1%}"
            if bkh_v is not None and va_v is not None
            else "—"
        )
        body += (
            f"<tr><td>{_html.escape(label)}</td>"
            f'<td class="num">{bkh_txt}</td><td class="num">{va_txt}</td>'
            f'<td class="num">{delta}</td>'
            f'<td class="num muted">{bn:,} / {vn:,}</td></tr>'
        )
    body += "</tbody></table>"

    bkh_n_layer1 = int(_load_stats_blob("bkh").get("n_total", 0) or 0)
    va_n_layer1 = int(_load_stats_blob("va").get("n_total", 0) or 0)
    body += (
        '<p class="sub-hdr" style="margin-top:16px">Layer 2 — LLM gates (primary)</p>'
        '<p class="chart-note">Layer 1 above = '
        f"{bkh_n_layer1:,} paired task_ids (BKH production vs VA staging). "
        f"Layer 2 BKH = stratified cal n={bkh_n} (not all 597 LLM-graded) · "
        f"VA = URL-map calibrated n={va_n}. Grounding N/A on BKH.</p>"
        '<table class="data-table"><thead><tr>'
        "<th>Gate</th><th class=\"num\">Threshold</th>"
        "<th class=\"num\">BKH cal</th><th class=\"num\">VA staging</th>"
        "<th class=\"num\">Δ (pp)</th></tr></thead><tbody>"
    )
    for _key, label, thr_pct in COMPARISON_LAYER2_METRICS:
        thr_pct / 100.0
        bkh_pct = layer2.get("BKH", {}).get(_key)
        va_pct = layer2.get("VA v2", {}).get(_key)
        if bkh_pct is None and va_pct is None:
            continue
        bkh_txt = "N/A" if bkh_pct is None else f"{bkh_pct:.0f}%"
        va_txt = f"{va_pct:.0f}%" if va_pct is not None else "—"
        delta = (
            f"{va_pct - bkh_pct:+.0f}"
            if bkh_pct is not None and va_pct is not None
            else "—"
        )
        d_cls = ""
        if bkh_pct is not None and va_pct is not None:
            d = va_pct - bkh_pct
            d_cls = "pass" if d >= 5 else ("fail" if d <= -5 else "warn")
        body += (
            f"<tr><td>{_html.escape(label)}</td><td class=\"num\">{thr_pct:.0f}%</td>"
            f'<td class="num">{bkh_txt}</td><td class="num">{va_txt}</td>'
            f'<td class="num {d_cls}">{delta}</td></tr>'
        )
    body += "</tbody></table>"

    link = report_href("va", "cohort_compare")

    return f"""
<div class="section-block" id="suite-corpus-compare">
  <div class="section-header" style="margin-bottom:12px">
    <div class="num">Compare</div>
    <h2>BKH vs VA — all primary metrics</h2>
    <p class="lead">Cross-corpus view on the 597 paired cohort (same task_ids). Full A/B dashboard:
    <a href="{link}">cohort compare</a> · profiles:
    <a href="{report_href('bkh', 'bkh_stats')}">BKH stats</a> ·
    <a href="{report_href('va', 'va_stats')}">VA stats</a>.</p>
  </div>
  {body}
</div>
"""
