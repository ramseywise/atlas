"""BKH ↔ VA A/B HTML blocks and chart data (demo §3 style)."""

from __future__ import annotations

import html as _html
import json

from evals.metrics.calibration.grader_scope import COMPARISON_LAYER2_METRICS


def ab_highlight_cards(va_report) -> str:
    """Three stat cards like eval_framework_report §3."""
    from evals.reports.html.results_section import _highlights

    h = _highlights(va_report, va_only=True)
    cards = []

    def _card(value: float, label: str, sub: str, *, ok: bool) -> str:
        color = "var(--green)" if ok else "var(--red)"
        return (
            f'<div class="stat-card">'
            f'<div class="num" style="color:{color}">{value:.0%}</div>'
            f'<div class="label">{_html.escape(label)}</div>'
            f'<div class="sub">{_html.escape(sub)}</div></div>'
        )

    if comp := h.get("completeness"):
        cards.append(
            _card(
                comp["value"],
                "VA Completeness",
                f"Target {comp['threshold']:.0%}",
                ok=comp["passed"],
            )
        )
    if esc := h.get("escalation"):
        cards.append(
            _card(esc["value"], "VA Escalation", f"Target {esc['threshold']:.0%}", ok=esc["passed"])
        )
    if ar := h.get("answer_relevancy"):
        cards.append(
            _card(ar["value"], "Answer Relevancy", f"Target {ar['threshold']:.0%}", ok=ar["passed"])
        )

    if not cards:
        return ""
    return f'<div class="grid-3">{"".join(cards)}</div>'


def ab_delta_table_html() -> str:
    """Primary gates: BKH baseline vs VA calibrated with Δ column."""
    from evals.reports.paths import bkh_stats_path
    from evals.reports.utils.figures import benchmark_layer2_cohorts

    layer2, bkh_n, n_va = benchmark_layer2_cohorts()
    bkh_n_layer1 = 0
    if bkh_stats_path().exists():
        data = json.loads(bkh_stats_path().read_text(encoding="utf-8"))
        stats = data.get("stats") or {}
        blob = next(iter(stats.values()), {}) if stats else data
        bkh_n_layer1 = int(blob.get("n_total", 0) or 0)
    rows = []
    for key, label, thr in COMPARISON_LAYER2_METRICS:
        bkh = layer2.get("BKH", {}).get(key)
        va = layer2.get("VA v2", {}).get(key)
        if bkh is None and va is None:
            continue
        delta = None
        delta_cls = ""
        if bkh is not None and va is not None:
            delta = va - bkh
            if delta >= 5:
                delta_cls = "pass"
            elif delta <= -5:
                delta_cls = "fail"
            else:
                delta_cls = "warn"
        bkh_txt = f"{bkh:.0f}%" if bkh is not None else "N/A"
        va_txt = f"{va:.0f}%" if va is not None else "N/A"
        d_txt = f"{delta:+.0f}pp" if delta is not None else "—"
        rows.append(
            f"<tr><td>{_html.escape(label)}</td>"
            f'<td class="num">{bkh_txt}</td>'
            f'<td class="num">{va_txt}</td>'
            f'<td class="num {delta_cls}">{d_txt}</td>'
            f'<td class="num">{thr:.0f}%</td></tr>'
        )

    if not rows:
        return ""

    return f"""
<div class="card" style="margin-top:0">
  <h3>Primary gates — BKH baseline vs VA calibrated</h3>
  <p class="chart-note">BKH Layer 1 n={bkh_n_layer1:,} paired · BKH Layer 2 n={bkh_n} cal · VA n={n_va} · Δ = VA v2 − BKH</p>
  <table class="data-table">
    <thead><tr>
      <th>Gate</th><th class="num">BKH baseline</th>
      <th class="num">VA staging</th><th class="num">Δ (pp)</th><th class="num">Threshold</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>
"""
