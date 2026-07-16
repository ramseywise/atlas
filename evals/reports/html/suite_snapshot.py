"""Primary gate snapshot cards — top of suite HTML (pass/fail at a glance)."""

from __future__ import annotations

import html as _html

from evals.metrics._constants import METRIC_FRIENDLY_NAMES, TIER_THRESHOLDS
from evals.metrics.base import MetricResult
from evals.metrics.suite import SuiteReport


def _find_metric(report: SuiteReport, name: str) -> MetricResult | None:
    for m in report.heuristic_metric_results + report.judge_metric_results:
        if m.metric_name == name:
            return m
    return None


def _conv_resolution_rate(stats: dict) -> tuple[float, int, int] | None:
    """Clean resolution rate at conversation level (one label per conv)."""
    co = stats.get("conv_outcome_breakdown") or {}
    resolved = int(co.get("resolved", 0))
    friction = int(co.get("resolved_with_friction", 0))
    unresolved = int(co.get("unresolved", 0))
    denom = resolved + friction + unresolved
    if denom <= 0:
        return None
    return resolved / denom, denom, resolved


def _gate_card(
    label: str,
    value_str: str,
    *,
    threshold_str: str,
    passed: bool | None,
    sub: str,
    layer: str,
) -> str:
    if passed is None:
        pill_cls, pill_txt = "warn", "N/A"
        accent = "var(--mid)"
    elif passed:
        pill_cls, pill_txt = "pass", "PASS"
        accent = "var(--green)"
    else:
        pill_cls, pill_txt = "fail", "FAIL"
        accent = "var(--red)"
    return (
        f'<div class="stat-card gate-card" style="border-top:3px solid {accent}">'
        f'<div class="gate-layer">{_html.escape(layer)}</div>'
        f'<div class="num" style="color:{accent};font-size:26px">{_html.escape(value_str)}</div>'
        f'<div class="label">{_html.escape(label)}</div>'
        f'<div class="sub">{sub}</div>'
        f'<span class="verdict-pill {pill_cls}" style="margin-top:8px;font-size:11px">{pill_txt}</span>'
        f'<div class="sub" style="margin-top:4px">gate {threshold_str}</div>'
        f"</div>"
    )


def _metric_gate_card(m: MetricResult, *, layer: str, threshold_override: float | None = None) -> str:
    thr = threshold_override if threshold_override is not None else m.threshold
    thr_str = f"{thr:.0%}" if thr > 0 else "—"
    if thr <= 0:
        passed = None
    else:
        passed = m.value >= thr
    label = METRIC_FRIENDLY_NAMES.get(m.metric_name, m.metric_name.replace("_", " "))
    sub = f"n={m.n_graded:,}"
    caveat = (m.breakdown or {}).get("caveat", "")
    if caveat and len(caveat) > 90:
        caveat = caveat[:87] + "…"
    if caveat:
        sub = f"{sub} · {caveat}"
    return _gate_card(
        label,
        f"{m.value:.1%}",
        threshold_str=thr_str,
        passed=passed,
        sub=sub,
        layer=layer,
    )


def suite_gate_snapshot_html(report: SuiteReport) -> str:
    """Primary metrics grid above Layer 1/2 tables."""
    is_va = bool(
        report.heuristic_stats.get("staging_calibration")
        or report.heuristic_stats.get("va_stats")
    )
    stats = report.heuristic_stats
    cards: list[str] = []

    # --- Layer 1 heuristics ---
    f1 = _find_metric(report, "retrieval_f1")
    if f1:
        cards.append(
            _metric_gate_card(
                f1,
                layer="Layer 1",
                threshold_override=0.50,
            )
        )

    sat = _find_metric(report, "satisfaction_rate")
    if sat:
        cards.append(_metric_gate_card(sat, layer="Layer 1"))

    conv = _conv_resolution_rate(stats)
    if conv is not None:
        rate, n_conv, n_resolved = conv
        thr = TIER_THRESHOLDS["resolution_rate"]
        cards.append(
            _gate_card(
                "Resolution (conversation)",
                f"{rate:.1%}",
                threshold_str=f"{thr:.0%}",
                passed=rate >= thr,
                sub=f"{n_resolved:,} / {n_conv:,} convs · sparse labels",
                layer="Layer 1",
            )
        )

    # --- Layer 2 LLM ---
    if is_va:
        judge_keys = (
            ("answer_relevancy", "Answer relevancy"),
            ("completeness", "Completeness"),
            ("grounding", "Grounding"),
            ("escalation", "Escalation"),
        )
        llm_note = (
            "Grounding uses passage-aware judge on VA staging. "
            "RAGAS ctx/faithfulness are in tracking fold (not calibrated without passages)."
        )
    else:
        judge_keys = (
            ("answer_relevancy", "Answer relevancy"),
            ("completeness", "Completeness"),
            ("escalation", "Escalation"),
        )
        llm_note = (
            "BKH = n≈50 stratified calibration sample. "
            "Grounding / RAGAS not run on production BKH — see "
            '<a href="../va/va_suite.html">va_suite</a> for grounding gates.'
        )

    for key, _label in judge_keys:
        m = _find_metric(report, key)
        if m:
            cards.append(_metric_gate_card(m, layer="Layer 2"))

    if not cards:
        return ""

    n_cols = min(len(cards), 4)
    grid_style = f"grid-template-columns: repeat({n_cols}, 1fr);" if n_cols < 4 else ""

    return f"""
<div class="section-block gate-snapshot">
  <div class="section-header" style="margin-bottom:12px">
    <div class="num">Gates</div>
    <h2>Primary metrics — pass / fail</h2>
    <p class="lead">Snapshot before Layer 1 &amp; 2 deep-dive tables below.</p>
  </div>
  <div class="grid-4 gate-grid" style="{grid_style}">
    {"".join(cards)}
  </div>
  <p class="muted" style="font-size:13px;margin:8px 0 0">{llm_note}</p>
</div>
"""
