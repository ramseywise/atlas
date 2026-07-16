"""Narrative HTML blocks ported from eval_framework_report.html (sections 01–03)."""

from __future__ import annotations

import html as _html
import json
from typing import Any

from evals.reports.utils.layout import dataset_root

_NARRATIVE_CSS = """
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}
.stat-box{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}
.stat-box .n{font-size:28px;font-weight:800;color:#1e3a5f}
.stat-box .l{font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;margin-top:4px}
.stat-box .s{font-size:12px;color:#028090;margin-top:2px}
.narrative{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:18px 22px;margin:18px 0;line-height:1.55}
.narrative p{margin:0 0 10px;font-size:14px}
.narrative p:last-child{margin-bottom:0}
.callout{border-radius:6px;padding:12px 16px;margin:14px 0;font-size:13px;line-height:1.45}
.callout.amber{background:#fef3c7;border-left:4px solid #f59e0b}
.callout.red{background:#fee2e2;border-left:4px solid #dc2626}
.callout.teal{background:#e0f4f7;border-left:4px solid #028090}
.sec-hdr{border-left:5px solid #028090;padding:8px 0 12px 16px;margin:24px 0 16px}
.sec-hdr .num{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:#028090;font-weight:600}
.sec-hdr h2{font-size:18px;color:#1e3a5f;margin:4px 0 6px}
.sec-hdr .lead{color:#64748b;font-size:14px;margin:0}
.data-table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
.data-table th{background:#1e3a5f;color:#fff;padding:6px 10px;text-align:left;font-size:11px}
.data-table td{padding:6px 10px;border-bottom:1px solid #e2e8f0}
.pass{color:#059669;font-weight:700}
.fail{color:#dc2626;font-weight:700}
.warn{color:#d97706;font-weight:700}
.sub-hdr{font-size:13px;font-weight:700;color:#1e3a5f;margin:28px 0 12px;text-transform:uppercase;letter-spacing:.05em}
.example-list{margin:0 0 12px;padding-left:20px;font-size:14px;line-height:1.5}
.example-list li{margin-bottom:8px}
"""


def narrative_css() -> str:
    """Legacy hook — theme CSS is injected via stats export + embed."""
    from evals.reports.utils.theme import REPORT_THEME_CSS

    return REPORT_THEME_CSS


def _fmt_num(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{int(n):,}"


def _extract_bkh_stats(stats: dict) -> dict:
    if "n_total" in stats:
        return stats
    inner = stats.get("stats") or {}
    if inner:
        return next(iter(inner.values()), stats)
    return stats


def bkh_dataset_section(stats: dict) -> str:
    """Section 01 — BKH corpus profile (goes on bkh_stats.html)."""
    from evals.reports.html.stats_profile import dataset_profile_section

    s = _extract_bkh_stats(stats)
    return dataset_profile_section(
        s,
        corpus="bkh",
        section_title="Dataset — What We Have and Its Limitations",
        lead=(
            f"{_fmt_num(s.get('n_total', 0))} turns of real user conversations with BookKeeper Hero. "
            "Understand corpus shape before interpreting any grader or pass rate."
        ),
        links={"suite": "bkh_suite.html", "calibration": "../va/calibration.html"},
    )


def va_staging_dataset_section(stats: dict) -> str:
    """Section 01 — VA staging archive profile (goes on va_stats.html)."""
    from evals.reports.html.stats_profile import dataset_profile_section

    s = _extract_bkh_stats(stats)
    return dataset_profile_section(
        s,
        corpus="va",
        section_title="Dataset — What We Have and Its Limitations",
        lead=(
            f"{_fmt_num(s.get('n_total', 0))} archived production turns from VA staging "
            "(<code>va_staging_responses/</code>). Same profile layout as BKH — detail tables below."
        ),
        links={"suite": "va_suite.html", "calibration": "calibration.html"},
    )


def _cohen_class(d: float) -> str:
    if d > 0.1:
        return "pass"
    if d > 0.05:
        return "warn"
    if d < -0.05:
        return "fail"
    return "warn"


def _use_for_grader(key: str, d: float) -> str:
    if d < -0.05:
        return "Do not use for A/B"
    if key in ("ragas_context_precision", "ragas_faithfulness"):
        return "Best discriminator (needs passage text)"
    if key == "grounding":
        return "Primary VA gate"
    if "escalation" in key:
        return "Reliable, low variance"
    if key in ("completeness", "deepeval_completeness"):
        return "Quality gate (check d on VA staging)"
    if d > 0.1:
        return "Strong on VA staging"
    return "Monitor — weak separation"


def layer1_heuristics_section(*, n_liked: int, n_disliked: int) -> str:
    """Layer 1 — free heuristic gates (eval_methods / calibration.html)."""
    from evals.metrics._constants import TIER_THRESHOLDS

    n = n_liked + n_disliked
    gate_rows = ""
    layer1_gates: list[tuple[str, str, float, str]] = [
        (
            "retrieval_precision",
            "Citation proxy — precision",
            TIER_THRESHOLDS.get("retrieval_precision", 0.75),
            "TP/(TP+FP): cited URL does not imply quality (often ~30% on BKH rated turns)",
        ),
        (
            "proxy_retrieval_recall",
            "Citation proxy — recall",
            TIER_THRESHOLDS.get("proxy_retrieval_recall", 0.65),
            "TP/(TP+FN): high recall — missing citation is a strong bad signal",
        ),
        (
            "known_response_rate",
            "Known response (1 − unknown)",
            TIER_THRESHOLDS.get("known_response_rate", 0.80),
            "response_type ≠ unknown; flags retrieval failures before LLM cost",
        ),
        (
            "weighted_resolution_score",
            "Weighted resolution",
            TIER_THRESHOLDS.get("weighted_resolution_score", 0.50),
            "Conv outcome: resolved=1.0, friction=0.4, unresolved=0.0",
        ),
        (
            "satisfaction_rate",
            "Satisfaction (rated only)",
            TIER_THRESHOLDS.get("satisfaction_rate", 0.70),
            "Aspirational north-star — eval samples skew dislike-heavy (~2.5–3:1)",
        ),
    ]
    for _key, label, thr, note in layer1_gates:
        gate_rows += (
            f"<tr><td>{_html.escape(label)}</td>"
            f"<td class='num'>{thr:.0%}</td>"
            f"<td>{_html.escape(note)}</td></tr>\n"
        )

    return f"""
<div class="sec-hdr">
  <div class="num">Layer 1</div>
  <h2>Heuristic metrics — free, pre-LLM</h2>
  <p class="lead">Run <code>eval_stats</code> first (CI-safe, no API cost). These signals come from
  citations, response_type, conversation metadata, and human like/dislike labels — not from an LLM judge.</p>
</div>
<div class="narrative">
  <p><strong>Why Layer 1 exists:</strong> The pipeline architecture treats heuristics as a cheap filter
  and diagnostic layer. You get corpus shape on <a href="bkh/bkh_stats.html">bkh_stats.html</a> and
  <a href="va/va_stats.html">va_stats.html</a> before spending tokens on Layer 2.</p>
  <p><strong>Ground truth for calibration:</strong> User sentiment (liked / disliked) is the oracle.
  Heuristics are &ldquo;good&rdquo; when they separate liked from disliked on rated turns — same rule as LLM graders
  (see grader methodology doc).</p>
</div>
<div class="callout teal"><strong>Example — citation proxy:</strong> &ldquo;Has sources&rdquo; has high
  <em>recall</em> (disliked turns often lack citations) but low <em>precision</em> (many cited responses are
  still disliked). That is why E_grounding failures exist: URL present, user still unhappy.</div>
<ul class="example-list">
  <li><strong>Has source / source count</strong> — structural retrieval proxy; 0 sources ⇒ short, evasive answers.</li>
  <li><strong>Known response</strong> — <code>response_type=unknown</code> means the KB returned nothing; distinct from escalation.</li>
  <li><strong>BKH↔VA overlap (v2)</strong> — expanded URL map (<code>human_validated_map.csv</code> + <code>kb_url_map_full.json</code>) adjusts ŷ before pass overrides.</li>
  <li><strong>Resolution score</strong> — conversation-level outcome from rated turns only; complements turn-level failure_type labels.</li>
</ul>
<div class="card" style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0">
  <h3 style="font-size:12px;color:#1e3a5f;text-transform:uppercase;margin:0 0 8px">Production pass gates (Layer 1)</h3>
  <p class="chart-note" style="margin:0 0 10px">From <code>evals/metrics/_constants.py</code> · dashed lines on KDE charts below</p>
  <table class="data-table">
    <thead><tr><th>Metric</th><th>Gate</th><th>Notes</th></tr></thead>
    <tbody>{gate_rows}</tbody>
  </table>
</div>
<p class="chart-note">Rated VA staging archive: {n_liked} liked / {n_disliked} disliked (n={n}) — KDE panels compare score distributions at these gates.</p>
"""


def _production_threshold(key: str) -> float:
    from evals.metrics._constants import get_threshold

    return get_threshold(key)


def _report_panel_threshold(key: str) -> float:
    from evals.metrics._constants import get_threshold
    from evals.reports.utils.figures import _CALIBRATION_THRESHOLD_OVERRIDES

    return _CALIBRATION_THRESHOLD_OVERRIDES.get(key, get_threshold(key))


def threshold_pass_comparison_table(rows: list[dict]) -> str:
    """Layer 2 — production vs report-panel threshold and pass-rate separation."""
    if not rows:
        return ""

    body = ""
    for r in sorted(rows, key=lambda x: x["d"], reverse=True):
        key = r["key"]
        prod = _production_threshold(key)
        panel = _report_panel_threshold(key)
        prod_cell = f"{prod:.0%}"
        panel_cell = f"{panel:.0%}"
        if abs(prod - panel) > 0.001:
            panel_cell = f"{panel:.0%} <span class='warn'>(viz)</span>"
        gap = r["liked_pass_pct"] - r["disliked_pass_pct"]
        gap_cls = "pass" if gap > 5 else "warn" if gap > 0 else "fail"
        d = r["d"]
        d_cls = _cohen_class(d)
        sign = "+" if d > 0 else ""
        body += (
            f"<tr><td>{_html.escape(r['label'])}</td>"
            f"<td class='num'>{prod_cell}</td>"
            f"<td class='num'>{panel_cell}</td>"
            f"<td class='num'>{r['liked_pass_pct']}%</td>"
            f"<td class='num'>{r['disliked_pass_pct']}%</td>"
            f"<td class='{gap_cls} num'>Δ{gap:+d}%</td>"
            f"<td class='{d_cls} num'>{sign}{d:.3f}</td></tr>\n"
        )

    return f"""
<div class="card" style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0">
  <h3 style="font-size:12px;color:#1e3a5f;text-transform:uppercase;margin:0 0 8px">Pass threshold selection (Layer 2)</h3>
  <p class="chart-note" style="margin:0 0 10px">
    <b>Production gate</b> = suite regression (<code>TIER_THRESHOLDS</code>).
    <b>Report panel</b> = threshold drawn on KDE/box charts when we zoom a noisy left tail (e.g. RAGAS faith 60% vs 50%).
    Pick gates where <b>Δ pass</b> (liked − disliked pass %) and <b>Cohen&rsquo;s d</b> agree — if d &lt; 0.05, the threshold is doing the work, not the score (prompt iteration needed).
  </p>
  <table class="data-table">
    <thead><tr>
      <th>Grader</th><th>Production</th><th>Report panel</th>
      <th>Liked pass %</th><th>Disliked pass %</th><th>Δ pass</th><th>Cohen&rsquo;s d</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table>
</div>
"""


def layer2_llm_judges_section(
    rows: list[dict],
    *,
    n_liked: int,
    n_disliked: int,
    vtag: str,
) -> str:
    """Layer 2 — LLM-as-judge intro, threshold table, leaderboard."""
    n = n_liked + n_disliked
    sorted_rows = sorted(rows, key=lambda r: r["d"], reverse=True)
    anti = [r for r in sorted_rows if r["d"] < -0.05]

    leaderboard = ""
    for r in sorted_rows:
        d = r["d"]
        cls = _cohen_class(d)
        sign = "+" if d > 0 else ""
        star = " ★" if r.get("is_default") else ""
        leaderboard += (
            f"<tr><td>{_html.escape(r['label'])}{star}</td>"
            f"<td class='{cls}'>{sign}{d:.3f}</td>"
            f"<td>{_html.escape(_use_for_grader(r['key'], d))}</td></tr>\n"
        )

    anti_note = ""
    if anti:
        names = ", ".join(_html.escape(r["label"]) for r in anti)
        anti_note = (
            f'<div class="callout red"><strong>Anti-correlated on VA staging:</strong> '
            f"{names} — do not use for SA A/B without re-calibration.</div>"
        )

    return f"""
<div class="sec-hdr">
  <div class="num">Layer 2</div>
  <h2>LLM-as-judge graders — paid quality layer</h2>
  <p class="lead">Custom Gemini judges + DeepEval/RAGAS cross-checks. A grader is useful only if it agrees
  with human sentiment on rated turns — not because the rubric sounds sophisticated.</p>
</div>
<div class="narrative">
  <p><strong>Interface contract:</strong> Each grader returns <code>score</code> (0–1) and
  <code>is_correct = score ≥ threshold</code>. Passage text must be passed to
  <strong>Grounding</strong> — URL-only context returns ~0.5 and invalidates calibration.</p>
  <p><strong>Grader families on VA staging ({vtag}, n={n} rated):</strong></p>
  <ul class="example-list">
    <li><strong>Custom judges</strong> — grounding (claim-level ratio), completeness (sub-questions),
    answer relevancy, escalation — domain-tuned prompts in <code>evals/graders/judges/</code>.</li>
    <li><strong>RAGAS</strong> — context precision &amp; faithfulness; need retrieved passage text; strongest
    discriminators when context is wired.</li>
    <li><strong>DeepEval</strong> — independent cross-check; compare Cohen&rsquo;s d before substituting for custom judges.</li>
    <li><strong>Heuristic Layer 2</strong> — <code>source_match</code> (URL overlap, no LLM) appears in quality JSON for overlap calibration.</li>
  </ul>
  <p><strong>Gate vs trace creation:</strong> Pipeline regression wants <em>recall</em> (catch bad responses).
  Golden trace labelling wants <em>precision</em>. Use the threshold table below to see pass-rate separation;
  use Cohen&rsquo;s d for raw score discrimination.</p>
</div>
{anti_note}
{threshold_pass_comparison_table(rows)}
<div class="card" style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0">
  <h3 style="font-size:12px;color:#1e3a5f;text-transform:uppercase;margin:0 0 8px">Discrimination ranking (Cohen&rsquo;s d) · ★ = default VA gate</h3>
  <table class="data-table">
    <thead><tr><th>Grader</th><th>Cohen&rsquo;s d</th><th>Use for</th></tr></thead>
    <tbody>{leaderboard or '<tr><td colspan="3">No rated data</td></tr>'}</tbody>
  </table>
</div>
"""


def eval_methods_report_body(
    rows: list[dict],
    *,
    n_liked: int,
    n_disliked: int,
    vtag: str,
) -> str:
    """Full eval methods narrative: Layer 1 + Layer 2."""
    return layer1_heuristics_section(
        n_liked=n_liked, n_disliked=n_disliked
    ) + layer2_llm_judges_section(
        rows,
        n_liked=n_liked,
        n_disliked=n_disliked,
        vtag=vtag,
    )


def calibration_methods_section(
    rows: list[dict],
    *,
    n_liked: int,
    n_disliked: int,
    vtag: str,
) -> str:
    """Deprecated alias — use :func:`eval_methods_report_body`."""
    return eval_methods_report_body(
        rows,
        n_liked=n_liked,
        n_disliked=n_disliked,
        vtag=vtag,
    )


def sample_sizes_table(*, staging: Any | None = None) -> str:
    """Section 03 explainer — what n=50, n≈597, n=500 each mean."""
    bkh_cal = dataset_root("bkh") / "quality_results/calibration_quality_v3.json"
    bkh_n = 50
    if bkh_cal.exists():
        bkh_n = json.loads(bkh_cal.read_text()).get("n_queries", 50)

    from evals.reports.paths import va_staging_all_responses_stats_path

    va_stats = va_staging_all_responses_stats_path()
    va_n = 597
    va_rated = "—"
    if va_stats.exists():
        s = _extract_bkh_stats(json.loads(va_stats.read_text()))
        va_n = int(s.get("n_total", va_n))
        n_l = int(s.get("n_liked", 0))
        n_d = int(s.get("n_disliked", 0))
        if n_l or n_d:
            va_rated = f"{n_l + n_d} ({n_l} liked / {n_d} disliked)"

    from evals.reports.paths import va_staging_all_quality_v1_path

    qual = va_staging_all_quality_v1_path()
    qual_n = "—"
    if qual.exists():
        d = json.loads(qual.read_text())
        qual_n = str(d.get("n_queries", len(d.get("query_results", []))))

    ov_line = ""
    rc_line = ""
    if staging is not None:
        ov = staging.overlap_summary or {}
        ov_line = (
            f"<tr><td>VA↔BKH overlap (paired)</td><td class='num'>{ov.get('n_paired', '—')}</td>"
            f"<td>Strict slug match + expanded via <code>kb_url_map</code> + human map</td></tr>"
            f"<tr><td>Expanded overlap rate</td><td class='num'>"
            f"{ov.get('expanded_overlap_rate', 0):.1%}</td>"
            f"<td>After alias remap — drives calibrated pass overrides</td></tr>"
        )
        rc = ov.get("reclassification") or {}
        if rc:
            rc_line = (
                f"<p class='chart-note' style='margin-top:8px'><b>Last reclassify:</b> "
                f"promoted_to_regression={rc.get('promoted_to_regression', 0)}, "
                f"capability_test={rc.get('promoted_to_capability_test', 0)}, "
                f"kb_indexing_gap={rc.get('promoted_to_kb_indexing_gap', 0)}. "
                f"Slices written under <code>va_staging_responses/eval_sets/reclassified/</code>.</p>"
            )

    return f"""
<div class="sec-hdr">
  <div class="num">Section 03</div>
  <h2>Results — BKH Baseline vs VA Staging Golden</h2>
  <p class="lead">Directional comparison only. Each row below is a <em>different dataset</em> —
  not a random subsample of the same pool.</p>
</div>
<table class="data-table">
  <thead><tr><th>Artifact</th><th>n</th><th>What it is</th></tr></thead>
  <tbody>
    <tr><td><code>bkh/stats/all_stats.json</code></td><td class="num">69,198</td>
      <td>Full BKH production corpus — Layer-1 heuristics only</td></tr>
    <tr><td><code>calibration_quality_v3.json</code></td><td class="num">{bkh_n}</td>
      <td>BKH human-rated stratified set (25 liked / 25 disliked) — <b>LLM bars labeled BKH n={bkh_n}</b></td></tr>
    <tr><td><code>va_staging_all_responses.jsonl</code></td><td class="num">{va_n}</td>
      <td>VA staging response archive; rated subset: {va_rated}</td></tr>
    <tr><td><code>va_staging_all_quality.json</code></td><td class="num">{qual_n}</td>
      <td>LLM-graded VA staging (v1) — Cohen&rsquo;s d / staging pass charts</td></tr>
    <tr><td><code>retrieval_eval_500.jsonl</code></td><td class="num">500</td>
      <td><b>Ablation only</b> — SA Bedrock agents (hc_adk / hc_lg / hc_rag). <em>Not</em> used in BKH↔VA charts</td></tr>
    {ov_line}
  </tbody>
</table>
<div class="callout teal"><strong>Reclassification (one pass):</strong>
  <code>prepare_va_staging()</code> runs overlap → threshold reclassify → calibration index.
  Called once in <code>make reports-validate</code>; cached for suite, comparison, and calibrated pass rates.
  Reclass does <em>not</em> change task count — it re-labels overlap rows into regression / capability / HITL slices.</div>
{rc_line}
<div class="callout amber"><strong>Ablation rerun (Bedrock):</strong>
  Start agents: <code>make sa-adk-bedrock-up sa-langgraph-bedrock-up</code> (ports 8011–8013).
  Run 500-task gate: <code>make eval-sa-500-top3</code> then <code>make grade-ablation-top</code>.
  Report: <code>make report-sa</code>. Archived §4 used ~44-task smoke — treat rankings as directional until 500-task completes.</div>
<div class="narrative">
  <p><strong>Why BKH LLM is n={bkh_n} but VA LLM is n≈{qual_n}:</strong> BKH suite grades the intentional calibration sample.
  VA suite grades the full staging archive. Side-by-side bars compare compatible metrics but different populations —
  read the table above before interpreting a gap.</p>
</div>
"""
