"""Generate an HTML comparison report across support agents and ablation configs.

Loads all ablation JSON files + va-staging smoke + BKH quality results and
produces a single-page HTML report similar to evals/reports/bkh/eval_suite/.

Usage:
    uv run python -m evals.reports compare
    uv run python -m evals.reports compare --output evals/reports/sa/comparison.html
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ABLATION_DIR = Path("data/datasets/support-agents/ablation")
QUALITY_DIR = Path("data/datasets/support-agents/quality")
VA_STAGING_SMOKE = Path("data/datasets/support-agents/va_staging/evals/smoke_20260509_212709.json")
BKH_QUALITY = Path("data/datasets/bkh/quality_results/graded_train.json")
# va_staging quality: latest graded run from va-agents (single production service)
VA_STAGING_QUALITY = Path("data/datasets/va_staging/runs/20260508_170125/responses_graded.json")
CORPUS_PATH = Path("data/articles/corpus_articles.jsonl")
DEFAULT_OUT = Path("evals/reports/sa/comparison.html")

# KB articles exist on two domains: the live help-center domain (cited by the
# agent) and the legacy KB domain (used by golden expected_urls).
LIVE_KB_DOMAIN = "help.kb-live.example"
LEGACY_KB_URL_RE = r"kb-legacy\.example/support/article/([^/\s\"'<>?#]+)"

# Display labels and ordering for each config file
_CONFIG_LABELS: dict[str, tuple[str, str]] = {
    "va_staging": ("va_staging", "gemini-3-flash-preview + ThinkingLevel.LOW + Bedrock HYBRID"),
    "adk_flash_thinking1024": (
        "hc_adk flash+thinking",
        "gemini-3-flash-preview, thinking=1024 (≈va_staging config)",
    ),
    "adk_flash": ("hc_adk flash", "gemini-3-flash-preview, thinking=0, Bedrock HYBRID"),
    "lg_flash": ("hc_lg flash", "gemini-3-flash-preview, CRAG=true, Bedrock HYBRID"),
    "lg_multi_query": ("hc_lg multi-query", "gemini-2.5-flash, MULTI_QUERY=true, CRAG=true"),
    "adk_rag_backend": ("hc_adk + hc_rag", "gemini-2.5-flash, hc_rag local retrieval backend"),
    "lg_rag_backend": (
        "hc_lg + hc_rag",
        "gemini-2.5-flash, CRAG=true, hc_rag local retrieval backend",
    ),
    "adk_thinking1024": ("hc_adk + thinking", "gemini-2.5-flash, thinking=1024, Bedrock HYBRID"),
    "lg_crag": ("hc_lg + CRAG", "gemini-2.5-flash, CRAG=true, Bedrock HYBRID"),
    "lg_no_crag": ("hc_lg no CRAG", "gemini-2.5-flash, CRAG=false, Bedrock HYBRID"),
    "lg_crag_thinking1024": ("hc_lg + CRAG + think", "gemini-2.5-flash, CRAG=true, thinking=1024"),
    "lg_llm_planner": ("hc_lg + LLM planner", "gemini-2.5-flash, CRAG=true, LLM_PLANNER=true"),
    "adk_baseline": ("hc_adk baseline", "gemini-2.5-flash, thinking=0, Bedrock HYBRID"),
    # ── YAML-config ablation runs (Sprint 3+) ────────────────────────────────
    "hc_adk_flash": ("hc_adk flash (v2)", "gemini-3-flash-preview, thinking=0, Bedrock HYBRID"),
    "hc_lg_crag_only": (
        "hc_lg CRAG baseline (v2)",
        "gemini-2.5-flash, CRAG=true, all other flags off",
    ),
    "hc_lg_multiquery": ("hc_lg multi-query (v2)", "gemini-2.5-flash, CRAG=true, MULTI_QUERY=true"),
    "hc_lg_hyde": (
        "hc_lg HyDE (v2)",
        "gemini-2.5-flash, CRAG=true, HyDE pre-retrieval (+200-400ms)",
    ),
    "hc_lg_hitl": (
        "hc_lg HITL gate (v2)",
        "gemini-2.5-flash, CRAG=true, confidence escalation gate threshold=0.3",
    ),
    "hc_lg_post_eval": (
        "hc_lg post-answer (v2)",
        "gemini-2.5-flash, CRAG=true, post-answer LLM judge (refine loop)",
    ),
    "hc_rag_full": (
        "hc_rag full pipeline",
        "planner→retriever→qa_policy→HITL×2→answer→post_answer_eval→summarizer",
    ),
}

# Estimated cost per 1K queries (USD). Based on Gemini public pricing:
#   flash-preview input $0.10/M, output $0.40/M, thinking $3.50/M thinking-tokens
#   2.5-flash     input $0.15/M, output $0.60/M, thinking $3.50/M thinking-tokens
# Assumptions per query: ~2 500 input tokens, ~600 output tokens, ~800 thinking tokens (if enabled).
# CRAG/multi-query adds a second grader call (~1 500 in, ~200 out).
# Bedrock costs ($0.003/query KB lookup) are excluded — same across all configs.
_CONFIG_COST: dict[str, tuple[str, str]] = {
    # (cost_per_1k_queries_usd_string, tooltip)
    "va_staging": ("~$0.45", "flash-preview + ThinkingLevel.LOW (~300 thinking tok/q)"),
    "adk_flash_thinking1024": ("~$0.45", "flash-preview + 1024 thinking tokens/query"),
    "adk_flash": ("~$0.12", "flash-preview, no thinking, 1 LLM call/query"),
    "lg_flash": ("~$0.20", "flash-preview + CRAG grader call (~+40%)"),
    "lg_multi_query": ("~$0.28", "2.5-flash + CRAG + multi-query expansion"),
    "adk_rag_backend": ("~$0.18", "2.5-flash, 1 LLM call/query, local RAG"),
    "lg_rag_backend": ("~$0.30", "2.5-flash + CRAG grader call, local RAG"),
    "adk_thinking1024": ("~$0.62", "2.5-flash + 1024 thinking tokens/query"),
    "lg_crag": ("~$0.30", "2.5-flash + CRAG grader call"),
    "lg_no_crag": ("~$0.18", "2.5-flash, 1 LLM call/query, no CRAG"),
    "lg_crag_thinking1024": ("~$0.92", "2.5-flash + CRAG + 1024 thinking tokens"),
    "lg_llm_planner": ("~$0.35", "2.5-flash + CRAG + LLM planner call"),
    "adk_baseline": ("~$0.18", "2.5-flash, 1 LLM call/query, no thinking"),
    # ── YAML-config ablation runs (Sprint 3+) ────────────────────────────────
    "hc_adk_flash": ("~$0.12", "flash-preview, no thinking, 1 LLM call/query"),
    "hc_lg_crag_only": ("~$0.30", "2.5-flash + CRAG grader call"),
    "hc_lg_multiquery": ("~$0.28", "2.5-flash + CRAG + multi-query expansion"),
    "hc_lg_hyde": ("~$0.32", "2.5-flash + CRAG + HyDE extra Gemini pre-call"),
    "hc_lg_hitl": ("~$0.30", "2.5-flash + CRAG + confidence escalation gate"),
    "hc_lg_post_eval": ("~$0.42", "2.5-flash + CRAG + post-answer LLM judge"),
    "hc_rag_full": ("~$0.55", "full pipeline: HITL, post-answer eval, summarizer"),
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_task_id(tid: str) -> str:
    """8a6d6d8c_3fa6_48bd_9e4b_44f17b447b3a_t1 → 8a6d6d8c-3fa6-48bd-9e4b-44f17b447b3a_turn_1"""
    m = re.match(r"^(.+)_t(\d+)$", tid)
    if m:
        return m.group(1).replace("_", "-") + "_turn_" + m.group(2)
    return tid.replace("_", "-")


def load_ablation_files() -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for key in _CONFIG_LABELS:
        p = ABLATION_DIR / f"{key}.json"
        if p.exists():
            configs[key] = json.loads(p.read_text())
    return configs


def load_va_staging_smoke() -> dict | None:
    if VA_STAGING_SMOKE.exists():
        return json.loads(VA_STAGING_SMOKE.read_text())
    return None


def load_bkh_quality() -> dict:
    if BKH_QUALITY.exists():
        return json.loads(BKH_QUALITY.read_text())
    return {}


def load_va_staging_quality() -> dict | None:
    if VA_STAGING_QUALITY.exists():
        return json.loads(VA_STAGING_QUALITY.read_text())
    return None


def _build_slug_map() -> dict[str, str]:
    if not CORPUS_PATH.exists():
        return {}
    slug_map: dict[str, str] = {}
    for line in CORPUS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        a = json.loads(line)
        url = a.get("url", "")
        if LIVE_KB_DOMAIN in url:
            m = re.search(r"/articles/\d+-(.+)$", url)
            if m:
                slug_map[m.group(1).rstrip("/")] = url
    return slug_map


def _resolve_legacy_url(legacy_url_str: str, slug_map: dict[str, str]) -> str | None:
    m = re.search(LEGACY_KB_URL_RE, legacy_url_str)
    if not m:
        return None
    slug = m.group(1).rstrip("/").split("?")[0]
    if slug in slug_map:
        return slug_map[slug]
    prefix = slug[:15]
    candidates = [u for s, u in slug_map.items() if s.startswith(prefix)]
    return candidates[0] if candidates else None


def compute_bkh_retrieval_stats(bkh: dict, configs: dict) -> dict:
    """Compute BKH retrieval MRR over the FULL annotated eval set.

    BKH only overlaps with a subset of our 44-task eval. Tasks where BKH has no
    response score 0 — so the denominator is the total number of annotated eval
    tasks (32), not the overlap count (7). This makes BKH MRR directly comparable
    to local agent MRR which is also computed over all 32 annotated tasks.
    """
    slug_map = _build_slug_map()

    # Build golden_url index from the largest ablation file: ablation_task_id → golden_urls
    ablation_golden: dict[str, list[str]] = {}
    for key in _CONFIG_LABELS:
        p = ABLATION_DIR / f"{key}.json"
        if p.exists():
            data = json.loads(p.read_text())
            for t in data.get("task_results", []):
                ablation_golden[t["task_id"]] = t.get("golden_urls", [])
            break

    # BKH task IDs use the normalized form; ablation IDs use underscore form
    def bkh_to_ablation_id(bkh_id: str) -> str:
        # '8a6d6d8c-3fa6-48bd-9e4b-44f17b447b3a_turn_1' → '8a6d6d8c_3fa6_48bd_9e4b_44f17b447b3a_t1'
        m = re.match(r"^(.+)_turn_(\d+)$", bkh_id)
        if m:
            return m.group(1).replace("-", "_") + "_t" + m.group(2)
        return bkh_id.replace("-", "_")

    # Score BKH on the tasks where it overlaps with annotated ablation tasks
    overlap_mrr: dict[str, float] = {}
    overlap_p1: dict[str, float] = {}
    overlap_p3: dict[str, float] = {}
    overlap_r3: dict[str, float] = {}
    n_overlap = 0

    for bkh_id, entry in bkh.items():
        ablation_id = bkh_to_ablation_id(bkh_id)
        golden_urls = ablation_golden.get(ablation_id, [])
        if not golden_urls:
            continue  # unannotated in our eval
        n_overlap += 1

        expected_urls = entry.get("expected_urls", [])
        if not expected_urls:
            overlap_mrr[ablation_id] = 0.0
            overlap_p1[ablation_id] = 0.0
            overlap_p3[ablation_id] = 0.0
            overlap_r3[ablation_id] = 0.0
            continue

        resolved_urls: list[str] = []
        for eu in expected_urls:
            r = _resolve_legacy_url(eu, slug_map)
            if r and r not in resolved_urls:
                resolved_urls.append(r)

        golden_set = set(golden_urls)
        mrr = 0.0
        for rank, url in enumerate(resolved_urls, 1):
            if url in golden_set:
                mrr = 1.0 / rank
                break
        top3 = resolved_urls[:3]
        top3_hits = sum(1 for u in top3 if u in golden_set)

        overlap_mrr[ablation_id] = mrr
        overlap_p1[ablation_id] = 1.0 if resolved_urls and resolved_urls[0] in golden_set else 0.0
        overlap_p3[ablation_id] = top3_hits / 3 if top3 else 0.0
        overlap_r3[ablation_id] = top3_hits / len(golden_set) if golden_set else 0.0

    # Denominator = all annotated eval tasks (BKH scores 0 on non-overlapping tasks)
    n_annotated_eval = sum(1 for urls in ablation_golden.values() if urls)

    p3_avg = sum(overlap_p3.values()) / n_annotated_eval if n_annotated_eval else 0.0
    r3_avg = sum(overlap_r3.values()) / n_annotated_eval if n_annotated_eval else 0.0

    return {
        "mrr": sum(overlap_mrr.values()) / n_annotated_eval if n_annotated_eval else 0.0,
        "mrr_overlap_only": sum(overlap_mrr.values()) / n_overlap if n_overlap else 0.0,
        "p@1": sum(overlap_p1.values()) / n_annotated_eval if n_annotated_eval else 0.0,
        "p@3": p3_avg,
        "r@3": r3_avg,
        "f1@3": 2 * p3_avg * r3_avg / (p3_avg + r3_avg) if (p3_avg + r3_avg) > 0 else 0.0,
        "n_annotated": n_overlap,
        "n_annotated_eval": n_annotated_eval,
        "n_tasks": len(bkh),
    }


def load_quality_grades() -> dict[str, dict]:
    """Load quality grader results for local agents from QUALITY_DIR/{config}.json.

    Returns config_key → {grader_summary, task_results} for all available files.
    """
    grades: dict[str, dict] = {}
    if not QUALITY_DIR.exists():
        return grades
    for p in QUALITY_DIR.glob("*.json"):
        with contextlib.suppress(Exception):
            grades[p.stem] = json.loads(p.read_text())
    return grades


def _aggregate_va_staging_quality(va_quality: dict | None) -> dict:
    """Extract grader_summary from va_staging quality file (488-task run)."""
    if not va_quality:
        return {}
    return va_quality.get("grader_summary", {})


def _aggregate_bkh_quality(bkh: dict, filter_to_ablation_ids: set[str] | None = None) -> dict:
    """Aggregate BKH grader results, optionally filtered to overlapping ablation task IDs."""
    agg: dict[str, dict] = {}
    for bkh_id, entry in bkh.items():
        if filter_to_ablation_ids is not None and bkh_id not in filter_to_ablation_ids:
            continue
        for metric, result in (entry.get("grader_results") or {}).items():
            if metric not in agg:
                agg[metric] = {"score_sum": 0.0, "n": 0, "n_correct": 0}
            agg[metric]["score_sum"] += result.get("score", 0)
            agg[metric]["n"] += 1
            agg[metric]["n_correct"] += int(result.get("is_correct", False))
    return {
        m: {
            "avg_score": v["score_sum"] / v["n"] if v["n"] else 0,
            "n_graded": v["n"],
            "pass_rate": v["n_correct"] / v["n"] if v["n"] else 0,
        }
        for m, v in agg.items()
    }


def _bkh_ablation_overlap_ids(bkh: dict, configs: dict) -> set[str]:
    """Return BKH task IDs (normalized form) that appear in any ablation file."""
    ablation_bkh_ids: set[str] = set()
    for data in configs.values():
        for t in data.get("task_results", []):
            norm = _normalize_task_id(t["task_id"])
            ablation_bkh_ids.add(norm)
    return {bid for bid in bkh if bid in ablation_bkh_ids}


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------


def compute_significance(configs: dict, top_n: int = 5) -> list[dict]:
    """Bootstrap 95% CI for MRR delta between all pairs of top-N configs.

    Returns list of {a, b, delta, ci_lo, ci_hi, n, significant} dicts sorted by |delta|.
    """
    if not _HAS_NUMPY:
        return []

    # Collect per-task MRR for each config (annotated tasks only)
    per_task: dict[str, dict[str, float]] = {}
    for key, data in configs.items():
        for t in data.get("task_results", []):
            if not t.get("golden_urls"):
                continue
            tid = t["task_id"]
            per_task.setdefault(tid, {})[key] = t.get("scores", {}).get("mrr", 0.0)

    # Rank configs by MRR — take top_n + va_staging always
    ranked = sorted(
        [k for k in configs if k != "va_staging"],
        key=lambda k: configs[k].get("aggregate", {}).get("mrr", 0),
        reverse=True,
    )
    keys = (["va_staging"] if "va_staging" in configs else []) + ranked[:top_n]

    # Build per-task arrays aligned on shared annotated tasks
    shared_tids = [tid for tid in per_task if all(k in per_task[tid] for k in keys)]

    rng = np.random.default_rng(42)
    results = []
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            a_mrr = np.array([per_task[tid][a] for tid in shared_tids])
            b_mrr = np.array([per_task[tid][b] for tid in shared_tids])
            n = len(shared_tids)
            obs_delta = float(a_mrr.mean() - b_mrr.mean())
            # Bootstrap
            boot_deltas = []
            for _ in range(5000):
                idx = rng.integers(0, n, n)
                boot_deltas.append(float(a_mrr[idx].mean() - b_mrr[idx].mean()))
            ci_lo, ci_hi = (
                float(np.percentile(boot_deltas, 2.5)),
                float(np.percentile(boot_deltas, 97.5)),
            )
            results.append(
                {
                    "a": a,
                    "b": b,
                    "delta": obs_delta,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n": n,
                    "significant": ci_lo > 0 or ci_hi < 0,
                }
            )

    return sorted(results, key=lambda x: abs(x["delta"]), reverse=True)


def compute_power_requirements() -> dict:
    """Return minimum annotated tasks needed for common MRR effect sizes at 80% power."""
    if not _HAS_NUMPY:
        return {}
    from scipy.stats import norm  # type: ignore[import]

    z_a, z_b = norm.ppf(0.975), norm.ppf(0.8)
    sd = 0.44  # typical SD of per-task reciprocal ranks (binary-ish)
    return {
        delta: int(((z_a + z_b) * sd / delta) ** 2 * 2) for delta in [0.05, 0.07, 0.10, 0.15, 0.20]
    }


# ---------------------------------------------------------------------------
# Discriminative tasks
# ---------------------------------------------------------------------------


def find_discriminative_tasks(
    config_keys: list[str], tasks: list[dict], top_keys: list[str] | None = None
) -> list[dict]:
    """Return annotated tasks ranked by how much configs disagree.

    Uses variance in per-task MRR across configs: tasks where half hit and half miss
    are most informative for distinguishing configs.
    """
    keys = top_keys or config_keys
    disc = []
    for row in tasks:
        if not row.get("golden_urls"):
            continue
        scores = []
        for k in keys:
            t = row["configs"].get(k)
            if t is not None:
                scores.append(t.get("scores", {}).get("mrr", 0.0))
        if len(scores) < 2:
            continue
        hit_rate = sum(1 for s in scores if s >= 1.0) / len(scores)
        # Disagreement: max at 0.5 hit_rate (half hit, half miss)
        disagreement = 1.0 - abs(2 * hit_rate - 1.0)
        disc.append({**row, "_disagreement": disagreement, "_hit_rate": hit_rate})
    return sorted(disc, key=lambda x: x["_disagreement"], reverse=True)


# ---------------------------------------------------------------------------
# Build unified task view
# ---------------------------------------------------------------------------


def build_task_matrix(
    configs: dict[str, dict], va_smoke: dict | None, bkh: dict
) -> tuple[list[str], list[dict]]:
    """Returns (ordered_config_keys, tasks_with_per_config_results)."""
    config_keys = [k for k in _CONFIG_LABELS if k in configs]

    # Build task_id → task dict from the largest ablation file
    all_task_ids: list[str] = []
    task_meta: dict[str, dict] = {}
    primary_key = config_keys[0] if config_keys else None
    if primary_key:
        for t in configs[primary_key].get("task_results", []):
            tid = t["task_id"]
            if tid not in task_meta:
                all_task_ids.append(tid)
                task_meta[tid] = {
                    "task_id": tid,
                    "query": t["query"],
                    "category": t.get("category", ""),
                    "golden_urls": t.get("golden_urls", []),
                }

    # Index each config's results by task_id
    config_results: dict[str, dict[str, dict]] = {}
    for key, data in configs.items():
        config_results[key] = {t["task_id"]: t for t in data.get("task_results", [])}

    # Index va-staging by normalized task_id → ablation-style task_id
    va_by_tid: dict[str, dict] = {}
    if va_smoke:
        for t in va_smoke.get("task_results", []):
            va_by_tid[t["task_id"]] = t

    # BKH lookup — normalize key in both directions
    bkh_by_tid: dict[str, dict] = {}
    for bkh_key, entry in bkh.items():
        bkh_by_tid[bkh_key] = entry

    rows = []
    for tid in all_task_ids:
        meta = task_meta[tid]
        row = {**meta, "configs": {}, "va_staging": None, "bkh": None}

        for key in config_keys:
            row["configs"][key] = config_results.get(key, {}).get(tid)

        if tid in va_by_tid:
            row["va_staging"] = va_by_tid[tid]

        norm = _normalize_task_id(tid)
        if norm in bkh_by_tid:
            row["bkh"] = bkh_by_tid[norm]

        rows.append(row)

    return config_keys, rows


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _mrr_class(v: float) -> str:
    if v >= 0.7:
        return "cell-great"
    if v >= 0.5:
        return "cell-good"
    if v >= 0.3:
        return "cell-ok"
    return "cell-bad"


def _lat_class(ms: float) -> str:
    if ms < 3000:
        return "cell-great"
    if ms < 6000:
        return "cell-good"
    if ms < 10000:
        return "cell-ok"
    return "cell-bad"


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _ms(v: float) -> str:
    return f"{v / 1000:.1f}s"


def _cost_cell(config_key: str) -> str:
    est, tip = _CONFIG_COST.get(config_key, ("—", "no estimate"))
    return (
        f'<td style="text-align:center;font-size:.82em" title="{tip} · per 1K queries">'
        f"<b>{est}</b></td>"
    )


def _task_result(v: float | None) -> str:
    if v is None:
        return '<span class="badge b-na">—</span>'
    if v >= 1.0:
        return '<span class="badge b-hit">HIT</span>'
    if v > 0:
        return '<span class="badge b-par">PAR</span>'
    return '<span class="badge b-miss">MISS</span>'


def _category_badge(cat: str) -> str:
    mapping = {
        "danish_support": ("cat-da", "🇩🇰 DA"),
        "english_support": ("cat-en", "🇬🇧 EN"),
        "escalation": ("cat-esc", "🔺 ESC"),
        "out_of_scope": ("cat-oos", "⊘ OOS"),
        "vague": ("cat-vague", "❓ VAGUE"),
    }
    cls, label = mapping.get(cat, ("cat-other", cat[:6]))
    return f'<span class="cat-badge {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------


def build_category_breakdown(
    config_keys: list[str], tasks: list[dict]
) -> dict[str, dict[str, dict]]:
    """category → config_key → {mrr_sum, n, n_annotated}"""
    cats: dict[str, dict[str, dict]] = {}
    for row in tasks:
        cat = row.get("category", "unknown")
        if cat not in cats:
            cats[cat] = {}
        for key in config_keys:
            if key not in cats[cat]:
                cats[cat][key] = {"mrr_sum": 0.0, "n": 0, "n_annotated": 0}
            t = row["configs"].get(key)
            if t is not None:
                cats[cat][key]["n"] += 1
                if row.get("golden_urls"):
                    cats[cat][key]["n_annotated"] += 1
                    cats[cat][key]["mrr_sum"] += t.get("scores", {}).get("mrr", 0.0)
    return cats


# ---------------------------------------------------------------------------
# CSS + HTML
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 0; background: #f5f6f8; color: #222; }
.page { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
h1 { font-size: 1.4em; margin: 0 0 4px; }
h2 { font-size: 1.05em; margin: 28px 0 8px; border-bottom: 1px solid #ddd;
     padding-bottom: 4px; color: #444; }
h3 { font-size: .95em; margin: 20px 0 6px; color: #555; }
.meta { color: #666; font-size: .85em; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 6px; overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 20px; }
th { background: #f0f2f5; text-align: left; padding: 9px 12px;
     font-size: .78em; color: #555; text-transform: uppercase; letter-spacing: .04em; }
td { padding: 8px 12px; font-size: .85em; border-top: 1px solid #eee; vertical-align: top; }
tr:hover td { background: #fafbfc; }
.badge { display:inline-block; padding:2px 7px; border-radius:3px; font-size:.75em; font-weight:600; }
.b-hit  { background:#d4edda; color:#155724; }
.b-par  { background:#fff3cd; color:#856404; }
.b-miss { background:#f8d7da; color:#721c24; }
.b-na   { background:#e9ecef; color:#6c757d; }
.b-info { background:#d1ecf1; color:#0c5460; }
.cat-badge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:.72em; font-weight:600; }
.cat-da    { background:#cfe2ff; color:#084298; }
.cat-en    { background:#d1ecf1; color:#0c5460; }
.cat-esc   { background:#f8d7da; color:#721c24; }
.cat-oos   { background:#e2d9f3; color:#4a235a; }
.cat-vague { background:#fff3cd; color:#856404; }
.cat-other { background:#e9ecef; color:#495057; }
.cell-great { background:#d4edda; }
.cell-good  { background:#d4f1d4; }
.cell-ok    { background:#fff3cd; }
.cell-bad   { background:#f8d7da; }
.stat-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
             gap:10px; margin-bottom:20px; }
.stat-card { background:#fff; border-radius:6px; padding:14px 16px;
             box-shadow:0 1px 3px rgba(0,0,0,.08); }
.stat-label { font-size:.72em; color:#666; text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
.stat-value { font-size:1.4em; font-weight:600; color:#222; }
.stat-sub   { font-size:.75em; color:#888; margin-top:2px; }
.highlight-row { background:#fffbeb !important; }
.query-text { font-size:.88em; color:#333; max-width:300px; }
.resp-excerpt { font-size:.78em; color:#555; max-width:300px; line-height:1.4; }
.note-box { background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
            padding:12px 16px; color:#856404; font-size:.85em; margin-bottom:16px; }
.legend { display:flex; gap:16px; margin-bottom:12px; flex-wrap:wrap; }
.legend-item { display:flex; align-items:center; gap:5px; font-size:.8em; }
details summary { cursor:pointer; font-weight:600; font-size:.85em; padding:6px 0; color:#555; }
details[open] summary { color:#222; }
"""


def _summary_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="stat-card">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f"{'<div class=stat-sub>' + sub + '</div>' if sub else ''}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _qual_cell(summary: dict, metric: str) -> str:
    """Render a quality metric cell.

    Primary (big):   standard LLM grader pass% — the agent's score
    Subscript (small): DeepEval calibrated pass% if available (run --calibrate),
                       otherwise ⚠️ uncal. to signal the score is unvalidated.
    """
    if metric not in summary:
        return '<td style="text-align:center;color:#aaa;font-size:.82em">—</td>'
    s = summary[metric]
    pr = s.get("pass_rate", 0)
    cls = "cell-great" if pr >= 0.75 else ("cell-good" if pr >= 0.55 else "cell-ok")

    # DeepEval calibration check — present if --calibrate was used
    deepeval_key = f"deepeval_{metric}"
    de = summary.get(deepeval_key)
    if de is not None:
        de_pr = de.get("pass_rate", 0)
        # Warn if DeepEval is materially lower (grader is lenient)
        gap = pr - de_pr
        de_color = "#dc2626" if gap > 0.20 else ("#92400e" if gap > 0.08 else "#16a34a")
        cal_html = (
            f'<span style="font-size:.70em;color:{de_color}" title="DeepEval calibrated pass%">'
            f"{de_pr * 100:.0f}% cal.</span>"
        )
    else:
        cal_html = '<span style="font-size:.68em;color:#94a3b8" title="Run --calibrate to validate">⚠️ uncal.</span>'

    return f'<td class="{cls}" style="text-align:center"><b>{pr * 100:.0f}%</b><br>{cal_html}</td>'


def _resolution_rate(data: dict) -> float | None:
    """Compute resolution rate (1 - escalation rate) from contact_support field."""
    tasks = data.get("task_results", [])
    if not tasks:
        return None
    n_contact = sum(1 for t in tasks if t.get("contact_support"))
    return 1.0 - n_contact / len(tasks)


def _res_cell(rate: float | None, na_colspan: int = 1) -> str:
    if rate is None:
        return f'<td style="text-align:center;color:#aaa;font-size:.82em" colspan="{na_colspan}">N/A</td>'
    cls = "cell-great" if rate >= 0.90 else ("cell-good" if rate >= 0.75 else "cell-ok")
    return f'<td class="{cls}" style="text-align:center;font-weight:600">{rate * 100:.0f}%</td>'


def _section_aggregate(
    config_keys: list[str],
    configs: dict,
    va_smoke: dict | None,
    bkh_stats: dict | None = None,
    quality_grades: dict | None = None,
    va_quality_summary: dict | None = None,
    bkh_quality_summary: dict | None = None,
) -> str:
    qual = quality_grades or {}
    quality_metrics = ["answer_relevancy", "completeness", "escalation"]
    # Columns: config + n + resolution + MRR + P@1 + P@3 + R@3 + F1@3 + avg_lat + p50 + cost + 3 quality = 14
    ncols = 14

    def _row(
        name: str,
        spec: str,
        agg: dict,
        n_tasks: int,
        n_ann: int,
        smoke_only: bool = False,
        lat_str: str | None = None,
        qual_summary: dict | None = None,
        resolution: float | None = None,
        config_key: str = "",
    ) -> str:
        mrr = agg.get("mrr", 0)
        p1 = agg.get("p@1", 0)
        p3 = agg.get("p@3", 0)
        r3 = agg.get("r@3", 0)
        f1_3 = agg.get("f1@3") if "f1@3" in agg else _f1(p3, r3)
        lat = agg.get("avg_latency_ms", 0)
        p50 = agg.get("p50_latency_ms", 0)
        smoke_tag = (
            ' <span class="badge b-info" style="font-size:.7em">smoke/10</span>'
            if smoke_only
            else ""
        )
        if lat_str is not None:
            lat_cell = lat_str
            p50_cell = ""
        else:
            lat_cell = f'<td class="{_lat_class(lat)}" style="text-align:center">{_ms(lat)}</td>'
            p50_cell = f'<td style="text-align:center;color:#888;font-size:.82em">{_ms(p50)}</td>'
        qs = qual_summary or {}
        qual_cells = "".join(_qual_cell(qs, m) for m in quality_metrics)
        cost_cell = (
            _cost_cell(config_key)
            if config_key
            else '<td style="text-align:center;color:#aaa">—</td>'
        )
        return (
            f"<tr>"
            f"<td><b>{name}</b>{smoke_tag}<br><span style='color:#888;font-size:.78em'>{spec}</span></td>"
            f"{_res_cell(resolution)}"
            f'<td class="{_mrr_class(mrr)}" style="text-align:center;font-weight:600">{_pct(mrr)}</td>'
            f'<td style="text-align:center">{_pct(p1)}</td>'
            f'<td style="text-align:center">{_pct(p3)}</td>'
            f'<td style="text-align:center">{_pct(r3)}</td>'
            f'<td style="text-align:center;font-weight:600">{_pct(f1_3)}</td>'
            f"<td style='text-align:center;color:#666;font-size:.82em' title='Tasks with golden URLs for retrieval / total tasks in run'>{n_ann}/{n_tasks}</td>"
            f"{qual_cells}"
            f"{lat_cell}{p50_cell}"
            f"{cost_cell}"
            f"</tr>"
        )

    def _section_header(label: str, color: str) -> str:
        return (
            f'<tr><td colspan="{ncols}" style="font-size:.72em;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.05em;color:{color};padding:8px 12px 2px">'
            f"{label}</td></tr>"
        )

    rows_html = ""

    # ── BKH (Bookkeeper Hero) — 3rd-party source baseline ───────────────────
    if bkh_stats and bkh_stats.get("n_annotated", 0) > 0:
        n_ann = bkh_stats["n_annotated"]
        n_ann_eval = bkh_stats.get("n_annotated_eval", 32)
        mrr_ol = bkh_stats.get("mrr_overlap_only", 0)
        rows_html += _section_header(
            "📚 3rd-party source baseline (BKH — Bookkeeper Hero)", "#5a1e6e"
        )
        bkh_agg = {
            "mrr": bkh_stats["mrr"],
            "p@1": bkh_stats["p@1"],
            "p@3": bkh_stats.get("p@3", 0.0),
            "r@3": bkh_stats.get("r@3", 0.0),
            "f1@3": bkh_stats.get("f1@3", 0.0),
        }
        rows_html += _row(
            "BKH (Bookkeeper Hero)",
            (
                f"3rd-party source — MRR over {n_ann_eval} annotated eval tasks "
                f"({n_ann} overlap; {_pct(mrr_ol)} on overlap-only); "
                f"quality on {n_ann} liked responses ⚠️ biased high"
            ),
            bkh_agg,
            44,
            n_ann,
            lat_str='<td style="text-align:center;color:#aaa;font-size:.82em" colspan="2">N/A</td>',
            qual_summary=bkh_quality_summary,
            resolution=None,
        )

    # ── va_staging (current production) — benchmark ──────────────────────────
    va_full = configs.get("va_staging", {})
    va_benchmark = va_full if va_full.get("n_tasks", 0) >= 44 else (va_smoke or {})
    is_smoke_only = va_benchmark is va_smoke and bool(va_benchmark)
    if va_benchmark:
        agg = va_benchmark.get("aggregate", {})
        rows_html += _section_header("⭐ Production benchmark (va_staging)", "#92400e")
        if quality_grades.get("va_staging"):
            va_q_note = " · quality: 44-task set ✓"
        elif va_quality_summary:
            va_q_note = " · quality: 488 production queries (proxy)"
        else:
            va_q_note = " · quality: run make grade-ablation CONFIG=va_staging"
        rows_html += _row(
            _CONFIG_LABELS.get("va_staging", ("va_staging", ""))[0],
            _CONFIG_LABELS.get("va_staging", ("va_staging", ""))[1] + va_q_note,
            agg,
            va_benchmark.get("n_tasks", 0),
            va_benchmark.get("n_annotated", 0),
            smoke_only=is_smoke_only,
            qual_summary=va_quality_summary,
            resolution=_resolution_rate(va_benchmark),
            config_key="va_staging",
        )

    # ── Local agents ──────────────────────────────────────────────────────────
    rows_html += _section_header("● Local agents (44-task eval)", "#1e3a5f")
    for key in config_keys:
        if key == "va_staging":
            continue
        data = configs[key]
        agg = data.get("aggregate", {})
        label, spec = _CONFIG_LABELS.get(key, (key, ""))
        q_summary = qual.get(key, {}).get("grader_summary", {}) if key in qual else None
        q_note = " · quality graded ✓" if q_summary else " · quality: run make grade-ablation-top"
        rows_html += _row(
            label,
            spec + q_note,
            agg,
            data.get("n_tasks", 0),
            data.get("n_annotated", 0),
            qual_summary=q_summary or {},
            resolution=_resolution_rate(data),
            config_key=key,
        )

    bkh_n = bkh_stats.get("n_annotated", 0) if bkh_stats else 0

    qual_header = "".join(
        f'<th style="text-align:center">{m.replace("_", " ").title()}<br>'
        f'<span style="font-weight:400;font-size:.8em">pass% / cal%</span></th>'
        for m in quality_metrics
    )

    return f"""
<h2>Aggregate comparison — retrieval &amp; quality</h2>
<div class="note-box">
  <b>Dataset alignment:</b> All retrieval metrics on same 44-task eval set.
  BKH MRR computed over all {bkh_stats.get("n_annotated_eval", 32)} annotated eval tasks (0 for {bkh_stats.get("n_annotated_eval", 32) - bkh_n} non-overlapping tasks).
  Resolution rate = % tasks where agent did not recommend contacting support (<code>contact_support=False</code>).
  <b>n ann/total</b> = tasks with golden URLs for retrieval / total tasks in run (not a pass count).
  Quality cells: <b>pass%</b> = standard LLM grader pass rate (agent score). Subscript = DeepEval calibrated pass% if <code>--calibrate</code> was run, otherwise <b>⚠️ uncal.</b> Red subscript = DeepEval &gt;20pp lower (standard grader too lenient). Completeness is lenient when passage context is unavailable.
  va_staging quality proxy: 488 production queries; run <code>make grade-ablation CONFIG=va_staging &amp;&amp; make grade-ablation-top</code> for 44-task scores.
</div>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th rowspan="2">Config</th>
      <th rowspan="2" style="text-align:center;background:#fef3c7">Resolution<br>Rate</th>
      <th colspan="5" style="text-align:center;background:#eef2ff">Retrieval (44-task)</th>
      <th rowspan="2" style="text-align:center;color:#888;font-size:.78em" title="Tasks with golden URLs for retrieval / total tasks in run">n ann/<br>total</th>
      <th colspan="3" style="text-align:center;background:#f0fdf4">Quality (uncalibrated)</th>
      <th colspan="2" style="text-align:center;background:#fff3e0">Latency</th>
      <th rowspan="2" style="text-align:center;background:#fef9c3" title="Estimated LLM cost per 1 000 queries (Gemini public pricing, ~2 500 input + 600 output tokens/query). Hover cells for assumptions. Bedrock KB excluded.">Cost<br><span style="font-weight:400;font-size:.78em">/1K q</span></th>
    </tr>
    <tr>
      <th style="text-align:center;background:#eef2ff">MRR</th>
      <th style="text-align:center;background:#eef2ff">P@1</th>
      <th style="text-align:center;background:#eef2ff">P@3</th>
      <th style="text-align:center;background:#eef2ff">R@3</th>
      <th style="text-align:center;background:#eef2ff;font-weight:700">F1@3</th>
      {qual_header}
      <th style="text-align:center;background:#fff3e0">Avg</th>
      <th style="text-align:center;background:#fff3e0">p50</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
"""


def _section_ablation_insights(config_keys: list[str], configs: dict) -> str:
    insights = [
        (
            "THINKING_BUDGET",
            "hc_adk",
            "adk_baseline",
            "adk_thinking1024",
            "Gemini extended thinking (1024 tokens). Gives the model reasoning budget before composing retrieval queries.",
        ),
        (
            "CRAG",
            "hc_lg",
            "lg_no_crag",
            "lg_crag",
            "Corrective RAG: grade passages → rewrite query → re-fetch if quality low. Confidence gate at 0.7 short-circuits when top passage score is high.",
        ),
        (
            "THINKING_BUDGET + CRAG",
            "hc_lg",
            "lg_crag",
            "lg_crag_thinking1024",
            "CRAG with thinking enabled on the answer node. Interaction effect: thinking changed citation behaviour, hurting CRAG grading alignment.",
        ),
        (
            "LLM_PLANNER",
            "hc_lg",
            "lg_crag",
            "lg_llm_planner",
            "LLM intent classifier (gemini-2.5-flash) vs deterministic regex router. Adds ~1s + 1 LLM call per query.",
        ),
    ]

    def _delta(a: float, b: float) -> str:
        d = a - b
        s = f"+{_pct(d)}" if d >= 0 else _pct(d)
        cls = (
            "cell-great"
            if d > 0.05
            else ("cell-good" if d > 0 else ("cell-bad" if d < -0.05 else "cell-ok"))
        )
        return f'<td class="{cls}" style="text-align:center;font-weight:600">{s}</td>'

    rows_html = ""
    for feature, agent, base_key, ablation_key, note in insights:
        base = configs.get(base_key, {}).get("aggregate", {})
        abl = configs.get(ablation_key, {}).get("aggregate", {})
        b_mrr = base.get("mrr", 0)
        a_mrr = abl.get("mrr", 0)
        b_r3 = base.get("r@3", 0)
        a_r3 = abl.get("r@3", 0)
        b_f1 = base.get("f1@3", _f1(base.get("p@3", 0), b_r3))
        a_f1 = abl.get("f1@3", _f1(abl.get("p@3", 0), a_r3))
        d_lat = abl.get("avg_latency_ms", 0) - base.get("avg_latency_ms", 0)
        lat_str = f"+{d_lat / 1000:.1f}s" if d_lat >= 0 else f"{d_lat / 1000:.1f}s"

        rows_html += (
            f"<tr>"
            f"<td><b>{feature}</b><br><span style='color:#888;font-size:.78em'>{agent}</span></td>"
            f"<td>{_CONFIG_LABELS.get(base_key, (base_key, ''))[0]}</td>"
            f"<td>{_CONFIG_LABELS.get(ablation_key, (ablation_key, ''))[0]}</td>"
            f'<td style="text-align:center">{_pct(b_mrr)} → {_pct(a_mrr)}</td>'
            f"{_delta(a_mrr, b_mrr)}"
            f'<td style="text-align:center">{_pct(b_r3)} → {_pct(a_r3)}</td>'
            f"{_delta(a_r3, b_r3)}"
            f'<td style="text-align:center;font-weight:600">{_pct(b_f1)} → {_pct(a_f1)}</td>'
            f'<td style="text-align:center;color:#666">{lat_str}</td>'
            f'<td style="font-size:.8em;color:#555;max-width:280px">{note}</td>'
            f"</tr>"
        )

    return f"""
<h2>Ablation insights — effect of each feature flag</h2>
<table>
  <thead><tr>
    <th>Feature</th><th>Baseline</th><th>Ablation</th>
    <th style="text-align:center">MRR</th><th style="text-align:center">ΔMRR</th>
    <th style="text-align:center">R@3</th><th style="text-align:center">ΔR@3</th>
    <th style="text-align:center">F1@3</th>
    <th style="text-align:center">Δlatency</th><th>Notes</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""


def _section_category_breakdown(config_keys: list[str], configs: dict, tasks: list[dict]) -> str:
    breakdown = build_category_breakdown(config_keys, tasks)

    short_labels = {k: _CONFIG_LABELS[k][0][:18] for k in config_keys}
    header_cells = "".join(
        f'<th style="text-align:center">{short_labels[k]}</th>' for k in config_keys
    )

    rows_html = ""
    for cat in ["danish_support", "english_support", "escalation", "out_of_scope", "vague"]:
        if cat not in breakdown:
            continue
        cat_data = breakdown[cat]
        badge = _category_badge(cat)
        n_tasks = next(iter(cat_data.values()), {}).get("n", 0)
        n_ann = next(iter(cat_data.values()), {}).get("n_annotated", 0)
        cells = ""
        for key in config_keys:
            d = cat_data.get(key, {})
            n_a = d.get("n_annotated", 0)
            mrr = d.get("mrr_sum", 0) / n_a if n_a > 0 else None
            if mrr is None:
                cells += '<td style="text-align:center;color:#aaa">—</td>'
            else:
                cells += f'<td class="{_mrr_class(mrr)}" style="text-align:center;font-weight:600">{_pct(mrr)}</td>'
        rows_html += (
            f"<tr>"
            f"<td>{badge}</td>"
            f"<td style='text-align:center;color:#888;font-size:.8em'>{n_ann}/{n_tasks}</td>"
            f"{cells}</tr>"
        )

    return f"""
<h2>Category breakdown (MRR)</h2>
<p style="color:#666;font-size:.85em">
  MRR computed only over annotated tasks (those with golden_urls).
  Escalation/OOS/vague tasks have no golden_urls by design — they measure routing correctness, not retrieval.
</p>
<table>
  <thead><tr>
    <th>Category</th><th style="text-align:center">ann/total</th>
    {header_cells}
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""


def _section_per_task(config_keys: list[str], tasks: list[dict], va_smoke: dict | None) -> str:
    short_labels = {k: _CONFIG_LABELS[k][0][:15] for k in config_keys}

    # va-staging task index
    va_by_tid: dict[str, dict] = {}
    if va_smoke:
        for t in va_smoke.get("task_results", []):
            va_by_tid[t["task_id"]] = t

    header_cols = "".join(
        f'<th style="text-align:center">{short_labels[k]}</th>' for k in config_keys
    )
    if va_smoke:
        header_cols += '<th style="text-align:center">va_staging<br><span style="font-weight:400;font-size:.85em">(smoke)</span></th>'
    header_cols += "<th>BKH response</th>"

    rows_html = ""
    prev_cat = None
    for row in tasks:
        cat = row.get("category", "")
        if cat != prev_cat:
            rows_html += f'<tr><td colspan="100" style="font-size:.72em;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:#555;padding:10px 12px 3px;background:#f8f9fa">{cat.replace("_", " ").upper()}</td></tr>'
            prev_cat = cat

        q = row["query"][:80] + ("…" if len(row["query"]) > 80 else "")
        has_gold = bool(row.get("golden_urls"))

        config_cells = ""
        for key in config_keys:
            t = row["configs"].get(key)
            if t is None:
                config_cells += '<td style="text-align:center;color:#aaa">—</td>'
            else:
                mrr = t.get("scores", {}).get("mrr") if has_gold else None
                config_cells += f'<td style="text-align:center">{_task_result(mrr)}</td>'

        # va-staging column
        va_cell = ""
        if va_smoke:
            vt = va_by_tid.get(row["task_id"])
            if vt is not None:
                mrr = vt.get("scores", {}).get("mrr") if has_gold else None
                va_cell = f'<td style="text-align:center">{_task_result(mrr)}</td>'
            else:
                va_cell = '<td style="text-align:center;color:#aaa">—</td>'

        # BKH column
        bkh_entry = row.get("bkh")
        if bkh_entry:
            resp = (bkh_entry.get("response") or "")[:120].replace("<", "&lt;")
            rating = bkh_entry.get("rating", "?")
            rating_badge = (
                '<span class="badge b-hit">👍</span>'
                if rating == 1.0
                else '<span class="badge b-miss">👎</span>'
                if rating == 0.0
                else f'<span class="badge b-info">{rating}</span>'
            )
            ft = bkh_entry.get("failure_type", "")
            ft_badge = f' <span class="badge b-na">{ft}</span>' if ft and ft != "none" else ""
            bkh_cell = (
                f'<td><span class="resp-excerpt">{resp}…</span><br>{rating_badge}{ft_badge}</td>'
            )
        else:
            bkh_cell = '<td style="color:#aaa;font-size:.8em">new task</td>'

        golden_note = ""
        if row.get("golden_urls"):
            slug = row["golden_urls"][0].split("/articles/")[-1][:40] if row["golden_urls"] else ""
            golden_note = f'<br><span style="color:#888;font-size:.75em">{slug}</span>'

        rows_html += (
            f"<tr>"
            f'<td class="query-text">{q}{golden_note}</td>'
            f"<td>{_category_badge(cat)}</td>"
            f"{config_cells}{va_cell}{bkh_cell}"
            f"</tr>"
        )

    return f"""
<h2>Per-task results</h2>
<div class="legend">
  <div class="legend-item"><span class="badge b-hit">HIT</span> golden URL found at rank 1</div>
  <div class="legend-item"><span class="badge b-par">PAR</span> golden URL found (not rank 1)</div>
  <div class="legend-item"><span class="badge b-miss">MISS</span> golden URL not retrieved</div>
  <div class="legend-item"><span class="badge b-na">—</span> unannotated task</div>
</div>
<div style="overflow-x:auto">
<table>
  <thead><tr>
    <th>Query</th><th>Cat</th>
    {header_cols}
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
"""


def _section_failure_modes(config_keys: list[str], tasks: list[dict]) -> str:
    def classify(row: dict, key: str) -> str:
        t = row["configs"].get(key)
        if t is None:
            return "no_data"
        has_gold = bool(row.get("golden_urls"))
        cat = row.get("category", "")
        if not has_gold:
            contact = (t.get("raw_response") or {}).get("contact_support", False)
            if cat == "escalation" and contact:
                return "correct_oos"
            if cat in ("out_of_scope", "vague", "escalation"):
                return "correct_oos"
            return "correct_oos"
        mrr = t.get("scores", {}).get("mrr", 0.0)
        if mrr >= 1.0:
            return "hit"
        retrieved = t.get("retrieved_urls", [])
        golden = row.get("golden_urls", [])
        if any(g in retrieved for g in golden):
            return "wrong_rank"
        if retrieved:
            return "wrong_article"
        return "no_retrieval"

    modes = ["hit", "wrong_rank", "wrong_article", "no_retrieval", "correct_oos"]
    counts: dict[str, dict[str, int]] = {k: dict.fromkeys(modes, 0) for k in config_keys}
    for row in tasks:
        for key in config_keys:
            m = classify(row, key)
            if m in modes:
                counts[key][m] = counts[key].get(m, 0) + 1

    short_labels = {k: _CONFIG_LABELS[k][0][:18] for k in config_keys}
    header = "".join(f'<th style="text-align:center">{short_labels[k]}</th>' for k in config_keys)
    rows_html = ""
    mode_labels = {
        "hit": ("HIT", "b-hit", "Golden URL cited at rank 1"),
        "wrong_rank": ("WRONG RANK", "b-par", "Golden URL retrieved but not at rank 1"),
        "wrong_article": ("WRONG ARTICLE", "b-miss", "Different articles retrieved; golden missed"),
        "no_retrieval": ("NO RETRIEVAL", "b-miss", "No sources returned"),
        "correct_oos": ("CORRECT OOS", "b-info", "Correctly handled non-retrieval query"),
    }
    for mode in modes:
        label, badge_cls, note = mode_labels[mode]
        cells = "".join(
            f'<td style="text-align:center">{counts[k][mode]}</td>' for k in config_keys
        )
        rows_html += (
            f"<tr>"
            f'<td><span class="badge {badge_cls}">{label}</span>'
            f'<br><span style="color:#888;font-size:.78em">{note}</span></td>'
            f"{cells}</tr>"
        )

    return f"""
<h2>Failure mode breakdown</h2>
<table>
  <thead><tr>
    <th>Failure mode</th>
    {header}
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""


def _section_quality_graders(va_quality: dict | None, bkh: dict) -> str:
    """Side-by-side LLM grader scores for va_staging (488 tasks) and BKH (10 tasks)."""

    def _grader_rows(grader_summary: dict) -> str:
        rows = ""
        order = ["answer_relevancy", "completeness", "escalation", "grounding"]
        for metric in order:
            if metric not in grader_summary:
                continue
            g = grader_summary[metric]
            avg = g.get("avg_score", 0)
            n = g.get("n_graded", 0)
            pr = g.get("pass_rate", 0)
            score_cls = "cell-great" if avg >= 0.85 else ("cell-good" if avg >= 0.70 else "cell-ok")
            rows += (
                f"<tr>"
                f"<td>{metric.replace('_', ' ').title()}</td>"
                f'<td style="text-align:center">{n}</td>'
                f'<td class="{score_cls}" style="text-align:center;font-weight:600">{avg:.3f}</td>'
                f'<td style="text-align:center">{pr * 100:.1f}%</td>'
                f"</tr>"
            )
        return rows

    # Build va_staging grader table
    if va_quality:
        va_gs = va_quality.get("grader_summary", {})
        va_n = va_quality.get("n_queries", 0)
        va_table = f"""
<h3>va_staging — {va_n} production queries</h3>
<table>
  <thead><tr>
    <th>Metric</th><th style="text-align:center">n graded</th>
    <th style="text-align:center">Avg score</th><th style="text-align:center">Pass rate</th>
  </tr></thead>
  <tbody>{_grader_rows(va_gs)}</tbody>
</table>"""
    else:
        va_table = '<p style="color:#aaa;font-size:.85em">va_staging quality results not found.</p>'

    # Build BKH grader table — aggregate across the 10 tasks
    bkh_grader_agg: dict[str, dict] = {}
    for entry in bkh.values():
        gr = entry.get("grader_results", {})
        for metric, result in gr.items():
            if metric not in bkh_grader_agg:
                bkh_grader_agg[metric] = {"score_sum": 0.0, "n": 0, "n_correct": 0}
            bkh_grader_agg[metric]["score_sum"] += result.get("score", 0)
            bkh_grader_agg[metric]["n"] += 1
            bkh_grader_agg[metric]["n_correct"] += int(result.get("is_correct", False))

    bkh_summary = {
        m: {
            "avg_score": v["score_sum"] / v["n"] if v["n"] else 0,
            "n_graded": v["n"],
            "pass_rate": v["n_correct"] / v["n"] if v["n"] else 0,
        }
        for m, v in bkh_grader_agg.items()
    }
    bkh_n = len(bkh)
    bkh_table = f"""
<h3>BKH (Bookkeeper Hero) quality results — {bkh_n} sampled tasks</h3>
<table>
  <thead><tr>
    <th>Metric</th><th style="text-align:center">n graded</th>
    <th style="text-align:center">Avg score</th><th style="text-align:center">Pass rate</th>
  </tr></thead>
  <tbody>{_grader_rows(bkh_summary)}</tbody>
</table>"""

    return f"""
<h2>LLM grader scores — response quality</h2>
<div class="note-box">
  <b>Comparability note:</b> va_staging evaluated on 488 production queries (full BKH production set);
  BKH quality results are for 10 sampled tasks from the regression eval set.
  Scores are <em>not directly comparable</em> — treat them as independent quality signals.
</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
  <div>{va_table}</div>
  <div>{bkh_table}</div>
</div>
"""


def _section_report_index() -> str:
    """Links to related eval reports."""
    links = [
        (
            "BKH eval suite",
            "../../bkh/all_suite.html",
            "Full BKH regression eval suite — per-task grader results across the full train/test split",
        ),
        (
            "BKH eval stats",
            "../../bkh/all_stats.html",
            "BKH aggregate statistics — category breakdown, pass rates, failure type distribution",
        ),
        (
            "VA staging quality",
            "../../va/va-staging_suite.html",
            "VA staging LLM grader suite (regenerate via eval_quality)",
        ),
    ]
    rows = "".join(
        f"<tr>"
        f'<td><a href="{href}" target="_blank"><b>{label}</b></a></td>'
        f'<td style="color:#666;font-size:.85em">{desc}</td>'
        f"</tr>"
        for label, href, desc in links
    )
    return f"""
<h2>Related reports</h2>
<table>
  <thead><tr><th>Report</th><th>Description</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _ci_bar_svg(lo: float, hi: float, delta: float, v_min: float, v_max: float) -> str:
    """Inline SVG forest-plot CI bar. Red dashed line at zero."""
    W, H, pad = 180, 18, 10
    chart_w = W - 2 * pad
    v_range = v_max - v_min or 0.4

    def to_x(v: float) -> int:
        return pad + int((v - v_min) / v_range * chart_w)

    x0, x1, xd = to_x(lo), to_x(hi), to_x(delta)
    x_zero = to_x(0.0)
    significant = lo > 0 or hi < 0
    bar_color = "#16a34a" if significant else "#64748b"

    return (
        f'<svg width="{W}" height="{H}" style="vertical-align:middle;overflow:visible">'
        # Background axis
        f'<line x1="{pad}" y1="{H // 2}" x2="{W - pad}" y2="{H // 2}" '
        f'stroke="#e2e8f0" stroke-width="1"/>'
        # Zero reference line (red dashed)
        f'<line x1="{x_zero}" y1="1" x2="{x_zero}" y2="{H - 1}" '
        f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="3,2"/>'
        # CI interval bar
        f'<line x1="{x0}" y1="{H // 2}" x2="{x1}" y2="{H // 2}" '
        f'stroke="{bar_color}" stroke-width="4" stroke-linecap="round"/>'
        # CI end caps
        f'<line x1="{x0}" y1="{H // 2 - 5}" x2="{x0}" y2="{H // 2 + 5}" '
        f'stroke="{bar_color}" stroke-width="1.5"/>'
        f'<line x1="{x1}" y1="{H // 2 - 5}" x2="{x1}" y2="{H // 2 + 5}" '
        f'stroke="{bar_color}" stroke-width="1.5"/>'
        # Point estimate diamond
        f'<polygon points="{xd},{H // 2 - 5} {xd + 4},{H // 2} {xd},{H // 2 + 5} {xd - 4},{H // 2}" '
        f'fill="{bar_color}" stroke="white" stroke-width="1"/>'
        f"</svg>"
    )


def _section_significance(sig_results: list[dict], power_req: dict) -> str:
    if not sig_results:
        return """
<h2>Statistical significance</h2>
<div class="note-box">Install numpy + scipy for bootstrap CI analysis: <code>uv add numpy scipy</code></div>
"""

    # Determine shared axis range across all comparisons
    all_vals = [r["ci_lo"] for r in sig_results] + [r["ci_hi"] for r in sig_results]
    v_min = min(all_vals) - 0.02
    v_max = max(all_vals) + 0.02

    rows_html = ""
    for r in sig_results[:12]:
        a_label = _CONFIG_LABELS.get(r["a"], (r["a"], ""))[0]
        b_label = _CONFIG_LABELS.get(r["b"], (r["b"], ""))[0]
        delta = r["delta"]
        lo, hi = r["ci_lo"], r["ci_hi"]
        sig = r["significant"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        ci_str = f"[{lo:+.3f}, {hi:+.3f}]"
        sig_badge = (
            '<span class="badge b-hit">SIG ✓</span>'
            if sig
            else '<span class="badge b-na">not sig</span>'
        )
        row_cls = ' class="highlight-row"' if sig else ""
        ci_bar = _ci_bar_svg(lo, hi, delta, v_min, v_max)
        rows_html += (
            f"<tr{row_cls}>"
            f"<td><b>{a_label}</b></td><td>{b_label}</td>"
            f'<td style="text-align:center;font-weight:600">{delta_str}</td>'
            f'<td style="text-align:center;padding:4px 8px">{ci_bar}<br>'
            f'<span style="font-family:monospace;font-size:.72em;color:#64748b">{ci_str}</span></td>'
            f'<td style="text-align:center">{r["n"]}</td>'
            f'<td style="text-align:center">{sig_badge}</td>'
            f"</tr>"
        )

    power_rows = ""
    for delta, n_needed in sorted(power_req.items()):
        power_rows += (
            f"<tr><td>Δ{delta:.2f} MRR</td>"
            f'<td style="text-align:center;font-weight:600">{n_needed}</td>'
            f'<td style="color:#888;font-size:.85em">{"Current" if n_needed <= 44 else ("Feasible" if n_needed <= 200 else "Hard")}</td>'
            f"</tr>"
        )

    return f"""
<h2>Statistical significance — bootstrap 95% CI on MRR delta</h2>
<div class="note-box">
  <b>⚠️ Key finding:</b> With N=32 annotated tasks, none of the inter-config differences are statistically significant.
  The bootstrap 95% CI includes zero for all pairwise comparisons (CI bars cross the red zero line).
  All ranking within the "Local agents" section should be treated as indicative, not conclusive.
  <b>Recommended next step: run 500-task eval</b> (<code>make eval-sa-500-top3</code>) to reach ~71 annotated tasks
  and detect the ~0.15 gap vs va_staging.
</div>
<div style="display:grid;grid-template-columns:2fr 1fr;gap:20px">
<div>
<h3>Pairwise comparisons (bootstrap 5000 iterations)</h3>
<p style="color:#666;font-size:.85em;margin-top:0">
  CI bars show 95% bootstrap confidence interval for Δ MRR (A − B).
  Red dashed line = zero. Diamond = point estimate.
  Green bar = significant; gray = not significant.
</p>
<table>
  <thead><tr>
    <th>Config A</th><th>Config B</th>
    <th style="text-align:center">Δ MRR</th>
    <th style="text-align:center">95% CI</th>
    <th style="text-align:center">n tasks</th>
    <th style="text-align:center">Significant?</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
<div>
<h3>Annotated tasks needed (80% power)</h3>
<table>
  <thead><tr><th>Effect size</th><th style="text-align:center">N needed</th><th>Feasibility</th></tr></thead>
  <tbody>{power_rows}</tbody>
</table>
<p style="color:#666;font-size:.82em;margin-top:8px">
  Assumes SD≈0.44 on per-task MRR, α=0.05, two-sided.
  Running 500 tasks → ~71 annotated → MDE ≈ 0.16 (vs Δ0.15 gap to va_staging).
</p>
</div>
</div>
"""


def _section_discriminative_tasks(
    config_keys: list[str], tasks: list[dict], top_keys: list[str]
) -> str:
    disc = find_discriminative_tasks(config_keys, tasks, top_keys)
    if not disc:
        return ""

    short = {k: _CONFIG_LABELS[k][0][:12] for k in top_keys if k in _CONFIG_LABELS}
    header_cols = "".join(f'<th style="text-align:center">{short[k]}</th>' for k in top_keys)

    rows_html = ""
    for row in disc[:20]:
        q = row["query"][:70] + ("…" if len(row["query"]) > 70 else "")
        cat = row.get("category", "")
        hit_rate = row["_hit_rate"]
        # Disagreement bar: fill proportional to disagreement score
        disc_score = row["_disagreement"]
        bar_w = int(disc_score * 60)
        bar = f'<div style="height:6px;width:{bar_w}px;background:#f59e0b;border-radius:3px;display:inline-block"></div>'

        slug = ""
        if row.get("golden_urls"):
            slug = row["golden_urls"][0].split("/articles/")[-1][:35]

        cells = ""
        for k in top_keys:
            t = row["configs"].get(k)
            if t is None:
                cells += '<td style="text-align:center;color:#aaa">—</td>'
            else:
                mrr = t.get("scores", {}).get("mrr") if row.get("golden_urls") else None
                cells += f'<td style="text-align:center">{_task_result(mrr)}</td>'

        rows_html += (
            f"<tr>"
            f'<td class="query-text">{q}<br><span style="color:#888;font-size:.75em">{slug}</span></td>'
            f"<td>{_category_badge(cat)}</td>"
            f'<td style="text-align:center">{bar} <span style="font-size:.78em;color:#888">{hit_rate:.0%}</span></td>'
            f"{cells}</tr>"
        )

    return f"""
<h2>Discriminative tasks — where configs disagree most</h2>
<p style="color:#666;font-size:.85em">
  Ranked by disagreement: tasks where exactly half the configs succeed are maximally informative.
  All-hit or all-miss tasks tell us nothing about which config is better.
  Showing top 20 most discriminative annotated tasks across top configs.
</p>
<div style="overflow-x:auto">
<table>
  <thead><tr>
    <th>Query</th><th>Cat</th>
    <th style="text-align:center">Disagr. / hit rate</th>
    {header_cols}
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>
"""


def _section_roadmap() -> str:
    return """
<h2>Research roadmap</h2>
<table>
  <thead><tr><th>Initiative</th><th>Expected gain</th><th>Cost</th><th>Status</th></tr></thead>
  <tbody>
    <tr>
      <td><b>Upgrade to gemini-3-flash-preview</b><br>
        <span style="color:#888;font-size:.8em">GEMINI_MODEL=gemini-3-flash-preview — biggest remaining lever vs va_staging</span></td>
      <td>~+0.04–0.08 MRR</td>
      <td>Same latency, slightly higher token cost</td>
      <td><span class="badge b-na">TODO</span></td>
    </tr>
    <tr>
      <td><b>Multi-query retrieval</b><br>
        <span style="color:#888;font-size:.8em">Generate 2–3 query reformulations → retrieve for each → merge/dedup.
        Matches va_staging's internal multi-query strategy.</span></td>
      <td>+0.05–0.10 MRR (especially vague/short queries)</td>
      <td>+1 LLM call, +N Bedrock calls. No grading loop.</td>
      <td><span class="badge b-na">TODO</span></td>
    </tr>
    <tr>
      <td><b>HyDE (Hypothetical Document Embeddings)</b><br>
        <span style="color:#888;font-size:.8em">Generate a fake "ideal article" → embed → retrieve by similarity.
        Bridges the semantic gap for abstract/accounting-domain queries.</span></td>
      <td>+0.03–0.07 MRR on abstract queries</td>
      <td>+1 LLM call per query, no retrieval loop</td>
      <td><span class="badge b-na">PLANNED</span></td>
    </tr>
    <tr>
      <td><b>hc_rag as retrieval backend for hc_adk / hc_lg</b><br>
        <span style="color:#888;font-size:.8em">VA_RETRIEVAL_MODE=rag / VA_RETRIEVAL_BACKEND=rag — test if DuckDB +
        multilingual-e5-large is competitive with Bedrock HYBRID, zero cloud cost.</span></td>
      <td>TBD (baseline: hc_rag alone ≈ 0.42 MRR)</td>
      <td>Zero Bedrock cost; local embedding only</td>
      <td><span class="badge b-na">TODO</span></td>
    </tr>
    <tr>
      <td><b>DPO / preference fine-tuning</b><br>
        <span style="color:#888;font-size:.8em">Curate golden vs rejected response pairs from this eval set,
        fine-tune retrieval query generation or answer synthesis.</span></td>
      <td>Potentially large, but requires 200+ preference pairs</td>
      <td>Significant ML infra; blocked on golden response annotation</td>
      <td><span class="badge b-miss">BLOCKED — need golden traces</span></td>
    </tr>
  </tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_report(out_path: Path) -> None:
    configs = load_ablation_files()
    va_smoke = load_va_staging_smoke()
    bkh = load_bkh_quality()
    va_quality = load_va_staging_quality()
    quality_grades = load_quality_grades()

    if not configs:
        print("No ablation JSON files found in", ABLATION_DIR)
        return

    bkh_stats = compute_bkh_retrieval_stats(bkh, configs)
    bkh_overlap_ids = _bkh_ablation_overlap_ids(bkh, configs)
    # Quality sources (prefer 44-task graded file; fall back to 488-query production file):
    # - va_staging: quality_grades["va_staging"] if run, else 488-query production graded.json
    # - BKH: filtered to 9 overlap task IDs (same tasks as retrieval eval)
    # - Local agents: from quality_grades (run make grade-ablation-top)
    va_quality_sum = quality_grades.get("va_staging", {}).get(
        "grader_summary"
    ) or _aggregate_va_staging_quality(va_quality)
    bkh_quality_sum = _aggregate_bkh_quality(bkh, filter_to_ablation_ids=bkh_overlap_ids)
    sig_results = compute_significance(configs)
    power_req = compute_power_requirements()
    config_keys, tasks = build_task_matrix(configs, va_smoke, bkh)

    n_tasks = len(tasks)
    n_annotated = sum(1 for t in tasks if t.get("golden_urls"))
    n_categories = len({t.get("category", "") for t in tasks})
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    header = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Support Agent Comparison — Eval Report</title>
  <style>{CSS}</style>
</head>
<body>
<div class="page">
  <h1>Support Agent Ablation &amp; Comparison Report</h1>
  <div class="meta">Generated {generated} &nbsp;·&nbsp;
    {len(config_keys)} configs &nbsp;·&nbsp;
    {n_tasks} tasks ({n_annotated} annotated) &nbsp;·&nbsp;
    {n_categories} categories
  </div>
  <div class="stat-grid">
"""

    # Top-level stat cards
    best_mrr = max(configs[k].get("aggregate", {}).get("mrr", 0) for k in config_keys)
    best_config = max(config_keys, key=lambda k: configs[k].get("aggregate", {}).get("mrr", 0))
    va_full = configs.get("va_staging", {})
    va_bench = va_full if va_full.get("n_tasks", 0) >= 44 else (va_smoke or {})
    va_mrr = va_bench.get("aggregate", {}).get("mrr", 0) if va_bench else 0
    bkh_mrr = bkh_stats.get("mrr", 0) if bkh_stats else 0

    header += _summary_card(
        "BKH MRR ⚠️", _pct(bkh_mrr), f"{bkh_stats.get('n_annotated', 0)} tasks (not comparable)"
    )
    header += _summary_card("va_staging MRR", _pct(va_mrr), "production benchmark")
    header += _summary_card(
        "Best local MRR", _pct(best_mrr), _CONFIG_LABELS.get(best_config, (best_config, ""))[0]
    )
    header += _summary_card(
        "Gap to va_staging", _pct(best_mrr - va_mrr), "best local vs production"
    )
    header += _summary_card("Tasks evaluated", str(n_tasks), f"{n_annotated} annotated")

    header += "</div>\n"

    # Top configs for discriminative analysis: va_staging + best 4 local
    top_local = sorted(
        [k for k in config_keys if k != "va_staging"],
        key=lambda k: configs[k].get("aggregate", {}).get("mrr", 0),
        reverse=True,
    )[:4]
    top_keys = (["va_staging"] if "va_staging" in config_keys else []) + top_local

    sections = [
        header,
        _section_report_index(),
        _section_aggregate(
            config_keys,
            configs,
            va_smoke,
            bkh_stats,
            quality_grades,
            va_quality_sum,
            bkh_quality_sum,
        ),
        _section_significance(sig_results, power_req),
        _section_discriminative_tasks(config_keys, tasks, top_keys),
        _section_quality_graders(va_quality, bkh),
        _section_ablation_insights(config_keys, configs),
        _section_category_breakdown(config_keys, configs, tasks),
        _section_failure_modes(config_keys, tasks),
        _section_per_task(config_keys, tasks, va_smoke),
        _section_roadmap(),
        "</div></body></html>",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(sections), encoding="utf-8")
    print(f"Report written → {out_path}")

    # Export SVG figures alongside the HTML report
    _export_figures_from_report(configs, config_keys)


def _export_figures_from_report(configs: dict, config_keys: list[str]) -> None:
    """Export key SVG figures derived from this report's live data.

    Called automatically after generate_report() so that `make report-sa`
    always refreshes both the HTML and the figures in one step.
    """
    try:
        from evals.reports.utils.figures import (
            _style,
            fig_bkh_pass_rates,
            fig_feature_impact,
            fig_mrr_comparison,
            fig_power_curve,
            fig_va_pass_rates,
        )

        _style()

        # Build display-labelled config dict for the MRR chart
        display_configs: dict = {}
        for key in config_keys:
            label, desc = _CONFIG_LABELS.get(key, (key, ""))
            mrr = configs[key].get("aggregate", {}).get("mrr", 0) if key in configs else 0
            display_configs[key] = {"display_label": label, "aggregate": {"mrr": mrr}}

        print("\nExporting SVG figures from live report data →")
        fig_mrr_comparison(display_configs)
        fig_feature_impact()  # uses hardcoded deltas (notebook-derived)
        fig_power_curve()
        fig_bkh_pass_rates()  # uses hardcoded calibration values
        fig_va_pass_rates()  # uses hardcoded VA staging values
    except ImportError as exc:
        print(f"  ⚠ SVG export skipped — matplotlib not available: {exc}")
    except Exception as exc:
        print(f"  ⚠ SVG export error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip SVG figure export (HTML report only)"
    )
    args = parser.parse_args()
    generate_report(args.output)


if __name__ == "__main__":
    main()
