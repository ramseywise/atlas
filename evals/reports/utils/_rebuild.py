"""Rebuild suite HTML from JSON — no LLM (avoids importing eval_quality at package load)."""

from __future__ import annotations

import json
from pathlib import Path

from evals.graders.judges.base import GraderOutput
from evals.metrics.base import PassRateMetric
from evals.metrics.comparison.url_reclassify import run_reclassification_pipeline
from evals.metrics.stats import eval_stats_metrics
from evals.metrics.suite import SuiteReport, evaluate_suite
from evals.pipelines.datasets import load_jsonl
from evals.reports.paths import DEFAULT_BKH_ALL
from evals.reports.suite import render_suite_html
from evals.reports.utils.layout import dataset_root, repo_root


def _metric_results_from_graded(query_results: list[dict]):
    by_grader: dict[str, list[GraderOutput]] = {}
    for qr in query_results:
        for gt, res in qr.get("grader_results", {}).items():
            by_grader.setdefault(gt, []).append(
                GraderOutput(
                    grader_type=gt,
                    score=res.get("score", 0.0),
                    is_correct=bool(res.get("is_correct", False)),
                    reasoning=res.get("reasoning", ""),
                    dimensions=res.get("dimensions") or {},
                    labels=res.get("labels") or {},
                )
            )
    return [PassRateMetric(gt).compute(outputs) for gt, outputs in by_grader.items() if outputs]


def _group_breakdown(
    query_results: list[dict],
    task_meta: dict[str, dict],
    *,
    prefer_sentiment: bool = False,
) -> dict:
    from evals.reports.html.eval_set_group import group_key_from_meta

    group_raw: dict[str, dict[str, list]] = {}
    for qr in query_results:
        group = group_key_from_meta(
            task_meta.get(qr["task_id"], {}),
            prefer_sentiment=prefer_sentiment,
        )
        for grader, res in qr.get("grader_results", {}).items():
            group_raw.setdefault(group, {}).setdefault(grader, []).append(res)
    return {
        group: {
            grader: {
                "pass_rate": sum(1 for r in outputs if r.get("is_correct")) / len(outputs),
                "avg_score": sum(r.get("score") or 0.0 for r in outputs) / len(outputs),
                "n": len(outputs),
            }
            for grader, outputs in graders.items()
        }
        for group, graders in group_raw.items()
    }


def _load_stats_json(stats_json_path: Path) -> dict:
    data = json.loads(stats_json_path.read_text())
    file_stats = data.get("stats", {})
    if not file_stats:
        return {}
    return next(iter(file_stats.values()))


def assemble_suite_report(
    *,
    graded_path: Path | None = None,
    source_path: Path | None = None,
    stats_json_path: Path | None = None,
    log_reclass_slices: bool = False,
    staging=None,
) -> tuple[SuiteReport, dict[str, Path | str]]:
    """Build SuiteReport from JSON — same logic as rebuild_suite_report, no HTML write."""
    from evals.pipelines.eval_quality import (
        compute_bkh_heuristic_stats,
        compute_va_heuristic_stats,
    )

    meta: dict[str, Path | str] = {}
    query_results: list[dict] = []
    dataset_name = "bkh"
    graded: dict = {}
    if graded_path and graded_path.exists():
        graded = json.loads(graded_path.read_text())
        query_results = graded.get("query_results", [])
        dataset_name = graded.get("dataset", graded_path.stem)
        meta["n_queries"] = len(query_results)
        meta["graded_path"] = graded_path

    task_meta: dict[str, dict] = {}
    source_tasks: list = []
    is_va = False
    if source_path and source_path.exists():
        from evals.graders.heuristic.calculate_stats import sentiment as _turn_sentiment

        source_tasks = load_jsonl(source_path)
        for t in source_tasks:
            meta_row = dict(t.metadata or {})
            meta_row["_turn_sentiment"] = _turn_sentiment(t)
            task_meta[t.task_id] = meta_row
        is_va = bool(
            source_tasks
            and (
                source_tasks[0].metadata.get("source") == "va_staging"
                or "golden_all_responses" in source_path.name
            )
        )
        meta["source_path"] = source_path
    else:
        for qr in query_results:
            task_meta[qr["task_id"]] = {}

    if is_va:
        dataset_name = "VA Staging Golden"
    elif dataset_name in ("bkh", "") or (stats_json_path and "bkh" in str(stats_json_path)):
        dataset_name = "BKH"

    if not source_path and graded_path and "calibration" in graded_path.name:
        cal = dataset_root("bkh") / "eval_sets" / "calibration_sample.jsonl"
        if cal.exists():
            source_path = cal
            from evals.graders.heuristic.calculate_stats import (
                sentiment as _turn_sentiment,
            )

            source_tasks = load_jsonl(cal)
            for t in source_tasks:
                meta_row = dict(t.metadata or {})
                meta_row["_turn_sentiment"] = _turn_sentiment(t)
                task_meta[t.task_id] = meta_row

    if stats_json_path and stats_json_path.exists():
        base_stats = _load_stats_json(stats_json_path)
        if not dataset_name or dataset_name == "bkh":
            dataset_name = stats_json_path.stem.replace("_stats", "")
    elif is_va:
        base_stats = compute_va_heuristic_stats(source_tasks)
    elif source_tasks:
        base_stats = compute_bkh_heuristic_stats(source_tasks)
    else:
        base_stats = {}

    cal_idx: dict[str, dict] = {}
    use_staging_judges = False
    breakdown_scenario: dict | None = None
    heuristic_metrics = eval_stats_metrics(base_stats) if base_stats else []

    if is_va and graded and graded_path and "v2" not in graded_path.name:
        from evals.metrics.calibration.pass_overrides import (
            GOLDEN_LLM_METRICS_V1,
            compute_llm_pass_summary,
            group_breakdown_calibrated,
            metric_results_from_calibrated,
        )

        use_staging_judges = True
        if staging is not None:
            cal_idx = staging.cal_index
        else:
            from evals.metrics.calibration.pass_overrides import build_calibration_index

            cal_idx = build_calibration_index()
        judge_metrics = metric_results_from_calibrated(query_results, cal_idx)
        breakdown = group_breakdown_calibrated(
            query_results,
            task_meta,
            cal_idx,
            prefer_sentiment=False,
        )
    else:
        judge_metrics = _metric_results_from_graded(query_results)
        from evals.metrics.calibration.grader_scope import BKH_CALIBRATION_LLM_KEYS

        judge_metrics = [m for m in judge_metrics if m.metric_name in BKH_CALIBRATION_LLM_KEYS]
        prefer_sent = bool(graded_path and "calibration" in graded_path.name)
        breakdown = _group_breakdown(
            query_results,
            task_meta,
            prefer_sentiment=prefer_sent,
        )
        if prefer_sent:
            breakdown_scenario = _group_breakdown(
                query_results,
                task_meta,
                prefer_sentiment=False,
            )

    if judge_metrics:
        judge_names = {m.metric_name for m in judge_metrics}
        heuristic_metrics = [
            m
            for m in heuristic_metrics
            if not (m.breakdown.get("status") == "wip" and m.metric_name in judge_names)
        ]

    is_cal_sample = not is_va and graded_path and "calibration" in graded_path.name
    heuristic_stats = {
        **base_stats,
        "group_breakdown": breakdown,
        "slice_grouping": "sentiment" if is_cal_sample else "eval_set",
        "is_calibration_sample": is_cal_sample,
    }
    if is_cal_sample and breakdown_scenario:
        heuristic_stats["group_breakdown_scenario"] = breakdown_scenario
    meta["is_va"] = is_va
    meta["is_calibration_sample"] = is_cal_sample

    if staging is not None:
        heuristic_stats["url_overlap"] = staging.overlap_summary
        meta["reclass_slices"] = staging.slice_paths
        if is_va and source_tasks:
            from evals.metrics.comparison.url_overlap import (
                compute_overlap_retrieval_proxy,
            )

            heuristic_stats["retrieval_proxy_overlap"] = compute_overlap_retrieval_proxy(
                source_tasks,
                staging.overlap_records,
            )
    elif (
        is_va and source_path and source_path.exists() and DEFAULT_BKH_ALL.exists() and source_tasks
    ):
        overlap_summary, slice_paths = run_reclassification_pipeline(
            source_path,
            DEFAULT_BKH_ALL,
            export_base_slices=True,
        )
        heuristic_stats["url_overlap"] = overlap_summary
        meta["reclass_slices"] = slice_paths
        if log_reclass_slices and slice_paths:
            print("  URL overlap + reclassification slices:")
            for name, p in sorted(slice_paths.items()):
                print(f"    {name}: {p}")

    if use_staging_judges:
        from evals.metrics.calibration.pass_overrides import (
            GOLDEN_LLM_METRICS_V1,
            compute_llm_pass_summary,
        )

        raw_gs = compute_llm_pass_summary(graded, GOLDEN_LLM_METRICS_V1, calibrated=False)
        cal_gs = compute_llm_pass_summary(
            graded,
            GOLDEN_LLM_METRICS_V1,
            calibrated=True,
            cal_index=cal_idx,
        )
        reclass = (heuristic_stats.get("url_overlap") or {}).get("reclassification") or {}
        by_reason = reclass.get("by_reason") or {}
        heuristic_stats["staging_calibration"] = {
            "raw": raw_gs,
            "calibrated": cal_gs,
            "metrics": [(k, label, thr) for k, label, thr in GOLDEN_LLM_METRICS_V1],
            "n_calibrated_up": sum(v.get("n_calibrated_up", 0) for v in cal_gs.values()),
            "n_kb_corpus_gap": reclass.get("n_grounding_corpus_gap", 0),
            "n_kb_corpus_promoted": int(by_reason.get("edge_kb_grounding_corpus_gap", 0)),
            "human_validated_map": str(
                repo_root() / "data" / "articles" / "kb_url_map" / "human_validated_map.csv"
            ),
            "url_map_adjusted": True,
        }

    overlap_rp = heuristic_stats.get("retrieval_proxy_overlap")
    if is_va and overlap_rp:
        from evals.metrics.enrichment.overlap_heuristics import (
            overlap_heuristic_metrics,
        )

        replace_keys = {"retrieval_precision", "proxy_retrieval_recall"}
        heuristic_metrics = [
            m for m in heuristic_metrics if m.metric_name not in replace_keys
        ] + overlap_heuristic_metrics(overlap_rp)

    suite = evaluate_suite(
        heuristic_metric_results=heuristic_metrics,
        judge_metric_results=judge_metrics,
        heuristic_stats=heuristic_stats,
        suite_name=dataset_name,
    )
    meta["heuristic_n"] = len(heuristic_metrics)
    meta["judge_n"] = len(judge_metrics)
    return suite, meta


def rebuild_suite_report(
    *,
    graded_path: Path | None = None,
    source_path: Path | None = None,
    stats_json_path: Path | None = None,
    output_path: Path | None = None,
    pdf: bool = False,
    staging=None,
) -> Path:
    """Build suite HTML from graded/stats JSON — no LLM calls."""
    suite, meta = assemble_suite_report(
        graded_path=graded_path,
        source_path=source_path,
        stats_json_path=stats_json_path,
        log_reclass_slices=staging is None,
        staging=staging,
    )

    if output_path is None:
        from evals.reports.utils.layout import layout_for_profile

        profile = "va" if meta.get("is_va") else "bkh"
        output_path = layout_for_profile(profile).suite_html()

    from evals.reports.html.results_section import suite_results_section

    results_html = suite_results_section(
        suite,
        is_va=bool(meta.get("is_va")),
        export_figure=True,
    )
    render_suite_html(suite, output_path, pdf=pdf, results_section_html=results_html)
    print(f"Report: {output_path}")
    n = meta.get("n_queries", "?")
    sample_note = ""
    if meta.get("is_calibration_sample"):
        sample_note = " · BKH LLM = stratified calibration sample (not full 69k corpus)"
    print(
        f"  {n} queries | "
        f"heuristic={meta.get('heuristic_n')} judge={meta.get('judge_n')} | "
        f"is_va={meta.get('is_va')}{sample_note}"
    )
    return output_path
