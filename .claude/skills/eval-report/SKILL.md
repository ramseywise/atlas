---
name: eval-report
description: "Build an HTML evaluation report from forecasting or segmentation results. Wraps evals/reports/ — generates figures, assembles HTML, opens in browser. Use after running the forecast or segmentation pipeline."
disable-model-invocation: true
allowed-tools: Read Bash Grep Glob Write
---

Build an evaluation report for: `$ARGUMENTS` (`forecast` | `segment` | both).

## Entry points

```python
# Forecast eval report
from evals.reports.report import build_eval_report
from evals.reports.figures import fig_forecast, fig_grader_pass_rates, fig_eval_history

# Segmentation report
from evals.reports.report import build_segment_report
from evals.reports.figures import fig_segments_scatter, fig_segment_eval, fig_segment_sizes_bar
```

Output goes to `evals/reports/output/` (gitignored).

## Forecast report workflow

1. Load eval results — check `evals/graders/graders.py` for `EvalHarness` output format
2. Build figures:
   - `fig_forecast(actuals, predictions, series_id)` → forecast vs actual SVG
   - `fig_grader_pass_rates(harness_result)` → pass/fail per grader
   - `fig_eval_history(drift_log_path)` → rolling MASE from `drift_log.jsonl`
3. Assemble: `build_eval_report(figures, harness_result)` → `output/eval_report.html`
4. Open: `open evals/reports/output/eval_report.html`

## Segmentation report workflow

1. Load `SegmentEvalReport` — from `core/segmentation/evaluation.py`
2. Build figures:
   - `fig_segments_scatter(embeddings, labels)` → 2D scatter (UMAP or PCA reduced)
   - `fig_segment_eval(eval_report)` → silhouette / CH / DB bar chart
   - `fig_segment_sizes_bar(eval_report)` → cluster size distribution
3. Assemble: `build_segment_report(figures, eval_report)` → `output/segment_report.html`
4. Open: `open evals/reports/output/segment_report.html`

## Pass thresholds (reference)

| Grader | Metric | Pass |
|--------|--------|------|
| MASE | vs naïve lag-7 | < 1.0 |
| SMAPE | | < 15% |
| Directional | % correct direction | > 55% |
| Coverage | % in 80% PI | ≥ 75% |
| Silhouette | | ≥ 0.25 |
| Davies-Bouldin | | ≤ 1.5 |
| Min cluster size | | ≥ 3 |

## Interpreting results

After opening the report, surface:
1. **Which graders failed** and by how much (not just pass/fail — show the delta)
2. **Drift trend** — is MASE improving or degrading across eval cycles?
3. **Segment quality** — are any clusters below minimum size? Is silhouette near threshold?
4. **Actionable next step**: retrain, tune hyperparameters, adjust features, or accept

If any grader is within 10% of its threshold: flag as "at risk" even if currently passing.

## Running from scratch

```bash
# Generate synthetic data and run forecast pipeline
uv run python -m pipelines.forecast

# Run segmentation pipeline
uv run python -m pipelines.segment

# Then build reports
uv run python -c "from evals.reports.report import build_eval_report; ..."
```
