---
name: ml-experiment
description: "Design, run, and interpret ML experiments. Covers model comparison, eval harness usage, hyperparameter tuning, and reading results against baselines. Use when comparing models, investigating forecast quality, or validating a new approach."
disable-model-invocation: true
allowed-tools: Read Bash Grep Glob Write
---

Run ML experiment for: `$ARGUMENTS` (describe the comparison or hypothesis).

## Before running an experiment

1. **State the hypothesis** — what do you expect to be true, and why?
2. **Define the baseline** — what does success look like vs the current model?
3. **Pin the data split** — use `generate_sequence_dataset()` or `generate_ml_dataset()` with fixed seed; log `n_rows`, `n_features`, `seed`
4. **Confirm eval metrics** — pick from the grader suite below; don't add new metrics mid-experiment

## Eval harness

```python
from evals.graders.graders import EvalHarness, MASEGrader, SMAPEGrader, DirectionalGrader

harness = EvalHarness(graders=[MASEGrader(), SMAPEGrader(), DirectionalGrader()])
result = harness.evaluate(actuals, predictions, baseline_predictions)
print(result.summary())  # pass/fail per grader + scores
```

Pass thresholds: `evals/metrics/constants.py::TIER_THRESHOLDS`

## Model comparison

```bash
make compare   # ARIMA vs Chronos comparison
```

Or directly:

```python
from evals.comparison import run_model_comparison
results = run_model_comparison(models=[arima, chronos], dataset=ds, n_splits=5)
```

Walk-forward CV: `src/cv_runner.py::run_walk_forward_cv()`

## Running the forecast agent

```bash
uv run python -m pipelines.forecast
```

Agent loop: Planner → Forecaster → Evaluator → Learner → (repeat)

Self-learning: `PlannerStrategy` updated via Haiku reflection each cycle. Drift trigger at `drift_ratio > 1.4`.

## Interpreting results

| Signal | Interpretation | Action |
|--------|---------------|--------|
| MASE > 1.0 | Worse than naïve lag-7 | Diagnose feature set; check preprocessing |
| SMAPE > 15% | Large percentage errors | Check outliers; try log transform |
| Directional < 55% | Not better than random | Model is not capturing trend |
| Coverage < 75% | PI too narrow | Increase uncertainty estimation |
| Drift ratio > 1.4 | Model degrading over time | LoRA fine-tune trigger (see `state.py`) |
| DriftGrader warning | Advisory only | Monitor; not a hard gate |

## Reproducibility checklist

Before reporting results:
- [ ] Fixed `random_state=42` in all models
- [ ] Seed logged in structlog output
- [ ] Data split identical to baseline (same `generate_*()` call + seed)
- [ ] Hyperparameters in config, not inline
- [ ] Model artifacts saved to `models/` with `joblib`
- [ ] Results logged as structured fields (not print statements)

## Experiment log format

```
## Experiment: [name]
Date: [today]
Hypothesis: [what you expected]
Dataset: generate_sequence_dataset(seed=42, n_series=7)
Models: [list]

### Results
| Model | MASE | SMAPE | Directional | Coverage | All passed? |
|-------|------|-------|-------------|---------- |-------------|

### Conclusion
[Did the hypothesis hold? What does this mean for the production model?]
### Next step
[One concrete action: accept, tune, discard, or investigate further]
```
