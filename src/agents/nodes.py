"""
LangGraph agent nodes for the cash flow forecasting system.

Four nodes, each receives AgentState and returns a partial state update:
  - planner_node     → updates strategy
  - forecaster_node  → updates forecasts
  - evaluator_node   → updates eval_report
  - learner_node     → updates learner_feedback, strategy, cycle_count, terminate
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from evals.graders.graders import EvalHarness
from src.agents.learner.bandit import BanditPolicy
from src.agents.learner.base import LearnerPolicy
from src.agents.learner.rule_based import RuleBasedPolicy
from src.agents.state import (
    AgentState,
    CategoryType,
    EvalReport,
    ForecastHorizon,
    ForecastResult,
    LearnerFeedback,
    ModelVariant,
    PlannerStrategy,
)

POLICY_REGISTRY: dict[str, type[LearnerPolicy]] = {
    "rule_based": RuleBasedPolicy,
    "bandit": BanditPolicy,
}

# ── Horizon map ───────────────────────────────────────────────────────────────

HORIZON_DAYS: dict[ForecastHorizon, int] = {
    ForecastHorizon.WEEK: 7,
    ForecastHorizon.FORTNIGHT: 14,
    ForecastHorizon.MONTH: 30,
    ForecastHorizon.QUARTER: 90,
}


# ── Planner Node ──────────────────────────────────────────────────────────────


def planner_node(state: AgentState) -> dict[str, Any]:
    """
    Selects forecast horizon, model variant, and feature flags.

    On cycle 0: uses defaults.
    On subsequent cycles: incorporates feedback from the previous Learner output.
    """
    feedback: LearnerFeedback | None = state.get("learner_feedback")

    if feedback is not None:
        # Adopt the updated strategy from the Learner
        strategy = feedback.updated_strategy
    else:
        # First cycle defaults
        strategy = PlannerStrategy(
            horizon=ForecastHorizon.MONTH,
            model_variant=ModelVariant.CHRONOS_TINY,
            context_multiplier=3.0,
            rationale="Initial default strategy — no prior feedback",
        )

    return {"strategy": strategy, "strategy_history": [strategy]}


# ── Forecaster Node ───────────────────────────────────────────────────────────


def _run_statsforecast_fallback(
    series_values: np.ndarray,
    horizon_days: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Statsforecast AutoETS fallback when Chronos is unavailable.
    Returns (point_forecast, lower_80, upper_80).
    """
    try:
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        n = len(series_values)
        df_sf = pd.DataFrame(
            {
                "unique_id": ["s1"] * n,
                "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                "y": series_values.tolist(),
            }
        )
        sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df_sf)
        pred = sf.predict(h=horizon_days, level=[80])
        point = pred["AutoETS"].tolist()
        lower = pred["AutoETS-lo-80"].tolist()
        upper = pred["AutoETS-hi-80"].tolist()
        return point, lower, upper
    except Exception:
        # Last resort: naïve seasonal (lag-7) with fixed ±15% intervals
        last = series_values[-7:] if len(series_values) >= 7 else series_values
        point = np.tile(last, math.ceil(horizon_days / len(last)))[:horizon_days].tolist()
        lower = (np.array(point) * 0.85).tolist()
        upper = (np.array(point) * 1.15).tolist()
        return point, lower, upper


def _run_chronos(
    series_values: np.ndarray,
    horizon_days: int,
    model_id: str,
) -> tuple[list[float], list[float], list[float]]:
    """
    Run Chronos foundation model forecast.
    Falls back to statsforecast on any import/CUDA error.
    """
    try:
        import torch
        from chronos import ChronosPipeline

        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        context = torch.tensor(series_values[np.newaxis, :], dtype=torch.float32)
        quantiles, mean = pipeline.predict_quantiles(
            context,
            prediction_length=horizon_days,
            quantile_levels=[0.1, 0.5, 0.9],
            num_samples=50,
        )
        point = mean[0].numpy().tolist()
        lower = quantiles[0, :, 0].numpy().tolist()
        upper = quantiles[0, :, 2].numpy().tolist()
        return point, lower, upper
    except Exception:
        # Chronos not installed or no GPU — graceful degradation
        return _run_statsforecast_fallback(series_values, horizon_days)


def forecaster_node(state: AgentState) -> dict[str, Any]:
    """
    Generates forecasts for each series using the strategy chosen by the Planner.
    """

    strategy: PlannerStrategy = state["strategy"]
    series_data = pl.from_dict(state["series_data"])
    horizon_days = HORIZON_DAYS[strategy.horizon]
    context_days = int(horizon_days * strategy.context_multiplier)

    model_id = strategy.model_variant.value
    use_chronos = model_id.startswith("amazon/")

    forecasts: list[ForecastResult] = []
    forecast_date = date.today()

    for series_id in series_data["series_id"].unique().to_list():
        series_df = (
            series_data.filter(pl.col("series_id") == series_id).sort("date").tail(context_days)
        )
        values = series_df["value"].to_numpy()
        # Derive category from sign column (synthetic data has no "category" column)
        sign = series_df["sign"][0] if "sign" in series_df.columns else "inflow"
        category = CategoryType.INCOME_RECURRING if sign == "inflow" else CategoryType.EXPENSE_FIXED

        if use_chronos:
            point, lower, upper = _run_chronos(values, horizon_days, model_id)
        else:
            point, lower, upper = _run_statsforecast_fallback(values, horizon_days)

        # Clamp so lower ≤ point ≤ upper — statsforecast can produce asymmetric intervals
        lower = [min(lo, pt) for lo, pt in zip(lower, point, strict=False)]
        upper = [max(hi, pt) for hi, pt in zip(upper, point, strict=False)]

        forecasts.append(
            ForecastResult(
                series_id=series_id,
                category=category,
                forecast_date=forecast_date,
                horizon=strategy.horizon,
                point_forecast=point,
                lower_80=lower,
                upper_80=upper,
                model_used=strategy.model_variant,
                forecast_steps=horizon_days,
            )
        )

    return {"forecasts": forecasts}


# ── Evaluator Node ────────────────────────────────────────────────────────────


def evaluator_node(state: AgentState) -> dict[str, Any]:
    """
    Runs the grader suite against available actuals.
    If no actuals yet (first cycle in simulation), generates synthetic actuals.
    """
    forecasts: list[ForecastResult] = state["forecasts"]
    actuals_raw: dict | None = state.get("actuals")
    series_data = pl.from_dict(state["series_data"])
    strategy: PlannerStrategy = state["strategy"]
    horizon_days = HORIZON_DAYS[strategy.horizon]

    # Build train arrays for MASE scaling
    train_arrays: dict[str, np.ndarray] = {}
    for sid in series_data["series_id"].unique().to_list():
        train_arrays[sid] = (
            series_data.filter(pl.col("series_id") == sid)
            .sort("date")
            .head(-horizon_days if horizon_days < len(series_data) else len(series_data))["value"]
            .to_numpy()
        )

    harness = EvalHarness(
        train_data_by_series=train_arrays,
        baseline_mase=0.85,
    )

    # Actuals: use provided or simulate from tail of training data
    actuals_by_series: dict[str, np.ndarray] = {}
    if actuals_raw:
        actuals_by_series = {k: np.array(v) for k, v in actuals_raw.items()}
    else:
        # Simulation mode: use the last horizon_days of training as pseudo-actuals
        for sid in series_data["series_id"].unique().to_list():
            tail = (
                series_data.filter(pl.col("series_id") == sid)
                .sort("date")
                .tail(horizon_days)["value"]
                .to_numpy()
            )
            actuals_by_series[sid] = tail

    cycle_id = state.get("cycle_id") or str(uuid.uuid4())[:8]
    report = harness.run(
        cycle_id=cycle_id,
        forecast_date=date.today(),
        forecasts=forecasts,
        actuals_by_series=actuals_by_series,
    )

    return {"eval_report": report, "eval_history": [report]}


# ── Learner Node ──────────────────────────────────────────────────────────────


LEARNER_SYSTEM_PROMPT = """You are a forecasting strategy advisor.
You receive a structured evaluation report from a cash flow forecasting agent
and must output a JSON object updating the forecasting strategy.

Rules:
- If MASE > 0.9: suggest switching to a larger Chronos model or increasing context_multiplier
- If SMAPE > 20%: suggest enabling use_covariates or decompose_by_category
- If DirectionalAccuracy < 55%: suggest increasing context_multiplier
- If Coverage80 < 70%: the model is overconfident — suggest a larger model
- If drift_ratio > 1.2: flag drift and set drift_detected=true
- drift_triggered_finetune should only be true if drift_ratio > 1.4
- Always return valid JSON matching the schema

Output JSON schema:
{
  "horizon": "7d|14d|30d|90d",
  "model_variant": "amazon/chronos-t5-tiny|amazon/chronos-t5-small|amazon/chronos-t5-mini|AutoARIMA|AutoETS",
  "context_multiplier": float (1.0-10.0),
  "feature_flags": {"use_covariates": bool, "use_calendar_features": bool, "decompose_by_category": bool},
  "rationale": "string",
  "drift_detected": bool,
  "drift_triggered_finetune": bool,
  "reflection_text": "string (2-3 sentences explaining reasoning)",
  "strategy_changes": ["list", "of", "changes"]
}"""


def _get_learner_policy(policy_name: str) -> LearnerPolicy:
    """Instantiate a learner policy from the registry."""
    policy_cls = POLICY_REGISTRY.get(policy_name)
    if policy_cls is None:
        policy_cls = RuleBasedPolicy
    return policy_cls()


def _generate_reflection(
    report: EvalReport, strategy: PlannerStrategy, cycle_count: int, max_cycles: int
) -> str:
    """Generate LLM reflection text (optional enrichment, not used for strategy selection)."""
    user_message = f"""Current cycle {cycle_count + 1}/{max_cycles}.

Eval report:
- MASE: {report.overall_mase}
- SMAPE: {report.overall_smape}%
- DirectionalAccuracy: {report.directional_accuracy}%
- Coverage80: {report.coverage_80}%
- DriftRatio: {report.drift_ratio}
- AllPassed: {report.all_passed}

Current strategy:
- horizon: {strategy.horizon.value}
- model_variant: {strategy.model_variant.value}
- context_multiplier: {strategy.context_multiplier}
- feature_flags: {strategy.feature_flags}

Summarize in 2-3 sentences what happened this cycle and what should change."""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            f"Cycle eval: MASE={report.overall_mase:.3f}, SMAPE={report.overall_smape:.1f}%, "
            f"Dir={report.directional_accuracy:.1f}%, Cov={report.coverage_80:.1f}%."
        )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=LEARNER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text.strip()
    except Exception:
        return (
            f"Cycle eval: MASE={report.overall_mase:.3f}, SMAPE={report.overall_smape:.1f}%, "
            f"Dir={report.directional_accuracy:.1f}%, Cov={report.coverage_80:.1f}%."
        )


def learner_node(state: AgentState) -> dict[str, Any]:
    """
    Reflects on the eval report and updates the forecasting strategy.
    Delegates strategy selection to pluggable LearnerPolicy (rule_based, bandit, rl).
    Optionally enriches with LLM reflection text.
    """
    report: EvalReport = state["eval_report"]
    strategy: PlannerStrategy = state["strategy"]
    cycle_count = state.get("cycle_count", 0)
    max_cycles = state.get("max_cycles", 5)
    policy_name = state.get("learner_policy_name", "rule_based")

    policy = _get_learner_policy(policy_name)

    # Update policy with current cycle's outcome
    policy.update(report, strategy)

    # Select next strategy
    new_strategy = policy.select_strategy(report, strategy, cycle_count)

    # Generate reflection text (LLM enrichment, decoupled from strategy)
    reflection_text = _generate_reflection(report, strategy, cycle_count, max_cycles)

    drift_detected = report.drift_ratio > 1.2
    drift_finetune = report.drift_ratio > 1.4
    changes = _diff_strategies(strategy, new_strategy)

    feedback = LearnerFeedback(
        cycle_id=report.cycle_id,
        updated_strategy=new_strategy,
        drift_detected=drift_detected,
        drift_triggered_finetune=drift_finetune,
        reflection_text=reflection_text,
        strategy_changes=changes,
    )

    terminate = cycle_count + 1 >= max_cycles or (report.all_passed and cycle_count >= 2)

    return {
        "learner_feedback": feedback,
        "strategy": new_strategy,
        "cycle_count": cycle_count + 1,
        "terminate": terminate,
    }


def _diff_strategies(old: PlannerStrategy, new: PlannerStrategy) -> list[str]:
    diffs = []
    if old.model_variant != new.model_variant:
        diffs.append(f"model: {old.model_variant.value} → {new.model_variant.value}")
    if abs(old.context_multiplier - new.context_multiplier) > 0.01:
        diffs.append(f"context_multiplier: {old.context_multiplier} → {new.context_multiplier}")
    if old.feature_flags != new.feature_flags:
        diffs.append(f"feature_flags: {old.feature_flags} → {new.feature_flags}")
    return diffs if diffs else ["No strategy change"]
