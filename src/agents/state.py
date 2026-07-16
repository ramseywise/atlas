"""
State schemas for the cash flow forecasting agent.
All inter-agent data flows through these typed models — no loose dicts.
"""

from __future__ import annotations

import operator
from datetime import date
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypedDict

# ── Enums ────────────────────────────────────────────────────────────────────


class ForecastHorizon(str, Enum):
    WEEK = "7d"
    FORTNIGHT = "14d"
    MONTH = "30d"
    QUARTER = "90d"


class ModelVariant(str, Enum):
    CHRONOS_TINY = "amazon/chronos-t5-tiny"
    CHRONOS_SMALL = "amazon/chronos-t5-small"
    CHRONOS_MINI = "amazon/chronos-t5-mini"
    STATSFORECAST_ARIMA = "AutoARIMA"
    STATSFORECAST_ETS = "AutoETS"


class CategoryType(str, Enum):
    INCOME_RECURRING = "income_recurring"
    INCOME_VARIABLE = "income_variable"
    EXPENSE_FIXED = "expense_fixed"
    EXPENSE_DISCRETIONARY = "expense_discretionary"


# ── Sub-models ───────────────────────────────────────────────────────────────


class PlannerStrategy(BaseModel):
    """Planner's selected configuration for the current cycle."""

    horizon: ForecastHorizon = ForecastHorizon.MONTH
    model_variant: ModelVariant = ModelVariant.CHRONOS_TINY
    context_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="How many horizon-lengths of history to feed as context",
    )
    feature_flags: dict[str, bool] = Field(
        default_factory=lambda: {
            "use_covariates": False,
            "use_calendar_features": True,
            "decompose_by_category": True,
        }
    )
    rationale: str = ""


class ForecastResult(BaseModel):
    """Output of the Forecaster agent for a single series."""

    series_id: str
    category: CategoryType
    forecast_date: date
    horizon: ForecastHorizon
    point_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    model_used: ModelVariant
    forecast_steps: int

    @model_validator(mode="after")
    def check_interval_consistency(self) -> ForecastResult:
        for lo, pt, hi in zip(self.lower_80, self.point_forecast, self.upper_80, strict=False):
            if not (lo <= pt <= hi):
                raise ValueError(
                    f"Interval violated: lower={lo:.2f} > point={pt:.2f} or point > upper={hi:.2f}"
                )
        return self


class GraderScore(BaseModel):
    """Single grader result."""

    grader_name: str
    metric_value: float
    threshold: float
    passed: bool
    detail: str = ""


class EvalReport(BaseModel):
    """Aggregated evaluation results for one forecast cycle."""

    cycle_id: str
    forecast_date: date
    series_scores: dict[str, list[GraderScore]]  # series_id -> grader scores
    overall_mase: float
    overall_smape: float
    directional_accuracy: float
    coverage_80: float
    drift_ratio: float  # current MASE / rolling baseline MASE
    all_passed: bool
    summary: str = ""


class LearnerFeedback(BaseModel):
    """Structured output of the Learner agent after reflection."""

    cycle_id: str
    updated_strategy: PlannerStrategy
    drift_detected: bool
    drift_triggered_finetune: bool
    reflection_text: str
    strategy_changes: list[str]  # human-readable diff of what changed


# ── LangGraph AgentState ─────────────────────────────────────────────────────


class AgentState(TypedDict):
    """
    The single source of truth passed between all graph nodes.
    LangGraph requires TypedDict; we use Annotated for reduction semantics.
    """

    # Inputs
    cycle_id: str
    series_data: dict  # polars DataFrame serialized as dict — deserialized per node
    actuals: dict | None  # for eval: actual values for the forecast period

    # Planner output
    strategy: PlannerStrategy | None

    # Forecaster output
    forecasts: list[ForecastResult]

    # Evaluator output
    eval_report: EvalReport | None

    # Learner output
    learner_feedback: LearnerFeedback | None

    # Cycle history — list accumulates across cycles via operator.add
    # Nodes must return these as single-element lists: {"eval_history": [report]}
    strategy_history: Annotated[list[PlannerStrategy], operator.add]
    eval_history: Annotated[list[EvalReport], operator.add]

    # Control flow
    cycle_count: int
    max_cycles: int
    terminate: bool
    error: str | None
