"""
State schemas for the crypto forecasting agent.
"""

from __future__ import annotations

import operator
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from src.agents.state import ModelVariant


class CryptoTimeframe(StrEnum):
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class PredictionType(StrEnum):
    DIRECTION = "direction"
    SPREAD = "spread"
    ABSOLUTE = "absolute"


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


TIMEFRAME_HOURS: dict[CryptoTimeframe, int] = {
    CryptoTimeframe.H1: 1,
    CryptoTimeframe.H4: 4,
    CryptoTimeframe.D1: 24,
    CryptoTimeframe.W1: 168,
}


class CryptoPlannerStrategy(BaseModel):
    """Strategy configuration for the crypto forecasting agent."""

    timeframe: CryptoTimeframe = CryptoTimeframe.D1
    model_variant: ModelVariant = ModelVariant.CHRONOS_TINY
    prediction_types: list[PredictionType] = Field(
        default_factory=lambda: [PredictionType.DIRECTION, PredictionType.ABSOLUTE]
    )
    context_bars: int = Field(default=90, ge=30, le=500)
    forecast_bars: int = Field(default=7, ge=1, le=90)
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT"])
    use_indicators: bool = True
    rationale: str = ""


class CryptoPrediction(BaseModel):
    """A single prediction output."""

    symbol: str
    prediction_type: PredictionType
    timestamp: datetime
    direction: Direction | None = None
    direction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    spread_value: float | None = None
    spread_pair: str | None = None
    point_forecast: list[float] | None = None
    lower_80: list[float] | None = None
    upper_80: list[float] | None = None
    model_used: ModelVariant = ModelVariant.CHRONOS_TINY


class CryptoEvalReport(BaseModel):
    """Evaluation metrics for crypto predictions."""

    cycle_id: str
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    directional_accuracy: float
    mase: float | None = None
    coverage_80: float | None = None
    all_passed: bool
    summary: str = ""


class CryptoLearnerFeedback(BaseModel):
    """Learner output for crypto agent."""

    cycle_id: str
    updated_strategy: CryptoPlannerStrategy
    drift_detected: bool = False
    reflection_text: str = ""
    strategy_changes: list[str] = Field(default_factory=list)


class CryptoAgentState(TypedDict):
    """LangGraph state for the crypto forecasting agent."""

    cycle_id: str
    ohlcv_data: dict
    symbols: list[str]

    strategy: CryptoPlannerStrategy | None
    predictions: list[CryptoPrediction]
    eval_report: CryptoEvalReport | None
    learner_feedback: CryptoLearnerFeedback | None

    strategy_history: Annotated[list[CryptoPlannerStrategy], operator.add]
    eval_history: Annotated[list[CryptoEvalReport], operator.add]

    cycle_count: int
    max_cycles: int
    terminate: bool
    error: str | None
    learner_policy_name: str
