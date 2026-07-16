"""
Smoke tests for the LangGraph agent loop.

No GPU or Chronos required — AutoETS fallback is used automatically.
These tests verify the loop mechanics, state transitions, and schema
validation without requiring external model dependencies.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from core.preprocessing.synthetic import generate_sequence_dataset
from src.agents.state import (
    CategoryType,
    ForecastHorizon,
    ForecastResult,
    ModelVariant,
    PlannerStrategy,
)


@pytest.fixture(scope="module")
def small_df():
    # All 7 series × 365 days — fast enough for smoke tests
    return generate_sequence_dataset(n_days=365, seed=42)


class TestPydanticStateSchemas:
    def test_forecast_result_interval_validation(self):
        with pytest.raises(ValidationError):
            ForecastResult(
                series_id="test",
                category=CategoryType.INCOME_RECURRING,
                forecast_date=date(2024, 1, 1),
                horizon=ForecastHorizon.MONTH,
                point_forecast=[100.0],
                lower_80=[200.0],  # lower > point — must raise
                upper_80=[300.0],
                model_used=ModelVariant.CHRONOS_TINY,
                forecast_steps=1,
            )

    def test_planner_strategy_defaults(self):
        s = PlannerStrategy()
        assert s.horizon == ForecastHorizon.MONTH
        assert 1.0 <= s.context_multiplier <= 10.0

    def test_planner_strategy_context_multiplier_bounds(self):
        with pytest.raises(ValidationError):
            PlannerStrategy(context_multiplier=0.5)  # below min=1.0


class TestAgentLoopSmoke:
    def test_single_cycle_completes(self, small_df):
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(series_df=small_df, max_cycles=1, verbose=False)
        assert final["cycle_count"] == 1
        assert final["eval_report"] is not None
        assert len(final["forecasts"]) > 0

    def test_terminates_on_max_cycles(self, small_df):
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(series_df=small_df, max_cycles=2, verbose=False)
        assert final["cycle_count"] <= 2
        assert final["terminate"] is True

    def test_strategy_history_accumulates(self, small_df):
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(series_df=small_df, max_cycles=2, verbose=False)
        assert len(final["strategy_history"]) >= 1

    def test_eval_report_fields_present(self, small_df):
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(series_df=small_df, max_cycles=1, verbose=False)
        report = final["eval_report"]
        assert report is not None
        assert hasattr(report, "overall_mase")
        assert hasattr(report, "all_passed")
        assert report.overall_mase >= 0

    def test_no_error_field(self, small_df):
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(series_df=small_df, max_cycles=1, verbose=False)
        assert final.get("error") is None
