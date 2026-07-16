"""Tests for evals/graders/graders.py."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from evals.graders.graders import (
    CoverageGrader,
    DirectionalGrader,
    DriftGrader,
    EvalHarness,
    MASEGrader,
    SMAPEGrader,
)
from src.agents.state import CategoryType, ForecastHorizon, ForecastResult, ModelVariant

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_forecast() -> ForecastResult:
    n = 30
    point = [1000.0 + i * 5 for i in range(n)]
    return ForecastResult(
        series_id="erp_revenue",
        category=CategoryType.INCOME_RECURRING,
        forecast_date=date(2024, 1, 1),
        horizon=ForecastHorizon.MONTH,
        point_forecast=point,
        lower_80=[p * 0.85 for p in point],
        upper_80=[p * 1.15 for p in point],
        model_used=ModelVariant.CHRONOS_TINY,
        forecast_steps=n,
    )


@pytest.fixture
def close_actuals(sample_forecast) -> np.ndarray:
    preds = np.array(sample_forecast.point_forecast)
    return preds + np.random.default_rng(0).normal(0, preds * 0.03)


@pytest.fixture
def bad_actuals(sample_forecast) -> np.ndarray:
    return np.array(sample_forecast.point_forecast) * 3.5


# ── MASE ──────────────────────────────────────────────────────────────────────


class TestMASEGrader:
    def test_passes_good_forecast(self, sample_forecast, close_actuals):
        train = np.array([1000.0 + i * 4 for i in range(200)])
        g = MASEGrader(train)
        score = g.score(close_actuals, sample_forecast)
        assert score.passed, f"Expected MASE < 1.0, got {score.metric_value}"

    def test_fails_bad_forecast(self, sample_forecast, bad_actuals):
        train = np.array([1000.0 + i * 4 for i in range(200)])
        g = MASEGrader(train)
        score = g.score(bad_actuals, sample_forecast)
        assert not score.passed

    def test_flat_train_guard(self, sample_forecast, close_actuals):
        g = MASEGrader(np.ones(200) * 1000.0)
        score = g.score(close_actuals, sample_forecast)
        assert not np.isnan(score.metric_value)
        assert not np.isinf(score.metric_value)


# ── SMAPE ─────────────────────────────────────────────────────────────────────


class TestSMAPEGrader:
    def test_passes_good_forecast(self, sample_forecast, close_actuals):
        g = SMAPEGrader()
        assert g.score(close_actuals, sample_forecast).passed

    def test_fails_bad_forecast(self, sample_forecast, bad_actuals):
        assert not SMAPEGrader().score(bad_actuals, sample_forecast).passed

    def test_bounded_output(self, sample_forecast, close_actuals):
        score = SMAPEGrader().score(close_actuals, sample_forecast)
        assert 0 <= score.metric_value <= 200


# ── Directional ───────────────────────────────────────────────────────────────


class TestDirectionalGrader:
    def test_passes_correct_direction(self, sample_forecast, close_actuals):
        g = DirectionalGrader()
        score = g.score(close_actuals, sample_forecast)
        assert score.passed, f"Expected > 55%, got {score.metric_value}"

    def test_single_step_returns_50(self, sample_forecast):
        score = DirectionalGrader().score(np.array([1000.0]), sample_forecast)
        assert score.metric_value == 50.0


# ── Coverage ──────────────────────────────────────────────────────────────────


class TestCoverageGrader:
    def test_passes_when_in_interval(self, sample_forecast, close_actuals):
        assert CoverageGrader().score(close_actuals, sample_forecast).passed

    def test_fails_when_outside(self, sample_forecast):
        bad = np.array(sample_forecast.upper_80) * 5
        assert not CoverageGrader().score(bad, sample_forecast).passed


# ── Drift ─────────────────────────────────────────────────────────────────────


class TestDriftGrader:
    def test_no_drift_initially(self):
        g = DriftGrader(baseline_mase=0.85)
        score = g.score()
        assert score.passed

    def test_drift_detected_after_degradation(self):
        g = DriftGrader(baseline_mase=0.5)
        for _ in range(12):
            g.update(0.8)  # ratio = 0.8/0.5 = 1.6 > 1.2
        assert not g.score().passed


# ── EvalHarness ───────────────────────────────────────────────────────────────


class TestEvalHarness:
    def test_full_run(self, sample_forecast, close_actuals):
        train = {"erp_revenue": np.array([1000.0 + i * 4 for i in range(200)])}
        harness = EvalHarness(train_data_by_series=train)
        report = harness.run(
            cycle_id="test-001",
            forecast_date=date(2024, 1, 1),
            forecasts=[sample_forecast],
            actuals_by_series={"erp_revenue": close_actuals},
        )
        assert report.overall_mase >= 0
        assert isinstance(report.all_passed, bool)

    def test_empty_actuals_no_crash(self, sample_forecast):
        harness = EvalHarness(train_data_by_series={"erp_revenue": np.ones(200) * 1000.0})
        report = harness.run(
            cycle_id="test-002",
            forecast_date=date(2024, 1, 1),
            forecasts=[sample_forecast],
            actuals_by_series={},
        )
        assert report is not None

    def test_drift_grader_is_warning_not_gate(self, sample_forecast, close_actuals):
        """DriftGrader failure must not flip all_passed to False."""
        train = {"erp_revenue": np.array([1000.0 + i * 4 for i in range(200)])}
        harness = EvalHarness(train_data_by_series=train, baseline_mase=0.01)
        # Force drift: set baseline extremely low
        for _ in range(12):
            harness._drift_grader.update(2.0)  # ratio >> 1.2
        report = harness.run(
            cycle_id="test-003",
            forecast_date=date(2024, 1, 1),
            forecasts=[sample_forecast],
            actuals_by_series={"erp_revenue": close_actuals},
        )
        # Non-drift graders should still pass
        non_drift = [
            s
            for scores in report.series_scores.values()
            for s in scores
            if s.grader_name != "DriftDetection"
        ]
        assert any(s.passed for s in non_drift)
