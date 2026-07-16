"""Tests for core/models/arima.py — ARIMA assumptions and forecasting."""

from __future__ import annotations

import numpy as np
import pytest

from evals.arima import (
    ARIMAForecaster,
    ARIMAForecastResult,
    AssumptionReport,
    check_arima_assumptions,
)


@pytest.fixture
def stationary_series() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(1000, 50, 400)


@pytest.fixture
def trending_series() -> np.ndarray:
    rng = np.random.default_rng(1)
    return np.cumsum(rng.normal(10, 5, 400)) + 1000.0


@pytest.fixture
def seasonal_series() -> np.ndarray:
    t = np.arange(400, dtype=float)
    return 1000 + 200 * np.sin(2 * np.pi * t / 7) + np.random.default_rng(2).normal(0, 20, 400)


class TestAssumptionChecks:
    def test_stationary_series_passes(self, stationary_series):
        report = check_arima_assumptions(stationary_series, series_id="test")
        assert isinstance(report, AssumptionReport)
        assert report.stationarity.recommended_d == 0
        assert "stationary" in report.stationarity.conclusion.lower()

    def test_trending_series_needs_differencing(self, trending_series):
        report = check_arima_assumptions(trending_series, series_id="trend")
        assert report.stationarity.recommended_d >= 1

    def test_insufficient_obs_flagged(self):
        vals = np.random.default_rng(0).normal(100, 5, 10)  # only 10 obs
        report = check_arima_assumptions(vals, seasonal_period=7)
        assert not report.min_obs_met
        assert any("Insufficient" in v for v in report.violations)

    def test_recommendation_present(self, stationary_series):
        report = check_arima_assumptions(stationary_series)
        assert len(report.recommendation) > 0


class TestARIMAForecaster:
    def test_auto_fit_predict(self, seasonal_series):
        forecaster = ARIMAForecaster("test", auto=True)
        forecaster.fit(seasonal_series)
        result = forecaster.predict(horizon=30)
        assert isinstance(result, ARIMAForecastResult)
        assert len(result.point_forecast) == 30
        assert len(result.lower_80) == 30
        assert len(result.upper_80) == 30

    def test_forecasts_non_negative(self, seasonal_series):
        forecaster = ARIMAForecaster("test", auto=True)
        forecaster.fit(seasonal_series)
        result = forecaster.predict(horizon=30)
        assert all(p >= 0.0 for p in result.point_forecast)
        assert all(lo >= 0.0 for lo in result.lower_80)

    def test_interval_ordering(self, seasonal_series):
        forecaster = ARIMAForecaster("test", auto=True)
        forecaster.fit(seasonal_series)
        result = forecaster.predict(horizon=30)
        for lo, pt, hi in zip(result.lower_80, result.point_forecast, result.upper_80):
            assert lo <= pt <= hi, f"Interval violated: lo={lo}, pt={pt}, hi={hi}"

    def test_predict_before_fit_raises(self):
        forecaster = ARIMAForecaster("test")
        with pytest.raises(RuntimeError, match="fit"):
            forecaster.predict(horizon=10)

    def test_check_assumptions_called_directly(self, stationary_series):
        forecaster = ARIMAForecaster("test", auto=False)
        report = forecaster.check_assumptions(stationary_series)
        assert report.series_id == "test"

    def test_comparison_vs_naive_string(self, seasonal_series):
        forecaster = ARIMAForecaster("test", auto=True)
        forecaster.fit(seasonal_series)
        result = forecaster.predict(horizon=30)
        summary = result.comparison_vs_naive()
        assert "Model" in summary

    def test_comparison_with_actuals(self, seasonal_series):
        forecaster = ARIMAForecaster("test", auto=True)
        forecaster.fit(seasonal_series[:300])
        result = forecaster.predict(horizon=30)
        actuals = seasonal_series[300:330]
        summary = result.comparison_vs_naive(actuals)
        assert "MASE" in summary
