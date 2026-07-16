"""Tests for core/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.preprocessing.preprocessing import (
    Preprocessor,
    ScalerParams,
    apply_scaler,
    detect_outliers,
    difference_series,
    fill_gaps,
    fit_scaler,
    inverse_scaler,
    check_stationarity,
    treat_outliers,
    undifference_series,
)
from core.preprocessing.synthetic import generate_sequence_dataset


@pytest.fixture(scope="module")
def sample_series() -> pl.DataFrame:
    df = generate_sequence_dataset(n_days=365, seed=42)
    return df.rename({"value": "amount"})


# ── Gap filling ───────────────────────────────────────────────────────────────


class TestFillGaps:
    def test_forward_fill_no_crash(self, sample_series):
        filled, gaps = fill_gaps(sample_series, value_col="amount")
        assert filled["amount"].null_count() == 0

    def test_interpolate_no_crash(self, sample_series):
        filled, gaps = fill_gaps(sample_series, value_col="amount", method="interpolate")
        assert filled["amount"].null_count() == 0

    def test_gap_count_reported(self, sample_series):
        _, gaps = fill_gaps(sample_series, value_col="amount")
        assert isinstance(gaps, dict)
        assert all(isinstance(v, int) for v in gaps.values())


# ── Outlier detection ─────────────────────────────────────────────────────────


class TestOutlierDetection:
    def test_iqr_detects_spike(self):
        vals = np.ones(100) * 100.0
        vals[50] = 10_000.0  # spike
        mask = detect_outliers(vals, method="iqr")
        assert mask[50], "Spike should be detected"
        assert mask.sum() < 10, "Too many false positives"

    def test_zscore_detects_outlier(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(100, 5, 200)
        vals[100] = 500.0
        mask = detect_outliers(vals, method="zscore")
        assert mask[100]

    def test_winsorise_clips_values(self):
        vals = np.array([1.0, 2.0, 3.0, 1000.0, 2.0, 1.5])
        mask = detect_outliers(vals, method="iqr")
        treated = treat_outliers(vals, mask, treatment="winsorise")
        assert treated.max() < 1000.0

    def test_flag_only_preserves_values(self):
        vals = np.array([1.0, 2.0, 3.0, 1000.0])
        mask = np.array([False, False, False, True])
        treated = treat_outliers(vals, mask, treatment="flag_only")
        assert treated[3] == 1000.0


# ── Stationarity ──────────────────────────────────────────────────────────────


class TestStationarity:
    def test_stationary_series(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(100, 5, 300)
        report = check_stationarity(vals, series_id="test")
        assert report.recommended_d == 0
        assert "stationary" in report.conclusion.lower()

    def test_trending_series(self):
        vals = np.arange(300, dtype=float) + np.random.default_rng(1).normal(0, 1, 300)
        report = check_stationarity(vals, series_id="trend")
        assert report.recommended_d >= 1

    def test_difference_roundtrip(self):
        rng = np.random.default_rng(42)
        vals = np.cumsum(rng.normal(0, 1, 100)) + 100.0
        diff = difference_series(vals, d=1)
        # Undifference should recover approximately
        recovered = undifference_series(diff[1:], vals[:1], d=1)
        assert np.allclose(recovered, vals[1:], atol=1.0)


# ── Scaling ───────────────────────────────────────────────────────────────────


class TestScaling:
    def test_log1p_roundtrip(self):
        vals = np.array([100.0, 200.0, 500.0, 1000.0])
        params = fit_scaler(vals, "log1p")
        scaled = apply_scaler(vals, params)
        recovered = inverse_scaler(scaled, params)
        assert np.allclose(vals, recovered, rtol=1e-5)

    def test_standard_roundtrip(self):
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        params = fit_scaler(vals, "standard")
        scaled = apply_scaler(vals, params)
        recovered = inverse_scaler(scaled, params)
        assert np.allclose(vals, recovered, rtol=1e-5)

    def test_minmax_range(self):
        vals = np.array([0.0, 50.0, 100.0])
        params = fit_scaler(vals, "minmax")
        scaled = apply_scaler(vals, params)
        assert scaled.min() >= -0.01 and scaled.max() <= 1.01

    def test_no_negative_after_inverse(self):
        vals = np.array([100.0, 500.0, 1000.0])
        params = fit_scaler(vals, "log1p")
        scaled = apply_scaler(vals, params)
        # Slightly out-of-distribution
        recovered = inverse_scaler(scaled - 5.0, params)
        assert recovered.min() >= 0.0


# ── Preprocessor (fit/transform) ─────────────────────────────────────────────


class TestPreprocessor:
    def test_fit_transform_no_crash(self, sample_series):
        pp = Preprocessor(value_col="amount")
        result = pp.fit_transform(sample_series)
        assert result.df is not None
        assert len(result.reports) > 0

    def test_transform_after_fit(self, sample_series):
        split_date = sample_series["date"].sort()[int(len(sample_series["date"].unique()) * 0.8)]
        train = sample_series.filter(pl.col("date") <= split_date)
        val = sample_series.filter(pl.col("date") > split_date)
        pp = Preprocessor(value_col="amount")
        pp.fit_transform(train)
        transformed_val = pp.transform(val)
        assert len(transformed_val) > 0

    def test_transform_before_fit_raises(self, sample_series):
        pp = Preprocessor(value_col="amount")
        with pytest.raises(RuntimeError, match="fit_transform"):
            pp.transform(sample_series)

    def test_inverse_transform_restores_scale(self, sample_series):
        # flag_only skips winsorisation so inverse must exactly round-trip
        pp = Preprocessor(value_col="amount", scale_method="log1p", outlier_treatment="flag_only")
        result = pp.fit_transform(sample_series)
        for sid in sample_series["series_id"].unique().to_list():
            orig = sample_series.filter(pl.col("series_id") == sid)["amount"].to_numpy()
            scaled = result.df.filter(pl.col("series_id") == sid)["amount"].to_numpy()
            recovered = pp.inverse_transform_array(sid, scaled)
            # Only compare rows where orig is non-zero — zeros are imputed gaps
            mask = orig > 0
            assert np.allclose(orig[mask], recovered[mask], rtol=0.01)

    def check_stationarity_summary_returns_df(self, sample_series):
        pp = Preprocessor(value_col="amount")
        pp.fit_transform(sample_series)
        df = pp.stationarity_summary()
        assert "series_id" in df.columns
        assert "recommended_d" in df.columns
