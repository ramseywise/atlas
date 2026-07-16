"""Tests for core/data/features.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.preprocessing.features import (
    add_calendar_features,
    add_ewm_features,
    add_lag_features,
    add_rolling_features,
    build_feature_matrix,
    correlation_importance,
    feature_names_for_ml,
)
from core.preprocessing.synthetic import generate_sequence_dataset


@pytest.fixture(scope="module")
def seq_df() -> pl.DataFrame:
    return generate_sequence_dataset(n_days=365, seed=42)


class TestLagFeatures:
    def test_lag_columns_added(self, seq_df):
        out = add_lag_features(seq_df, value_col="value", lags=[1, 7])
        assert "lag_1d" in out.columns
        assert "lag_7d" in out.columns

    def test_no_lookahead(self, seq_df):
        out = add_lag_features(seq_df, value_col="value", lags=[1])
        for sid in out["series_id"].unique().to_list():
            s = out.filter(pl.col("series_id") == sid).sort("date")
            vals = s["value"].to_numpy()
            lags = s["lag_1d"].to_numpy()
            for i in range(1, min(20, len(vals))):
                if not (np.isnan(lags[i]) or np.isnan(vals[i - 1])):
                    assert abs(lags[i] - vals[i - 1]) < 1e-6

    def test_first_row_is_nan(self, seq_df):
        for sid in seq_df["series_id"].unique().to_list():
            out = add_lag_features(seq_df.filter(pl.col("series_id") == sid), lags=[1])
            assert np.isnan(out.sort("date")["lag_1d"][0])


class TestRollingFeatures:
    def test_rolling_columns_added(self, seq_df):
        out = add_rolling_features(seq_df, windows=[7, 30])
        assert "roll_mean_7d" in out.columns
        assert "roll_std_30d" in out.columns

    def test_no_future_leakage(self, seq_df):
        # roll_mean_7d at position i uses only positions [i-6..i]
        for sid in seq_df["series_id"].unique().to_list()[:1]:
            s = add_rolling_features(
                seq_df.filter(pl.col("series_id") == sid), windows=[7]
            ).sort("date")
            vals = s["value"].to_numpy()
            means = s["roll_mean_7d"].to_numpy()
            for i in range(6, min(20, len(vals))):
                expected = np.mean(vals[i - 6 : i + 1])
                assert abs(means[i] - expected) < 1e-4


class TestEWMFeatures:
    def test_ewm_columns_added(self, seq_df):
        out = add_ewm_features(seq_df, spans=[7])
        assert "ewm_7d" in out.columns

    def test_ewm_values_in_range(self, seq_df):
        out = add_ewm_features(seq_df.filter(pl.col("series_id") == "erp_revenue"), spans=[7])
        ewm = out["ewm_7d"].drop_nulls().to_numpy()
        # EWM of positive values should stay positive
        assert (ewm >= 0).all()


class TestCalendarFeatures:
    def test_calendar_columns_added(self, seq_df):
        out = add_calendar_features(seq_df)
        for col in ["day_of_week", "day_of_month", "month", "quarter", "is_month_end"]:
            assert col in out.columns

    def test_fourier_columns_added(self, seq_df):
        out = add_calendar_features(seq_df, include_fourier=True)
        assert "weekly_sin_1" in out.columns
        assert "annual_cos_2" in out.columns

    def test_fourier_bounded(self, seq_df):
        out = add_calendar_features(seq_df, include_fourier=True)
        for col in ["weekly_sin_1", "weekly_cos_1"]:
            vals = out[col].to_numpy()
            assert vals.min() >= -1.01
            assert vals.max() <= 1.01


class TestCorrelationImportance:
    def test_returns_ranked_features(self, seq_df):
        s = seq_df.filter(pl.col("series_id") == "erp_revenue").sort("date")
        out = add_lag_features(s, lags=[1, 7, 14]).drop_nulls()
        feature_cols = ["lag_1d", "lag_7d", "lag_14d"]
        X = out.select(feature_cols).to_numpy()
        y = out["value"].to_numpy()
        result = correlation_importance(X, y, feature_cols)
        assert len(result.ranked) == 3
        assert result.ranked[0][1] >= result.ranked[-1][1]

    def test_importances_non_negative(self, seq_df):
        s = seq_df.filter(pl.col("series_id") == "erp_revenue").sort("date")
        out = add_lag_features(s, lags=[1, 7]).drop_nulls()
        X = out.select(["lag_1d", "lag_7d"]).to_numpy()
        y = out["value"].to_numpy()
        result = correlation_importance(X, y, ["lag_1d", "lag_7d"])
        assert all(v >= 0.0 for v in result.importances.values())


class TestBuildFeatureMatrix:
    def test_full_pipeline_no_crash(self, seq_df):
        out = build_feature_matrix(seq_df, value_col="value")
        assert len(out) == len(seq_df)

    def test_feature_names_excludes_non_features(self, seq_df):
        out = build_feature_matrix(seq_df, value_col="value")
        names = feature_names_for_ml(out)
        assert "date" not in names
        assert "value" not in names
        assert len(names) > 10  # should have substantial feature set
