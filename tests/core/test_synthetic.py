"""Tests for core/data/synthetic.py — both ML and sequence dataset variants."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.preprocessing.synthetic import (
    generate_ml_dataset,
    generate_sequence_dataset,
    temporal_split,
    walk_forward_cv,
)

# ── Sequence dataset ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def seq_df() -> pl.DataFrame:
    return generate_sequence_dataset(n_days=730, seed=42)


class TestSequenceDataset:
    def test_reproducibility(self):
        df1 = generate_sequence_dataset(n_days=100, seed=42)
        df2 = generate_sequence_dataset(n_days=100, seed=42)
        assert df1.equals(df2)

    def test_different_seeds_differ(self):
        df1 = generate_sequence_dataset(n_days=100, seed=42)
        df2 = generate_sequence_dataset(n_days=100, seed=99)
        assert not df1.equals(df2)

    def test_schema(self, seq_df):
        required = {"date", "series_id", "source", "sign", "value", "is_anomaly"}
        assert required.issubset(set(seq_df.columns))

    def test_values_non_negative(self, seq_df):
        assert seq_df["value"].min() >= 0.0

    def test_all_pipeline_sources_present(self, seq_df):
        sources = set(seq_df["source"].unique().to_list())
        assert len(sources) >= 5  # at least 5 pipeline sources

    def test_anomaly_rate_in_range(self, seq_df):
        rate = seq_df["is_anomaly"].mean()
        assert 0.01 <= rate <= 0.15

    def test_row_count(self, seq_df):
        n_series = seq_df["series_id"].n_unique()
        assert len(seq_df) == n_series * 730

    def test_date_coverage(self, seq_df):
        n_dates = seq_df["date"].n_unique()
        assert n_dates == 730


# ── ML tabular dataset ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ml_df() -> pl.DataFrame:
    return generate_ml_dataset(n_days=365, horizon_days=30, seed=42)


class TestMLDataset:
    def test_lag_columns_present(self, ml_df):
        for lag in [1, 7, 14, 30]:
            assert f"lag_{lag}d" in ml_df.columns, f"Missing lag_{lag}d"

    def test_rolling_columns_present(self, ml_df):
        for w in [7, 14, 30, 90]:
            assert f"roll_mean_{w}d" in ml_df.columns

    def test_calendar_columns_present(self, ml_df):
        for col in ["day_of_week", "day_of_month", "month", "is_month_end"]:
            assert col in ml_df.columns

    def test_target_column_present(self, ml_df):
        assert "target_30d" in ml_df.columns

    def test_no_lookahead_in_lags(self, ml_df):
        # For each series, lag_1 at row i should equal value at row i-1
        for sid in ml_df["series_id"].unique().sort().to_list():
            s = ml_df.filter(pl.col("series_id") == sid).sort("date")
            vals = s["value"].to_numpy()
            lags = s["lag_1d"].to_numpy()
            # Check non-nan rows
            for i in range(1, min(10, len(vals))):
                if not (np.isnan(lags[i]) or np.isnan(vals[i - 1])):
                    assert abs(lags[i] - vals[i - 1]) < 1e-6, f"Lag mismatch at row {i}"


# ── Temporal splits ───────────────────────────────────────────────────────────


class TestTemporalSplits:
    def test_no_date_leakage(self, seq_df):
        split = temporal_split(seq_df)
        assert split.train["date"].max() < split.val["date"].min()
        assert split.val["date"].max() < split._test["date"].min()

    def test_proportions(self, seq_df):
        split = temporal_split(seq_df, val_frac=0.15, test_frac=0.20)
        total = seq_df["date"].n_unique()
        n_test = split._test["date"].n_unique()
        assert abs(n_test / total - 0.20) < 0.05

    def test_get_test_guard(self, seq_df):
        split = temporal_split(seq_df)
        with pytest.raises(RuntimeError, match="acknowledged=True"):
            split.get_test()

    def test_get_test_with_acknowledgement(self, seq_df):
        split = temporal_split(seq_df)
        test = split.get_test(acknowledged=True)
        assert len(test) > 0

    def test_summary_string(self, seq_df):
        split = temporal_split(seq_df)
        s = split.summary()
        assert "Train" in s and "Val" in s and "HELD OUT" in s


class TestWalkForwardCV:
    def test_val_windows_non_overlapping(self, seq_df):
        split = temporal_split(seq_df)
        folds = walk_forward_cv(split.train, horizon_days=30, min_train_days=180, step_days=30)
        ranges = [(f.val["date"].min(), f.val["date"].max()) for f in folds]
        for i in range(len(ranges) - 1):
            assert ranges[i][1] < ranges[i + 1][0]

    def test_train_expands(self, seq_df):
        split = temporal_split(seq_df)
        folds = walk_forward_cv(split.train, horizon_days=30, min_train_days=180, step_days=30)
        sizes = [f.train["date"].n_unique() for f in folds]
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1]

    def test_at_least_one_fold(self, seq_df):
        split = temporal_split(seq_df)
        folds = walk_forward_cv(split.train, horizon_days=30, min_train_days=180)
        assert len(folds) >= 1
