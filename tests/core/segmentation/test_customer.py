"""Unit tests for core/preprocessing/customer.py."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.preprocessing.customer import (
    CustomerProfile,
    build_customer_profiles,
)
from core.preprocessing.synthetic import generate_sequence_dataset


@pytest.fixture(scope="module")
def canonical_df():
    """Single-customer canonical DataFrame derived from synthetic data."""
    df = generate_sequence_dataset(n_days=365, seed=42)
    return df.with_columns(
        [
            pl.lit("cust-001").alias("customer_id"),
            pl.col("source").alias("source"),
            pl.col("sign").alias("sign"),
            pl.col("value").alias("amount"),
        ]
    )


class TestBuildCustomerProfiles:
    def test_returns_one_profile_per_customer(self, canonical_df):
        profiles = build_customer_profiles(canonical_df)
        assert "cust-001" in profiles
        assert len(profiles) == 1

    def test_profile_fields_populated(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        assert profile.total_inflow > 0
        assert profile.total_outflow >= 0
        assert profile.n_active_series > 0
        assert profile.daily_net_std >= 0

    def test_inflow_share_in_range(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        assert 0.0 <= profile.inflow_share <= 1.0

    def test_activity_recency_in_range(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        assert 0.0 <= profile.activity_last_30d <= 1.0
        assert 0.0 <= profile.activity_last_90d <= 1.0

    def test_multiple_customers(self):
        frames = []
        for i in range(3):
            df = generate_sequence_dataset(n_days=200, seed=i)
            frames.append(
                df.with_columns(
                    [
                        pl.lit(f"cust-{i:03d}").alias("customer_id"),
                        pl.col("source").alias("source"),
                        pl.col("sign").alias("sign"),
                        pl.col("value").alias("amount"),
                    ]
                )
            )
        multi_df = pl.concat(frames)
        profiles = build_customer_profiles(multi_df)
        assert len(profiles) == 3


class TestCustomerProfileVector:
    def test_feature_vector_shape(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        vec = profile.to_feature_vector()
        assert vec.shape == (len(CustomerProfile.feature_names()),)
        assert vec.dtype == np.float32

    def test_no_nan_in_vector(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        vec = profile.to_feature_vector()
        assert not np.any(np.isnan(vec))

    def test_feature_names_length_matches_vector(self, canonical_df):
        profile = build_customer_profiles(canonical_df)["cust-001"]
        assert len(CustomerProfile.feature_names()) == len(profile.to_feature_vector())
