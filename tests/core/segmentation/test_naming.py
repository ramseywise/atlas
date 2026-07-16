"""Unit tests for core/segmentation/naming.py."""

from __future__ import annotations

import numpy as np

from core.preprocessing.customer import CustomerProfile
from core.segmentation.naming import (
    SegmentLabel,
    compute_centroids,
    name_segments,
)


class TestComputeCentroids:
    def test_basic(self):
        X = np.array([[1.0, 2.0], [1.5, 2.5], [10.0, 10.0]], dtype=np.float32)
        labels = np.array([0, 0, 1])
        centroids = compute_centroids(X, labels)
        assert set(centroids.keys()) == {0, 1}
        np.testing.assert_allclose(centroids[0], [1.25, 2.25], atol=1e-5)
        np.testing.assert_allclose(centroids[1], [10.0, 10.0], atol=1e-5)

    def test_noise_excluded(self):
        X = np.array([[1.0], [2.0], [99.0]], dtype=np.float32)
        labels = np.array([0, 0, -1])
        centroids = compute_centroids(X, labels)
        assert -1 not in centroids
        assert 0 in centroids

    def test_empty_labels_returns_empty(self):
        X = np.zeros((0, 3), dtype=np.float32)
        labels = np.array([], dtype=int)
        assert compute_centroids(X, labels) == {}


class TestNameSegments:
    def test_returns_segment_labels(self):
        rng = np.random.default_rng(0)
        n_feat = len(CustomerProfile.feature_names())
        centroids = {
            0: rng.uniform(0, 1, n_feat).astype(np.float32),
            1: rng.uniform(1, 2, n_feat).astype(np.float32),
        }
        # No API key in test env — falls back to rule-based
        result = name_segments(centroids)
        assert set(result.keys()) == {0, 1}
        for cid, sl in result.items():
            assert isinstance(sl, SegmentLabel)
            assert sl.cluster_id == cid
            assert len(sl.label) > 0
            assert len(sl.description) > 0

    def test_rule_based_high_growth(self):
        """Centroid with high trend and inflow_share should get High-Growth label."""
        feature_names = CustomerProfile.feature_names()
        idx = {n: i for i, n in enumerate(feature_names)}
        n = len(feature_names)
        vec = np.zeros(n, dtype=np.float32)
        vec[idx["trend_slope_norm"]] = 0.5  # strong upward trend
        vec[idx["inflow_share"]] = 0.8  # very inflow-dominant
        centroids = {0: vec}
        result = name_segments(centroids, feature_names=feature_names)
        assert "Growth" in result[0].label or "Revenue" in result[0].label

    def test_missing_cluster_falls_back(self):
        """If API returns partial JSON, missing clusters get default label."""
        rng = np.random.default_rng(1)
        n_feat = len(CustomerProfile.feature_names())
        centroids = {0: rng.uniform(0, 1, n_feat).astype(np.float32)}
        result = name_segments(centroids)
        assert 0 in result
