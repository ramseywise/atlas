"""Unit tests for core/segmentation/algorithms.py."""

from __future__ import annotations

import numpy as np
import pytest

from core.segmentation.algorithms import (
    ClusterResult,
    fit_agglomerative,
    fit_gmm,
    fit_kmeans,
    select_best,
)


@pytest.fixture
def blobs():
    """Three well-separated Gaussian blobs, 30 points each."""
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.3, (30, 2))
    b = rng.normal([5, 0], 0.3, (30, 2))
    c = rng.normal([2.5, 4], 0.3, (30, 2))
    return np.vstack([a, b, c]).astype(np.float32)


class TestKMeans:
    def test_returns_cluster_result(self, blobs):
        result = fit_kmeans(blobs, n_clusters=3)
        assert isinstance(result, ClusterResult)
        assert result.algorithm == "kmeans"
        assert result.n_clusters == 3
        assert len(result.labels) == len(blobs)

    def test_no_noise(self, blobs):
        result = fit_kmeans(blobs, n_clusters=3)
        assert result.noise_fraction == 0.0
        assert -1 not in result.labels

    def test_inertia_in_metadata(self, blobs):
        result = fit_kmeans(blobs, n_clusters=3)
        assert "inertia" in result.metadata
        assert result.metadata["inertia"] > 0

    def test_recovers_three_blobs(self, blobs):
        result = fit_kmeans(blobs, n_clusters=3)
        # Each blob (30 points) should map to one cluster
        sizes = np.bincount(result.labels)
        assert set(sizes) == {30}


class TestGMM:
    def test_returns_cluster_result(self, blobs):
        result = fit_gmm(blobs, n_components=3)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 3

    def test_bic_aic_in_metadata(self, blobs):
        result = fit_gmm(blobs, n_components=3)
        assert "bic" in result.metadata
        assert "aic" in result.metadata


class TestAgglomerative:
    def test_basic(self, blobs):
        result = fit_agglomerative(blobs, n_clusters=3)
        assert result.n_clusters == 3
        assert len(set(result.labels)) == 3


class TestSelectBest:
    def test_selects_good_clustering(self, blobs):
        result = select_best(blobs, min_cluster_size=5)
        # Should find ~3 clusters on well-separated blobs
        assert result.n_clusters >= 2
        assert len(result.labels) == len(blobs)

    def test_falls_back_gracefully_on_tiny_data(self):
        # Only 4 points — HDBSCAN will struggle, should fall back without error
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (4, 3)).astype(np.float32)
        result = select_best(X, min_cluster_size=2)
        assert len(result.labels) == 4
