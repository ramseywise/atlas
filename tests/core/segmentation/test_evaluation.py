"""Unit tests for core/segmentation/evaluation.py."""

from __future__ import annotations

import numpy as np
import pytest

from core.segmentation.algorithms import ClusterResult
from core.segmentation.evaluation import (
    DB_THRESHOLD,
    MIN_CLUSTER_SIZE,
    SILHOUETTE_THRESHOLD,
    SegmentEvalReport,
    evaluate_clusters,
)


@pytest.fixture
def blobs_and_labels():
    rng = np.random.default_rng(42)
    a = rng.normal([0, 0], 0.3, (10, 2))
    b = rng.normal([5, 0], 0.3, (10, 2))
    c = rng.normal([2.5, 4], 0.3, (10, 2))
    X = np.vstack([a, b, c]).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    result = ClusterResult(
        algorithm="kmeans", labels=labels, n_clusters=3, noise_fraction=0.0, metadata={}
    )
    return X, result


class TestEvaluateClusters:
    def test_returns_report(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result, cycle=1)
        assert isinstance(report, SegmentEvalReport)

    def test_silhouette_passes_on_clean_blobs(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result)
        assert report.silhouette >= SILHOUETTE_THRESHOLD
        assert report.silhouette_passed

    def test_db_passes_on_clean_blobs(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result)
        assert report.davies_bouldin <= DB_THRESHOLD
        assert report.db_passed

    def test_min_size_passes(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result)
        assert report.min_cluster_size >= MIN_CLUSTER_SIZE
        assert report.min_size_passed

    def test_all_passed(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result)
        assert report.all_passed

    def test_cluster_sizes_correct(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result)
        assert report.cluster_sizes == {0: 10, 1: 10, 2: 10}

    def test_noise_excluded_from_metrics(self):
        rng = np.random.default_rng(0)
        a = rng.normal([0, 0], 0.3, (10, 2))
        b = rng.normal([5, 0], 0.3, (10, 2))
        X = np.vstack([a, b]).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10)
        labels[0] = -1  # mark one as noise
        result = ClusterResult(
            algorithm="hdbscan", labels=labels, n_clusters=2, noise_fraction=1 / 20, metadata={}
        )
        report = evaluate_clusters(X, result)
        # Noise customer should not appear in cluster_sizes
        assert -1 not in report.cluster_sizes

    def test_too_few_points_returns_zeros(self):
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        labels = np.array([0, 1])
        result = ClusterResult(
            algorithm="kmeans", labels=labels, n_clusters=2, noise_fraction=0.0, metadata={}
        )
        report = evaluate_clusters(X, result)
        # Should not raise; metrics fall back to zero / 999
        assert report.silhouette == 0.0
        assert len(report.warnings) > 0

    def test_summary_string(self, blobs_and_labels):
        X, result = blobs_and_labels
        report = evaluate_clusters(X, result, cycle=2)
        s = report.summary()
        assert "cycle=2" in s
        assert "PASS" in s or "FAIL" in s
