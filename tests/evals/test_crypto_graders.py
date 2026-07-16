"""Tests for crypto-specific graders (Sharpe, Sortino, MaxDrawdown)."""

from __future__ import annotations

import numpy as np

from evals.graders.crypto_graders import (
    CryptoEvalHarness,
    MaxDrawdownGrader,
    SharpeGrader,
    SortinoGrader,
)


class TestSharpeGrader:
    def test_positive_returns_positive_sharpe(self):
        grader = SharpeGrader(threshold=0.5)
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02, 0.01, 0.015, 0.02])
        result = grader.score(returns)
        assert result["metric_value"] > 0
        assert result["passed"] is True

    def test_negative_returns_negative_sharpe(self):
        grader = SharpeGrader(threshold=0.5)
        returns = np.array([-0.02, -0.03, -0.01, -0.02, -0.015])
        result = grader.score(returns)
        assert result["metric_value"] < 0
        assert result["passed"] is False

    def test_zero_variance_returns_zero(self):
        grader = SharpeGrader(threshold=0.5)
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        result = grader.score(returns)
        assert result["metric_value"] == 0.0

    def test_insufficient_data(self):
        grader = SharpeGrader(threshold=0.5)
        returns = np.array([0.01])
        result = grader.score(returns)
        assert result["passed"] is False


class TestSortinoGrader:
    def test_all_positive_returns(self):
        grader = SortinoGrader(threshold=0.7)
        returns = np.array([0.01, 0.02, 0.015, 0.03, 0.01])
        result = grader.score(returns)
        assert result["metric_value"] > 0  # All positive → high Sortino
        assert result["passed"] is True

    def test_mixed_returns(self):
        grader = SortinoGrader(threshold=0.7)
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.01, 0.02, -0.01, 0.015])
        result = grader.score(returns)
        assert result["metric_value"] > 0

    def test_heavy_downside(self):
        grader = SortinoGrader(threshold=0.7)
        returns = np.array([-0.05, -0.03, -0.04, 0.01, -0.02])
        result = grader.score(returns)
        assert result["metric_value"] < 0
        assert result["passed"] is False


class TestMaxDrawdownGrader:
    def test_no_drawdown(self):
        grader = MaxDrawdownGrader(threshold=0.15)
        returns = np.array([0.01, 0.02, 0.01, 0.015, 0.02])
        result = grader.score(returns)
        assert result["metric_value"] == 0.0
        assert result["passed"] is True

    def test_significant_drawdown(self):
        grader = MaxDrawdownGrader(threshold=0.15)
        returns = np.array([0.05, 0.03, -0.10, -0.08, -0.05, 0.02, 0.03])
        result = grader.score(returns)
        assert result["metric_value"] > 0.10
        assert result["passed"] is False

    def test_small_drawdown_passes(self):
        grader = MaxDrawdownGrader(threshold=0.15)
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        result = grader.score(returns)
        assert result["metric_value"] < 0.15
        assert result["passed"] is True


class TestCryptoEvalHarness:
    def test_harness_runs_all_graders(self):
        harness = CryptoEvalHarness()
        returns = np.array([0.01, 0.02, -0.005, 0.015, -0.01, 0.02, 0.01, -0.005])
        results = harness.run(returns)
        assert len(results) == 3
        grader_names = {r["grader_name"] for r in results}
        assert grader_names == {"sharpe_ratio", "sortino_ratio", "max_drawdown"}
