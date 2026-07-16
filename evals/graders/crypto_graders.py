"""
Crypto-specific evaluation graders.

Financial performance metrics: Sharpe, Sortino, Max Drawdown.
Composable via CryptoEvalHarness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from evals.metrics.crypto_constants import (
    DEFAULT_MAX_DRAWDOWN_THRESHOLD,
    DEFAULT_SHARPE_THRESHOLD,
    DEFAULT_SORTINO_THRESHOLD,
)


class BaseCryptoGrader(ABC):
    name: str
    threshold: float

    @abstractmethod
    def score(self, returns: np.ndarray) -> dict: ...

    def _make_result(self, value: float, passed: bool, detail: str = "") -> dict:
        return {
            "grader_name": self.name,
            "metric_value": round(value, 4),
            "threshold": self.threshold,
            "passed": passed,
            "detail": detail,
        }


class SharpeGrader(BaseCryptoGrader):
    """Annualized Sharpe ratio. Higher is better; passes if > threshold."""

    name = "sharpe_ratio"

    def __init__(
        self, threshold: float = DEFAULT_SHARPE_THRESHOLD, periods_per_year: float = 365.0
    ) -> None:
        self.threshold = threshold
        self.periods_per_year = periods_per_year

    def score(self, returns: np.ndarray) -> dict:
        if len(returns) < 2 or returns.std() == 0:
            return self._make_result(0.0, False, "Insufficient data or zero variance")

        sharpe = float(returns.mean() / returns.std() * np.sqrt(self.periods_per_year))
        return self._make_result(sharpe, sharpe > self.threshold)


class SortinoGrader(BaseCryptoGrader):
    """Sortino ratio — only penalizes downside deviation. Higher is better."""

    name = "sortino_ratio"

    def __init__(
        self, threshold: float = DEFAULT_SORTINO_THRESHOLD, periods_per_year: float = 365.0
    ) -> None:
        self.threshold = threshold
        self.periods_per_year = periods_per_year

    def score(self, returns: np.ndarray) -> dict:
        if len(returns) < 2:
            return self._make_result(0.0, False, "Insufficient data")

        downside = returns[returns < 0]
        downside_std = downside.std() if len(downside) > 1 else returns.std()
        if downside_std == 0:
            return self._make_result(0.0, False, "Zero downside deviation")

        sortino = float(returns.mean() / downside_std * np.sqrt(self.periods_per_year))
        return self._make_result(sortino, sortino > self.threshold)


class MaxDrawdownGrader(BaseCryptoGrader):
    """Maximum peak-to-trough drawdown. Lower is better; passes if < threshold."""

    name = "max_drawdown"

    def __init__(self, threshold: float = DEFAULT_MAX_DRAWDOWN_THRESHOLD) -> None:
        self.threshold = threshold

    def score(self, returns: np.ndarray) -> dict:
        if len(returns) < 2:
            return self._make_result(0.0, True, "Insufficient data")

        cumulative = np.cumprod(1.0 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / peak
        max_dd = float(drawdowns.max())

        return self._make_result(max_dd, max_dd < self.threshold)


class CryptoEvalHarness:
    """Compose crypto graders and run them against a returns array."""

    def __init__(self, periods_per_year: float = 365.0) -> None:
        self.graders = [
            SharpeGrader(periods_per_year=periods_per_year),
            SortinoGrader(periods_per_year=periods_per_year),
            MaxDrawdownGrader(),
        ]

    def run(self, returns: np.ndarray) -> list[dict]:
        return [g.score(returns) for g in self.graders]
