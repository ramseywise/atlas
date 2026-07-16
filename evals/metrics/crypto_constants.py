"""Threshold constants for crypto evaluation graders."""

from __future__ import annotations

DEFAULT_SHARPE_THRESHOLD = 0.5
DEFAULT_SORTINO_THRESHOLD = 0.7
DEFAULT_MAX_DRAWDOWN_THRESHOLD = 0.15

CRYPTO_TIER_THRESHOLDS = {
    "tier1": {
        "sharpe_min": 1.5,
        "sortino_min": 2.0,
        "max_drawdown_max": 0.08,
        "directional_min": 62.0,
    },
    "tier2": {
        "sharpe_min": 1.0,
        "sortino_min": 1.3,
        "max_drawdown_max": 0.12,
        "directional_min": 58.0,
    },
    "tier3": {
        "sharpe_min": 0.5,
        "sortino_min": 0.7,
        "max_drawdown_max": 0.15,
        "directional_min": 55.0,
    },
}
