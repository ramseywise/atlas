"""
Pass thresholds and tier definitions for the eval grader suite.

TIER_THRESHOLDS defines what constitutes a "passing" forecast at each
quality tier. Tier 1 = production-ready; Tier 3 = baseline beat only.
"""

from __future__ import annotations

TIER_THRESHOLDS: dict[str, dict[str, float]] = {
    "tier1": {
        "mase_max": 0.7,
        "smape_max": 10.0,
        "directional_min": 62.0,
        "coverage_min": 78.0,
        "drift_ratio_max": 1.1,
    },
    "tier2": {
        "mase_max": 0.85,
        "smape_max": 12.0,
        "directional_min": 58.0,
        "coverage_min": 76.0,
        "drift_ratio_max": 1.15,
    },
    "tier3": {
        "mase_max": 1.0,
        "smape_max": 15.0,
        "directional_min": 55.0,
        "coverage_min": 75.0,
        "drift_ratio_max": 1.2,
    },
}

# Default thresholds used by graders (Tier 3 — beats naïve baseline)
DEFAULT_MASE_THRESHOLD: float = 1.0
DEFAULT_SMAPE_THRESHOLD: float = 15.0
DEFAULT_DIRECTIONAL_THRESHOLD: float = 55.0
DEFAULT_COVERAGE_THRESHOLD: float = 75.0
DEFAULT_DRIFT_THRESHOLD: float = 1.2
