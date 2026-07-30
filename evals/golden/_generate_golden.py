"""
Golden dataset generator.

Generates evals/golden/{forecast,segment,crypto}_golden_qa.jsonl using the
same synthetic data sources the agents consume during training and eval.

Run once:  uv run python -m evals.golden._generate_golden
Re-run to regenerate (datasets are deterministic — seed=42).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent
SEED = 42
RNG = np.random.default_rng(SEED)

# ── Forecast golden dataset ────────────────────────────────────────────────────
# Each entry captures one forecast eval scenario:
#   train_values  — the training history fed to MASEGrader
#   forecast      — the point_forecast the grader evaluates
#   lower_80 / upper_80  — interval for CoverageGrader
#   actuals       — what actually happened in the forecast window
#
# Series IDs match PipelineSource values used by generate_sequence_dataset().
# Category values match CategoryType enum used by ForecastResult.
# Amounts are in the same units ($USD/day) as the SaaS-growth archetype.


def _sin_series(
    n: int, base: float, trend: float, weekly: float, monthly: float, noise_std: float, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    vals = (
        base
        + trend * t
        + weekly * np.sin(2 * np.pi * t / 7)
        + monthly * np.sin(2 * np.pi * t / 30.44)
        + rng.normal(0, noise_std, n)
    )
    return np.clip(vals, 0.0, None).tolist()


def _naive_forecast(
    train: list[float], horizon: int, spread: float = 0.08
) -> tuple[list[float], list[float], list[float]]:
    """Naïve seasonal forecast: repeat last 7 days, with ±spread CI."""
    last = train[-7:] if len(train) >= 7 else train
    point = np.tile(last, math.ceil(horizon / len(last)))[:horizon].tolist()
    lo = (np.array(point) * (1 - spread)).tolist()
    hi = (np.array(point) * (1 + spread)).tolist()
    return point, lo, hi


def _good_forecast(
    train: list[float], horizon: int
) -> tuple[list[float], list[float], list[float]]:
    """Good forecast: close to actuals with tight intervals."""
    base = float(np.mean(train[-30:]))
    trend = (train[-1] - train[-30]) / 30 if len(train) >= 30 else 0.0
    rng = np.random.default_rng(SEED + 1)
    point = [max(0.0, base + trend * i + rng.normal(0, base * 0.02)) for i in range(horizon)]
    lo = [p * 0.88 for p in point]
    hi = [p * 1.12 for p in point]
    return point, lo, hi


def _bad_forecast(train: list[float], horizon: int) -> tuple[list[float], list[float], list[float]]:
    """Bad forecast: systematically off by 3×."""
    base = float(np.mean(train[-30:])) * 3.0
    point = [base] * horizon
    lo = [p * 0.9 for p in point]
    hi = [p * 1.1 for p in point]
    return point, lo, hi


def _close_actuals(forecast: list[float], noise_frac: float = 0.03) -> list[float]:
    rng = np.random.default_rng(SEED + 2)
    return [max(0.0, v + rng.normal(0, v * noise_frac)) for v in forecast]


def _noisy_actuals(forecast: list[float], noise_frac: float = 0.25) -> list[float]:
    rng = np.random.default_rng(SEED + 3)
    return [max(0.0, v + rng.normal(0, v * noise_frac)) for v in forecast]


def _anomaly_actuals(forecast: list[float], idx: int = 10) -> list[float]:
    """Actuals with a spike anomaly at position idx."""
    out = list(forecast)
    out[idx] = forecast[idx] * 5.0
    return out


def _flat_actuals(n: int, value: float = 5000.0) -> list[float]:
    return [value] * n


def build_forecast_entries() -> list[dict]:
    entries = []

    # ── Series 1: erp_revenue — SaaS growth, steady upward trend ──────────────
    series_id = "erp_revenue"
    category = "income_recurring"
    train = _sin_series(500, base=8000, trend=8, weekly=0, monthly=1500, noise_std=1000, seed=10)
    horizon = 30

    # 1-10: good forecasts, close actuals — should pass all graders
    for i in range(10):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _close_actuals(point, noise_frac=0.02 + i * 0.003)
        entries.append(
            {
                "id": f"forecast-erp-good-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (SaaS growth, steady trend)",
                "category": "good_forecast",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "saas_growth",
                    "base_amount": 8000,
                    "trend_rate": 8,
                    "n_train_days": 500,
                },
                "expected_outcome": {
                    "mase_lt": 1.0,
                    "smape_lt": 15.0,
                    "directional_gt": 55.0,
                    "coverage_gte": 75.0,
                },
            }
        )

    # 11-15: bad forecasts — should fail MASE and SMAPE
    for i in range(5):
        point, lo, hi = _bad_forecast(train, horizon)
        actuals = _close_actuals(train[-horizon:], noise_frac=0.05)
        entries.append(
            {
                "id": f"forecast-erp-bad-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (overestimate: 3x off)",
                "category": "bad_forecast",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "saas_growth",
                    "base_amount": 8000,
                    "trend_rate": 8,
                    "n_train_days": 500,
                },
                "expected_outcome": {
                    "mase_gte": 1.0,
                    "smape_gte": 15.0,
                    "all_passed": False,
                },
            }
        )

    # ── Series 2: payroll — weekly seasonality, outflow ──────────────────────
    series_id = "payroll"
    category = "expense_fixed"
    train = _sin_series(400, base=12000, trend=15, weekly=0, monthly=1200, noise_std=400, seed=20)
    horizon = 14

    # 16-20: good forecasts (14d, fixed expense)
    for i in range(5):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _close_actuals(point, noise_frac=0.015)
        entries.append(
            {
                "id": f"forecast-payroll-good-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (fixed expense, low noise)",
                "category": "good_forecast",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "saas_growth",
                    "base_amount": 12000,
                    "trend_rate": 15,
                    "n_train_days": 400,
                },
                "expected_outcome": {
                    "mase_lt": 1.0,
                    "smape_lt": 10.0,
                    "directional_gt": 55.0,
                    "coverage_gte": 75.0,
                },
            }
        )

    # ── Series 3: accounts_receivable — high variance, lumpy ─────────────────
    series_id = "accounts_receivable"
    category = "income_variable"
    train = _sin_series(400, base=4000, trend=2, weekly=800, monthly=2500, noise_std=1500, seed=30)
    horizon = 30

    # 21-25: noisy actuals — coverage grader challenged by wide actual variance
    for i in range(5):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _noisy_actuals(point, noise_frac=0.18 + i * 0.02)
        entries.append(
            {
                "id": f"forecast-ar-noisy-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (high variance, lumpy AR collections)",
                "category": "high_variance",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "smb_services",
                    "base_amount": 4000,
                    "trend_rate": 2,
                    "n_train_days": 400,
                    "monthly_phase": 0.3,
                    "anomaly_prob": 0.06,
                },
                "expected_outcome": {
                    "note": "Coverage may fail due to genuine high variance in AR",
                },
            }
        )

    # ── Series 4: sub_billing — anomaly spike in actuals ─────────────────────
    series_id = "sub_billing"
    category = "income_recurring"
    train = _sin_series(365, base=18000, trend=18, weekly=0, monthly=2000, noise_std=600, seed=40)
    horizon = 30

    # 26-28: actuals contain anomaly spike — tests grader robustness
    for i in range(3):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _anomaly_actuals(point, idx=5 + i * 5)
        entries.append(
            {
                "id": f"forecast-sub-anomaly-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (anomaly spike in actuals)",
                "category": "anomaly_in_actuals",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "saas_growth",
                    "base_amount": 18000,
                    "trend_rate": 18,
                    "n_train_days": 365,
                    "anomaly_prob": 0.02,
                    "anomaly_multiplier": 3.0,
                },
                "expected_outcome": {
                    "note": "SMAPE will spike due to anomaly; directional accuracy affected",
                },
            }
        )

    # ── Series 5: marketplace_gmv — weekend peak, high volume ────────────────
    series_id = "marketplace_gmv"
    category = "income_variable"
    train = _sin_series(
        500, base=55000, trend=30, weekly=15000, monthly=5000, noise_std=8000, seed=50
    )
    horizon = 7

    # 29-31: short horizon (7d) with strong weekly seasonality
    for i in range(3):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _close_actuals(point, noise_frac=0.05)
        entries.append(
            {
                "id": f"forecast-gmv-weekly-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (marketplace GMV, weekly peak)",
                "category": "short_horizon_seasonal",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "marketplace",
                    "base_amount": 55000,
                    "trend_rate": 30,
                    "weekly_amp": 15000,
                    "n_train_days": 500,
                    "weekly_phase": 0.5,
                },
                "expected_outcome": {
                    "mase_lt": 1.0,
                    "smape_lt": 15.0,
                    "directional_gt": 55.0,
                    "coverage_gte": 75.0,
                },
            }
        )

    # ── Series 6: equity_funding — sparse lumpy inflows (edge case) ──────────
    series_id = "equity_funding"
    category = "income_variable"
    # Very sparse: mostly zeros with rare large spikes (anomaly_multiplier=80)
    train_base = _sin_series(400, base=0, trend=0, weekly=0, monthly=0, noise_std=500, seed=60)
    # Add a few large funding events
    rng_eq = np.random.default_rng(60)
    train_arr = np.array(train_base)
    spikes = rng_eq.integers(50, 390, size=5)
    train_arr[spikes] += rng_eq.uniform(30000, 80000, size=5)
    train = np.clip(train_arr, 0.0, None).tolist()
    horizon = 30

    # 32-34: sparse series — tests grader edge case
    for i in range(3):
        point, lo, hi = _naive_forecast(train, horizon, spread=0.5)
        actuals = _flat_actuals(horizon, value=0.0)  # no funding this period
        entries.append(
            {
                "id": f"forecast-equity-sparse-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (sparse lumpy equity tranches)",
                "category": "sparse_series",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "early_stage_founder",
                    "base_amount": 0,
                    "anomaly_prob": 0.015,
                    "anomaly_multiplier": 80.0,
                    "n_train_days": 400,
                },
                "expected_outcome": {
                    "note": "MASE undefined if naïve_MAE≈0; grader uses 1e-8 guard",
                },
            }
        )

    # ── Series 7: tax_provision — annual spike only (Q4) ─────────────────────
    series_id = "tax_provision"
    category = "expense_discretionary"
    train = _sin_series(730, base=1800, trend=1.5, weekly=0, monthly=0, noise_std=200, seed=70)
    # Inject annual Q4 spike
    train_arr = np.array(train)
    for yr in range(2):
        q4_idx = 330 + yr * 365
        if q4_idx < len(train_arr):
            train_arr[q4_idx : q4_idx + 30] += 5000
    train = np.clip(train_arr, 0.0, None).tolist()
    horizon = 90

    # 35-37: long horizon (90d quarter forecast), annual-only seasonality
    for i in range(3):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _close_actuals(point, noise_frac=0.04)
        entries.append(
            {
                "id": f"forecast-tax-quarter-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (quarterly tax provision, Q4 spike)",
                "category": "long_horizon",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "smb_services",
                    "base_amount": 1800,
                    "trend_rate": 1.5,
                    "annual_amp": 5000,
                    "n_train_days": 730,
                },
                "expected_outcome": {
                    "mase_lt": 1.0,
                    "smape_lt": 15.0,
                    "directional_gt": 55.0,
                    "coverage_gte": 75.0,
                },
            }
        )

    # ── Series 8: retail erp_revenue — Q4 spike, thin margins ────────────────
    series_id = "erp_revenue_retail"
    category = "income_variable"
    train = _sin_series(730, base=8000, trend=1, weekly=2000, monthly=1000, noise_std=2500, seed=80)
    # Add Q4 annual spike
    train_arr = np.array(train)
    for yr in range(2):
        q4_idx = 300 + yr * 365
        if q4_idx < len(train_arr):
            train_arr[q4_idx : q4_idx + 60] += 18000
    train = np.clip(train_arr, 0.0, None).tolist()
    horizon = 30

    # 38-40: retail Q4 in-window (spike captured)
    for i in range(3):
        point, lo, hi = _good_forecast(train, horizon)
        # Actuals for a Q4 window — above forecast if seasonal
        actuals = _close_actuals(point, noise_frac=0.08)
        entries.append(
            {
                "id": f"forecast-retail-q4-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (retail Q4 peak, weekly + annual seasonality)",
                "category": "multi_level_seasonality",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "retail_seasonal",
                    "base_amount": 8000,
                    "trend_rate": 1,
                    "weekly_amp": 2000,
                    "annual_amp": 18000,
                    "annual_phase": 0.25,
                    "n_train_days": 730,
                },
                "expected_outcome": {
                    "mase_lt": 1.0,
                    "smape_lt": 15.0,
                },
            }
        )

    # ── Series 9: multi-step reasoning — drift scenario ───────────────────────
    series_id = "erp_revenue"
    category = "income_recurring"
    # Two-year train, then sudden regime shift (revenue drops 40%)
    train_pre = _sin_series(
        600, base=14000, trend=5, weekly=0, monthly=4000, noise_std=2500, seed=90
    )
    train_post = _sin_series(
        200, base=8400, trend=2, weekly=0, monthly=2400, noise_std=2000, seed=91
    )
    train = train_pre + train_post
    horizon = 30

    # 41-43: drift — forecast trained on pre-shift, actuals from post-shift
    for i in range(3):
        point, lo, hi = _good_forecast(train_pre, horizon)  # stale model on pre-shift data
        # Actuals reflect the new lower regime
        rng_d = np.random.default_rng(SEED + i)
        actuals = [max(0.0, 8400 + rng_d.normal(0, 2000)) for _ in range(horizon)]
        entries.append(
            {
                "id": f"forecast-drift-regime-{i + 1:02d}",
                "query": (
                    f"Forecast {horizon}d for {series_id} after 40% revenue regime shift "
                    "(model trained pre-shift)"
                ),
                "category": "drift_regime_shift",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "professional_services",
                    "pre_shift_base": 14000,
                    "post_shift_base": 8400,
                    "regime_shift_at_day": 600,
                    "n_train_days": 800,
                },
                "expected_outcome": {
                    "mase_gte": 1.0,
                    "note": "DriftGrader should flag ratio > 1.2 after sufficient history",
                },
            }
        )

    # ── Series 10: missing data (short train) — edge case ──────────────────────
    series_id = "bank_operating"
    category = "income_variable"
    # Very short history — below typical min_train_days=200
    train = _sin_series(
        45, base=3000, trend=0.5, weekly=1500, monthly=500, noise_std=1000, seed=100
    )
    horizon = 7

    # 44-46: short train — graders should not crash
    for i in range(3):
        point, lo, hi = _naive_forecast(train, horizon, spread=0.15)
        actuals = _close_actuals(point, noise_frac=0.10)
        entries.append(
            {
                "id": f"forecast-short-train-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (only {len(train)}d history — cold start)",
                "category": "short_history_edge_case",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "retail_seasonal",
                    "base_amount": 3000,
                    "weekly_amp": 1500,
                    "n_train_days": 45,
                    "note": "Deliberately below min_train_days=200",
                },
                "expected_outcome": {
                    "note": "Graders must not crash; low directional accuracy expected",
                },
            }
        )

    # ── Series 11: inventory_purchase — manufacturing, large irregular spikes ──
    series_id = "inventory_purch"
    category = "expense_discretionary"
    train = _sin_series(
        500, base=12000, trend=1, weekly=1000, monthly=5000, noise_std=4000, seed=110
    )
    horizon = 30

    # 47-50: manufacturing cost series
    for i in range(4):
        point, lo, hi = _good_forecast(train, horizon)
        actuals = _noisy_actuals(point, noise_frac=0.20 + i * 0.05)
        entries.append(
            {
                "id": f"forecast-inventory-{i + 1:02d}",
                "query": f"Forecast {horizon}d for {series_id} (manufacturing inventory, lumpy)",
                "category": "high_variance",
                "series_id": series_id,
                "category_type": category,
                "horizon": horizon,
                "train_values": train,
                "point_forecast": point,
                "lower_80": lo,
                "upper_80": hi,
                "actuals": actuals,
                "source_data": {
                    "archetype": "manufacturing",
                    "base_amount": 12000,
                    "trend_rate": 1,
                    "monthly_amp": 5000,
                    "anomaly_prob": 0.07,
                    "anomaly_multiplier": 4.0,
                    "n_train_days": 500,
                },
                "expected_outcome": {
                    "note": "High noise; coverage grader may fail at ±15% interval",
                },
            }
        )

    assert len(entries) >= 30, f"Expected >=30 forecast entries, got {len(entries)}"
    return entries


# ── Segment golden dataset ─────────────────────────────────────────────────────
# Each entry stores:
#   embedding_matrix — 2D float array (n_customers × n_features)
#   labels           — cluster labels (-1 = noise)
#   algorithm        — which clustering algo was used
#   n_clusters, noise_fraction — ClusterResult fields
#
# Feature space mirrors CustomerProfile.feature_names() (14 features).
# Archetypes from synthetic.py drive the cluster separation.


def _archetype_centroid(archetype: str) -> np.ndarray:
    """Return a representative 14-feature centroid for each archetype."""
    # Features: total_inflow, total_outflow, net_position, n_active_series,
    #           daily_net_std, inflow_cv, outflow_cv, weekly_autocorr,
    #           monthly_autocorr, trend_slope_norm, activity_last_30d,
    #           activity_last_90d, inflow_share, top_source_share
    centroids = {
        "early_stage_founder": np.array(
            [
                50_000,
                200_000,
                -150_000,
                5,
                3000,
                1.5,
                0.3,
                0.1,
                0.15,
                -0.02,
                0.08,
                0.22,
                0.20,
                0.45,
            ],
            dtype=np.float32,
        ),
        "smb_services": np.array(
            [
                1_800_000,
                1_500_000,
                300_000,
                5,
                2000,
                0.4,
                0.2,
                0.05,
                0.25,
                0.01,
                0.07,
                0.21,
                0.55,
                0.42,
            ],
            dtype=np.float32,
        ),
        "saas_growth": np.array(
            [
                5_000_000,
                4_200_000,
                800_000,
                5,
                5000,
                0.25,
                0.3,
                0.02,
                0.18,
                0.04,
                0.08,
                0.23,
                0.54,
                0.38,
            ],
            dtype=np.float32,
        ),
        "manufacturing": np.array(
            [
                6_000_000,
                5_800_000,
                200_000,
                6,
                10_000,
                0.5,
                0.4,
                0.12,
                0.35,
                0.005,
                0.07,
                0.21,
                0.51,
                0.36,
            ],
            dtype=np.float32,
        ),
        "retail_seasonal": np.array(
            [
                2_500_000,
                2_400_000,
                100_000,
                5,
                8000,
                0.7,
                0.4,
                0.30,
                0.20,
                0.002,
                0.15,
                0.35,
                0.51,
                0.40,
            ],
            dtype=np.float32,
        ),
        "professional_services": np.array(
            [
                4_000_000,
                3_800_000,
                200_000,
                5,
                4000,
                0.35,
                0.15,
                0.03,
                0.30,
                0.015,
                0.07,
                0.22,
                0.51,
                0.43,
            ],
            dtype=np.float32,
        ),
        "marketplace": np.array(
            [
                18_000_000,
                17_100_000,
                900_000,
                4,
                25_000,
                0.20,
                0.20,
                0.40,
                0.10,
                0.03,
                0.08,
                0.22,
                0.51,
                0.62,
            ],
            dtype=np.float32,
        ),
    }
    return centroids[archetype]


def _make_cluster(
    archetype: str, n: int, noise_scale: float = 0.08, seed_offset: int = 0
) -> np.ndarray:
    """Generate n customer embedding vectors around an archetype centroid."""
    rng = np.random.default_rng(SEED + seed_offset)
    centroid = _archetype_centroid(archetype)
    # Scale noise relative to feature magnitude
    noise = rng.normal(0, np.abs(centroid) * noise_scale + 1e-6, (n, len(centroid)))
    return (centroid + noise).astype(np.float32)


def build_segment_entries() -> list[dict]:
    entries = []

    archetypes = [
        "early_stage_founder",
        "smb_services",
        "saas_growth",
        "manufacturing",
        "retail_seasonal",
        "professional_services",
        "marketplace",
    ]

    # ── Group 1: clean well-separated blobs (3 archetypes, 10 each) ───────────
    # Should produce high silhouette, low DB, pass all graders
    for trial in range(3):
        groups = [("saas_growth", 10), ("manufacturing", 10), ("marketplace", 10)]
        blocks = [
            _make_cluster(a, n, noise_scale=0.05, seed_offset=trial * 100 + i)
            for i, (a, n) in enumerate(groups)
        ]
        X = np.vstack(blocks).tolist()
        labels = [0] * 10 + [1] * 10 + [2] * 10
        entries.append(
            {
                "id": f"segment-clean-3arc-{trial + 1:02d}",
                "query": "Cluster 30 customers from 3 well-separated archetypes",
                "category": "clean_separation",
                "embedding_matrix": X,
                "labels": labels,
                "algorithm": "kmeans",
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "n_customers": 30,
                "source_data": {
                    "archetypes": ["saas_growth", "manufacturing", "marketplace"],
                    "n_per_archetype": 10,
                    "noise_scale": 0.05,
                },
                "expected_outcome": {
                    "silhouette_gte": 0.25,
                    "db_lte": 1.5,
                    "min_cluster_size_gte": 3,
                    "all_passed": True,
                },
            }
        )

    # ── Group 2: all 7 archetypes, realistic scale ────────────────────────────
    for trial in range(8):
        sizes = [6, 8, 10, 5, 6, 7, 8]  # 50 customers total
        blocks = [
            _make_cluster(a, n, noise_scale=0.08, seed_offset=trial * 200 + i)
            for i, (a, n) in enumerate(zip(archetypes, sizes, strict=False))
        ]
        X = np.vstack(blocks).tolist()
        labels = [j for j, n in enumerate(sizes) for _ in range(n)]
        entries.append(
            {
                "id": f"segment-all-7arc-{trial + 1:02d}",
                "query": "Cluster 50 customers across all 7 business archetypes",
                "category": "full_archetype_mix",
                "embedding_matrix": X,
                "labels": labels,
                "algorithm": "kmeans",
                "n_clusters": 7,
                "noise_fraction": 0.0,
                "n_customers": 50,
                "source_data": {
                    "archetypes": archetypes,
                    "n_per_archetype": sizes,
                    "noise_scale": 0.08,
                },
                "expected_outcome": {
                    "silhouette_gte": 0.25,
                    "db_lte": 1.5,
                    "min_cluster_size_gte": 3,
                },
            }
        )

    # ── Group 3: noisy / overlapping archetypes ───────────────────────────────
    # High intra-cluster variance — silhouette may fail, tests grader robustness
    for trial in range(6):
        groups = [("smb_services", 12), ("professional_services", 12), ("saas_growth", 11)]
        blocks = [
            _make_cluster(a, n, noise_scale=0.35, seed_offset=trial * 300 + i)
            for i, (a, n) in enumerate(groups)
        ]
        X = np.vstack(blocks).tolist()
        labels = [0] * 12 + [1] * 12 + [2] * 11
        entries.append(
            {
                "id": f"segment-overlapping-{trial + 1:02d}",
                "query": "Cluster customers from 3 similar-revenue archetypes (high overlap)",
                "category": "overlapping_archetypes",
                "embedding_matrix": X,
                "labels": labels,
                "algorithm": "kmeans",
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "n_customers": 35,
                "source_data": {
                    "archetypes": ["smb_services", "professional_services", "saas_growth"],
                    "n_per_archetype": [12, 12, 11],
                    "noise_scale": 0.35,
                    "note": "High overlap: smb_services and professional_services have similar scale",
                },
                "expected_outcome": {
                    "note": "Silhouette may be below 0.25 due to genuine archetype similarity",
                },
            }
        )

    # ── Group 4: HDBSCAN noise fraction — some customers unclassified ─────────
    for trial in range(6):
        groups = [("early_stage_founder", 8), ("marketplace", 10), ("retail_seasonal", 8)]
        blocks = [
            _make_cluster(a, n, noise_scale=0.06, seed_offset=trial * 400 + i)
            for i, (a, n) in enumerate(groups)
        ]
        X_clean = np.vstack(blocks)
        # Add 4 outliers (random noise far from any centroid)
        rng = np.random.default_rng(SEED + trial * 400 + 99)
        outliers = rng.uniform(-500_000, 500_000, (4, 14)).astype(np.float32)
        X = np.vstack([X_clean, outliers]).tolist()
        n_clean = 26
        labels = [0] * 8 + [1] * 10 + [2] * 8 + [-1, -1, -1, -1]
        entries.append(
            {
                "id": f"segment-hdbscan-noise-{trial + 1:02d}",
                "query": "HDBSCAN clustering with 4 outlier noise customers",
                "category": "noise_fraction",
                "embedding_matrix": X,
                "labels": labels,
                "algorithm": "hdbscan",
                "n_clusters": 3,
                "noise_fraction": 4 / 30,
                "n_customers": 30,
                "source_data": {
                    "archetypes": ["early_stage_founder", "marketplace", "retail_seasonal"],
                    "n_clean": n_clean,
                    "n_outliers": 4,
                    "noise_scale": 0.06,
                },
                "expected_outcome": {
                    "silhouette_gte": 0.25,
                    "db_lte": 1.5,
                    "min_cluster_size_gte": 3,
                },
            }
        )

    # ── Group 5: tiny dataset (below threshold scaling) ───────────────────────
    for trial in range(3):
        groups = [("saas_growth", 5), ("retail_seasonal", 5), ("early_stage_founder", 5)]
        blocks = [
            _make_cluster(a, n, noise_scale=0.05, seed_offset=trial * 500 + i)
            for i, (a, n) in enumerate(groups)
        ]
        X = np.vstack(blocks).tolist()
        labels = [0] * 5 + [1] * 5 + [2] * 5
        entries.append(
            {
                "id": f"segment-tiny-{trial + 1:02d}",
                "query": "Cluster 15 customers — below N=50 relaxed-threshold zone",
                "category": "small_dataset_edge_case",
                "embedding_matrix": X,
                "labels": labels,
                "algorithm": "kmeans",
                "n_clusters": 3,
                "noise_fraction": 0.0,
                "n_customers": 15,
                "source_data": {
                    "archetypes": ["saas_growth", "retail_seasonal", "early_stage_founder"],
                    "n_per_archetype": 5,
                    "noise_scale": 0.05,
                    "note": "N<50 → evaluate_clusters applies relaxed thresholds",
                },
                "expected_outcome": {
                    "note": "Relaxed silhouette threshold (scaled by N/50); min_size=5 passes",
                },
            }
        )

    # ── Group 6: multi-step reasoning — segment drift (archetype reassignment) ─
    # Round 1: 3 clusters found. Round 2: one cluster splits into 2.
    for trial in range(4):
        # Initial: 3 clean clusters
        groups_r1 = [("saas_growth", 12), ("manufacturing", 12), ("marketplace", 12)]
        blocks_r1 = [
            _make_cluster(a, n, noise_scale=0.06, seed_offset=trial * 600 + i)
            for i, (a, n) in enumerate(groups_r1)
        ]
        X_r1 = np.vstack(blocks_r1).tolist()
        labels_r1 = [0] * 12 + [1] * 12 + [2] * 12

        # After split: saas_growth diverges into early-stage and growth
        groups_r2 = [
            ("early_stage_founder", 6),
            ("saas_growth", 6),
            ("manufacturing", 12),
            ("marketplace", 12),
        ]
        blocks_r2 = [
            _make_cluster(a, n, noise_scale=0.06, seed_offset=trial * 600 + i + 10)
            for i, (a, n) in enumerate(groups_r2)
        ]
        X_r2 = np.vstack(blocks_r2).tolist()
        labels_r2 = [0] * 6 + [1] * 6 + [2] * 12 + [3] * 12
        entries.append(
            {
                "id": f"segment-drift-split-{trial + 1:02d}",
                "query": "Segment drift: SaaS cluster splits into 2 sub-archetypes between cycles",
                "category": "segment_drift_multi_step",
                "round_1": {
                    "embedding_matrix": X_r1,
                    "labels": labels_r1,
                    "algorithm": "kmeans",
                    "n_clusters": 3,
                    "noise_fraction": 0.0,
                    "n_customers": 36,
                },
                "round_2": {
                    "embedding_matrix": X_r2,
                    "labels": labels_r2,
                    "algorithm": "kmeans",
                    "n_clusters": 4,
                    "noise_fraction": 0.0,
                    "n_customers": 36,
                },
                "source_data": {
                    "note": "Agent should detect cluster count change from 3→4 as drift signal",
                    "archetypes_r1": ["saas_growth", "manufacturing", "marketplace"],
                    "archetypes_r2": [
                        "early_stage_founder",
                        "saas_growth",
                        "manufacturing",
                        "marketplace",
                    ],
                },
                "expected_outcome": {
                    "note": "Both rounds pass silhouette; n_clusters drift detectable via eval_history",
                },
            }
        )

    assert len(entries) >= 30, f"Expected >=30 segment entries, got {len(entries)}"
    return entries


# ── Crypto golden dataset ──────────────────────────────────────────────────────
# Each entry represents one evaluation scenario for CryptoEvalHarness.
# Fields:
#   returns   — daily log returns (what the graders actually consume)
#   ohlcv     — dict of {symbol_key: {open, high, low, close, volume}} lists
#               used by crypto_evaluator_node helpers
#   symbols   — list of trading pairs


def _make_ohlcv(
    n: int,
    start_price: float,
    daily_return_mu: float,
    daily_return_std: float,
    seed: int,
) -> dict:
    """Generate synthetic OHLCV data for one symbol."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(daily_return_mu, daily_return_std, n)
    close_prices = start_price * np.exp(np.cumsum(log_returns))
    close_prices = np.clip(close_prices, 1.0, None)

    # OHLC: open = previous close; high/low add symmetric wick
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = start_price
    wick = rng.uniform(0.002, 0.015, n) * close_prices
    high_prices = np.maximum(close_prices, open_prices) + wick
    low_prices = np.minimum(close_prices, open_prices) - wick
    volumes = rng.lognormal(mean=10.0, sigma=0.8, size=n) * 1_000

    return {
        "open": open_prices.tolist(),
        "high": high_prices.tolist(),
        "low": low_prices.tolist(),
        "close": close_prices.tolist(),
        "volume": volumes.tolist(),
    }


def _make_returns(ohlcv: dict) -> list[float]:
    """Compute daily log returns from close prices."""
    close = np.array(ohlcv["close"])
    return np.diff(np.log(close)).tolist()


def build_crypto_entries() -> list[dict]:
    entries = []

    # ── Scenario 1: BTC steady bull run — Sharpe > 0.5, low drawdown ─────────
    # mu=0.004/std=0.015 verified to pass Sharpe>0.5 and MaxDD<0.15 for all 6 seeds.
    for trial in range(6):
        ohlcv = _make_ohlcv(
            180, start_price=30000, daily_return_mu=0.004, daily_return_std=0.015, seed=SEED + trial
        )
        returns = _make_returns(ohlcv)
        entries.append(
            {
                "id": f"crypto-btc-bull-{trial + 1:02d}",
                "query": "Evaluate BTC/USDT returns during steady 6-month bull run",
                "category": "bull_run",
                "symbol": "BTC/USDT",
                "symbol_key": "BTC_USDT",
                "ohlcv": {"BTC_USDT": ohlcv},
                "returns": returns,
                "n_bars": 180,
                "timeframe": "1d",
                "source_data": {
                    "start_price": 30000,
                    "daily_return_mu": 0.004,
                    "daily_return_std": 0.015,
                    "note": "~146% annualized return; Sharpe > 2.5 expected; MaxDD < 0.12",
                },
                "expected_outcome": {
                    "sharpe_gt": 0.5,
                    "sortino_gt": 0.7,
                    "max_drawdown_lt": 0.15,
                    "all_passed": True,
                },
            }
        )

    # ── Scenario 2: ETH moderate growth ─────────────────────────────────────
    for trial in range(5):
        ohlcv = _make_ohlcv(
            120,
            start_price=1800,
            daily_return_mu=0.001,
            daily_return_std=0.030,
            seed=SEED + 100 + trial,
        )
        returns = _make_returns(ohlcv)
        entries.append(
            {
                "id": f"crypto-eth-moderate-{trial + 1:02d}",
                "query": "Evaluate ETH/USDT returns during moderate growth period",
                "category": "moderate_growth",
                "symbol": "ETH/USDT",
                "symbol_key": "ETH_USDT",
                "ohlcv": {"ETH_USDT": ohlcv},
                "returns": returns,
                "n_bars": 120,
                "timeframe": "1d",
                "source_data": {
                    "start_price": 1800,
                    "daily_return_mu": 0.001,
                    "daily_return_std": 0.030,
                    "note": "Moderate drift; Sharpe borderline 0.5",
                },
                "expected_outcome": {
                    "sharpe_gt": 0.0,
                    "max_drawdown_lt": 0.20,
                },
            }
        )

    # ── Scenario 3: BTC bear market crash (drawdown > 15%) ───────────────────
    for trial in range(5):
        ohlcv = _make_ohlcv(
            180,
            start_price=60000,
            daily_return_mu=-0.003,
            daily_return_std=0.035,
            seed=SEED + 200 + trial,
        )
        returns = _make_returns(ohlcv)
        entries.append(
            {
                "id": f"crypto-btc-bear-{trial + 1:02d}",
                "query": "Evaluate BTC/USDT during bear market (max drawdown expected > 15%)",
                "category": "bear_market",
                "symbol": "BTC/USDT",
                "symbol_key": "BTC_USDT",
                "ohlcv": {"BTC_USDT": ohlcv},
                "returns": returns,
                "n_bars": 180,
                "timeframe": "1d",
                "source_data": {
                    "start_price": 60000,
                    "daily_return_mu": -0.003,
                    "daily_return_std": 0.035,
                    "note": "Negative drift; max drawdown typically > 25%",
                },
                "expected_outcome": {
                    "sharpe_lt": 0.0,
                    "max_drawdown_gt": 0.15,
                    "all_passed": False,
                },
            }
        )

    # ── Scenario 4: low-volatility stablecoin-adjacent (BTC sideways) ────────
    for trial in range(4):
        ohlcv = _make_ohlcv(
            90,
            start_price=28000,
            daily_return_mu=0.0001,
            daily_return_std=0.008,
            seed=SEED + 300 + trial,
        )
        returns = _make_returns(ohlcv)
        entries.append(
            {
                "id": f"crypto-btc-sideways-{trial + 1:02d}",
                "query": "Evaluate BTC/USDT during low-volatility sideways consolidation",
                "category": "low_volatility_sideways",
                "symbol": "BTC/USDT",
                "symbol_key": "BTC_USDT",
                "ohlcv": {"BTC_USDT": ohlcv},
                "returns": returns,
                "n_bars": 90,
                "timeframe": "1d",
                "source_data": {
                    "start_price": 28000,
                    "daily_return_mu": 0.0001,
                    "daily_return_std": 0.008,
                    "note": "Near-zero drift, very low std; Sharpe near zero",
                },
                "expected_outcome": {
                    "note": "Sharpe near 0; SortinoGrader borderline; MaxDrawdown should pass",
                },
            }
        )

    # ── Scenario 5: BTC/ETH spread (multi-symbol) ────────────────────────────
    for trial in range(4):
        ohlcv_btc = _make_ohlcv(
            90,
            start_price=45000,
            daily_return_mu=0.002,
            daily_return_std=0.028,
            seed=SEED + 400 + trial,
        )
        ohlcv_eth = _make_ohlcv(
            90,
            start_price=2200,
            daily_return_mu=0.0025,
            daily_return_std=0.032,
            seed=SEED + 401 + trial,
        )
        returns_btc = _make_returns(ohlcv_btc)
        entries.append(
            {
                "id": f"crypto-btc-eth-spread-{trial + 1:02d}",
                "query": "Evaluate BTC/ETH spread prediction (multi-symbol)",
                "category": "multi_symbol_spread",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "symbol_key": "BTC_USDT",  # primary for single-return eval
                "ohlcv": {
                    "BTC_USDT": ohlcv_btc,
                    "ETH_USDT": ohlcv_eth,
                },
                "returns": returns_btc,  # graders use BTC returns; spread via ohlcv
                "n_bars": 90,
                "timeframe": "1d",
                "source_data": {
                    "btc_start": 45000,
                    "eth_start": 2200,
                    "note": "Spread = BTC close / ETH close at forecast time",
                },
                "expected_outcome": {
                    "sharpe_gt": 0.0,
                    "max_drawdown_lt": 0.20,
                },
            }
        )

    # ── Scenario 6: high-vol crash then recovery (multi-step) ─────────────────
    for trial in range(3):
        rng = np.random.default_rng(SEED + 500 + trial)
        # Phase 1: crash 60 days
        log_r1 = rng.normal(-0.006, 0.04, 60)
        # Phase 2: recovery 60 days
        log_r2 = rng.normal(0.004, 0.03, 60)
        log_returns = np.concatenate([log_r1, log_r2])
        close_prices = 50000 * np.exp(np.cumsum(log_returns))
        close_prices = np.clip(close_prices, 1.0, None)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = 50000
        wick = rng.uniform(0.002, 0.02, 120) * close_prices
        ohlcv = {
            "open": open_prices.tolist(),
            "high": (np.maximum(close_prices, open_prices) + wick).tolist(),
            "low": (np.minimum(close_prices, open_prices) - wick).tolist(),
            "close": close_prices.tolist(),
            "volume": rng.lognormal(10.5, 0.9, 120).tolist(),
        }
        returns = np.diff(np.log(close_prices)).tolist()
        entries.append(
            {
                "id": f"crypto-crash-recovery-{trial + 1:02d}",
                "query": "Evaluate BTC/USDT through crash (60d) then recovery (60d) — multi-step reasoning",
                "category": "crash_then_recovery",
                "symbol": "BTC/USDT",
                "symbol_key": "BTC_USDT",
                "ohlcv": {"BTC_USDT": ohlcv},
                "returns": returns,
                "n_bars": 120,
                "timeframe": "1d",
                "source_data": {
                    "crash_phase_days": 60,
                    "crash_mu": -0.006,
                    "recovery_phase_days": 60,
                    "recovery_mu": 0.004,
                    "start_price": 50000,
                    "note": "Net return near zero; drawdown occurs mid-period then recovers",
                },
                "expected_outcome": {
                    "note": "MaxDrawdown > 15% from crash peak; Sharpe near zero over full window",
                },
            }
        )

    # ── Scenario 7: 4h timeframe (intraday pattern) ───────────────────────────
    for trial in range(3):
        # 4h bars: 6 per day × 90 days = 540 bars
        ohlcv = _make_ohlcv(
            540,
            start_price=40000,
            daily_return_mu=0.0004,
            daily_return_std=0.012,
            seed=SEED + 600 + trial,
        )
        returns = _make_returns(ohlcv)
        entries.append(
            {
                "id": f"crypto-btc-4h-{trial + 1:02d}",
                "query": "Evaluate BTC/USDT on 4h bars — intraday pattern, 540 bars",
                "category": "intraday_4h",
                "symbol": "BTC/USDT",
                "symbol_key": "BTC_USDT",
                "ohlcv": {"BTC_USDT": ohlcv},
                "returns": returns,
                "n_bars": 540,
                "timeframe": "4h",
                "source_data": {
                    "start_price": 40000,
                    "bars_per_day": 6,
                    "daily_return_mu": 0.0004,
                    "daily_return_std": 0.012,
                    "periods_per_year": 2190,
                    "note": "periods_per_year=365*24/4=2190 for annualization",
                },
                "expected_outcome": {
                    "note": "Sharpe annualized with 2190 periods/year; higher than 1d equivalent",
                },
            }
        )

    assert len(entries) >= 30, f"Expected >=30 crypto entries, got {len(entries)}"
    return entries


# ── Write JSONL files ──────────────────────────────────────────────────────────


def write_jsonl(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(entries)} entries → {path}")


def main() -> None:
    forecast_entries = build_forecast_entries()
    segment_entries = build_segment_entries()
    crypto_entries = build_crypto_entries()

    write_jsonl(forecast_entries, GOLDEN_DIR / "forecast_golden_qa.jsonl")
    write_jsonl(segment_entries, GOLDEN_DIR / "segment_golden_qa.jsonl")
    write_jsonl(crypto_entries, GOLDEN_DIR / "crypto_golden_qa.jsonl")

    print(
        f"\nTotals: forecast={len(forecast_entries)}, "
        f"segment={len(segment_entries)}, crypto={len(crypto_entries)}"
    )


if __name__ == "__main__":
    main()
