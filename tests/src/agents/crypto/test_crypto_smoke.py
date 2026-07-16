"""Smoke tests for the crypto forecasting agent loop."""

from __future__ import annotations

import numpy as np
import polars as pl

from src.agents.crypto.graph import run_crypto_agent
from src.agents.crypto.state import (
    CryptoAgentState,
    CryptoEvalReport,
    CryptoPlannerStrategy,
    CryptoPrediction,
    CryptoTimeframe,
    Direction,
    PredictionType,
)


def _make_ohlcv_df(n: int = 100) -> pl.DataFrame:
    """Generate synthetic OHLCV DataFrame for testing."""
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    close = 42000.0 + np.cumsum(rng.normal(0, 100, n))

    start = date(2024, 1, 1)
    end = start + timedelta(days=n - 1)
    timestamps = pl.date_range(start, end, interval="1d", eager=True)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": close + rng.normal(0, 50, n),
            "high": close + rng.uniform(50, 200, n),
            "low": close - rng.uniform(50, 200, n),
            "close": close,
            "volume": rng.uniform(1000, 10000, n),
        }
    )


class TestCryptoAgentSmoke:
    def test_single_cycle_completes(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=1,
            learner_policy="rule_based",
            verbose=False,
        )

        assert result["cycle_count"] == 1
        assert result["terminate"] is True
        assert len(result["predictions"]) > 0

    def test_produces_direction_predictions(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=1,
            verbose=False,
        )

        dir_preds = [
            p for p in result["predictions"] if p.prediction_type == PredictionType.DIRECTION
        ]
        assert len(dir_preds) > 0
        assert dir_preds[0].direction in (Direction.UP, Direction.DOWN, Direction.NEUTRAL)

    def test_produces_absolute_predictions(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=1,
            verbose=False,
        )

        abs_preds = [
            p for p in result["predictions"] if p.prediction_type == PredictionType.ABSOLUTE
        ]
        assert len(abs_preds) > 0
        assert abs_preds[0].point_forecast is not None
        assert len(abs_preds[0].point_forecast) == 7  # default forecast_bars

    def test_multi_symbol_spread(self):
        btc_df = _make_ohlcv_df()
        eth_df = _make_ohlcv_df()
        eth_df = eth_df.with_columns(pl.col("close") / 15)  # ETH ~$2800

        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": btc_df, "ETH_USDT": eth_df},
            symbols=["BTC/USDT", "ETH/USDT"],
            max_cycles=1,
            verbose=False,
        )

        spread_preds = [
            p for p in result["predictions"] if p.prediction_type == PredictionType.SPREAD
        ]
        assert len(spread_preds) > 0
        assert spread_preds[0].spread_value is not None
        assert spread_preds[0].spread_value > 10  # BTC/ETH ratio > 10

    def test_bandit_policy_works(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=2,
            learner_policy="bandit",
            verbose=False,
        )

        assert result["cycle_count"] >= 1
        assert len(result.get("strategy_history", [])) >= 1

    def test_eval_report_populated(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=1,
            verbose=False,
        )

        report = result["eval_report"]
        assert isinstance(report, CryptoEvalReport)
        assert report.directional_accuracy >= 0
        assert report.max_drawdown >= 0

    def test_strategy_history_accumulates(self):
        ohlcv = _make_ohlcv_df()
        result = run_crypto_agent(
            ohlcv_data={"BTC_USDT": ohlcv},
            symbols=["BTC/USDT"],
            max_cycles=3,
            learner_policy="rule_based",
            verbose=False,
        )

        history = result.get("strategy_history", [])
        assert len(history) >= 1
