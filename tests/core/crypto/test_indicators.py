"""Tests for technical indicator computation."""

from __future__ import annotations

import numpy as np
import polars as pl

from core.preprocessing.crypto.indicators import (
    add_all_indicators,
    compute_atr,
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_volume_sma,
)


def _make_ohlcv(n: int = 50) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    from datetime import date, timedelta

    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(1000, 10000, n)

    start = date(2024, 1, 1)
    end = start + timedelta(days=n - 1)
    timestamps = pl.date_range(start, end, interval="1d", eager=True)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestRSI:
    def test_rsi_column_exists(self):
        df = compute_rsi(_make_ohlcv())
        assert "rsi" in df.columns

    def test_rsi_bounded(self):
        df = compute_rsi(_make_ohlcv())
        rsi_values = df["rsi"].drop_nulls().to_numpy()
        assert np.all(rsi_values >= 0)
        assert np.all(rsi_values <= 100)

    def test_rsi_custom_period(self):
        df = compute_rsi(_make_ohlcv(100), period=7)
        rsi_values = df["rsi"].drop_nulls()
        assert len(rsi_values) > 0


class TestMACD:
    def test_macd_columns_exist(self):
        df = compute_macd(_make_ohlcv())
        assert "macd_line" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns

    def test_histogram_is_difference(self):
        df = compute_macd(_make_ohlcv(100))
        df_clean = df.drop_nulls(subset=["macd_line", "macd_signal", "macd_histogram"])
        if len(df_clean) > 0:
            diff = (df_clean["macd_line"] - df_clean["macd_signal"]).to_numpy()
            hist = df_clean["macd_histogram"].to_numpy()
            np.testing.assert_allclose(diff, hist, atol=1e-10)


class TestBollinger:
    def test_bollinger_columns_exist(self):
        df = compute_bollinger(_make_ohlcv())
        assert "bb_middle" in df.columns
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns

    def test_upper_above_lower(self):
        df = compute_bollinger(_make_ohlcv(100))
        df_clean = df.drop_nulls(subset=["bb_upper", "bb_lower"])
        if len(df_clean) > 0:
            assert (df_clean["bb_upper"] >= df_clean["bb_lower"]).all()


class TestATR:
    def test_atr_column_exists(self):
        df = compute_atr(_make_ohlcv())
        assert "atr" in df.columns

    def test_atr_non_negative(self):
        df = compute_atr(_make_ohlcv(100))
        atr_values = df["atr"].drop_nulls().to_numpy()
        assert np.all(atr_values >= 0)


class TestVolumeSMA:
    def test_volume_columns_exist(self):
        df = compute_volume_sma(_make_ohlcv())
        assert "volume_sma" in df.columns
        assert "volume_ratio" in df.columns


class TestAddAllIndicators:
    def test_adds_all_columns(self):
        df = add_all_indicators(_make_ohlcv(100))
        expected = {
            "rsi",
            "macd_line",
            "macd_signal",
            "macd_histogram",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "atr",
            "volume_sma",
            "volume_ratio",
        }
        assert expected.issubset(set(df.columns))

    def test_preserves_original_columns(self):
        original = _make_ohlcv()
        result = add_all_indicators(original)
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in result.columns
