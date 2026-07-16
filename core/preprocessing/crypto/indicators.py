"""
Technical indicator computation using pure Polars expressions.

All functions take a DataFrame with OHLCV columns and return it with new columns added.
"""

from __future__ import annotations

import polars as pl


def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Relative Strength Index (0-100). Oversold < 30, Overbought > 70."""
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=period)

    return df.with_columns(
        (100.0 - 100.0 / (1.0 + gain / loss)).alias("rsi"),
    )


def compute_macd(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> pl.DataFrame:
    """MACD line, signal line, and histogram."""
    return (
        df.with_columns(
            (
                pl.col("close").ewm_mean(span=fast, adjust=False)
                - pl.col("close").ewm_mean(span=slow, adjust=False)
            ).alias("macd_line"),
        )
        .with_columns(
            pl.col("macd_line").ewm_mean(span=signal_period, adjust=False).alias("macd_signal"),
        )
        .with_columns(
            (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram"),
        )
    )


def compute_bollinger(
    df: pl.DataFrame,
    period: int = 20,
    std_mult: float = 2.0,
) -> pl.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower."""
    return (
        df.with_columns(
            pl.col("close").rolling_mean(window_size=period).alias("bb_middle"),
            pl.col("close").rolling_std(window_size=period).alias("_bb_std"),
        )
        .with_columns(
            (pl.col("bb_middle") + std_mult * pl.col("_bb_std")).alias("bb_upper"),
            (pl.col("bb_middle") - std_mult * pl.col("_bb_std")).alias("bb_lower"),
        )
        .drop("_bb_std")
    )


def compute_atr(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Average True Range — volatility indicator."""
    high_low = pl.col("high") - pl.col("low")
    high_close = (pl.col("high") - pl.col("close").shift(1)).abs()
    low_close = (pl.col("low") - pl.col("close").shift(1)).abs()

    tr = (
        pl.when(high_low >= high_close)
        .then(pl.when(high_low >= low_close).then(high_low).otherwise(low_close))
        .otherwise(pl.when(high_close >= low_close).then(high_close).otherwise(low_close))
    )

    return df.with_columns(
        tr.rolling_mean(window_size=period).alias("atr"),
    )


def compute_volume_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Volume simple moving average for volume spike detection."""
    return df.with_columns(
        pl.col("volume").rolling_mean(window_size=period).alias("volume_sma"),
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=period)).alias(
            "volume_ratio"
        ),
    )


def add_all_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Add all technical indicators to an OHLCV DataFrame."""
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger(df)
    df = compute_atr(df)
    df = compute_volume_sma(df)
    return df
