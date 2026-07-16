"""
Async CCXT-based crypto OHLCV fetcher.

Supports any exchange in the CCXT library (Binance, Coinbase, Kraken, etc.).
Returns Polars DataFrames with standardized columns.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import polars as pl

OHLCV_SCHEMA = {
    "timestamp": pl.Datetime("ms"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class CryptoFetcher:
    """Fetch OHLCV candle data from crypto exchanges via CCXT."""

    def __init__(self, exchange: str = "binance") -> None:
        self.exchange_id = exchange
        self._exchange = None

    async def _get_exchange(self):
        if self._exchange is None:
            import ccxt.async_support as ccxt_async

            exchange_cls = getattr(ccxt_async, self.exchange_id)
            self._exchange = exchange_cls({"enableRateLimit": True})
        return self._exchange

    async def fetch_ohlcv(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        since: datetime | None = None,
        limit: int = 500,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV candles from the exchange.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH/USDT")
            timeframe: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d", "1w")
            since: Start time (UTC). If None, fetches most recent candles.
            limit: Maximum number of candles to fetch.

        Returns:
            Polars DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        exchange = await self._get_exchange()
        since_ms = int(since.timestamp() * 1000) if since else None

        ohlcv = await exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit,
        )

        if not ohlcv:
            return pl.DataFrame(schema=OHLCV_SCHEMA)

        df = pl.DataFrame(
            ohlcv,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        ).with_columns(
            pl.col("timestamp").cast(pl.Datetime("ms")),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )

        return df

    async def fetch_multiple(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        since: datetime | None = None,
        limit: int = 500,
    ) -> dict[str, pl.DataFrame]:
        """Fetch OHLCV for multiple symbols concurrently."""
        tasks = [
            self.fetch_ohlcv(symbol=sym, timeframe=timeframe, since=since, limit=limit)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, pl.DataFrame] = {}
        for sym, result in zip(symbols, results, strict=False):
            if isinstance(result, pl.DataFrame):
                output[sym] = result
        return output

    async def close(self) -> None:
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


def fetch_ohlcv_sync(
    symbol: str = "BTC/USDT",
    exchange: str = "binance",
    timeframe: str = "1d",
    since: datetime | None = None,
    limit: int = 500,
) -> pl.DataFrame:
    """Synchronous convenience wrapper around CryptoFetcher."""

    async def _run() -> pl.DataFrame:
        async with CryptoFetcher(exchange=exchange) as fetcher:
            return await fetcher.fetch_ohlcv(
                symbol=symbol, timeframe=timeframe, since=since, limit=limit
            )

    return asyncio.run(_run())
