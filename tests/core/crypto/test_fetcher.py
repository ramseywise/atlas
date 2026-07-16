"""Tests for CryptoFetcher with mocked CCXT exchange."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from core.preprocessing.crypto.fetcher import CryptoFetcher


MOCK_OHLCV = [
    [1704067200000, 42000.0, 42500.0, 41800.0, 42200.0, 1500.0],
    [1704153600000, 42200.0, 42800.0, 42100.0, 42600.0, 1200.0],
    [1704240000000, 42600.0, 43000.0, 42400.0, 42900.0, 1800.0],
]


@pytest.fixture
def mock_exchange():
    exchange = AsyncMock()
    exchange.fetch_ohlcv = AsyncMock(return_value=MOCK_OHLCV)
    exchange.close = AsyncMock()
    return exchange


@pytest.mark.asyncio
async def test_fetch_ohlcv_returns_dataframe(mock_exchange):
    fetcher = CryptoFetcher(exchange="binance")
    fetcher._exchange = mock_exchange

    df = await fetcher.fetch_ohlcv(symbol="BTC/USDT", timeframe="1d", limit=3)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume"}


@pytest.mark.asyncio
async def test_fetch_ohlcv_correct_values(mock_exchange):
    fetcher = CryptoFetcher(exchange="binance")
    fetcher._exchange = mock_exchange

    df = await fetcher.fetch_ohlcv(symbol="BTC/USDT")

    assert df["close"].to_list() == [42200.0, 42600.0, 42900.0]
    assert df["volume"].to_list() == [1500.0, 1200.0, 1800.0]


@pytest.mark.asyncio
async def test_fetch_ohlcv_empty_response(mock_exchange):
    mock_exchange.fetch_ohlcv = AsyncMock(return_value=[])
    fetcher = CryptoFetcher(exchange="binance")
    fetcher._exchange = mock_exchange

    df = await fetcher.fetch_ohlcv(symbol="BTC/USDT")

    assert len(df) == 0


@pytest.mark.asyncio
async def test_fetch_multiple(mock_exchange):
    fetcher = CryptoFetcher(exchange="binance")
    fetcher._exchange = mock_exchange

    result = await fetcher.fetch_multiple(symbols=["BTC/USDT", "ETH/USDT"])

    assert "BTC/USDT" in result
    assert "ETH/USDT" in result
    assert len(result["BTC/USDT"]) == 3


@pytest.mark.asyncio
async def test_context_manager(mock_exchange):
    with patch(
        "core.preprocessing.crypto.fetcher.CryptoFetcher._get_exchange", return_value=mock_exchange
    ):
        fetcher = CryptoFetcher(exchange="binance")
        fetcher._exchange = mock_exchange
        async with fetcher:
            df = await fetcher.fetch_ohlcv(symbol="BTC/USDT")
            assert len(df) == 3
    mock_exchange.close.assert_called_once()


@pytest.mark.asyncio
async def test_since_parameter_converted(mock_exchange):
    fetcher = CryptoFetcher(exchange="binance")
    fetcher._exchange = mock_exchange
    since = datetime(2024, 1, 1, tzinfo=timezone.utc)

    await fetcher.fetch_ohlcv(symbol="BTC/USDT", since=since)

    call_kwargs = mock_exchange.fetch_ohlcv.call_args
    assert call_kwargs[1]["since"] == int(since.timestamp() * 1000)
