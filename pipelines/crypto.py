"""
Crypto prediction pipeline — end-to-end runner.

Usage:
    uv run python -m pipelines.crypto
    uv run python -m pipelines.crypto --symbol BTC/USDT --exchange binance --timeframe 1d --cycles 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from core.preprocessing.crypto.fetcher import CryptoFetcher
from core.preprocessing.crypto.indicators import add_all_indicators
from src.agents.crypto.graph import run_crypto_agent
from src.agents.crypto.paper_trading import PaperTrader
from src.agents.crypto.state import Direction, PredictionType

console = Console()

PREDICTIONS_LOG = Path("data/crypto/predictions.jsonl")
PAPER_TRADING_STATE = Path("data/crypto/paper_trader.json")


def run(
    symbols: list[str] | None = None,
    exchange: str = "binance",
    timeframe: str = "1d",
    max_cycles: int = 3,
    learner_policy: str = "bandit",
    limit: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Full crypto prediction pipeline:
      1. Fetch OHLCV data for symbols
      2. Compute technical indicators
      3. Run crypto forecasting agent
      4. Log predictions
      5. Update paper trading positions
      6. Score prior predictions against actuals
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]

    console.print(Panel.fit("[bold cyan]Crypto Prediction Pipeline[/bold cyan]"))

    # 1. Fetch OHLCV
    console.print(Rule("1. Fetch OHLCV Data"))
    ohlcv_data = asyncio.run(_fetch_data(symbols, exchange, timeframe, limit))
    for sym, df in ohlcv_data.items():
        console.print(f"  {sym}: {len(df)} candles")

    # 2. Add indicators
    console.print(Rule("2. Technical Indicators"))
    for sym in ohlcv_data:
        ohlcv_data[sym] = add_all_indicators(ohlcv_data[sym])
    console.print(f"  Added RSI, MACD, Bollinger, ATR, Volume SMA to {len(ohlcv_data)} series")

    # 3. Run agent
    console.print(Rule("3. Crypto Agent"))
    final_state = run_crypto_agent(
        ohlcv_data=ohlcv_data,
        symbols=symbols,
        max_cycles=max_cycles,
        learner_policy=learner_policy,
        verbose=verbose,
    )

    # 4. Log predictions
    console.print(Rule("4. Log Predictions"))
    predictions = final_state.get("predictions", [])
    _log_predictions(predictions)
    console.print(f"  Logged {len(predictions)} predictions → {PREDICTIONS_LOG}")

    # 5. Paper trading
    console.print(Rule("5. Paper Trading"))
    _update_paper_trading(predictions, ohlcv_data)

    console.print(Rule("Done"))
    return final_state


async def _fetch_data(
    symbols: list[str],
    exchange: str,
    timeframe: str,
    limit: int,
) -> dict[str, pl.DataFrame]:
    async with CryptoFetcher(exchange=exchange) as fetcher:
        result = await fetcher.fetch_multiple(symbols=symbols, timeframe=timeframe, limit=limit)
    return {k.replace("/", "_"): v for k, v in result.items()}


def _log_predictions(predictions: list) -> None:
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_LOG.open("a") as f:
        for pred in predictions:
            entry = {
                "timestamp": pred.timestamp.isoformat(),
                "symbol": pred.symbol,
                "type": pred.prediction_type.value,
                "direction": pred.direction.value if pred.direction else None,
                "confidence": pred.direction_confidence,
                "point_forecast": pred.point_forecast[:3] if pred.point_forecast else None,
                "spread_value": pred.spread_value,
                "model": pred.model_used.value,
            }
            f.write(json.dumps(entry) + "\n")


def _update_paper_trading(predictions: list, ohlcv_data: dict) -> None:
    trader = (
        PaperTrader.load(PAPER_TRADING_STATE) if PAPER_TRADING_STATE.exists() else PaperTrader()
    )

    # Close existing positions based on current prices
    current_prices: dict[str, float] = {}
    for sym_key, df in ohlcv_data.items():
        if len(df) > 0:
            symbol = sym_key.replace("_", "/")
            current_prices[symbol] = df["close"].to_list()[-1]

    closed = trader.close_all(current_prices)
    if closed:
        console.print(f"  Closed {len(closed)} positions")

    # Open new positions from direction predictions
    dir_preds = [p for p in predictions if p.prediction_type == PredictionType.DIRECTION]
    for pred in dir_preds:
        if pred.direction in (Direction.UP, Direction.DOWN) and pred.symbol in current_prices:
            confidence = pred.direction_confidence or 0.5
            if confidence >= 0.3:
                trader.open_position(
                    symbol=pred.symbol,
                    direction=pred.direction,
                    entry_price=current_prices[pred.symbol],
                )

    stats = trader.stats()
    console.print(f"  Trades: {stats['total_trades']} | Win rate: {stats['win_rate']:.1%}")
    console.print(f"  P&L: ${stats['cumulative_pnl']:,.2f} | Sharpe: {stats['sharpe']:.2f}")

    trader.save(PAPER_TRADING_STATE)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Crypto prediction pipeline")
    parser.add_argument("--symbol", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--policy", type=str, default="bandit")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    run(
        symbols=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        max_cycles=args.cycles,
        learner_policy=args.policy,
        limit=args.limit,
    )


if __name__ == "__main__":
    _cli()
