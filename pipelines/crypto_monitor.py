"""
Crypto monitoring pipeline — designed to run on a schedule (cron/systemd timer).

Fetches latest candle, scores outstanding predictions, logs results.

Usage:
    uv run python -m pipelines.crypto_monitor
    uv run python -m pipelines.crypto_monitor --symbol BTC/USDT --exchange binance
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from core.preprocessing.crypto.fetcher import CryptoFetcher

console = Console()

PREDICTIONS_LOG = Path("data/crypto/predictions.jsonl")
MONITOR_LOG = Path("data/crypto/monitor_log.jsonl")


def run_monitoring_cycle(
    symbols: list[str] | None = None,
    exchange: str = "binance",
    timeframe: str = "1d",
) -> dict:
    """
    One monitoring cycle:
      1. Fetch latest prices
      2. Load outstanding predictions
      3. Score predictions against actuals
      4. Log results
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]

    console.print("[bold cyan]Crypto Monitor — scoring predictions[/bold cyan]")

    # Fetch current prices
    current_prices = asyncio.run(_fetch_current_prices(symbols, exchange, timeframe))
    console.print(f"  Current prices: {current_prices}")

    # Load and score predictions
    scored = _score_predictions(current_prices)

    # Log results
    _log_monitoring(scored)

    # Print summary
    _print_summary(scored)

    return {"scored": len(scored), "prices": current_prices}


async def _fetch_current_prices(
    symbols: list[str],
    exchange: str,
    timeframe: str,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    async with CryptoFetcher(exchange=exchange) as fetcher:
        for symbol in symbols:
            df = await fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1)
            if len(df) > 0:
                prices[symbol] = df["close"].to_list()[-1]
    return prices


def _score_predictions(current_prices: dict[str, float]) -> list[dict]:
    """Score outstanding direction predictions against current prices."""
    if not PREDICTIONS_LOG.exists():
        return []

    scored: list[dict] = []
    with PREDICTIONS_LOG.open() as f:
        for line in f:
            pred = json.loads(line.strip())
            if pred.get("type") != "direction":
                continue
            if pred.get("direction") is None:
                continue

            symbol = pred["symbol"]
            if symbol not in current_prices:
                continue

            current_price = current_prices[symbol]
            # We need entry price — use point_forecast baseline or skip
            entry_price = (
                pred.get("point_forecast", [None])[0] if pred.get("point_forecast") else None
            )
            if entry_price is None:
                continue

            actual_direction = "up" if current_price > entry_price else "down"
            correct = pred["direction"] == actual_direction
            pnl_pct = (current_price - entry_price) / entry_price if entry_price != 0 else 0.0

            if pred["direction"] == "down":
                pnl_pct = -pnl_pct

            scored.append(
                {
                    "timestamp": pred["timestamp"],
                    "symbol": symbol,
                    "predicted": pred["direction"],
                    "actual": actual_direction,
                    "correct": correct,
                    "confidence": pred.get("confidence"),
                    "pnl_pct": round(pnl_pct, 4),
                }
            )

    return scored


def _log_monitoring(scored: list[dict]) -> None:
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a") as f:
        for entry in scored:
            entry["scored_at"] = datetime.now(tz=UTC).isoformat()
            f.write(json.dumps(entry) + "\n")


def _print_summary(scored: list[dict]) -> None:
    if not scored:
        console.print("  No predictions to score")
        return

    table = Table(title="Prediction Scorecard", show_lines=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Predicted")
    table.add_column("Actual")
    table.add_column("Correct")
    table.add_column("P&L %")

    for entry in scored[-10:]:
        table.add_row(
            entry["symbol"],
            entry["predicted"],
            entry["actual"],
            "yes" if entry["correct"] else "no",
            f"{entry['pnl_pct']:.2%}",
        )
    console.print(table)

    correct_count = sum(1 for s in scored if s["correct"])
    total = len(scored)
    console.print(f"\n  Accuracy: {correct_count}/{total} ({correct_count / total:.1%})")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Crypto monitoring cycle")
    parser.add_argument("--symbol", nargs="+", default=["BTC/USDT", "ETH/USDT"])
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1d")
    args = parser.parse_args()
    run_monitoring_cycle(symbols=args.symbol, exchange=args.exchange, timeframe=args.timeframe)


if __name__ == "__main__":
    _cli()
