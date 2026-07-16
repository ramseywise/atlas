"""
Paper trading engine for crypto predictions.

Tracks simulated positions and computes portfolio statistics
without risking real capital.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src.agents.crypto.state import Direction


@dataclass
class Trade:
    """A completed paper trade."""

    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 4),
        }


@dataclass
class OpenPosition:
    """An open paper position."""

    symbol: str
    direction: Direction
    entry_price: float
    entry_time: datetime
    size: float = 1.0


@dataclass
class PaperTrader:
    """Simulated trading engine tracking predictions as positions."""

    trades: list[Trade] = field(default_factory=list)
    open_positions: list[OpenPosition] = field(default_factory=list)
    initial_capital: float = 10000.0

    def open_position(
        self,
        symbol: str,
        direction: Direction,
        entry_price: float,
        timestamp: datetime | None = None,
        size: float = 1.0,
    ) -> None:
        if timestamp is None:
            timestamp = datetime.now(tz=UTC)

        self.open_positions.append(
            OpenPosition(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                entry_time=timestamp,
                size=size,
            )
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime | None = None,
    ) -> Trade | None:
        if timestamp is None:
            timestamp = datetime.now(tz=UTC)

        pos = None
        for p in self.open_positions:
            if p.symbol == symbol:
                pos = p
                break

        if pos is None:
            return None

        self.open_positions.remove(pos)

        if pos.direction == Direction.UP:
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size

        pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price != 0 else 0.0

        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
        )
        self.trades.append(trade)
        return trade

    def close_all(self, prices: dict[str, float], timestamp: datetime | None = None) -> list[Trade]:
        """Close all open positions at given prices."""
        closed = []
        for pos in list(self.open_positions):
            if pos.symbol in prices:
                trade = self.close_position(pos.symbol, prices[pos.symbol], timestamp)
                if trade:
                    closed.append(trade)
        return closed

    @property
    def cumulative_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def sharpe_from_trades(self) -> float:
        if len(self.trades) < 2:
            return 0.0
        returns = np.array([t.pnl_pct for t in self.trades])
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(365))

    def stats(self) -> dict:
        return {
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "cumulative_pnl": round(self.cumulative_pnl, 2),
            "sharpe": round(self.sharpe_from_trades, 4),
            "open_positions": len(self.open_positions),
            "capital": round(self.initial_capital + self.cumulative_pnl, 2),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "initial_capital": self.initial_capital,
            "trades": [t.to_dict() for t in self.trades],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> PaperTrader:
        data = json.loads(path.read_text())
        trader = cls(initial_capital=data["initial_capital"])
        for t in data["trades"]:
            trader.trades.append(
                Trade(
                    symbol=t["symbol"],
                    direction=Direction(t["direction"]),
                    entry_price=t["entry_price"],
                    exit_price=t["exit_price"],
                    entry_time=datetime.fromisoformat(t["entry_time"]),
                    exit_time=datetime.fromisoformat(t["exit_time"]),
                    pnl=t["pnl"],
                    pnl_pct=t["pnl_pct"],
                )
            )
        return trader
