"""
LangGraph StateGraph for the crypto forecasting agent.

Graph topology:
  START → crypto_planner → crypto_forecaster → crypto_evaluator → crypto_learner → (loop or END)
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import polars as pl
from langgraph.graph import END, START, StateGraph

from src.agents.crypto.nodes import (
    crypto_evaluator_node,
    crypto_forecaster_node,
    crypto_learner_node,
    crypto_planner_node,
)
from src.agents.crypto.state import CryptoAgentState


def should_continue(state: CryptoAgentState) -> Literal["crypto_planner", "__end__"]:
    if state.get("terminate", False):
        return END
    if state.get("error"):
        return END
    return "crypto_planner"


def build_crypto_graph() -> StateGraph:
    """Build and compile the crypto forecasting agent graph."""
    graph = StateGraph(CryptoAgentState)

    graph.add_node("crypto_planner", crypto_planner_node)
    graph.add_node("crypto_forecaster", crypto_forecaster_node)
    graph.add_node("crypto_evaluator", crypto_evaluator_node)
    graph.add_node("crypto_learner", crypto_learner_node)

    graph.add_edge(START, "crypto_planner")
    graph.add_edge("crypto_planner", "crypto_forecaster")
    graph.add_edge("crypto_forecaster", "crypto_evaluator")
    graph.add_edge("crypto_evaluator", "crypto_learner")
    graph.add_conditional_edges("crypto_learner", should_continue)

    return graph.compile()


def run_crypto_agent(
    ohlcv_data: dict[str, pl.DataFrame],
    symbols: list[str] | None = None,
    max_cycles: int = 3,
    learner_policy: str = "bandit",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the crypto forecasting agent loop.

    Args:
        ohlcv_data: Dict of symbol_key → Polars DataFrame with OHLCV columns
        symbols: Trading pairs to forecast (e.g., ["BTC/USDT", "ETH/USDT"])
        max_cycles: Maximum iterations
        learner_policy: Strategy adaptation policy ("rule_based", "bandit")
        verbose: Print summary

    Returns:
        Final CryptoAgentState as dict
    """
    if symbols is None:
        symbols = list(ohlcv_data.keys())
        symbols = [s.replace("_", "/") for s in symbols]

    serialized_data = {
        k: v.to_dict(as_series=False) if isinstance(v, pl.DataFrame) else v
        for k, v in ohlcv_data.items()
    }

    graph = build_crypto_graph()

    initial_state: CryptoAgentState = {
        "cycle_id": str(uuid.uuid4())[:8],
        "ohlcv_data": serialized_data,
        "symbols": symbols,
        "strategy": None,
        "predictions": [],
        "eval_report": None,
        "learner_feedback": None,
        "strategy_history": [],
        "eval_history": [],
        "cycle_count": 0,
        "max_cycles": max_cycles,
        "terminate": False,
        "error": None,
        "learner_policy_name": learner_policy,
    }

    final_state = graph.invoke(initial_state)

    if verbose:
        _print_crypto_summary(final_state)

    return final_state


def _print_crypto_summary(state: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cycles = state.get("cycle_count", 0)
    console.print(f"\n[bold green]Crypto agent completed — {cycles} cycle(s)[/bold green]")

    eval_history = state.get("eval_history", [])
    if eval_history:
        table = Table(title="Crypto Eval History", show_lines=True)
        table.add_column("Cycle", style="cyan")
        table.add_column("Sharpe")
        table.add_column("Sortino")
        table.add_column("MaxDD")
        table.add_column("Dir%")
        table.add_column("Passed")

        for i, report in enumerate(eval_history):
            passed = "yes" if report.all_passed else "no"
            table.add_row(
                str(i + 1),
                f"{report.sharpe_ratio:.3f}",
                f"{report.sortino_ratio:.3f}",
                f"{report.max_drawdown:.2%}",
                f"{report.directional_accuracy:.1f}%",
                passed,
            )
        console.print(table)

    predictions = state.get("predictions", [])
    if predictions:
        console.print(f"\n[bold]Predictions ({len(predictions)}):[/bold]")
        for pred in predictions[:10]:
            if pred.prediction_type.value == "direction":
                console.print(
                    f"  {pred.symbol} → {pred.direction.value} "
                    f"(confidence: {pred.direction_confidence:.1%})"
                )
            elif pred.prediction_type.value == "absolute":
                console.print(
                    f"  {pred.symbol} → next {len(pred.point_forecast)} bars: "
                    f"[{pred.point_forecast[0]:.2f} ... {pred.point_forecast[-1]:.2f}]"
                )
            elif pred.prediction_type.value == "spread":
                console.print(f"  {pred.spread_pair} ratio: {pred.spread_value:.4f}")
