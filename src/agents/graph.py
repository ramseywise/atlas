"""
LangGraph StateGraph for the cash flow forecasting agent.

Graph topology:
  START → planner → forecaster → evaluator → learner → (loop or END)

The learner decides whether to continue cycling or terminate based on:
  - max_cycles reached
  - all graders passing for 2+ consecutive cycles
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import polars as pl
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    evaluator_node,
    forecaster_node,
    learner_node,
    planner_node,
)
from src.agents.state import AgentState

# ── Routing ───────────────────────────────────────────────────────────────────


def should_continue(state: AgentState) -> Literal["planner", "__end__"]:
    """Route: loop back to planner, or terminate."""
    if state.get("terminate", False):
        return END
    if state.get("error"):
        return END
    return "planner"


# ── Graph Factory ─────────────────────────────────────────────────────────────


def build_forecasting_graph() -> StateGraph:
    """Build and compile the forecasting agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("forecaster", forecaster_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("learner", learner_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "forecaster")
    graph.add_edge("forecaster", "evaluator")
    graph.add_edge("evaluator", "learner")
    graph.add_conditional_edges("learner", should_continue)

    return graph.compile()


# ── Runner ────────────────────────────────────────────────────────────────────


def run_forecasting_agent(
    series_df: pl.DataFrame,
    actuals: dict[str, list[float]] | None = None,
    max_cycles: int = 5,
    learner_policy: str = "rule_based",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the full forecasting agent loop.

    Args:
        series_df:   Polars DataFrame with columns [date, series_id, category, value]
        actuals:     Optional dict of series_id → actual values for the forecast period.
                     If None, uses pseudo-actuals from the tail of training data.
        max_cycles:  Maximum number of plan→forecast→eval→learn iterations
        learner_policy: Policy for strategy adaptation ("rule_based", "bandit")
        verbose:     Print cycle summaries

    Returns:
        Final AgentState as a dict
    """
    graph = build_forecasting_graph()

    initial_state: AgentState = {
        "cycle_id": str(uuid.uuid4())[:8],
        "series_data": series_df.to_dict(as_series=False),
        "actuals": actuals,
        "strategy": None,
        "forecasts": [],
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
        _print_run_summary(final_state)

    return final_state


def _print_run_summary(state: dict[str, Any]) -> None:
    """Print a structured run summary."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    cycles = state.get("cycle_count", 0)

    console.print(f"\n[bold green]✓ Forecasting agent completed — {cycles} cycle(s)[/bold green]")

    # Eval history table
    eval_history = state.get("eval_history", [])
    if eval_history:
        table = Table(title="Eval History", show_lines=True)
        table.add_column("Cycle", style="cyan")
        table.add_column("MASE")
        table.add_column("SMAPE")
        table.add_column("Dir%")
        table.add_column("Cov80%")
        table.add_column("Drift")
        table.add_column("Passed")

        for i, report in enumerate(eval_history):
            drift_flag = "⚠️" if report.drift_ratio > 1.2 else "✓"
            passed = "✅" if report.all_passed else "❌"
            table.add_row(
                str(i + 1),
                f"{report.overall_mase:.3f}",
                f"{report.overall_smape:.1f}%",
                f"{report.directional_accuracy:.1f}%",
                f"{report.coverage_80:.1f}%",
                f"{report.drift_ratio:.2f} {drift_flag}",
                passed,
            )
        console.print(table)

    # Strategy evolution
    strategy_history = state.get("strategy_history", [])
    if strategy_history:
        console.print("\n[bold]Strategy Evolution:[/bold]")
        for i, s in enumerate(strategy_history):
            console.print(
                f"  Cycle {i + 1}: {s.model_variant.value} | "
                f"ctx×{s.context_multiplier} | {s.horizon.value}"
            )

    # Final learner feedback
    feedback = state.get("learner_feedback")
    if feedback and feedback.drift_triggered_finetune:
        console.print(
            Panel(
                "[bold red]⚠️  DRIFT DETECTED — Fine-tune trigger logged[/bold red]\n"
                f"{feedback.reflection_text}",
                title="Drift Alert",
            )
        )
