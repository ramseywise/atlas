"""
Forecast pipeline — end-to-end runner for the cash flow forecasting agent.

Usage:
    uv run python -m pipelines.forecast
    uv run python -m pipelines.forecast --days 730 --cycles 4
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from core.preprocessing.synthetic import generate_sequence_dataset, temporal_split
from src.agents.graph import run_forecasting_agent
from src.cv_runner import run_walk_forward_cv

console = Console()


def run(
    n_days: int = 3 * 365,
    max_cycles: int = 4,
    cv_folds: int = 4,
    seed: int = 42,
    drift_log: Path = Path("evals/reports/drift_log.jsonl"),
    verbose: bool = True,
) -> dict:
    """
    Full forecast pipeline:
      1. Generate synthetic data
      2. Temporal split
      3. Walk-forward CV
      4. Full agent run on train+val
      5. Log drift events

    Returns the final agent state.
    """
    console.print(Panel.fit("[bold cyan]Forecast Pipeline[/bold cyan]"))

    # 1. Data
    console.print(Rule("1. Synthetic Data"))
    df = generate_sequence_dataset(
        start_date=date(2021, 1, 1),
        n_days=n_days,
        seed=seed,
    )
    console.print(
        f"  {len(df):,} rows | {df['series_id'].n_unique()} series | "
        f"{df['date'].min()} → {df['date'].max()}"
    )

    # 2. Split
    console.print(Rule("2. Temporal Split"))
    split = temporal_split(df, val_frac=0.15, test_frac=0.20)
    console.print(split.summary())

    # 3. CV
    console.print(Rule("3. Walk-Forward CV"))
    cv_summary = run_walk_forward_cv(
        full_train_df=split.train,
        horizon_days=30,
        min_train_days=200,
        step_days=30,
        max_folds=cv_folds,
        max_agent_cycles=2,
        verbose=verbose,
    )
    console.print(f"  CV mean MASE: {cv_summary.mean_mase:.3f} ± {cv_summary.std_mase:.3f}")

    # 4. Full run
    console.print(Rule("4. Full Agent Run (train+val)"))
    train_and_val = split.train.vstack(split.val)
    final_state = run_forecasting_agent(
        series_df=train_and_val,
        max_cycles=max_cycles,
        verbose=verbose,
    )

    # 5. Drift log
    drift_log.parent.mkdir(parents=True, exist_ok=True)
    feedback = final_state.get("learner_feedback")
    if feedback and feedback.drift_detected:
        entry = {
            "cycle_id": feedback.cycle_id,
            "drift_ratio": final_state["eval_report"].drift_ratio
            if final_state.get("eval_report")
            else None,
            "finetune_triggered": feedback.drift_triggered_finetune,
            "reflection": feedback.reflection_text,
        }
        with drift_log.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        console.print(f"\n[yellow]Drift logged → {drift_log}[/yellow]")
    else:
        console.print("\n[green]No drift detected[/green]")

    console.print(Rule("Done"))
    return final_state


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Cash flow forecast pipeline")
    parser.add_argument("--days", type=int, default=3 * 365)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(n_days=args.days, max_cycles=args.cycles, cv_folds=args.folds, seed=args.seed)


if __name__ == "__main__":
    _cli()
