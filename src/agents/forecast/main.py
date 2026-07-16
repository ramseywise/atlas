"""
Main entrypoint for the cash flow forecasting agent demo.

Runs:
  1. Synthetic sequence data generation
  2. Temporal split (train / val / test — no leakage)
  3. Walk-forward CV on the train split
  4. Full agent run on train+val
  5. Model comparison: ARIMA vs Chronos vs Agent
  6. Drift log inspection
"""

from __future__ import annotations

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


def main() -> None:
    console.print(Panel.fit("[bold cyan]Cash Flow Forecasting Agent[/bold cyan]"))

    # ── 1. Generate synthetic data ────────────────────────────────────────────
    console.print(Rule("1. Synthetic Data Generation"))
    df = generate_sequence_dataset(
        start_date=date(2021, 1, 1),
        n_days=3 * 365,
        seed=42,
    )
    console.print(f"Generated {len(df):,} rows × {len(df.columns)} cols")
    console.print(f"Series: {df['series_id'].unique().to_list()}")
    console.print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    console.print(Rule("2. Temporal Split (no leakage)"))
    split = temporal_split(df, val_frac=0.15, test_frac=0.20)
    console.print(split.summary())
    console.print("[dim]Test set held out — not touched until final evaluation[/dim]")

    # ── 3. Walk-forward CV on train split ─────────────────────────────────────
    console.print(Rule("3. Walk-Forward Cross-Validation (Agent Loop)"))

    # cv_runner uses 'value' col from sequence dataset
    cv_summary = run_walk_forward_cv(
        full_train_df=split.train,
        horizon_days=30,
        min_train_days=200,
        step_days=30,
        max_folds=4,
        max_agent_cycles=2,
        verbose=True,
    )

    # ── 4. Full agent run on train+val ────────────────────────────────────────
    console.print(Rule("4. Full Agent Run (train + val)"))
    train_and_val = split.train.vstack(split.val)
    final_state = run_forecasting_agent(
        series_df=train_and_val,
        max_cycles=4,
        verbose=True,
    )

    # ── 5. Model comparison ───────────────────────────────────────────────────
    console.print(Rule("5. Model Comparison: ARIMA vs Chronos vs Agent"))
    try:
        from evals.comparison import run_model_comparison
        comparison = run_model_comparison(
            train_df=split.train,
            horizon_days=30,
            min_train_days=200,
            step_days=30,
            max_folds=2,       # keep fast for demo
            models=["arima", "chronos"],   # exclude agent to avoid duplicate CV
            verbose=True,
        )
    except Exception as e:
        console.print(f"[yellow]Model comparison skipped: {e}[/yellow]")

    # ── 6. Drift log ──────────────────────────────────────────────────────────
    drift_log_path = Path("evals/reports/drift_log.jsonl")
    drift_log_path.parent.mkdir(parents=True, exist_ok=True)

    feedback = final_state.get("learner_feedback")
    if feedback and feedback.drift_detected:
        entry = {
            "cycle_id": feedback.cycle_id,
            "drift_ratio": final_state["eval_report"].drift_ratio
            if final_state.get("eval_report") else None,
            "finetune_triggered": feedback.drift_triggered_finetune,
            "reflection": feedback.reflection_text,
        }
        with drift_log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        console.print(f"\n[yellow]Drift event logged → {drift_log_path}[/yellow]")
    else:
        console.print("\n[green]No drift detected this run[/green]")

    console.print(Rule("Done"))
    console.print(f"CV mean MASE: {cv_summary.mean_mase:.3f} ± {cv_summary.std_mase:.3f}")
    console.print(
        f"Final run cycles: {final_state['cycle_count']} | "
        f"All graders passed: {final_state['eval_report'].all_passed if final_state.get('eval_report') else 'N/A'}"
    )


if __name__ == "__main__":
    main()
