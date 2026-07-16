"""
Learner policy comparison harness.

Runs the same walk-forward CV folds with different learner policies and
compares convergence speed, terminal MASE, and strategy stability.
"""

from __future__ import annotations

from datetime import date

from rich.console import Console
from rich.table import Table

from core.preprocessing.synthetic import generate_sequence_dataset, temporal_split
from src.agents.graph import run_forecasting_agent

console = Console()


def run_learner_comparison(
    policies: list[str] | None = None,
    n_days: int = 2 * 365,
    max_cycles: int = 4,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Compare learner policies on the same dataset.

    Returns dict of policy_name → {final_mase, final_smape, cycles_to_pass, strategy_changes}.
    """
    if policies is None:
        policies = ["rule_based", "bandit"]

    df = generate_sequence_dataset(start_date=date(2022, 1, 1), n_days=n_days, seed=seed)
    split = temporal_split(df, val_frac=0.15, test_frac=0.20)
    train_val = split.train.vstack(split.val)

    results: dict[str, dict] = {}

    for policy_name in policies:
        console.print(f"\n[bold cyan]Running policy: {policy_name}[/bold cyan]")
        final_state = run_forecasting_agent(
            series_df=train_val,
            max_cycles=max_cycles,
            learner_policy=policy_name,
            verbose=False,
        )

        eval_history = final_state.get("eval_history", [])
        last_report = eval_history[-1] if eval_history else None

        cycles_to_pass = None
        for i, report in enumerate(eval_history):
            if report.all_passed:
                cycles_to_pass = i + 1
                break

        total_changes = sum(len(s.rationale) > 0 for s in final_state.get("strategy_history", []))

        results[policy_name] = {
            "final_mase": last_report.overall_mase if last_report else None,
            "final_smape": last_report.overall_smape if last_report else None,
            "cycles_run": final_state.get("cycle_count", 0),
            "cycles_to_pass": cycles_to_pass,
            "strategy_changes": total_changes,
            "all_passed": last_report.all_passed if last_report else False,
        }

    _print_comparison(results)
    return results


def _print_comparison(results: dict[str, dict]) -> None:
    table = Table(title="Learner Policy Comparison", show_lines=True)
    table.add_column("Policy", style="cyan")
    table.add_column("Final MASE")
    table.add_column("Final SMAPE")
    table.add_column("Cycles Run")
    table.add_column("Cycles to Pass")
    table.add_column("Strategy Changes")
    table.add_column("All Passed")

    for name, data in results.items():
        table.add_row(
            name,
            f"{data['final_mase']:.3f}" if data["final_mase"] else "N/A",
            f"{data['final_smape']:.1f}%" if data["final_smape"] else "N/A",
            str(data["cycles_run"]),
            str(data["cycles_to_pass"]) if data["cycles_to_pass"] else "—",
            str(data["strategy_changes"]),
            "yes" if data["all_passed"] else "no",
        )

    console.print(table)


if __name__ == "__main__":
    run_learner_comparison()
