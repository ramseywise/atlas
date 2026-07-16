"""
Segmentation pipeline — end-to-end runner for the customer segmentation agent.

Generates synthetic multi-customer data, writes it to a temp Parquet file,
runs the segmentation agent, prints segment assignments and quality metrics.

Usage:
    uv run python -m pipelines.segment
    uv run python -m pipelines.segment --customers 20 --days 365 --cycles 3
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from core.preprocessing.synthetic import generate_multi_customer_dataset
from src.agents.segment.graph import run_segmentation_agent

console = Console()


def _make_multi_customer_df(
    n_customers: int,
    n_days: int,
    seed: int,
) -> pl.DataFrame:
    """
    Synthesise a multi-customer CashFlowRecord DataFrame using archetype-based generation.

    Each customer is assigned a CustomerArchetype (weighted random), producing
    genuinely different cash flow shapes suitable for meaningful segmentation.
    """
    return generate_multi_customer_dataset(
        n_customers=n_customers,
        n_days=n_days,
        seed=seed,
    ).with_columns(pl.col("value").alias("amount"))


def run(
    n_customers: int = 50,
    n_days: int = 365,
    max_cycles: int = 3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Full segmentation pipeline:
      1. Generate synthetic multi-customer cash flow data
      2. Write to temp Parquet
      3. Run segmentation agent
      4. Print results

    Returns the final agent state.
    """
    console.print(Panel.fit("[bold magenta]Segmentation Pipeline[/bold magenta]"))

    # 1. Data
    console.print(Rule("1. Multi-Customer Synthetic Data"))
    df = _make_multi_customer_df(n_customers=n_customers, n_days=n_days, seed=seed)
    archetype_counts = (
        df.select(["customer_id", "archetype"]).unique()
        .group_by("archetype").len().sort("archetype")
    ) if "archetype" in df.columns else None
    console.print(f"  {n_customers} customers | {len(df):,} rows | {n_days} days each")
    if archetype_counts is not None:
        for row in archetype_counts.iter_rows(named=True):
            console.print(f"    {row['archetype']}: {row['len']} customers")

    # 2. Write to Parquet
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_path = f.name
    df.write_parquet(tmp_path)
    console.print(f"  Written → {tmp_path}")

    # 3. Run agent
    console.print(Rule("2. Segmentation Agent"))
    final_state = run_segmentation_agent(
        customer_df_path=tmp_path,
        max_cycles=max_cycles,
        verbose=verbose,
    )

    # 4. Results table
    result = final_state.get("result")
    report = final_state.get("eval_report")

    if result and report:
        console.print(Rule("3. Results"))
        table = Table(title="Segment Assignments", show_lines=True)
        table.add_column("Segment ID", style="cyan")
        table.add_column("Label")
        table.add_column("Description")
        table.add_column("Size", style="magenta")

        for cid, info in result["segment_names"].items():
            size = report.cluster_sizes.get(cid, "?")
            table.add_row(str(cid), info["label"], info["description"], str(size))

        console.print(table)
        console.print(
            f"\n  Quality: sil={report.silhouette:.3f} "
            f"db={report.davies_bouldin:.3f} "
            f"{'[green]PASS[/green]' if report.all_passed else '[red]FAIL[/red]'}"
        )

    Path(tmp_path).unlink(missing_ok=True)
    return final_state


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Customer segmentation pipeline")
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(
        n_customers=args.customers,
        n_days=args.days,
        max_cycles=args.cycles,
        seed=args.seed,
    )


if __name__ == "__main__":
    _cli()
