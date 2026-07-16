"""
Walk-forward cross-validation runner for the forecasting agent.

Follows strict statistical inference rules:
- Expanding window: train grows, val window slides forward
- No leakage between folds
- Reports per-fold metrics and aggregate stats
- Separate from test set (test set never touched here)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from src.agents.graph import run_forecasting_agent
from src.agents.state import EvalReport
from core.preprocessing.synthetic import WalkForwardFold, walk_forward_cv

console = Console()


@dataclass
class CVResult:
    """Results from a single CV fold."""

    fold_idx: int
    train_end: date
    val_end: date
    train_rows: int
    val_rows: int
    eval_report: EvalReport | None
    cycles_used: int
    error: str | None = None

    @property
    def mase(self) -> float:
        return self.eval_report.overall_mase if self.eval_report else float("nan")

    @property
    def smape(self) -> float:
        return self.eval_report.overall_smape if self.eval_report else float("nan")

    @property
    def dir_acc(self) -> float:
        return self.eval_report.directional_accuracy if self.eval_report else float("nan")

    @property
    def cov_80(self) -> float:
        return self.eval_report.coverage_80 if self.eval_report else float("nan")


@dataclass
class CVSummary:
    """Aggregate CV results across all folds."""

    folds: list[CVResult]
    mean_mase: float
    std_mase: float
    mean_smape: float
    std_smape: float
    mean_dir: float
    mean_cov: float
    n_folds: int
    n_passed: int

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_folds if self.n_folds > 0 else 0.0

    def print_summary(self) -> None:
        table = Table(title=f"Walk-Forward CV Summary — {self.n_folds} folds", show_lines=True)
        table.add_column("Fold", style="cyan")
        table.add_column("Train → Val")
        table.add_column("MASE")
        table.add_column("SMAPE")
        table.add_column("Dir%")
        table.add_column("Cov80%")
        table.add_column("Passed")

        for r in self.folds:
            passed = "✅" if (r.eval_report and r.eval_report.all_passed) else "❌"
            table.add_row(
                str(r.fold_idx + 1),
                f"{r.train_end} → {r.val_end}",
                f"{r.mase:.3f}",
                f"{r.smape:.1f}%",
                f"{r.dir_acc:.1f}%",
                f"{r.cov_80:.1f}%",
                passed,
            )

        console.print(table)
        console.print(
            f"\n[bold]Aggregate:[/bold] "
            f"MASE={self.mean_mase:.3f}±{self.std_mase:.3f} | "
            f"SMAPE={self.mean_smape:.1f}% | "
            f"Dir={self.mean_dir:.1f}% | "
            f"Cov={self.mean_cov:.1f}% | "
            f"Pass rate={self.pass_rate:.0%}"
        )


def run_walk_forward_cv(
    full_train_df: pl.DataFrame,
    horizon_days: int = 30,
    min_train_days: int = 365,
    step_days: int = 30,
    max_folds: int = 6,
    max_agent_cycles: int = 3,
    verbose: bool = True,
) -> CVSummary:
    """
    Run walk-forward cross-validation using the full forecasting agent.

    Each fold:
      1. Slices training data up to fold.train_end
      2. Runs the agent (full plan→forecast→eval→learn loop)
      3. Evaluates against fold.val data as actuals
      4. Records metrics

    Args:
        full_train_df:   DataFrame with [date, series_id, category, value]
                         This is the TRAIN split only — test never touched here.
        horizon_days:    Forecast horizon per fold (days)
        min_train_days:  Minimum training history before first fold
        step_days:       How far each fold advances (non-overlapping val windows)
        max_folds:       Cap on number of folds
        max_agent_cycles: Agent internal iterations per fold (keep low for CV speed)
        verbose:         Print per-fold progress

    Returns:
        CVSummary with per-fold and aggregate metrics
    """
    folds = walk_forward_cv(
        df=full_train_df,
        horizon_days=horizon_days,
        min_train_days=min_train_days,
        step_days=step_days,
        max_folds=max_folds,
    )

    if verbose:
        console.print(f"\n[bold blue]Walk-Forward CV: {len(folds)} folds[/bold blue]")

    results: list[CVResult] = []

    for fold in folds:
        if verbose:
            console.print(
                f"  Fold {fold.fold_idx + 1}: "
                f"train→{fold.train_end} | val {fold.val_end}"
            )

        # Build actuals dict from validation fold
        actuals_by_series: dict[str, list[float]] = {}
        for sid in fold.val["series_id"].unique().to_list():
            vals = (
                fold.val.filter(pl.col("series_id") == sid)
                .sort("date")
                .head(horizon_days)["value"]
                .to_list()
            )
            actuals_by_series[sid] = vals

        try:
            final_state = run_forecasting_agent(
                series_df=fold.train,
                actuals=actuals_by_series,
                max_cycles=max_agent_cycles,
                verbose=False,
            )
            eval_report: EvalReport | None = final_state.get("eval_report")
            cycles_used = final_state.get("cycle_count", 0)

            result = CVResult(
                fold_idx=fold.fold_idx,
                train_end=fold.train_end,
                val_end=fold.val_end,
                train_rows=len(fold.train),
                val_rows=len(fold.val),
                eval_report=eval_report,
                cycles_used=cycles_used,
            )
        except Exception as e:
            result = CVResult(
                fold_idx=fold.fold_idx,
                train_end=fold.train_end,
                val_end=fold.val_end,
                train_rows=len(fold.train),
                val_rows=len(fold.val),
                eval_report=None,
                cycles_used=0,
                error=str(e),
            )
            if verbose:
                console.print(f"    [red]ERROR: {e}[/red]")

        results.append(result)

    # Aggregate
    valid = [r for r in results if r.eval_report is not None]
    n_passed = sum(1 for r in valid if r.eval_report and r.eval_report.all_passed)

    mase_vals = [r.mase for r in valid]
    smape_vals = [r.smape for r in valid]
    dir_vals = [r.dir_acc for r in valid]
    cov_vals = [r.cov_80 for r in valid]

    summary = CVSummary(
        folds=results,
        mean_mase=float(np.nanmean(mase_vals)) if mase_vals else float("nan"),
        std_mase=float(np.nanstd(mase_vals)) if mase_vals else float("nan"),
        mean_smape=float(np.nanmean(smape_vals)) if smape_vals else float("nan"),
        std_smape=float(np.nanstd(smape_vals)) if smape_vals else float("nan"),
        mean_dir=float(np.nanmean(dir_vals)) if dir_vals else float("nan"),
        mean_cov=float(np.nanmean(cov_vals)) if cov_vals else float("nan"),
        n_folds=len(results),
        n_passed=n_passed,
    )

    if verbose:
        summary.print_summary()

    return summary
