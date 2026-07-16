"""
Model comparison harness: ARIMA vs Chronos vs Agent Loop.

Runs all three model families on the same walk-forward CV folds and
produces a side-by-side metric table. This is the primary tool for
deciding which model (or combination) to deploy per customer / series.

Comparison dimensions:
  - Forecast accuracy: MASE, SMAPE, directional accuracy, coverage
  - Interval calibration: coverage_80 (want 75–85%)
  - Latency: fit + predict wall time per fold
  - Cost: LLM calls (agent loop only)
  - Iteration improvements: agent loop vs Chronos baseline across cycles

Usage:
    result = run_model_comparison(train_df, horizon_days=30)
    result.print_table()
    result.as_dataframe()  # → polars DataFrame for notebook analysis
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

from core.preprocessing.synthetic import WalkForwardFold, walk_forward_cv, TemporalSplit
from evals.arima import ARIMAForecaster, ARIMAForecastResult
from evals.graders.graders import MASEGrader, SMAPEGrader, DirectionalGrader, CoverageGrader


# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class SingleModelFoldResult:
    model_name: str
    fold_idx: int
    train_end: date
    val_end: date
    mase: float
    smape: float
    directional_accuracy: float
    coverage_80: float
    fit_seconds: float
    predict_seconds: float
    error: str | None = None

    @property
    def all_passed(self) -> bool:
        return (
            self.mase < 1.0
            and self.smape < 15.0
            and self.directional_accuracy > 55.0
            and self.coverage_80 >= 75.0
        )


@dataclass
class ModelComparisonResult:
    folds: list[SingleModelFoldResult]
    model_names: list[str]

    def summary_by_model(self) -> dict[str, dict[str, float]]:
        summaries: dict[str, dict[str, float]] = {}
        for name in self.model_names:
            model_folds = [f for f in self.folds if f.model_name == name and f.error is None]
            if not model_folds:
                continue
            summaries[name] = {
                "mean_mase": float(np.mean([f.mase for f in model_folds])),
                "mean_smape": float(np.mean([f.smape for f in model_folds])),
                "mean_dir": float(np.mean([f.directional_accuracy for f in model_folds])),
                "mean_cov80": float(np.mean([f.coverage_80 for f in model_folds])),
                "mean_fit_s": float(np.mean([f.fit_seconds for f in model_folds])),
                "n_folds": len(model_folds),
                "pass_rate": sum(1 for f in model_folds if f.all_passed) / len(model_folds),
            }
        return summaries

    def as_dataframe(self) -> pl.DataFrame:
        rows = []
        for f in self.folds:
            rows.append({
                "model": f.model_name,
                "fold": f.fold_idx,
                "train_end": str(f.train_end),
                "val_end": str(f.val_end),
                "mase": f.mase,
                "smape": f.smape,
                "dir_acc": f.directional_accuracy,
                "cov_80": f.coverage_80,
                "fit_s": f.fit_seconds,
                "passed": f.all_passed,
                "error": f.error or "",
            })
        return pl.DataFrame(rows)

    def print_table(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title="Model Comparison — Walk-Forward CV", show_lines=True)
            table.add_column("Model", style="bold cyan")
            table.add_column("Folds")
            table.add_column("MASE ↓")
            table.add_column("SMAPE ↓")
            table.add_column("Dir% ↑")
            table.add_column("Cov80%")
            table.add_column("Fit(s) ↓")
            table.add_column("Pass%")

            for name, s in self.summary_by_model().items():
                table.add_row(
                    name,
                    str(int(s["n_folds"])),
                    f"{s['mean_mase']:.3f}",
                    f"{s['mean_smape']:.1f}%",
                    f"{s['mean_dir']:.1f}%",
                    f"{s['mean_cov80']:.1f}%",
                    f"{s['mean_fit_s']:.2f}s",
                    f"{s['pass_rate']:.0%}",
                )
            console.print(table)
        except ImportError:
            for name, s in self.summary_by_model().items():
                print(f"{name}: MASE={s['mean_mase']:.3f} SMAPE={s['mean_smape']:.1f}% "
                      f"Dir={s['mean_dir']:.1f}% Cov={s['mean_cov80']:.1f}% "
                      f"Pass={s['pass_rate']:.0%}")


# ── Per-model runners ─────────────────────────────────────────────────────────


def _score_fold_arima(
    fold: WalkForwardFold,
    series_id: str,
    horizon_days: int,
    seasonal_period: int = 7,
) -> SingleModelFoldResult:
    """Run AutoARIMA on one series for one fold."""
    train_vals = (
        fold.train.filter(pl.col("series_id") == series_id)
        .sort("date")["amount" if "amount" in fold.train.columns else "value"]
        .to_numpy().astype(float)
    )
    val_vals = (
        fold.val.filter(pl.col("series_id") == series_id)
        .sort("date").head(horizon_days)["amount" if "amount" in fold.val.columns else "value"]
        .to_numpy().astype(float)
    )

    if len(train_vals) < 2 * seasonal_period or len(val_vals) == 0:
        return SingleModelFoldResult(
            model_name="ARIMA", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=float("nan"), smape=float("nan"),
            directional_accuracy=float("nan"), coverage_80=float("nan"),
            fit_seconds=0.0, predict_seconds=0.0,
            error="Insufficient data",
        )

    try:
        forecaster = ARIMAForecaster(series_id=series_id, auto=True)
        t0 = time.perf_counter()
        forecaster.fit(train_vals, seasonal_period=seasonal_period)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = forecaster.predict(horizon=horizon_days)
        pred_s = time.perf_counter() - t0

        actuals = val_vals[: len(result.point_forecast)]
        preds = np.array(result.point_forecast[: len(actuals)])
        lower = np.array(result.lower_80[: len(actuals)])
        upper = np.array(result.upper_80[: len(actuals)])

        # MASE
        diff = train_vals[seasonal_period:] - train_vals[:-seasonal_period]
        naive_mae = np.mean(np.abs(diff)) or 1e-8
        mase = float(np.mean(np.abs(actuals - preds)) / naive_mae)

        # SMAPE
        denom = np.where((np.abs(actuals) + np.abs(preds)) / 2 < 1e-8, 1e-8,
                         (np.abs(actuals) + np.abs(preds)) / 2)
        smape = float(100.0 * np.mean(np.abs(actuals - preds) / denom))

        # Directional
        dir_acc = float(100.0 * np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        )) if len(actuals) > 1 else 50.0

        # Coverage
        cov = float(100.0 * np.mean((actuals >= lower) & (actuals <= upper)))

        return SingleModelFoldResult(
            model_name="ARIMA", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=mase, smape=smape,
            directional_accuracy=dir_acc, coverage_80=cov,
            fit_seconds=fit_s, predict_seconds=pred_s,
        )

    except Exception as e:
        return SingleModelFoldResult(
            model_name="ARIMA", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=float("nan"), smape=float("nan"),
            directional_accuracy=float("nan"), coverage_80=float("nan"),
            fit_seconds=0.0, predict_seconds=0.0,
            error=str(e),
        )


def _score_fold_chronos(
    fold: WalkForwardFold,
    series_id: str,
    horizon_days: int,
    model_id: str = "amazon/chronos-t5-tiny",
) -> SingleModelFoldResult:
    """Run Chronos (or AutoETS fallback) on one series for one fold."""
    value_col = "amount" if "amount" in fold.train.columns else "value"
    train_vals = (
        fold.train.filter(pl.col("series_id") == series_id)
        .sort("date")[value_col].to_numpy().astype(float)
    )
    val_vals = (
        fold.val.filter(pl.col("series_id") == series_id)
        .sort("date").head(horizon_days)[value_col].to_numpy().astype(float)
    )

    if len(train_vals) == 0 or len(val_vals) == 0:
        return SingleModelFoldResult(
            model_name="Chronos", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=float("nan"), smape=float("nan"),
            directional_accuracy=float("nan"), coverage_80=float("nan"),
            fit_seconds=0.0, predict_seconds=0.0, error="No data",
        )

    try:
        t0 = time.perf_counter()
        point, lower, upper = _run_chronos_or_fallback(train_vals, horizon_days, model_id)
        elapsed = time.perf_counter() - t0

        actuals = val_vals[: len(point)]
        preds = np.array(point[: len(actuals)])
        lower_arr = np.array(lower[: len(actuals)])
        upper_arr = np.array(upper[: len(actuals)])

        seasonal_period = 7
        diff = train_vals[seasonal_period:] - train_vals[:-seasonal_period]
        naive_mae = np.mean(np.abs(diff)) or 1e-8
        mase = float(np.mean(np.abs(actuals - preds)) / naive_mae)

        denom = np.where((np.abs(actuals) + np.abs(preds)) / 2 < 1e-8, 1e-8,
                         (np.abs(actuals) + np.abs(preds)) / 2)
        smape = float(100.0 * np.mean(np.abs(actuals - preds) / denom))
        dir_acc = float(100.0 * np.mean(
            np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        )) if len(actuals) > 1 else 50.0
        cov = float(100.0 * np.mean((actuals >= lower_arr) & (actuals <= upper_arr)))

        return SingleModelFoldResult(
            model_name="Chronos", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=mase, smape=smape,
            directional_accuracy=dir_acc, coverage_80=cov,
            fit_seconds=0.0, predict_seconds=elapsed,
        )

    except Exception as e:
        return SingleModelFoldResult(
            model_name="Chronos", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=float("nan"), smape=float("nan"),
            directional_accuracy=float("nan"), coverage_80=float("nan"),
            fit_seconds=0.0, predict_seconds=0.0, error=str(e),
        )


def _run_chronos_or_fallback(
    values: np.ndarray, horizon: int, model_id: str
) -> tuple[list[float], list[float], list[float]]:
    """Chronos → AutoETS → naïve lag-7 fallback chain."""
    try:
        import torch
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(model_id, device_map="cpu",
                                                    torch_dtype=torch.float32)
        ctx = torch.tensor(values[np.newaxis, :], dtype=torch.float32)
        quantiles, mean = pipeline.predict_quantiles(ctx, prediction_length=horizon,
                                                      quantile_levels=[0.1, 0.5, 0.9],
                                                      num_samples=50)
        return (mean[0].numpy().tolist(),
                quantiles[0, :, 0].numpy().tolist(),
                quantiles[0, :, 2].numpy().tolist())
    except Exception:
        pass

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS
        import pandas as pd
        n = len(values)
        df_sf = pd.DataFrame({"unique_id": ["s1"] * n,
                               "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                               "y": values.tolist()})
        sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df_sf)
        pred = sf.predict(h=horizon, level=[80])
        return pred["AutoETS"].tolist(), pred["AutoETS-lo-80"].tolist(), pred["AutoETS-hi-80"].tolist()
    except Exception:
        pass

    import math
    last = values[-7:] if len(values) >= 7 else values
    point = list(np.tile(last, math.ceil(horizon / len(last)))[:horizon])
    return point, [p * 0.85 for p in point], [p * 1.15 for p in point]


# ── Agent loop runner ─────────────────────────────────────────────────────────


def _score_fold_agent(
    fold: WalkForwardFold,
    horizon_days: int,
    max_cycles: int = 3,
) -> list[SingleModelFoldResult]:
    """
    Run the full LangGraph agent loop on one fold.
    Returns one result per cycle so iteration improvement is visible.
    """
    from src.agents.graph import run_forecasting_agent
    from src.agents.state import EvalReport

    value_col = "amount" if "amount" in fold.val.columns else "value"
    actuals_by_series: dict[str, list[float]] = {}
    for sid in fold.val["series_id"].unique().to_list():
        actuals_by_series[sid] = (
            fold.val.filter(pl.col("series_id") == sid)
            .sort("date").head(horizon_days)[value_col].to_list()
        )

    results: list[SingleModelFoldResult] = []
    try:
        t0 = time.perf_counter()
        final_state = run_forecasting_agent(
            series_df=fold.train, actuals=actuals_by_series,
            max_cycles=max_cycles, verbose=False,
        )
        elapsed = time.perf_counter() - t0

        for i, report in enumerate(final_state.get("eval_history", [])):
            results.append(SingleModelFoldResult(
                model_name=f"Agent-cycle{i + 1}",
                fold_idx=fold.fold_idx,
                train_end=fold.train_end, val_end=fold.val_end,
                mase=report.overall_mase, smape=report.overall_smape,
                directional_accuracy=report.directional_accuracy,
                coverage_80=report.coverage_80,
                fit_seconds=elapsed / max(len(final_state.get("eval_history", [])), 1),
                predict_seconds=0.0,
            ))
    except Exception as e:
        results.append(SingleModelFoldResult(
            model_name="Agent-cycle1", fold_idx=fold.fold_idx,
            train_end=fold.train_end, val_end=fold.val_end,
            mase=float("nan"), smape=float("nan"),
            directional_accuracy=float("nan"), coverage_80=float("nan"),
            fit_seconds=0.0, predict_seconds=0.0, error=str(e),
        ))
    return results


# ── Main comparison runner ────────────────────────────────────────────────────


def run_model_comparison(
    train_df: pl.DataFrame,
    horizon_days: int = 30,
    min_train_days: int = 180,
    step_days: int = 30,
    max_folds: int = 4,
    series_ids: list[str] | None = None,
    models: list[Literal["arima", "chronos", "agent"]] | None = None,
    max_agent_cycles: int = 3,
    verbose: bool = True,
) -> ModelComparisonResult:
    """
    Run walk-forward CV for each model family and return comparison results.

    Args:
        train_df:       Training DataFrame (test set excluded — never touched here)
        horizon_days:   Forecast horizon per fold
        series_ids:     Subset of series to compare on (None = all)
        models:         Which model families to include (None = all three)
        verbose:        Print progress
    """
    models = models or ["arima", "chronos", "agent"]
    folds = walk_forward_cv(
        train_df, horizon_days=horizon_days,
        min_train_days=min_train_days, step_days=step_days, max_folds=max_folds,
    )

    value_col = "amount" if "amount" in train_df.columns else "value"
    all_series = series_ids or train_df["series_id"].unique().sort().to_list()
    all_results: list[SingleModelFoldResult] = []
    model_names: list[str] = []

    for fold in folds:
        if verbose:
            try:
                from rich.console import Console
                Console().print(f"  Fold {fold.fold_idx + 1}: train→{fold.train_end} | val→{fold.val_end}")
            except ImportError:
                print(f"  Fold {fold.fold_idx + 1}: train→{fold.train_end} | val→{fold.val_end}")

        for sid in all_series:
            if "arima" in models:
                r = _score_fold_arima(fold, sid, horizon_days)
                all_results.append(r)
                if "ARIMA" not in model_names:
                    model_names.append("ARIMA")

            if "chronos" in models:
                r = _score_fold_chronos(fold, sid, horizon_days)
                all_results.append(r)
                if "Chronos" not in model_names:
                    model_names.append("Chronos")

        if "agent" in models:
            agent_results = _score_fold_agent(fold, horizon_days, max_agent_cycles)
            all_results.extend(agent_results)
            for r in agent_results:
                if r.model_name not in model_names:
                    model_names.append(r.model_name)

    result = ModelComparisonResult(folds=all_results, model_names=model_names)
    if verbose:
        result.print_table()
    return result
