"""
Feature engineering and importance analysis for cash flow forecasting.

Two audiences:
  1. Traditional ML (XGBoost, LightGBM) — tabular lag/window/calendar features
  2. ARIMA — external regressors (ARIMAX), Fourier terms for seasonality

Feature groups:
  - Lag features: direct history windows (t-1, t-7, t-14, t-28, t-30)
  - Rolling statistics: mean, std, min, max, skew over multiple windows
  - Calendar: day-of-week, month-end, quarter-end, holiday proximity, fiscal calendar
  - Trend: simple linear trend component, ewm (exponentially weighted mean)
  - Cross-series: net cash position, inflow/outflow ratio, burn rate
  - Fourier: sine/cosine terms for weekly and annual seasonality (for ARIMAX)

Feature importance:
  - Permutation importance (model-agnostic, works post-fit)
  - Mutual information (linear + nonlinear dependencies, no model required)
  - Correlation analysis (Pearson + Spearman for lag selection)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

# ── Lag and rolling features ──────────────────────────────────────────────────


LAG_DAYS: list[int] = [1, 7, 14, 28, 30]
ROLL_WINDOWS: list[int] = [7, 14, 30, 90]


def add_lag_features(
    df: pl.DataFrame,
    value_col: str = "value",
    series_col: str = "series_id",
    lags: list[int] = LAG_DAYS,
) -> pl.DataFrame:
    """
    Add lag features per series (no cross-series leakage).
    Each series is shifted independently.
    """
    frames: list[pl.DataFrame] = []
    for sid in df[series_col].unique().sort().to_list():
        s = df.filter(pl.col(series_col) == sid).sort("date")
        vals = s[value_col].to_numpy().astype(float)
        n = len(vals)
        new_cols: dict[str, list] = {}
        for lag in lags:
            lagged = np.full(n, float("nan"))
            lagged[lag:] = vals[: n - lag]
            new_cols[f"lag_{lag}d"] = lagged.tolist()
        frames.append(s.with_columns([pl.Series(k, v) for k, v in new_cols.items()]))
    return pl.concat(frames).sort([series_col, "date"])


def add_rolling_features(
    df: pl.DataFrame,
    value_col: str = "value",
    series_col: str = "series_id",
    windows: list[int] = ROLL_WINDOWS,
) -> pl.DataFrame:
    """
    Rolling mean, std, min, max, and skewness per series.
    Rolling stats computed on [i-window+1 .. i] (no lookahead).
    """
    frames: list[pl.DataFrame] = []
    for sid in df[series_col].unique().sort().to_list():
        s = df.filter(pl.col(series_col) == sid).sort("date")
        vals = s[value_col].to_numpy().astype(float)
        n = len(vals)
        new_cols: dict[str, list] = {}
        for w in windows:
            means, stds, mins, maxs = (
                [float("nan")] * n,
                [float("nan")] * n,
                [float("nan")] * n,
                [float("nan")] * n,
            )
            for i in range(w - 1, n):
                window = vals[i - w + 1 : i + 1]
                means[i] = float(np.mean(window))
                stds[i] = float(np.std(window, ddof=1))
                mins[i] = float(np.min(window))
                maxs[i] = float(np.max(window))
            new_cols[f"roll_mean_{w}d"] = means
            new_cols[f"roll_std_{w}d"] = stds
            new_cols[f"roll_min_{w}d"] = mins
            new_cols[f"roll_max_{w}d"] = maxs
        frames.append(s.with_columns([pl.Series(k, v) for k, v in new_cols.items()]))
    return pl.concat(frames).sort([series_col, "date"])


def add_ewm_features(
    df: pl.DataFrame,
    value_col: str = "value",
    series_col: str = "series_id",
    spans: list[int] | None = None,
) -> pl.DataFrame:
    """
    Exponentially weighted mean for each span.
    EWM is more responsive to recent changes than simple rolling mean —
    useful for catching trend shifts early.
    """
    if spans is None:
        spans = [7, 14, 30]
    frames: list[pl.DataFrame] = []
    for sid in df[series_col].unique().sort().to_list():
        s = df.filter(pl.col(series_col) == sid).sort("date")
        vals = s[value_col].to_numpy().astype(float)
        new_cols: dict[str, list] = {}
        for span in spans:
            alpha = 2.0 / (span + 1)
            ewm = np.full(len(vals), float("nan"))
            for i, v in enumerate(vals):
                if np.isnan(v):
                    continue
                if np.isnan(ewm[i - 1]) if i > 0 else True:
                    ewm[i] = v
                else:
                    ewm[i] = alpha * v + (1 - alpha) * ewm[i - 1]
            new_cols[f"ewm_{span}d"] = ewm.tolist()
        frames.append(s.with_columns([pl.Series(k, v) for k, v in new_cols.items()]))
    return pl.concat(frames).sort([series_col, "date"])


# ── Calendar features ─────────────────────────────────────────────────────────


def add_calendar_features(
    df: pl.DataFrame,
    date_col: str = "date",
    include_fourier: bool = True,
) -> pl.DataFrame:
    """
    Add calendar and Fourier features.

    Calendar:
      day_of_week (0=Mon), day_of_month, month, quarter
      is_month_end, is_quarter_end, is_year_end
      days_to_month_end (useful for AR/AP cycle modelling)

    Fourier terms (for ARIMAX and gradient boosting):
      Weekly: sin/cos of (2π * dow / 7)
      Monthly: sin/cos of (2π * dom / 30.44)
      Annual: sin/cos of (2π * doy / 365.25)
    Fourier terms allow the model to capture smooth seasonality without
    dummy-variable explosion. Use k=1..3 harmonics per period.
    """
    df = df.with_columns(
        [
            pl.col(date_col).dt.weekday().alias("day_of_week"),
            pl.col(date_col).dt.day().alias("day_of_month"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.quarter().alias("quarter"),
            (pl.col(date_col).dt.month_end() == pl.col(date_col))
            .cast(pl.Int8)
            .alias("is_month_end"),
            pl.col(date_col).dt.ordinal_day().alias("day_of_year"),
        ]
    )

    df = df.with_columns(
        [
            (
                (pl.col(date_col).dt.month().is_in([3, 6, 9, 12]))
                & (pl.col(date_col).dt.month_end() == pl.col(date_col))
            )
            .cast(pl.Int8)
            .alias("is_quarter_end"),
            ((pl.col(date_col).dt.month() == 12) & (pl.col(date_col).dt.day() == 31))
            .cast(pl.Int8)
            .alias("is_year_end"),
        ]
    )

    # Days to month end — proxy for AR/AP pressure
    df = df.with_columns(
        [
            (pl.col(date_col).dt.month_end().dt.day() - pl.col(date_col).dt.day()).alias(
                "days_to_month_end"
            )
        ]
    )

    if include_fourier:
        dow = df["day_of_week"].to_numpy().astype(float)
        dom = df["day_of_month"].to_numpy().astype(float)
        doy = df["day_of_year"].to_numpy().astype(float)

        fourier_cols: dict[str, np.ndarray] = {}
        for k in range(1, 3):
            fourier_cols[f"weekly_sin_{k}"] = np.sin(2 * np.pi * k * dow / 7)
            fourier_cols[f"weekly_cos_{k}"] = np.cos(2 * np.pi * k * dow / 7)
            fourier_cols[f"monthly_sin_{k}"] = np.sin(2 * np.pi * k * dom / 30.44)
            fourier_cols[f"monthly_cos_{k}"] = np.cos(2 * np.pi * k * dom / 30.44)
            fourier_cols[f"annual_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
            fourier_cols[f"annual_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

        df = df.with_columns([pl.Series(k, v.tolist()) for k, v in fourier_cols.items()])

    return df


# ── Cross-series (company-level) features ─────────────────────────────────────


def add_company_level_features(
    df: pl.DataFrame,
    customer_id_col: str = "customer_id",
    date_col: str = "date",
    amount_col: str = "amount",
    sign_col: str = "sign",
) -> pl.DataFrame:
    """
    Derive company-level aggregate features from multi-pipeline data.

    net_cashflow:      inflows - outflows on that day
    inflow_outflow_ratio: total inflows / total outflows (burn rate proxy)
    rolling_burn_30d:  30-day rolling net cash position
    cash_runway_signal: rolling net / rolling std (volatility-adjusted runway)

    These are joined back onto the per-series rows so each series
    gets the company context as an additional feature dimension.
    """
    from core.preprocessing.ingestion import CashFlowSign

    daily = (
        df.with_columns(
            pl.when(pl.col(sign_col) == CashFlowSign.INFLOW.value)
            .then(pl.col(amount_col))
            .otherwise(-pl.col(amount_col))
            .alias("_signed")
        )
        .group_by([customer_id_col, date_col])
        .agg(
            [
                pl.col("_signed").sum().alias("net_cashflow"),
                pl.col("_signed").filter(pl.col("_signed") > 0).sum().alias("_total_in"),
                pl.col("_signed").filter(pl.col("_signed") < 0).abs().sum().alias("_total_out"),
            ]
        )
        .with_columns(
            (pl.col("_total_in") / (pl.col("_total_out") + 1.0)).alias("inflow_outflow_ratio")
        )
        .sort([customer_id_col, date_col])
    )

    # Rolling 30d net cash position
    company_frames: list[pl.DataFrame] = []
    for cid in daily[customer_id_col].unique().to_list():
        c = daily.filter(pl.col(customer_id_col) == cid)
        net = c["net_cashflow"].to_numpy().astype(float)
        n = len(net)
        roll_net = [float("nan")] * n
        roll_std = [float("nan")] * n
        for i in range(29, n):
            window = net[i - 29 : i + 1]
            roll_net[i] = float(np.sum(window))
            roll_std[i] = float(np.std(window, ddof=1)) + 1e-8
        runway = [
            roll_net[i] / roll_std[i] if not np.isnan(roll_net[i]) else float("nan")
            for i in range(n)
        ]
        company_frames.append(
            c.with_columns(
                [
                    pl.Series("rolling_net_30d", roll_net),
                    pl.Series("cash_runway_signal", runway),
                ]
            )
        )

    company_df = pl.concat(company_frames).drop(["_total_in", "_total_out"])
    return df.join(company_df, on=[customer_id_col, date_col], how="left")


# ── Feature importance ────────────────────────────────────────────────────────


@dataclass
class FeatureImportanceResult:
    method: str
    importances: dict[str, float]  # feature_name → score
    ranked: list[tuple[str, float]]  # sorted desc by score

    def top_k(self, k: int = 15) -> list[tuple[str, float]]:
        return self.ranked[:k]

    def as_dataframe(self) -> pl.DataFrame:
        names, scores = zip(*self.ranked, strict=False) if self.ranked else ([], [])
        return pl.DataFrame({"feature": list(names), "importance": list(scores)})


def mutual_information_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    discrete_features: bool = False,
) -> FeatureImportanceResult:
    """
    Mutual information between each feature and the target.
    Model-agnostic — works before any model is fitted.
    Captures both linear and nonlinear dependencies.

    Use for initial feature selection and understanding which lags/windows matter.
    """
    try:
        from sklearn.feature_selection import mutual_info_regression

        scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=42)
        importance = dict(zip(feature_names, scores.tolist(), strict=False))
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return FeatureImportanceResult("mutual_information", importance, ranked)
    except ImportError:
        # Fallback: absolute Pearson correlation
        return correlation_importance(X, y, feature_names)


def correlation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    method: Literal["pearson", "spearman"] = "pearson",
) -> FeatureImportanceResult:
    """
    |Pearson| or |Spearman| correlation of each feature with target.

    Spearman is more robust to monotonic nonlinear relationships and
    outliers — preferred for cash flow data.

    Use for lag selection: plot lag_k vs target correlation to find
    dominant autocorrelation lags (informs ARIMA p order).
    """
    scores: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        valid = ~(np.isnan(col) | np.isnan(y))
        if valid.sum() < 10:
            scores[name] = 0.0
            continue
        if method == "pearson":
            c = float(np.corrcoef(col[valid], y[valid])[0, 1])
        else:
            from scipy.stats import spearmanr

            c, _ = spearmanr(col[valid], y[valid])
        scores[name] = abs(float(c)) if not np.isnan(c) else 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return FeatureImportanceResult(f"{method}_correlation", scores, ranked)


def permutation_importance(
    model,  # fitted sklearn-compatible model
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 10,
    scoring_fn=None,  # callable(y_true, y_pred) → float, higher = better
    random_state: int = 42,
) -> FeatureImportanceResult:
    """
    Permutation importance: shuffle each feature, measure performance drop.
    Works for any fitted model (sklearn, XGBoost, LightGBM).

    Score drop = importance. If drop is negative, the feature was hurting.
    Use this after fitting a baseline model to validate feature selection.
    """
    rng = np.random.default_rng(random_state)

    def default_neg_mase(yt, yp):
        mae = np.mean(np.abs(yt - yp))
        naive_mae = np.mean(np.abs(np.diff(yt)))
        return -mae / (naive_mae + 1e-8)

    score_fn = scoring_fn or default_neg_mase
    baseline_preds = model.predict(X)
    baseline_score = score_fn(y, baseline_preds)

    importances: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        drops: list[float] = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_score = score_fn(y, model.predict(X_perm))
            drops.append(baseline_score - perm_score)
        importances[name] = float(np.mean(drops))

    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return FeatureImportanceResult("permutation", importances, ranked)


# ── Full feature pipeline ─────────────────────────────────────────────────────


def build_feature_matrix(
    df: pl.DataFrame,
    value_col: str = "value",
    series_col: str = "series_id",
    date_col: str = "date",
    include_fourier: bool = True,
    include_ewm: bool = True,
    lags: list[int] = LAG_DAYS,
    roll_windows: list[int] = ROLL_WINDOWS,
) -> pl.DataFrame:
    """
    One-shot feature builder: lag + rolling + ewm + calendar + Fourier.

    Returns a DataFrame ready for ML models. NaN rows (early history)
    are retained — caller should drop via .drop_nulls() before fitting.
    """
    df = add_lag_features(df, value_col, series_col, lags)
    df = add_rolling_features(df, value_col, series_col, roll_windows)
    if include_ewm:
        df = add_ewm_features(df, value_col, series_col)
    df = add_calendar_features(df, date_col, include_fourier)
    return df


def feature_names_for_ml(df: pl.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """
    Return the list of feature column names suitable for ML model input.
    Excludes known non-feature columns and any specified extras.
    """
    non_features = {
        "date",
        "series_id",
        "customer_id",
        "source",
        "sign",
        "is_anomaly",
        "currency",
        "amount",
        "value",
        # target columns
        *[f"target_{k}d" for k in range(1, 91)],
    }
    if exclude:
        non_features.update(exclude)
    return [c for c in df.columns if c not in non_features]
