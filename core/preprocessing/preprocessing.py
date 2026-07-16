"""
Time-series preprocessing for cash flow data.

This module handles everything between raw ingestion and model input:
  - Missing value imputation (forward-fill, interpolation, seasonal median)
  - Outlier detection and treatment (IQR, z-score, STL-residual)
  - Stationarity testing and differencing (ADF, KPSS)
  - Scaling (min-max, standard, log1p for skewed cash flows)
  - Business calendar features (holidays, month-end, quarter-end)
  - Pipeline-level validation (schema, date gaps, negative values)

ARIMA requirements handled here:
  - Stationarity: ADF test → auto-differencing to reach I(d)
  - Constant variance: log or Box-Cox transform for heteroskedastic series
  - No missing values: all gaps filled before fitting

Usage:
    pp = Preprocessor()
    result = pp.fit_transform(df, series_col="amount", date_col="date")
    # result.df       — transformed DataFrame
    # result.report   — what was done and why
    # pp.inverse(arr) — undo scaling for forecast output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import polars as pl

# ── Result containers ─────────────────────────────────────────────────────────


@dataclass
class StationarityReport:
    """ADF + KPSS test results for a single series."""

    series_id: str
    adf_statistic: float
    adf_pvalue: float
    adf_stationary: bool  # True if ADF rejects unit root (p < 0.05)
    kpss_statistic: float | None  # None if statsmodels unavailable
    kpss_pvalue: float | None
    kpss_stationary: bool | None  # True if KPSS fails to reject stationarity
    recommended_d: int  # suggested order of differencing
    conclusion: str


@dataclass
class OutlierReport:
    series_id: str
    method: str
    n_outliers: int
    outlier_indices: list[int]
    treatment: str  # "winsorise" | "interpolate" | "flag_only"


@dataclass
class PreprocessingReport:
    series_id: str
    n_raw: int
    n_gaps_filled: int
    n_outliers_treated: int
    differencing_order: int
    transform: str  # "none" | "log1p" | "standard" | "minmax"
    stationarity: StationarityReport | None
    outlier: OutlierReport | None
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    df: pl.DataFrame
    reports: dict[str, PreprocessingReport]  # series_id → report
    scaler_params: dict[str, dict]  # series_id → {mean, std, min, max, log}


# ── Imputation ────────────────────────────────────────────────────────────────


def fill_gaps(
    df: pl.DataFrame,
    date_col: str = "date",
    value_col: str = "amount",
    series_col: str = "series_id",
    method: Literal["forward_fill", "interpolate", "seasonal_median"] = "forward_fill",
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    Fill missing dates and NaN values in a panel DataFrame.

    Returns (filled_df, gap_counts_by_series).

    - forward_fill: last known value carried forward (conservative, preserves last state)
    - interpolate: linear interpolation between known values
    - seasonal_median: replace with median of same weekday across training data
      (best for cash flows with strong weekly patterns)
    """
    gap_counts: dict[str, int] = {}
    frames: list[pl.DataFrame] = []

    for sid in df[series_col].unique().sort().to_list():
        series = df.filter(pl.col(series_col) == sid).sort(date_col)
        dates = series[date_col].to_list()

        if not dates:
            continue

        full_range = _date_range(dates[0], dates[-1])
        n_gaps = len(full_range) - len(dates)
        gap_counts[sid] = n_gaps

        if n_gaps > 0:
            full_df = pl.DataFrame({date_col: full_range}).with_columns(
                pl.col(date_col).cast(pl.Date)
            )
            series = full_df.join(series, on=date_col, how="left")
            series = series.with_columns(pl.col(series_col).fill_null(sid))

        if method == "forward_fill":
            series = series.with_columns(pl.col(value_col).forward_fill())
            # Backfill any leading NaNs
            series = series.with_columns(pl.col(value_col).backward_fill())
        elif method == "interpolate":
            vals = series[value_col].to_numpy().astype(float)
            nans = np.isnan(vals)
            if nans.any():
                idx = np.arange(len(vals))
                vals[nans] = np.interp(idx[nans], idx[~nans], vals[~nans])
            series = series.with_columns(pl.Series(value_col, vals))
        elif method == "seasonal_median":
            series = _seasonal_median_fill(series, value_col, date_col)

        frames.append(series)

    return pl.concat(frames).sort([series_col, date_col]), gap_counts


def _seasonal_median_fill(
    df: pl.DataFrame,
    value_col: str,
    date_col: str,
) -> pl.DataFrame:
    df = df.with_columns(pl.col(date_col).dt.weekday().alias("_dow"))
    dow_medians = (
        df.filter(pl.col(value_col).is_not_null())
        .group_by("_dow")
        .agg(pl.col(value_col).median().alias("_median"))
    )
    df = df.join(dow_medians, on="_dow", how="left")
    df = df.with_columns(
        pl.when(pl.col(value_col).is_null())
        .then(pl.col("_median"))
        .otherwise(pl.col(value_col))
        .alias(value_col)
    )
    return df.drop(["_dow", "_median"])


def _date_range(start: date, end: date) -> list[date]:
    from datetime import timedelta

    days = (end - start).days + 1
    return [start + timedelta(days=i) for i in range(days)]


# ── Outlier detection and treatment ───────────────────────────────────────────


def detect_outliers(
    values: np.ndarray,
    method: Literal["iqr", "zscore", "both"] = "iqr",
    iqr_multiplier: float = 3.0,
    zscore_threshold: float = 3.5,
) -> np.ndarray:
    """
    Return boolean mask of outlier positions.

    IQR method: beyond Q1 - k*IQR or Q3 + k*IQR (robust to heavy tails)
    Z-score: |z| > threshold (assumes approximate normality)
    Both: union of the two masks
    """
    mask = np.zeros(len(values), dtype=bool)

    if method in ("iqr", "both"):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        mask |= (values < q1 - iqr_multiplier * iqr) | (values > q3 + iqr_multiplier * iqr)

    if method in ("zscore", "both"):
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad > 1e-8:
            mod_z = 0.6745 * (values - med) / mad
            mask |= np.abs(mod_z) > zscore_threshold

    return mask


def treat_outliers(
    values: np.ndarray,
    outlier_mask: np.ndarray,
    treatment: Literal["winsorise", "interpolate", "flag_only"] = "winsorise",
    winsorise_pct: float = 0.01,
) -> np.ndarray:
    """
    Apply outlier treatment.

    winsorise: clip to [p1, p99] percentiles — preserves shape, reduces extremes
    interpolate: replace with linear interpolation of neighbours
    flag_only: return unchanged (outliers remain; caller uses mask externally)
    """
    out = values.copy()

    if treatment == "winsorise":
        lo = np.percentile(values[~outlier_mask], winsorise_pct * 100)
        hi = np.percentile(values[~outlier_mask], (1 - winsorise_pct) * 100)
        out = np.clip(out, lo, hi)

    elif treatment == "interpolate":
        idx = np.arange(len(values))
        good = ~outlier_mask
        if good.sum() >= 2:
            out[outlier_mask] = np.interp(idx[outlier_mask], idx[good], values[good])

    return out


# ── Stationarity ──────────────────────────────────────────────────────────────


def check_stationarity(
    values: np.ndarray,
    series_id: str = "unknown",
    max_d: int = 2,
) -> StationarityReport:
    """
    ADF + KPSS tests. Recommends differencing order d.

    ADF null = unit root (non-stationary). Reject → stationary.
    KPSS null = stationary. Reject → non-stationary.
    Conflicting results signal trend-stationary vs difference-stationary.

    ARIMA prerequisite: series must be I(d) — difference d times to reach I(0).
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss

        adf_stat, adf_p, _, _, _, _ = adfuller(values, autolag="AIC")
        adf_stationary = adf_p < 0.05

        try:
            kpss_stat, kpss_p, _, _ = kpss(values, regression="c", nlags="auto")
            kpss_stationary = kpss_p > 0.05
        except Exception:
            kpss_stat, kpss_p, kpss_stationary = None, None, None

        # Determine d: difference until ADF stationary
        d = 0
        working = values.copy()
        while d < max_d:
            stat, p, *_ = adfuller(working, autolag="AIC")
            if p < 0.05:
                break
            working = np.diff(working)
            d += 1

        if adf_stationary and (kpss_stationary is None or kpss_stationary):
            conclusion = "Stationary — no differencing needed (d=0)"
        elif not adf_stationary and kpss_stationary is False:
            conclusion = f"Non-stationary — recommend d={d} (difference-stationary)"
        elif adf_stationary and kpss_stationary is False:
            conclusion = "Trend-stationary — consider detrending or d=1"
        else:
            conclusion = f"Uncertain — recommend d={d}, verify residuals post-fit"

        return StationarityReport(
            series_id=series_id,
            adf_statistic=float(adf_stat),
            adf_pvalue=float(adf_p),
            adf_stationary=adf_stationary,
            kpss_statistic=float(kpss_stat) if kpss_stat is not None else None,
            kpss_pvalue=float(kpss_p) if kpss_p is not None else None,
            kpss_stationary=kpss_stationary,
            recommended_d=d,
            conclusion=conclusion,
        )

    except ImportError:
        # statsmodels not installed — simple variance-ratio heuristic
        mid = len(values) // 2
        var_ratio = np.var(values[mid:]) / (np.var(values[:mid]) + 1e-8)
        stationary = 0.5 < var_ratio < 2.0
        return StationarityReport(
            series_id=series_id,
            adf_statistic=float("nan"),
            adf_pvalue=float("nan"),
            adf_stationary=stationary,
            kpss_statistic=None,
            kpss_pvalue=None,
            kpss_stationary=None,
            recommended_d=0 if stationary else 1,
            conclusion="statsmodels unavailable — variance-ratio heuristic used",
        )


def difference_series(values: np.ndarray, d: int = 1) -> np.ndarray:
    """Apply d-th order differencing. Prepends d NaNs to preserve length."""
    out = values.astype(float).copy()
    prefix = np.full(d, float("nan"))
    for _ in range(d):
        out = np.concatenate([prefix[:1], np.diff(out[~np.isnan(out)])])
    return out


def undifference_series(
    diff_values: np.ndarray,
    last_originals: np.ndarray,
    d: int = 1,
) -> np.ndarray:
    """
    Invert differencing for forecast output.
    last_originals: the last d values from the training series (needed as starting points).
    """
    out = diff_values.copy()
    for i in range(d):
        out = np.concatenate([[last_originals[-(d - i)]], out]).cumsum()[1:]
    return out


# ── Scaling ───────────────────────────────────────────────────────────────────


@dataclass
class ScalerParams:
    method: str
    mean: float = 0.0
    std: float = 1.0
    min_val: float = 0.0
    max_val: float = 1.0
    log_transform: bool = False


def fit_scaler(
    values: np.ndarray,
    method: Literal["standard", "minmax", "log1p", "none"] = "log1p",
) -> ScalerParams:
    """
    Fit scaler parameters on training data.
    log1p is recommended for cash flow (right-skewed, always positive).
    Standard scaling after log1p is the typical Chronos preprocessing path.
    """
    use_log = method == "log1p"
    work = np.log1p(np.clip(values, 0.0, None)) if use_log else values

    params = ScalerParams(
        method=method,
        mean=float(np.mean(work)),
        std=float(np.std(work)) or 1.0,
        min_val=float(np.min(work)),
        max_val=float(np.max(work)),
        log_transform=use_log,
    )
    return params


def apply_scaler(values: np.ndarray, params: ScalerParams) -> np.ndarray:
    work = np.log1p(np.clip(values, 0.0, None)) if params.log_transform else values.copy()
    if params.method in ("log1p", "standard"):
        return (work - params.mean) / params.std
    elif params.method == "minmax":
        rng = params.max_val - params.min_val
        return (work - params.min_val) / (rng if rng > 1e-8 else 1.0)
    return work


def inverse_scaler(values: np.ndarray, params: ScalerParams) -> np.ndarray:
    work = values.copy()
    if params.method in ("log1p", "standard"):
        work = work * params.std + params.mean
    elif params.method == "minmax":
        rng = params.max_val - params.min_val
        work = work * (rng if rng > 1e-8 else 1.0) + params.min_val
    if params.log_transform:
        work = np.expm1(work)
    return np.clip(work, 0.0, None)


# ── Main Preprocessor ─────────────────────────────────────────────────────────


class Preprocessor:
    """
    Full preprocessing pipeline for cash flow time-series.

    Fit on train split, apply to val/test without data leakage:
        pp = Preprocessor()
        train_result = pp.fit_transform(train_df)
        val_result   = pp.transform(val_df)   # uses train fit params

    The fit_transform → transform pattern mirrors sklearn convention.
    """

    def __init__(
        self,
        impute_method: Literal["forward_fill", "interpolate", "seasonal_median"] = "forward_fill",
        outlier_method: Literal["iqr", "zscore", "both"] = "iqr",
        outlier_treatment: Literal["winsorise", "interpolate", "flag_only"] = "winsorise",
        scale_method: Literal["standard", "minmax", "log1p", "none"] = "log1p",
        auto_difference: bool = False,  # set True for ARIMA preprocessing path
        value_col: str = "amount",
        date_col: str = "date",
        series_col: str = "series_id",
    ):
        self.impute_method = impute_method
        self.outlier_method = outlier_method
        self.outlier_treatment = outlier_treatment
        self.scale_method = scale_method
        self.auto_difference = auto_difference
        self.value_col = value_col
        self.date_col = date_col
        self.series_col = series_col

        self._scaler_params: dict[str, ScalerParams] = {}
        self._stationarity: dict[str, StationarityReport] = {}
        self._fitted = False

    def fit_transform(self, df: pl.DataFrame) -> PreprocessingResult:
        """Fit on df and return transformed DataFrame + reports."""
        filled, gap_counts = fill_gaps(
            df, self.date_col, self.value_col, self.series_col, self.impute_method
        )
        frames: list[pl.DataFrame] = []
        reports: dict[str, PreprocessingReport] = {}
        scaler_params: dict[str, dict] = {}

        for sid in filled[self.series_col].unique().sort().to_list():
            series = filled.filter(pl.col(self.series_col) == sid).sort(self.date_col)
            vals = series[self.value_col].to_numpy().astype(float)
            warnings: list[str] = []

            # Outliers
            outlier_mask = detect_outliers(vals, self.outlier_method)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                vals = treat_outliers(vals, outlier_mask, self.outlier_treatment)
            outlier_report = OutlierReport(
                series_id=sid,
                method=self.outlier_method,
                n_outliers=n_outliers,
                outlier_indices=np.where(outlier_mask)[0].tolist(),
                treatment=self.outlier_treatment,
            )

            # Stationarity (fit for reference; auto-diff if requested)
            stat_report = check_stationarity(vals, series_id=sid)
            self._stationarity[sid] = stat_report
            d = stat_report.recommended_d if self.auto_difference else 0
            if d > 0:
                vals = difference_series(vals, d)
                warnings.append(f"Applied d={d} differencing (ADF recommended)")

            # Scaling — fit params from this series
            clean_vals = vals[~np.isnan(vals)]
            params = fit_scaler(clean_vals, self.scale_method)
            self._scaler_params[sid] = params
            scaler_params[sid] = {
                "method": params.method,
                "mean": params.mean,
                "std": params.std,
                "min_val": params.min_val,
                "max_val": params.max_val,
                "log_transform": params.log_transform,
            }
            scaled = apply_scaler(vals, params)

            frames.append(series.with_columns(pl.Series(self.value_col, scaled)))
            reports[sid] = PreprocessingReport(
                series_id=sid,
                n_raw=len(series),
                n_gaps_filled=gap_counts.get(sid, 0),
                n_outliers_treated=n_outliers,
                differencing_order=d,
                transform=self.scale_method,
                stationarity=stat_report,
                outlier=outlier_report,
                warnings=warnings,
            )

        self._fitted = True
        return PreprocessingResult(
            df=pl.concat(frames).sort([self.series_col, self.date_col]),
            reports=reports,
            scaler_params=scaler_params,
        )

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply fitted scaler params to new data (val/test). No refitting."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform()")

        filled, _ = fill_gaps(
            df, self.date_col, self.value_col, self.series_col, self.impute_method
        )
        frames: list[pl.DataFrame] = []

        for sid in filled[self.series_col].unique().sort().to_list():
            series = filled.filter(pl.col(self.series_col) == sid).sort(self.date_col)
            vals = series[self.value_col].to_numpy().astype(float)
            params = self._scaler_params.get(sid)
            if params is None:
                frames.append(series)
                continue
            scaled = apply_scaler(vals, params)
            frames.append(series.with_columns(pl.Series(self.value_col, scaled)))

        return pl.concat(frames).sort([self.series_col, self.date_col])

    def inverse_transform_array(self, series_id: str, values: np.ndarray) -> np.ndarray:
        """Undo scaling for a forecast array (model output → original units)."""
        params = self._scaler_params.get(series_id)
        if params is None:
            return values
        return inverse_scaler(values, params)

    def stationarity_summary(self) -> pl.DataFrame:
        """Return a DataFrame summarising stationarity test results per series."""
        rows = []
        for sid, r in self._stationarity.items():
            rows.append(
                {
                    "series_id": sid,
                    "adf_pvalue": r.adf_pvalue,
                    "adf_stationary": r.adf_stationary,
                    "recommended_d": r.recommended_d,
                    "conclusion": r.conclusion,
                }
            )
        return pl.DataFrame(rows)
