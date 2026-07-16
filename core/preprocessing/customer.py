"""
Customer profile builder.

Aggregates a multi-series CashFlowRecord DataFrame (one customer, many pipeline
sources) into a single flat feature vector per customer — the input to clustering.

Feature groups:
  - Volume:     total inflow/outflow, net position, series count
  - Volatility: std of daily net, coefficient of variation per source
  - Seasonality: weekly/monthly periodicity strength (via autocorrelation)
  - Trend:      linear slope of 90-day rolling net cash
  - Recency:    fraction of activity in last 30 / 90 days
  - Mix:        inflow/outflow ratio, top source by volume share
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class CustomerProfile:
    customer_id: str

    # Volume
    total_inflow: float
    total_outflow: float
    net_position: float
    n_active_series: int

    # Volatility
    daily_net_std: float
    inflow_cv: float        # coefficient of variation on daily inflows
    outflow_cv: float

    # Seasonality (autocorrelation at lag-7 and lag-30)
    weekly_autocorr: float
    monthly_autocorr: float

    # Trend (OLS slope of 90-day rolling net, normalised by mean absolute net)
    trend_slope_norm: float

    # Recency
    activity_last_30d: float   # fraction of total volume in last 30 days
    activity_last_90d: float

    # Mix
    inflow_share: float        # inflow / (inflow + outflow)
    top_source: str            # source with highest volume
    top_source_share: float    # that source's share of total volume

    def to_feature_vector(self) -> np.ndarray:
        return np.array([
            self.total_inflow,
            self.total_outflow,
            self.net_position,
            self.n_active_series,
            self.daily_net_std,
            self.inflow_cv,
            self.outflow_cv,
            self.weekly_autocorr,
            self.monthly_autocorr,
            self.trend_slope_norm,
            self.activity_last_30d,
            self.activity_last_90d,
            self.inflow_share,
            self.top_source_share,
        ], dtype=np.float32)

    @classmethod
    def feature_names(cls) -> list[str]:
        return [
            "total_inflow", "total_outflow", "net_position", "n_active_series",
            "daily_net_std", "inflow_cv", "outflow_cv",
            "weekly_autocorr", "monthly_autocorr",
            "trend_slope_norm",
            "activity_last_30d", "activity_last_90d",
            "inflow_share", "top_source_share",
        ]


def build_customer_profiles(
    df: pl.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "date",
    amount_col: str = "amount",
    sign_col: str = "sign",
    source_col: str = "source",
) -> dict[str, CustomerProfile]:
    """
    Build one CustomerProfile per customer_id.

    Args:
        df: Canonical CashFlowRecord DataFrame (may contain multiple customers).

    Returns:
        Dict mapping customer_id → CustomerProfile.
    """
    profiles: dict[str, CustomerProfile] = {}

    for cid in df[customer_col].unique().to_list():
        cdf = df.filter(pl.col(customer_col) == cid)
        profiles[cid] = _profile_one_customer(
            cdf, cid, date_col, amount_col, sign_col, source_col
        )

    return profiles


def _profile_one_customer(
    df: pl.DataFrame,
    customer_id: str,
    date_col: str,
    amount_col: str,
    sign_col: str,
    source_col: str,
) -> CustomerProfile:
    df = df.sort(date_col)
    max_date = df[date_col].max()

    signed = df.with_columns(
        pl.when(pl.col(sign_col) == "inflow")
          .then(pl.col(amount_col))
          .otherwise(-pl.col(amount_col))
          .alias("signed_amount")
    )

    # ── Volume ────────────────────────────────────────────────────────────────
    total_inflow = float(df.filter(pl.col(sign_col) == "inflow")[amount_col].sum())
    total_outflow = float(df.filter(pl.col(sign_col) == "outflow")[amount_col].sum())
    net_position = total_inflow - total_outflow
    n_active_series = df[source_col].n_unique()

    # ── Daily net aggregation (for volatility + trend) ────────────────────────
    daily = (
        signed.group_by(date_col)
        .agg(pl.col("signed_amount").sum().alias("net"))
        .sort(date_col)
    )
    daily_net = daily["net"].to_numpy()

    daily_net_std = float(np.std(daily_net)) if len(daily_net) > 1 else 0.0

    # ── Volatility per sign ───────────────────────────────────────────────────
    def _cv(series: pl.Series) -> float:
        vals = series.to_numpy()
        mu = np.mean(vals)
        return float(np.std(vals) / (abs(mu) + 1e-8))

    inflow_daily = (
        df.filter(pl.col(sign_col) == "inflow")
        .group_by(date_col).agg(pl.col(amount_col).sum().alias("v"))
        .sort(date_col)["v"]
    )
    outflow_daily = (
        df.filter(pl.col(sign_col) == "outflow")
        .group_by(date_col).agg(pl.col(amount_col).sum().alias("v"))
        .sort(date_col)["v"]
    )
    inflow_cv = _cv(inflow_daily) if len(inflow_daily) > 1 else 0.0
    outflow_cv = _cv(outflow_daily) if len(outflow_daily) > 1 else 0.0

    # ── Seasonality (lag-7 and lag-30 autocorrelation) ────────────────────────
    weekly_autocorr = _autocorr(daily_net, lag=7)
    monthly_autocorr = _autocorr(daily_net, lag=30)

    # ── Trend (OLS slope on 90-day rolling net) ────────────────────────────────
    trend_slope_norm = _trend_slope(daily_net, window=90)

    # ── Recency ───────────────────────────────────────────────────────────────
    total_vol = total_inflow + total_outflow + 1e-8

    def _recency_share(days: int) -> float:
        cutoff = max_date - pl.duration(days=days)
        recent = df.filter(pl.col(date_col) >= cutoff)[amount_col].sum()
        return float(recent) / total_vol

    activity_last_30d = _recency_share(30)
    activity_last_90d = _recency_share(90)

    # ── Mix ───────────────────────────────────────────────────────────────────
    inflow_share = total_inflow / total_vol

    source_vol = (
        df.group_by(source_col)
        .agg(pl.col(amount_col).sum().alias("vol"))
        .sort("vol", descending=True)
    )
    top_source = str(source_vol[source_col][0])
    top_source_share = float(source_vol["vol"][0]) / total_vol

    return CustomerProfile(
        customer_id=customer_id,
        total_inflow=total_inflow,
        total_outflow=total_outflow,
        net_position=net_position,
        n_active_series=n_active_series,
        daily_net_std=daily_net_std,
        inflow_cv=inflow_cv,
        outflow_cv=outflow_cv,
        weekly_autocorr=weekly_autocorr,
        monthly_autocorr=monthly_autocorr,
        trend_slope_norm=trend_slope_norm,
        activity_last_30d=activity_last_30d,
        activity_last_90d=activity_last_90d,
        inflow_share=inflow_share,
        top_source=top_source,
        top_source_share=top_source_share,
    )


def _autocorr(arr: np.ndarray, lag: int) -> float:
    if len(arr) <= lag:
        return 0.0
    a = arr[:-lag] - np.mean(arr)
    b = arr[lag:] - np.mean(arr)
    denom = np.std(arr) ** 2 * len(a) + 1e-8
    return float(np.sum(a * b) / denom)


def _trend_slope(arr: np.ndarray, window: int = 90) -> float:
    """OLS slope over last `window` points, normalised by mean absolute value."""
    work = arr[-window:] if len(arr) >= window else arr
    if len(work) < 2:
        return 0.0
    x = np.arange(len(work), dtype=float)
    x -= x.mean()
    y = work - work.mean()
    slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
    scale = np.mean(np.abs(work)) + 1e-8
    return slope / scale
