"""
Company-level cash flow data ingestion.

Models the real-world situation: a company has several data pipelines
(ERP, payroll, AR/AP, bank feeds, subscription billing) each delivering
data on different schemas and cadences. This module normalises them all
into the canonical CashFlowRecord format before any processing.

Design:
  - Each pipeline has its own adapter (raw schema → CashFlowRecord)
  - IngestPipeline orchestrates loading, validating, and merging
  - Multi-customer: each customer gets their own DataFrame keyed by customer_id
  - All dates normalised to UTC date (no tz-aware datetimes in the store)

Usage:
    pipeline = IngestPipeline(customer_id="acme-001")
    df = pipeline.load_from_synthetic()        # for dev/testing
    df = pipeline.load_from_csv(path, source)  # for real data
    df = pipeline.merge_sources([df1, df2])    # combine pipelines
"""

from __future__ import annotations

from dataclasses import field
from datetime import date
from pathlib import Path

import polars as pl
from pydantic import BaseModel, field_validator

from core.preprocessing.synthetic import (
    CashFlowSign,
    PipelineSource,
    generate_sequence_dataset,
)

# ── Canonical record schema ────────────────────────────────────────────────────


class CashFlowRecord(BaseModel):
    """
    Single normalised cash flow observation.
    All pipeline adapters produce this schema.
    """

    customer_id: str
    date: date
    series_id: str  # pipeline-scoped series name
    source: PipelineSource
    sign: CashFlowSign
    amount: float  # always positive; sign field carries direction
    currency: str = "USD"
    is_anomaly: bool = False
    metadata: dict = field(default_factory=dict)

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"amount must be non-negative, got {v}")
        return v

    class Config:
        use_enum_values = True


# Canonical Polars schema — all ingested DataFrames conform to this
CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "customer_id": pl.Utf8,
    "date": pl.Date,
    "series_id": pl.Utf8,
    "source": pl.Utf8,
    "sign": pl.Utf8,
    "amount": pl.Float64,
    "currency": pl.Utf8,
    "is_anomaly": pl.Boolean,
}


# ── Pipeline adapters ─────────────────────────────────────────────────────────


def adapt_erp_export(df: pl.DataFrame, customer_id: str) -> pl.DataFrame:
    """
    Adapter for ERP CSV exports (SAP, NetSuite, QuickBooks typical schema).

    Expected raw columns: transaction_date, account_code, debit, credit, description
    Maps debit → outflow, credit → inflow.
    """
    required = {"transaction_date", "account_code", "debit", "credit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ERP export missing columns: {missing}")

    return df.with_columns(
        [
            pl.col("transaction_date").cast(pl.Date).alias("date"),
            pl.lit(customer_id).alias("customer_id"),
            pl.col("account_code").alias("series_id"),
            pl.lit(PipelineSource.ERP_REVENUE.value).alias("source"),
            pl.when(pl.col("credit") > 0)
            .then(pl.lit(CashFlowSign.INFLOW.value))
            .otherwise(pl.lit(CashFlowSign.OUTFLOW.value))
            .alias("sign"),
            (pl.col("credit") + pl.col("debit")).abs().alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.lit(False).alias("is_anomaly"),
        ]
    ).select(list(CANONICAL_SCHEMA.keys()))


def adapt_payroll_export(df: pl.DataFrame, customer_id: str) -> pl.DataFrame:
    """
    Adapter for payroll exports (ADP, Gusto typical schema).

    Expected raw columns: pay_date, gross_pay, net_pay, employee_count
    """
    required = {"pay_date", "gross_pay"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Payroll export missing columns: {missing}")

    return df.with_columns(
        [
            pl.col("pay_date").cast(pl.Date).alias("date"),
            pl.lit(customer_id).alias("customer_id"),
            pl.lit("payroll").alias("series_id"),
            pl.lit(PipelineSource.PAYROLL.value).alias("source"),
            pl.lit(CashFlowSign.OUTFLOW.value).alias("sign"),
            pl.col("gross_pay").cast(pl.Float64).alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.lit(False).alias("is_anomaly"),
        ]
    ).select(list(CANONICAL_SCHEMA.keys()))


def adapt_bank_feed(
    df: pl.DataFrame, customer_id: str, account_type: str = "operating"
) -> pl.DataFrame:
    """
    Adapter for bank feed exports (Plaid, direct OFX/CSV typical schema).

    Expected raw columns: date, amount, description
    Negative amount = outflow (bank convention), positive = inflow.
    """
    required = {"date", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Bank feed missing columns: {missing}")

    source = (
        PipelineSource.BANK_OPERATING.value
        if account_type == "operating"
        else PipelineSource.BANK_RESERVE.value
    )

    return df.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.lit(customer_id).alias("customer_id"),
            pl.lit(f"bank_{account_type}").alias("series_id"),
            pl.lit(source).alias("source"),
            pl.when(pl.col("amount") >= 0)
            .then(pl.lit(CashFlowSign.INFLOW.value))
            .otherwise(pl.lit(CashFlowSign.OUTFLOW.value))
            .alias("sign"),
            pl.col("amount").abs().alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.lit(False).alias("is_anomaly"),
        ]
    ).select(list(CANONICAL_SCHEMA.keys()))


def adapt_ar_aging(df: pl.DataFrame, customer_id: str) -> pl.DataFrame:
    """
    Adapter for AR aging reports (invoice collection data).

    Expected raw columns: collection_date, invoice_id, amount_collected
    """
    required = {"collection_date", "amount_collected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"AR aging missing columns: {missing}")

    return df.with_columns(
        [
            pl.col("collection_date").cast(pl.Date).alias("date"),
            pl.lit(customer_id).alias("customer_id"),
            pl.lit("accounts_receivable").alias("series_id"),
            pl.lit(PipelineSource.AR.value).alias("source"),
            pl.lit(CashFlowSign.INFLOW.value).alias("sign"),
            pl.col("amount_collected").cast(pl.Float64).alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.lit(False).alias("is_anomaly"),
        ]
    ).select(list(CANONICAL_SCHEMA.keys()))


def adapt_subscription_billing(df: pl.DataFrame, customer_id: str) -> pl.DataFrame:
    """
    Adapter for subscription billing exports (Stripe, Recurly typical schema).

    Expected raw columns: charge_date, mrr, new_mrr, churned_mrr
    """
    required = {"charge_date", "mrr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Subscription billing missing columns: {missing}")

    return df.with_columns(
        [
            pl.col("charge_date").cast(pl.Date).alias("date"),
            pl.lit(customer_id).alias("customer_id"),
            pl.lit("sub_billing").alias("series_id"),
            pl.lit(PipelineSource.SUBSCRIPTION_BILLING.value).alias("source"),
            pl.lit(CashFlowSign.INFLOW.value).alias("sign"),
            pl.col("mrr").cast(pl.Float64).alias("amount"),
            pl.lit("USD").alias("currency"),
            pl.lit(False).alias("is_anomaly"),
        ]
    ).select(list(CANONICAL_SCHEMA.keys()))


# ── IngestPipeline ─────────────────────────────────────────────────────────────


ADAPTER_MAP: dict[PipelineSource, object] = {
    PipelineSource.ERP_REVENUE: adapt_erp_export,
    PipelineSource.PAYROLL: adapt_payroll_export,
    PipelineSource.BANK_OPERATING: adapt_bank_feed,
    PipelineSource.BANK_RESERVE: adapt_bank_feed,
    PipelineSource.AR: adapt_ar_aging,
    PipelineSource.SUBSCRIPTION_BILLING: adapt_subscription_billing,
}


class IngestPipeline:
    """
    Orchestrates loading and merging from multiple data pipeline sources
    for a single customer.

    Each customer gets their own pipeline instance so customer_id is always
    stamped on every record — multi-tenant safe.
    """

    def __init__(self, customer_id: str):
        self.customer_id = customer_id

    def load_from_csv(
        self,
        path: Path | str,
        source: PipelineSource,
        **adapter_kwargs,
    ) -> pl.DataFrame:
        """Load a real CSV from a pipeline export and normalise to canonical schema."""
        raw = pl.read_csv(str(path), try_parse_dates=True)
        adapter = ADAPTER_MAP.get(source)
        if adapter is None:
            raise ValueError(f"No adapter registered for {source}")
        return adapter(raw, self.customer_id, **adapter_kwargs)  # type: ignore[call-arg]

    def load_from_synthetic(
        self,
        n_days: int = 3 * 365,
        seed: int = 42,
    ) -> pl.DataFrame:
        """
        Load synthetic data as if it came from real pipelines.
        Stamps customer_id and renames 'value' → 'amount' to match canonical schema.
        """
        seq = generate_sequence_dataset(n_days=n_days, seed=seed)
        return (
            seq.rename({"value": "amount"})
            .with_columns(
                [
                    pl.lit(self.customer_id).alias("customer_id"),
                    pl.lit("USD").alias("currency"),
                ]
            )
            .select(list(CANONICAL_SCHEMA.keys()))
        )

    def merge_sources(self, frames: list[pl.DataFrame]) -> pl.DataFrame:
        """
        Merge DataFrames from multiple pipeline sources.
        All must conform to CANONICAL_SCHEMA.
        Validates schema, deduplicates, sorts.
        """
        for i, df in enumerate(frames):
            missing = set(CANONICAL_SCHEMA.keys()) - set(df.columns)
            if missing:
                raise ValueError(f"Frame {i} missing canonical columns: {missing}")

        merged = (
            pl.concat(frames)
            .unique(subset=["customer_id", "date", "series_id"])
            .sort(["customer_id", "series_id", "date"])
        )

        return merged

    def daily_cashflow_summary(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aggregate all pipeline sources into a single daily net cash flow per customer.

        Inflows are positive, outflows are negative in the net column.
        This is the company-level view suitable for forecasting.
        """
        return (
            df.with_columns(
                pl.when(pl.col("sign") == CashFlowSign.INFLOW.value)
                .then(pl.col("amount"))
                .otherwise(-pl.col("amount"))
                .alias("signed_amount")
            )
            .group_by(["customer_id", "date"])
            .agg(
                [
                    pl.col("signed_amount").sum().alias("net_cashflow"),
                    pl.col("signed_amount")
                    .filter(pl.col("sign") == CashFlowSign.INFLOW.value)
                    .sum()
                    .alias("total_inflow"),
                    pl.col("signed_amount")
                    .abs()
                    .filter(pl.col("sign") == CashFlowSign.OUTFLOW.value)
                    .sum()
                    .alias("total_outflow"),
                    pl.col("series_id").n_unique().alias("n_sources"),
                ]
            )
            .sort(["customer_id", "date"])
        )
