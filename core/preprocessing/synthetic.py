"""
Synthetic cash flow data generator.

Two dataset variants built on top of archetype-based customer simulation:

  generate_sequence_dataset()      — single-customer panel (Chronos / statsforecast)
  generate_multi_customer_dataset()— multi-customer panel with realistic archetypes
  generate_ml_dataset()            — tabular, feature-engineered rows for sklearn/XGBoost

Customer archetypes encode genuinely different business shapes so that clustering
finds meaningful segments rather than noise-level variation:

  EARLY_STAGE_FOUNDER   — pre-revenue / seed-stage: lumpy equity in, high burn, no AR
  SMB_SERVICES          — small services firm: steady project revenue, payroll-heavy
  SAAS_GROWTH           — scaling SaaS: rising MRR, sales-driven spend spikes
  MANUFACTURING         — inventory-driven: large irregular AR/AP, thin margins
  RETAIL_SEASONAL       — consumer retail: Q4 spike, supplier AP, thin operating buffer
  PROFESSIONAL_SERVICES — consulting/agency: milestone billing, low COGS, people costs
  MARKETPLACE           — two-sided platform: high GMV throughput, net-revenue model

Statistical properties:
  - Multi-level seasonality: weekly (payroll), monthly (AR/AP), annual (tax/Q4)
  - Heteroskedastic noise — higher variance on discretionary/variable sources
  - ~3–8% anomaly injection depending on archetype
  - Reproducible: pass seed= explicitly; RANDOM_SEED = 42 default

Temporal split rules (no leakage):
  - All splits are strictly chronological — never shuffle
  - Walk-forward CV uses expanding train window
  - Test set guarded behind get_test(acknowledged=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Literal

import numpy as np
import polars as pl

RANDOM_SEED = 42


# ── Enums ─────────────────────────────────────────────────────────────────────


class PipelineSource(str, Enum):
    """Real-world data pipeline sources that contribute to company cash flow."""
    ERP_REVENUE = "erp_revenue"
    ERP_COGS = "erp_cogs"
    PAYROLL = "payroll"
    AR = "accounts_receivable"
    AP = "accounts_payable"
    BANK_OPERATING = "bank_operating"
    BANK_RESERVE = "bank_reserve"
    SUBSCRIPTION_BILLING = "sub_billing"
    TAX_PROVISION = "tax_provision"
    EQUITY_FUNDING = "equity_funding"       # investor capital tranches
    INVENTORY_PURCHASE = "inventory_purch"  # stock/raw-material buying
    SALES_COMMISSION = "sales_commission"   # outbound sales spend
    MARKETPLACE_GMV = "marketplace_gmv"     # gross transaction volume (inflow)
    MARKETPLACE_PAYOUT = "marketplace_payout"  # seller payouts (outflow)


class CashFlowSign(str, Enum):
    INFLOW = "inflow"
    OUTFLOW = "outflow"


# ── Series configuration ───────────────────────────────────────────────────────


@dataclass
class SeriesConfig:
    name: str
    source: PipelineSource
    sign: CashFlowSign
    base_amount: float
    trend_rate: float       # additive daily trend (positive = growing)
    weekly_amp: float       # weekly seasonal amplitude
    monthly_amp: float      # monthly (30.44d) seasonal amplitude
    annual_amp: float       # annual (365.25d) seasonal amplitude
    noise_std: float
    anomaly_prob: float = 0.05
    anomaly_multiplier: float = 3.0
    # Phase shifts let different archetypes peak at different times in cycle
    weekly_phase: float = 0.0
    monthly_phase: float = 0.0
    annual_phase: float = 0.0


# ── Customer archetypes ────────────────────────────────────────────────────────


@dataclass
class CustomerArchetype:
    """
    A distinct business type with its own cash flow shape.

    series: the pipeline sources active for this archetype.
    weight: relative sampling probability in generate_multi_customer_dataset().
    noise_scale: per-instance jitter applied to base_amounts (0.2 = ±20% variation
                 within archetype, preserving inter-archetype differences).
    """
    name: str
    label: str          # human-readable segment name for the rule-based namer
    description: str    # one-sentence description
    series: list[SeriesConfig]
    weight: float = 1.0
    noise_scale: float = 0.15   # instance-level variation within archetype


# ── Archetype definitions ──────────────────────────────────────────────────────

ARCHETYPES: list[CustomerArchetype] = [

    # ── 1. Early-stage founder ─────────────────────────────────────────────────
    # Pre-product-market-fit. Revenue near zero. Lives on equity tranches (lumpy).
    # Burn is high relative to income. No AR pipeline yet.
    CustomerArchetype(
        name="early_stage_founder",
        label="Pre-Revenue Startup",
        description="Equity-funded startup with minimal revenue and high burn rate.",
        weight=1.5,
        noise_scale=0.25,
        series=[
            SeriesConfig(
                name="equity_funding", source=PipelineSource.EQUITY_FUNDING,
                sign=CashFlowSign.INFLOW,
                base_amount=0.0, trend_rate=0.0,
                weekly_amp=0.0, monthly_amp=0.0, annual_amp=0.0,
                noise_std=500.0,
                anomaly_prob=0.015, anomaly_multiplier=80.0,  # rare large tranches
            ),
            SeriesConfig(
                name="sub_billing", source=PipelineSource.SUBSCRIPTION_BILLING,
                sign=CashFlowSign.INFLOW,
                base_amount=300.0, trend_rate=4.0,
                weekly_amp=0.0, monthly_amp=80.0, annual_amp=100.0,
                noise_std=120.0, anomaly_prob=0.02,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=4_500.0, trend_rate=8.0,  # headcount growing fast
                weekly_amp=0.0, monthly_amp=500.0, annual_amp=800.0,
                noise_std=200.0, anomaly_prob=0.01,
            ),
            SeriesConfig(
                name="accounts_payable", source=PipelineSource.AP,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_200.0, trend_rate=3.0,
                weekly_amp=100.0, monthly_amp=300.0, annual_amp=0.0,
                noise_std=400.0, anomaly_prob=0.04,
            ),
            SeriesConfig(
                name="bank_operating", source=PipelineSource.BANK_OPERATING,
                sign=CashFlowSign.INFLOW,
                base_amount=200.0, trend_rate=1.0,
                weekly_amp=50.0, monthly_amp=100.0, annual_amp=0.0,
                noise_std=150.0, anomaly_prob=0.03,
            ),
        ],
    ),

    # ── 2. SMB services ────────────────────────────────────────────────────────
    # 10–50 person professional services firm. Steady project revenue (~$2M ARR).
    # Payroll is the dominant cost. Net-30 invoicing. Some quarterly seasonality.
    CustomerArchetype(
        name="smb_services",
        label="SMB Services",
        description="Small professional services firm with steady project billing and payroll-heavy costs.",
        weight=2.0,
        noise_scale=0.12,
        series=[
            SeriesConfig(
                name="erp_revenue", source=PipelineSource.ERP_REVENUE,
                sign=CashFlowSign.INFLOW,
                base_amount=5_500.0, trend_rate=3.0,
                weekly_amp=0.0, monthly_amp=1_200.0, annual_amp=2_000.0,
                noise_std=800.0, anomaly_prob=0.04, anomaly_multiplier=3.0,
            ),
            SeriesConfig(
                name="accounts_receivable", source=PipelineSource.AR,
                sign=CashFlowSign.INFLOW,
                base_amount=4_000.0, trend_rate=2.0,
                weekly_amp=800.0, monthly_amp=2_500.0, annual_amp=1_500.0,
                noise_std=1_500.0, anomaly_prob=0.06, anomaly_multiplier=4.0,
                monthly_phase=0.3,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=6_000.0, trend_rate=1.5,
                weekly_amp=0.0, monthly_amp=600.0, annual_amp=1_200.0,
                noise_std=200.0, anomaly_prob=0.01,
            ),
            SeriesConfig(
                name="accounts_payable", source=PipelineSource.AP,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_500.0, trend_rate=0.5,
                weekly_amp=200.0, monthly_amp=400.0, annual_amp=500.0,
                noise_std=400.0, anomaly_prob=0.03,
            ),
            SeriesConfig(
                name="tax_provision", source=PipelineSource.TAX_PROVISION,
                sign=CashFlowSign.OUTFLOW,
                base_amount=600.0, trend_rate=0.2,
                weekly_amp=0.0, monthly_amp=0.0, annual_amp=2_000.0,
                noise_std=100.0, anomaly_prob=0.01, anomaly_multiplier=4.0,
            ),
        ],
    ),

    # ── 3. SaaS growth ────────────────────────────────────────────────────────
    # $5–15M ARR, MRR growing ~5%/mo. Low COGS. Sales and marketing is the main
    # outflow. Annual contract spikes in Q4.
    CustomerArchetype(
        name="saas_growth",
        label="SaaS Growth",
        description="Scaling SaaS business with rising MRR, low COGS, and sales-driven spend spikes.",
        weight=2.0,
        noise_scale=0.10,
        series=[
            SeriesConfig(
                name="sub_billing", source=PipelineSource.SUBSCRIPTION_BILLING,
                sign=CashFlowSign.INFLOW,
                base_amount=18_000.0, trend_rate=18.0,  # fast MRR growth
                weekly_amp=0.0, monthly_amp=2_000.0, annual_amp=8_000.0,
                noise_std=600.0, anomaly_prob=0.02, anomaly_multiplier=3.0,
                annual_phase=0.1,  # Q4-skewed
            ),
            SeriesConfig(
                name="erp_revenue", source=PipelineSource.ERP_REVENUE,
                sign=CashFlowSign.INFLOW,
                base_amount=8_000.0, trend_rate=8.0,
                weekly_amp=0.0, monthly_amp=1_500.0, annual_amp=5_000.0,
                noise_std=1_000.0, anomaly_prob=0.03, anomaly_multiplier=4.0,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=12_000.0, trend_rate=15.0,  # heavy hiring
                weekly_amp=0.0, monthly_amp=1_200.0, annual_amp=3_000.0,
                noise_std=400.0, anomaly_prob=0.01,
            ),
            SeriesConfig(
                name="sales_commission", source=PipelineSource.SALES_COMMISSION,
                sign=CashFlowSign.OUTFLOW,
                base_amount=4_000.0, trend_rate=12.0,
                weekly_amp=0.0, monthly_amp=800.0, annual_amp=4_000.0,
                noise_std=1_200.0, anomaly_prob=0.05, anomaly_multiplier=3.5,
            ),
            SeriesConfig(
                name="tax_provision", source=PipelineSource.TAX_PROVISION,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_800.0, trend_rate=1.5,
                weekly_amp=0.0, monthly_amp=0.0, annual_amp=5_000.0,
                noise_std=200.0, anomaly_prob=0.01, anomaly_multiplier=4.0,
            ),
        ],
    ),

    # ── 4. Manufacturing / commodity ──────────────────────────────────────────
    # Capital-intensive. Large irregular AR (enterprise buyers, net-60).
    # Big inventory purchase spikes. Thin margins. Seasonal demand.
    CustomerArchetype(
        name="manufacturing",
        label="Manufacturing",
        description="Capital-intensive manufacturer with lumpy AR collections and inventory-driven cost cycles.",
        weight=1.0,
        noise_scale=0.20,
        series=[
            SeriesConfig(
                name="accounts_receivable", source=PipelineSource.AR,
                sign=CashFlowSign.INFLOW,
                base_amount=22_000.0, trend_rate=2.0,
                weekly_amp=2_000.0, monthly_amp=8_000.0, annual_amp=12_000.0,
                noise_std=6_000.0, anomaly_prob=0.08, anomaly_multiplier=5.0,
                monthly_phase=0.5,  # collections cluster mid-month
            ),
            SeriesConfig(
                name="erp_revenue", source=PipelineSource.ERP_REVENUE,
                sign=CashFlowSign.INFLOW,
                base_amount=15_000.0, trend_rate=1.0,
                weekly_amp=500.0, monthly_amp=3_000.0, annual_amp=10_000.0,
                noise_std=3_000.0, anomaly_prob=0.04,
            ),
            SeriesConfig(
                name="inventory_purch", source=PipelineSource.INVENTORY_PURCHASE,
                sign=CashFlowSign.OUTFLOW,
                base_amount=12_000.0, trend_rate=1.0,
                weekly_amp=1_000.0, monthly_amp=5_000.0, annual_amp=8_000.0,
                noise_std=4_000.0, anomaly_prob=0.07, anomaly_multiplier=4.0,
                monthly_phase=0.1,
            ),
            SeriesConfig(
                name="accounts_payable", source=PipelineSource.AP,
                sign=CashFlowSign.OUTFLOW,
                base_amount=8_000.0, trend_rate=0.5,
                weekly_amp=800.0, monthly_amp=3_000.0, annual_amp=4_000.0,
                noise_std=2_000.0, anomaly_prob=0.05,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=9_000.0, trend_rate=0.5,
                weekly_amp=0.0, monthly_amp=900.0, annual_amp=1_500.0,
                noise_std=300.0, anomaly_prob=0.01,
            ),
            SeriesConfig(
                name="tax_provision", source=PipelineSource.TAX_PROVISION,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_200.0, trend_rate=0.1,
                weekly_amp=0.0, monthly_amp=0.0, annual_amp=5_000.0,
                noise_std=200.0, anomaly_prob=0.01,
            ),
        ],
    ),

    # ── 5. Retail / seasonal ──────────────────────────────────────────────────
    # Consumer retail. Q4 is 40% of annual revenue. Tight margins. Constant
    # supplier AP. Bank operating is primary cash visibility.
    CustomerArchetype(
        name="retail_seasonal",
        label="Retail Seasonal",
        description="Consumer retailer with dominant Q4 revenue spike and thin year-round margins.",
        weight=1.0,
        noise_scale=0.18,
        series=[
            SeriesConfig(
                name="erp_revenue", source=PipelineSource.ERP_REVENUE,
                sign=CashFlowSign.INFLOW,
                base_amount=8_000.0, trend_rate=1.0,
                weekly_amp=2_000.0, monthly_amp=1_000.0, annual_amp=18_000.0,
                noise_std=2_500.0, anomaly_prob=0.04,
                annual_phase=0.25,  # peaks in Nov/Dec
            ),
            SeriesConfig(
                name="bank_operating", source=PipelineSource.BANK_OPERATING,
                sign=CashFlowSign.INFLOW,
                base_amount=3_000.0, trend_rate=0.5,
                weekly_amp=1_500.0, monthly_amp=500.0, annual_amp=5_000.0,
                noise_std=1_000.0, anomaly_prob=0.05,
                annual_phase=0.25,
            ),
            SeriesConfig(
                name="inventory_purch", source=PipelineSource.INVENTORY_PURCHASE,
                sign=CashFlowSign.OUTFLOW,
                base_amount=5_500.0, trend_rate=0.5,
                weekly_amp=500.0, monthly_amp=800.0, annual_amp=10_000.0,
                noise_std=2_000.0, anomaly_prob=0.06, anomaly_multiplier=3.5,
                annual_phase=-0.1,  # buys stock ahead of Q4
            ),
            SeriesConfig(
                name="accounts_payable", source=PipelineSource.AP,
                sign=CashFlowSign.OUTFLOW,
                base_amount=4_000.0, trend_rate=0.3,
                weekly_amp=400.0, monthly_amp=600.0, annual_amp=2_000.0,
                noise_std=800.0, anomaly_prob=0.04,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=3_500.0, trend_rate=0.2,
                weekly_amp=0.0, monthly_amp=350.0, annual_amp=500.0,
                noise_std=150.0, anomaly_prob=0.01,
            ),
        ],
    ),

    # ── 6. Professional services / consulting ─────────────────────────────────
    # Milestone-based billing. High utilisation, low COGS. People costs dominate.
    # Longer payment cycles (net-45 to net-60). Low anomaly rate.
    CustomerArchetype(
        name="professional_services",
        label="Professional Services",
        description="Consulting or agency business with milestone billing, high utilisation, and people-cost dominance.",
        weight=1.5,
        noise_scale=0.10,
        series=[
            SeriesConfig(
                name="erp_revenue", source=PipelineSource.ERP_REVENUE,
                sign=CashFlowSign.INFLOW,
                base_amount=14_000.0, trend_rate=5.0,
                weekly_amp=0.0, monthly_amp=4_000.0, annual_amp=3_000.0,
                noise_std=2_500.0, anomaly_prob=0.05, anomaly_multiplier=3.0,
                monthly_phase=0.6,  # milestone at end of month
            ),
            SeriesConfig(
                name="accounts_receivable", source=PipelineSource.AR,
                sign=CashFlowSign.INFLOW,
                base_amount=10_000.0, trend_rate=4.0,
                weekly_amp=500.0, monthly_amp=3_500.0, annual_amp=2_000.0,
                noise_std=2_000.0, anomaly_prob=0.04, anomaly_multiplier=3.5,
                monthly_phase=0.7,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=16_000.0, trend_rate=4.0,
                weekly_amp=0.0, monthly_amp=1_600.0, annual_amp=3_500.0,
                noise_std=400.0, anomaly_prob=0.01,
            ),
            SeriesConfig(
                name="accounts_payable", source=PipelineSource.AP,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_800.0, trend_rate=0.5,
                weekly_amp=100.0, monthly_amp=300.0, annual_amp=500.0,
                noise_std=300.0, anomaly_prob=0.02,
            ),
            SeriesConfig(
                name="tax_provision", source=PipelineSource.TAX_PROVISION,
                sign=CashFlowSign.OUTFLOW,
                base_amount=1_400.0, trend_rate=0.4,
                weekly_amp=0.0, monthly_amp=0.0, annual_amp=4_500.0,
                noise_std=150.0, anomaly_prob=0.01,
            ),
        ],
    ),

    # ── 7. Marketplace / platform ─────────────────────────────────────────────
    # Two-sided platform. High gross transaction volume, low net take-rate (~10%).
    # Payout velocity is the key risk. Strong weekend/event-driven weekly pattern.
    CustomerArchetype(
        name="marketplace",
        label="Marketplace Platform",
        description="Two-sided platform with high GMV throughput, rapid seller payouts, and event-driven weekly cycles.",
        weight=1.0,
        noise_scale=0.15,
        series=[
            SeriesConfig(
                name="marketplace_gmv", source=PipelineSource.MARKETPLACE_GMV,
                sign=CashFlowSign.INFLOW,
                base_amount=55_000.0, trend_rate=30.0,
                weekly_amp=15_000.0, monthly_amp=5_000.0, annual_amp=20_000.0,
                noise_std=8_000.0, anomaly_prob=0.04, anomaly_multiplier=3.0,
                weekly_phase=0.5,  # peaks Friday/Saturday
            ),
            SeriesConfig(
                name="marketplace_payout", source=PipelineSource.MARKETPLACE_PAYOUT,
                sign=CashFlowSign.OUTFLOW,
                base_amount=49_500.0, trend_rate=27.0,  # ~90% payout ratio
                weekly_amp=13_000.0, monthly_amp=4_500.0, annual_amp=18_000.0,
                noise_std=7_000.0, anomaly_prob=0.03, anomaly_multiplier=2.5,
                weekly_phase=0.6,  # payouts lag GMV by ~1 day
            ),
            SeriesConfig(
                name="sub_billing", source=PipelineSource.SUBSCRIPTION_BILLING,
                sign=CashFlowSign.INFLOW,
                base_amount=3_000.0, trend_rate=5.0,  # seller subscription fees
                weekly_amp=0.0, monthly_amp=500.0, annual_amp=1_000.0,
                noise_std=300.0, anomaly_prob=0.02,
            ),
            SeriesConfig(
                name="payroll", source=PipelineSource.PAYROLL,
                sign=CashFlowSign.OUTFLOW,
                base_amount=5_000.0, trend_rate=8.0,
                weekly_amp=0.0, monthly_amp=500.0, annual_amp=800.0,
                noise_std=200.0, anomaly_prob=0.01,
            ),
        ],
    ),
]

# Lookup by archetype name
ARCHETYPE_BY_NAME: dict[str, CustomerArchetype] = {a.name: a for a in ARCHETYPES}

# Original single-company default (preserves backward compat for forecast agent)
DEFAULT_SERIES: list[SeriesConfig] = ARCHETYPES[2].series  # SaaS growth as representative default


# ── Raw signal generator ───────────────────────────────────────────────────────


def _generate_raw_series(
    config: SeriesConfig,
    start_date: date,
    n_days: int,
    rng: np.random.Generator,
    amount_scale: float = 1.0,
) -> pl.DataFrame:
    """Generate one synthetic series as a daily panel."""
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    t = np.arange(n_days, dtype=float)

    base = config.base_amount * amount_scale
    trend = base + config.trend_rate * amount_scale * t
    weekly = config.weekly_amp * amount_scale * np.sin(2 * np.pi * t / 7 + config.weekly_phase)
    monthly = config.monthly_amp * amount_scale * np.sin(2 * np.pi * t / 30.44 + config.monthly_phase)
    annual = config.annual_amp * amount_scale * np.sin(2 * np.pi * t / 365.25 + config.annual_phase)
    noise_scale = config.noise_std * amount_scale * (1.0 + 0.0005 * t)
    noise = rng.normal(0.0, noise_scale)
    values = trend + weekly + monthly + annual + noise

    anomaly_mask = rng.random(n_days) < config.anomaly_prob
    direction = rng.choice([-1.0, 1.0], size=n_days)
    values[anomaly_mask] *= config.anomaly_multiplier * direction[anomaly_mask]
    values = np.clip(values, 0.0, None)

    return pl.DataFrame({
        "date": dates,
        "series_id": config.name,
        "source": config.source.value,
        "sign": config.sign.value,
        "value": values.tolist(),
        "is_anomaly": anomaly_mask.tolist(),
    }).with_columns(pl.col("date").cast(pl.Date))


# ── Single-customer sequence dataset (Chronos / statsforecast format) ──────────


def generate_sequence_dataset(
    start_date: date = date(2021, 1, 1),
    n_days: int = 3 * 365,
    series_configs: list[SeriesConfig] | None = None,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Panel time-series dataset for Chronos / statsforecast.

    Schema: [date, series_id, source, sign, value, is_anomaly]
    One row per (date, series) — strictly sorted by (series_id, date).

    Uses the SaaS-growth archetype series by default (backward-compatible).
    """
    rng = np.random.default_rng(seed)
    configs = series_configs or DEFAULT_SERIES
    frames = [_generate_raw_series(c, start_date, n_days, rng) for c in configs]
    return pl.concat(frames).sort(["series_id", "date"])


# ── Multi-customer dataset with archetype-based variation ──────────────────────


def generate_multi_customer_dataset(
    n_customers: int = 50,
    n_days: int = 365,
    start_date: date = date(2023, 1, 1),
    seed: int = RANDOM_SEED,
    archetype_weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """
    Generate a realistic multi-customer cash flow dataset.

    Each customer is assigned a CustomerArchetype (with weighted random sampling),
    then given instance-level amount jitter (noise_scale) so customers within the
    same archetype differ in scale while preserving the archetype's distinctive
    cash flow *shape*.

    Schema: [date, series_id, source, sign, value, is_anomaly, customer_id, archetype]

    Args:
        n_customers:       Total number of customers to generate.
        n_days:            Days of history per customer.
        start_date:        Start date for all series.
        seed:              Master RNG seed — fully reproducible.
        archetype_weights: Override default archetype weights (dict of name → weight).
                           If None, uses each archetype's built-in weight.

    Returns:
        Polars DataFrame sorted by (customer_id, series_id, date).
    """
    rng = np.random.default_rng(seed)

    archetypes = ARCHETYPES
    weights = np.array([
        archetype_weights.get(a.name, a.weight) if archetype_weights else a.weight
        for a in archetypes
    ])
    weights = weights / weights.sum()

    assigned: list[CustomerArchetype] = list(
        rng.choice(archetypes, size=n_customers, p=weights)  # type: ignore[arg-type]
    )

    frames: list[pl.DataFrame] = []
    for i, archetype in enumerate(assigned):
        customer_id = f"cust-{i:03d}"
        # Per-instance amount scale: log-normal so some customers are 2–3× larger
        # within their archetype without changing the shape
        amount_scale = float(rng.lognormal(mean=0.0, sigma=archetype.noise_scale))

        for config in archetype.series:
            series_rng = np.random.default_rng(seed + i * 100 + hash(config.name) % 10_000)
            df = _generate_raw_series(config, start_date, n_days, series_rng, amount_scale)
            df = df.with_columns([
                pl.lit(customer_id).alias("customer_id"),
                pl.lit(archetype.name).alias("archetype"),
            ])
            frames.append(df)

    return pl.concat(frames).sort(["customer_id", "series_id", "date"])


# ── ML tabular dataset (sklearn / XGBoost format) ─────────────────────────────


def generate_ml_dataset(
    start_date: date = date(2021, 1, 1),
    n_days: int = 3 * 365,
    series_configs: list[SeriesConfig] | None = None,
    horizon_days: int = 30,
    seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Tabular dataset for traditional ML (XGBoost, LightGBM, sklearn regressors).

    Each row is one (series, date) observation with:
    - Lag features: t-1, t-7, t-14, t-30
    - Rolling stats: mean/std over 7, 14, 30, 90 day windows
    - Calendar features: day_of_week, day_of_month, month, quarter, is_month_end
    - Target: value at t+horizon_days (direct multi-step forecast target)
    - target_1d through target_7d for sequence-to-sequence training
    """
    seq = generate_sequence_dataset(start_date, n_days, series_configs, seed)
    frames: list[pl.DataFrame] = []

    for sid in seq["series_id"].unique().sort().to_list():
        s = seq.filter(pl.col("series_id") == sid).sort("date")
        v = s["value"].to_numpy()
        n = len(v)

        lags: dict[str, list] = {
            "lag_1d": [float("nan")] * 1 + v[:-1].tolist(),
            "lag_7d": [float("nan")] * 7 + v[:-7].tolist(),
            "lag_14d": [float("nan")] * 14 + v[:-14].tolist(),
            "lag_30d": [float("nan")] * 30 + v[:-30].tolist(),
        }
        roll_mean_7 = _rolling_mean(v, 7)
        roll_std_7 = _rolling_std(v, 7)
        roll_mean_14 = _rolling_mean(v, 14)
        roll_mean_30 = _rolling_mean(v, 30)
        roll_std_30 = _rolling_std(v, 30)
        roll_mean_90 = _rolling_mean(v, 90)

        dates = s["date"].to_list()
        dow = [d.weekday() for d in dates]
        dom = [d.day for d in dates]
        month = [d.month for d in dates]
        quarter = [(d.month - 1) // 3 + 1 for d in dates]
        is_month_end = [
            1 if (dates[i] + timedelta(days=1)).day == 1 else 0
            for i in range(n)
        ]
        is_quarter_end = [
            1 if month[i] in (3, 6, 9, 12) and is_month_end[i] else 0
            for i in range(n)
        ]

        target = [v[i + horizon_days] if i + horizon_days < n else float("nan") for i in range(n)]
        targets_1_7 = {
            f"target_{k}d": [v[i + k] if i + k < n else float("nan") for i in range(n)]
            for k in range(1, 8)
        }

        row_df = s.with_columns([
            pl.Series("lag_1d", lags["lag_1d"]),
            pl.Series("lag_7d", lags["lag_7d"]),
            pl.Series("lag_14d", lags["lag_14d"]),
            pl.Series("lag_30d", lags["lag_30d"]),
            pl.Series("roll_mean_7d", roll_mean_7),
            pl.Series("roll_std_7d", roll_std_7),
            pl.Series("roll_mean_14d", roll_mean_14),
            pl.Series("roll_mean_30d", roll_mean_30),
            pl.Series("roll_std_30d", roll_std_30),
            pl.Series("roll_mean_90d", roll_mean_90),
            pl.Series("day_of_week", dow).cast(pl.Int8),
            pl.Series("day_of_month", dom).cast(pl.Int8),
            pl.Series("month", month).cast(pl.Int8),
            pl.Series("quarter", quarter).cast(pl.Int8),
            pl.Series("is_month_end", is_month_end).cast(pl.Int8),
            pl.Series("is_quarter_end", is_quarter_end).cast(pl.Int8),
            pl.Series(f"target_{horizon_days}d", target),
            *[pl.Series(k, v2) for k, v2 in targets_1_7.items()],
        ])
        frames.append(row_df)

    return pl.concat(frames).sort(["series_id", "date"])


def _rolling_mean(arr: np.ndarray, window: int) -> list[float]:
    out = [float("nan")] * len(arr)
    for i in range(window - 1, len(arr)):
        out[i] = float(np.mean(arr[i - window + 1 : i + 1]))
    return out


def _rolling_std(arr: np.ndarray, window: int) -> list[float]:
    out = [float("nan")] * len(arr)
    for i in range(window - 1, len(arr)):
        out[i] = float(np.std(arr[i - window + 1 : i + 1], ddof=1))
    return out


# ── Temporal splits (shared for both dataset types) ───────────────────────────


@dataclass
class TemporalSplit:
    """
    Strictly chronological train/val/test split.
    Test is guarded — call get_test(acknowledged=True) for final evaluation only.
    """
    train: pl.DataFrame
    val: pl.DataFrame
    _test: pl.DataFrame
    split_dates: dict[str, date]

    def get_test(self, *, acknowledged: bool = False) -> pl.DataFrame:
        if not acknowledged:
            raise RuntimeError(
                "Call get_test(acknowledged=True) only for final evaluation. "
                "Never touch test data during development or CV."
            )
        return self._test

    def summary(self) -> str:
        return (
            f"Train: {self.split_dates['train_start']} → {self.split_dates['train_end']} "
            f"({len(self.train):,} rows)\n"
            f"Val:   {self.split_dates['val_start']} → {self.split_dates['val_end']} "
            f"({len(self.val):,} rows)\n"
            f"Test:  {self.split_dates['test_start']} → {self.split_dates['test_end']} "
            f"({len(self._test):,} rows) [HELD OUT]"
        )


def temporal_split(
    df: pl.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.20,
) -> TemporalSplit:
    """Chronological split on unique dates. Works for both ML and sequence datasets."""
    sorted_df = df.sort("date")
    dates = sorted_df["date"].unique().sort()
    n = len(dates)

    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    assert n_train >= 2, "Train split too small"

    train_end = dates[n_train - 1]
    val_end = dates[n_train + n_val - 1]

    train = sorted_df.filter(pl.col("date") <= train_end)
    val = sorted_df.filter((pl.col("date") > train_end) & (pl.col("date") <= val_end))
    test = sorted_df.filter(pl.col("date") > val_end)

    return TemporalSplit(
        train=train,
        val=val,
        _test=test,
        split_dates={
            "train_start": dates[0],
            "train_end": train_end,
            "val_start": dates[n_train],
            "val_end": val_end,
            "test_start": dates[n_train + n_val],
            "test_end": dates[-1],
        },
    )


@dataclass
class WalkForwardFold:
    train: pl.DataFrame
    val: pl.DataFrame
    fold_idx: int
    train_end: date
    val_end: date


def walk_forward_cv(
    df: pl.DataFrame,
    horizon_days: int = 30,
    min_train_days: int = 365,
    step_days: int = 30,
    max_folds: int = 12,
) -> list[WalkForwardFold]:
    """
    Expanding-window walk-forward cross-validation.
    Works on both ML tabular and sequence datasets (date column must exist).
    """
    sorted_df = df.sort("date")
    dates = sorted_df["date"].unique().sort().to_list()
    n = len(dates)

    folds: list[WalkForwardFold] = []
    cursor = min_train_days
    fold_idx = 0

    while cursor + horizon_days <= n and fold_idx < max_folds:
        train_end = dates[cursor - 1]
        val_end = dates[min(cursor + horizon_days - 1, n - 1)]

        train = sorted_df.filter(pl.col("date") <= train_end)
        val = sorted_df.filter(
            (pl.col("date") > train_end) & (pl.col("date") <= val_end)
        )

        if len(val) > 0:
            folds.append(WalkForwardFold(
                train=train, val=val,
                fold_idx=fold_idx, train_end=train_end, val_end=val_end,
            ))

        cursor += step_days
        fold_idx += 1

    return folds
