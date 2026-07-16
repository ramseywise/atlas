"""
Seed MetricDefinition nodes into the knowledge graph on first boot.
Call: python -m core.knowledge.seeds
"""
from __future__ import annotations

from core.knowledge.graph import AtlasGraph

METRIC_SEEDS = [
    {
        "name": "burn_ratio",
        "definition": "Monthly cash outflows divided by monthly cash inflows. A ratio > 1.0 means the business spends more than it earns.",
        "formula": "avg_monthly_outflow / avg_monthly_inflow",
    },
    {
        "name": "runway",
        "definition": "Estimated months until cash reserves reach zero, calculated from current cash balance divided by net monthly burn.",
        "formula": "cash_balance / max(monthly_outflow - monthly_inflow, 1)",
    },
    {
        "name": "inflow_cv",
        "definition": "Coefficient of variation for monthly inflows — standard deviation divided by mean. Higher values indicate more volatile revenue.",
        "formula": "inflow_std / avg_monthly_inflow",
    },
    {
        "name": "inflow_growth_rate",
        "definition": "Revenue growth rate comparing the second half of a customer's history to the first half.",
        "formula": "(avg_inflow_second_half - avg_inflow_first_half) / abs(avg_inflow_first_half)",
    },
    {
        "name": "net_cashflow",
        "definition": "Total inflows minus total outflows over the measurement period.",
        "formula": "total_inflow - total_outflow",
    },
    {
        "name": "mase",
        "definition": "Mean Absolute Scaled Error — forecast error relative to a naïve seasonal baseline. Values < 1.0 indicate the model beats the baseline.",
        "formula": "mean(|forecast - actual|) / mean(|actual_t - actual_{t-7}|)",
    },
    {
        "name": "smape",
        "definition": "Symmetric Mean Absolute Percentage Error — percentage forecast error symmetric around zero. Target < 15%.",
        "formula": "2 * mean(|forecast - actual| / (|forecast| + |actual|))",
    },
]


def seed(uri: str | None = None, user: str | None = None, password: str | None = None):
    kwargs = {}
    if uri:
        kwargs["uri"] = uri
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password

    g = AtlasGraph(**kwargs)
    g.bootstrap_schema()
    for m in METRIC_SEEDS:
        g.upsert_metric_definition(m["name"], m["definition"], m.get("formula"))
    g.close()
    print(f"Seeded {len(METRIC_SEEDS)} metric definitions.")


if __name__ == "__main__":
    seed()
