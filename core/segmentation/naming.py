"""
LLM-powered segment naming.

Haiku receives cluster centroids (feature values) and returns a human-readable
label + one-sentence description for each cluster.

Centroid features come from CustomerProfile.feature_names() — the same vector
that was used for clustering.

Falls back to rule-based names if ANTHROPIC_API_KEY is absent or the call fails.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from core.preprocessing.customer import CustomerProfile


@dataclass
class SegmentLabel:
    cluster_id: int
    label: str          # short name, e.g. "High-Growth SaaS"
    description: str    # one sentence


NAMING_SYSTEM_PROMPT = """You are a B2B financial analyst naming customer segments.
You receive cluster centroids as JSON (feature_name → centroid_value) for each cluster.
Return a JSON object mapping cluster_id (string) to {"label": "...", "description": "..."}.

Known archetypes to draw from (use as inspiration, not as a strict lookup):
  Pre-Revenue Startup, SaaS Growth, SMB Services, Manufacturing,
  Retail Seasonal, Professional Services, Marketplace Platform.

Rules:
- label: 2-4 words, describe the segment's dominant financial behaviour
- description: one sentence explaining what makes this group distinct from the OTHERS
- Use business language — reference the archetype taxonomy above when it fits
- Do not use cluster numbers in the label
- CRITICAL: Every label must be unique — no two clusters may share the same label
- If two clusters are similar, differentiate by degree or dominant trait
  (e.g. "Early SaaS" vs "Scale-Stage SaaS", "Asset-Heavy Mfg" vs "Lean Mfg")
- Output only valid JSON, no markdown
"""


def name_segments(
    centroids: dict[int, np.ndarray],
    feature_names: list[str] | None = None,
) -> dict[int, SegmentLabel]:
    """
    Generate human-readable labels for each cluster centroid.

    Args:
        centroids:     {cluster_id: centroid_vector} — mean feature vector per cluster
        feature_names: column names matching centroid vectors (default: CustomerProfile.feature_names())

    Returns:
        {cluster_id: SegmentLabel}
    """
    if feature_names is None:
        feature_names = CustomerProfile.feature_names()

    centroid_json = {
        str(cid): {
            name: round(float(val), 4)
            for name, val in zip(feature_names, vec)
        }
        for cid, vec in centroids.items()
    }

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    raw: dict | None = None

    if api_key:
        raw = _call_haiku(centroid_json, api_key)

    if raw is None:
        raw = _rule_based_names(centroids, feature_names)

    result: dict[int, SegmentLabel] = {}
    for cid in centroids:
        entry = raw.get(str(cid), {})
        result[cid] = SegmentLabel(
            cluster_id=cid,
            label=entry.get("label", f"Segment {cid}"),
            description=entry.get("description", ""),
        )
    return result


def _call_haiku(centroid_json: dict, api_key: str) -> dict | None:
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_msg = f"Cluster centroids:\n{json.dumps(centroid_json, indent=2)}\n\nName each segment."

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=NAMING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        return json.loads(text)
    except Exception:
        return None


def _rule_based_names(
    centroids: dict[int, np.ndarray],
    feature_names: list[str],
) -> dict:
    """
    Heuristic labelling matching the CustomerArchetype taxonomy in synthetic.py.

    Priority order within each cluster: GMV-scale (marketplace) → high burn /
    equity (founder) → strong MRR trend (SaaS) → seasonal spike (retail) →
    high AR volatility (manufacturing) → payroll-dominant (professional services
    or SMB) → fallback balanced/constrained.
    """
    idx = {name: i for i, name in enumerate(feature_names)}

    def _get(vec: np.ndarray, name: str, default: float = 0.0) -> float:
        return float(vec[idx[name]]) if name in idx else default

    def _classify(vec: np.ndarray) -> tuple[str, str]:
        inflow = _get(vec, "total_inflow")
        outflow = _get(vec, "total_outflow")
        net = _get(vec, "net_position")
        inflow_share = _get(vec, "inflow_share", 0.5)
        trend = _get(vec, "trend_slope_norm")
        inflow_cv = _get(vec, "inflow_cv")
        outflow_cv = _get(vec, "outflow_cv")
        weekly_ac = _get(vec, "weekly_autocorr")
        n_series = _get(vec, "n_active_series", 4)
        top_share = _get(vec, "top_source_share", 0.5)

        # Absolute scale guard — use total volumes not just ratios
        total_vol = inflow + outflow

        # Marketplace: extreme gross volume, payout ratio ~90%, weekly cycle
        if total_vol > 60_000 and outflow / max(inflow, 1) > 0.75 and weekly_ac > 0.2:
            return "Marketplace Platform", "High GMV throughput with rapid seller payouts and event-driven weekly cycles."

        # Pre-revenue startup: inflow_share very low (< 0.25), deep negative net
        if inflow_share < 0.25 and net < -500_000:
            return "Pre-Revenue Startup", "Equity-funded startup with minimal revenue and high operational burn."

        # Retail seasonal: strong weekly autocorrelation is the fingerprint of
        # weekend-driven consumer spend + high inflow_cv from Q4 spike
        if weekly_ac > 0.35 and inflow_cv > 0.6:
            return "Retail Seasonal", "Consumer retailer with dominant seasonal revenue spike and thin margins."

        # Manufacturing: high inflow_cv (lumpy enterprise AR) + high outflow_cv
        # (inventory purchase spikes) + multiple active series
        if inflow_cv > 0.8 and outflow_cv > 0.6 and n_series >= 5:
            return "Manufacturing", "Capital-intensive manufacturer with lumpy AR collections and inventory-driven costs."

        # SaaS growth: strong positive trend, good inflow share, low volatility
        if trend > 0.05 and inflow_share > 0.45 and inflow_cv < 0.7:
            return "SaaS Growth", "Scaling subscription business with rising MRR and sales-driven spend."

        # Professional services: inflow_share moderate-high, low top_share
        # (revenue spread across multiple clients), low outflow_cv
        if inflow_share > 0.45 and top_share < 0.55 and outflow_cv < 0.5:
            return "Professional Services", "Consulting firm with milestone billing and people-cost dominance."

        # SMB services: inflow_share positive, net positive or near-zero, modest scale
        if inflow_share > 0.40 and net >= -100_000:
            return "SMB Services", "Small services firm with steady project revenue and payroll-heavy costs."

        # Fallback: net-negative residual
        if net < 0:
            return "Cash-Constrained", "Net outflow position — operational costs exceed current revenue."

        return "Balanced Operations", "Moderate inflow/outflow mix with stable cash position."

    # Rank centroids by total_inflow descending so larger clusters get first pick
    ranked = sorted(centroids.items(), key=lambda kv: -_get(kv[1], "total_inflow"))

    result: dict[str, dict] = {}
    used_labels: set[str] = set()

    for cid, vec in ranked:
        label, desc = _classify(vec)
        if label in used_labels:
            # Differentiate by appending the most distinctive scalar
            net = _get(vec, "net_position")
            trend = _get(vec, "trend_slope_norm")
            qualifier = "High-Burn" if net < -1000 else ("Fast-Growing" if trend > 0.1 else "Stable")
            label = f"{qualifier} {label}"
        used_labels.add(label)
        result[str(cid)] = {"label": label, "description": desc}

    return result


def compute_centroids(
    X: np.ndarray,
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Compute mean feature vector per cluster (excludes noise label -1)."""
    centroids = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask = labels == cid
        centroids[int(cid)] = X[mask].mean(axis=0)
    return centroids
