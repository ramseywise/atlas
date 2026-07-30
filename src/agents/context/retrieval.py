"""
Historical forecast retrieval tool for the forecast agent.

Retrieves completed forecast runs from the episodic memory store, filtered
by customer, horizon, and/or time range. Provides the "Select" lever from
agent-context.md §1 — fetch only the relevant subset before injecting.

Design notes:
  - Source: the same file-backed ForecastMemoryStore used by memory.py
  - No vector embeddings needed at this scale; filter + recency sort suffices
  - Returns structured results (not raw JSON) so callers get typed data
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.agents.context.memory import EpisodicEntry, load_memory

logger = logging.getLogger(__name__)

# ── Public retrieval API ───────────────────────────────────────────────────────


def retrieve_historical_forecasts(
    customer_id: str | None = None,
    horizon: str | None = None,
    passed_only: bool = False,
    limit: int = 10,
) -> list[EpisodicEntry]:
    """
    Retrieve historical forecast entries from the episodic memory store.

    Args:
        customer_id: Filter to this customer (or DEFAULT_CUSTOMER_ID if None).
        horizon:     Filter to a specific horizon value (e.g. "30d"). No filter if None.
        passed_only: If True, return only runs where all_passed=True.
        limit:       Maximum number of entries to return (newest first).

    Returns:
        List of EpisodicEntry objects, newest first, up to `limit`.
    """
    store = load_memory(customer_id)
    entries = store.episodic

    if horizon:
        entries = [e for e in entries if e.horizon == horizon]

    if passed_only:
        entries = [e for e in entries if e.all_passed]

    # Newest first
    entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)

    result = entries[:limit]
    logger.debug(
        "context.retrieval.fetched customer_id=%s horizon=%s passed_only=%s count=%d",
        customer_id,
        horizon,
        passed_only,
        len(result),
    )
    return result


def retrieve_best_strategy(
    customer_id: str | None = None,
    horizon: str | None = None,
) -> str | None:
    """
    Return the model variant that achieved the lowest MASE for the given horizon.

    Args:
        customer_id: Customer to look up (or DEFAULT_CUSTOMER_ID if None).
        horizon:     Forecast horizon (e.g. "30d"). If None, searches all horizons.

    Returns:
        Model variant string (e.g. "amazon/chronos-t5-small"), or None if no history.
    """
    store = load_memory(customer_id)
    facts = store.semantic

    if not facts.best_strategy_by_horizon:
        return None

    if horizon and horizon in facts.best_strategy_by_horizon:
        return facts.best_strategy_by_horizon[horizon]

    if horizon:
        return None  # no data for this horizon yet

    # No horizon filter — return best across all horizons
    best_h = min(
        facts.best_mase_by_horizon,
        key=lambda h: facts.best_mase_by_horizon[h],
        default=None,
    )
    return facts.best_strategy_by_horizon.get(best_h) if best_h else None


def format_retrieval_context(entries: list[EpisodicEntry]) -> str:
    """
    Format retrieved entries as a prompt-injectable string block.

    Priority ordering follows agent-context.md §2:
    retrieved content is "current task context" — stable tier, keep through compaction.
    """
    if not entries:
        return ""
    lines = ["<retrieved_forecasts>"]
    for e in entries:
        status = "PASSED" if e.all_passed else "FAILED"
        drift = " [DRIFT]" if e.drift_detected else ""
        lines.append(
            f"  {e.timestamp[:10]} | {e.strategy_variant} | {e.horizon} | "
            f"MASE={e.overall_mase:.3f} | SMAPE={e.overall_smape:.1f}% | {status}{drift}"
        )
    lines.append("</retrieved_forecasts>")
    return "\n".join(lines)


def _now_utc() -> str:
    return datetime.now(tz=UTC).isoformat()
