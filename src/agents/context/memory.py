"""
Persistent memory store for the forecast agent.

Implements a file-backed JSON store keyed by customer_id (or "default").
Tier mapping (agent-memory.md §1):
  - Episodic  — past run summaries, appended per run, pruned to EPISODIC_MAX_ENTRIES
  - Semantic  — aggregated facts (best strategy per horizon, drift counts)
  - Procedural — stored in the system prompt (not this file)

Lifetime: cross-session (survives process restarts).
Single-writer: only the context_save_node writes; all other nodes read.

Storage path: DATA_DIR / "agent_memory" / "<customer_id>.json"
Override via ATLAS_MEMORY_DIR env var.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

EPISODIC_MAX_ENTRIES = 20  # prune oldest when exceeded
DEFAULT_CUSTOMER_ID = "default"

# ── Memory path resolution ─────────────────────────────────────────────────────


def _memory_dir() -> Path:
    """Resolve the memory store directory. Respects ATLAS_MEMORY_DIR env var."""
    env_override = os.environ.get("ATLAS_MEMORY_DIR")
    if env_override:
        return Path(env_override)
    # Default: <repo-root>/data/agent_memory
    repo_root = Path(__file__).parents[4]
    return repo_root / "data" / "agent_memory"


# ── Sub-models ─────────────────────────────────────────────────────────────────


class EpisodicEntry(BaseModel):
    """One completed forecast run — append-only tier."""

    run_id: str
    timestamp: str  # ISO 8601 UTC
    cycle_count: int
    overall_mase: float
    overall_smape: float
    drift_detected: bool
    strategy_variant: str  # ModelVariant.value
    horizon: str  # ForecastHorizon.value
    all_passed: bool
    reflection: str = ""


class SemanticFacts(BaseModel):
    """Aggregated cross-run knowledge — update-aware tier."""

    best_strategy_by_horizon: dict[str, str] = Field(
        default_factory=dict,
        description="horizon -> model_variant that achieved lowest MASE",
    )
    best_mase_by_horizon: dict[str, float] = Field(
        default_factory=dict,
        description="horizon -> best MASE seen",
    )
    total_drift_events: int = 0
    total_runs: int = 0
    last_updated: str = ""  # ISO 8601 UTC


class ForecastMemoryStore(BaseModel):
    """Top-level memory object persisted per customer."""

    customer_id: str
    episodic: list[EpisodicEntry] = Field(default_factory=list)
    semantic: SemanticFacts = Field(default_factory=SemanticFacts)


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _store_path(customer_id: str) -> Path:
    return _memory_dir() / f"{customer_id}.json"


def load_memory(customer_id: str | None = None) -> ForecastMemoryStore:
    """Load memory for a customer. Returns empty store if none exists yet."""
    cid = customer_id or DEFAULT_CUSTOMER_ID
    path = _store_path(cid)
    if not path.exists():
        logger.debug("context.memory.init customer_id=%s", cid)
        return ForecastMemoryStore(customer_id=cid)
    try:
        data = json.loads(path.read_text())
        store = ForecastMemoryStore.model_validate(data)
        logger.debug(
            "context.memory.loaded customer_id=%s episodes=%d",
            cid,
            len(store.episodic),
        )
        return store
    except Exception as exc:
        logger.warning("context.memory.load_failed customer_id=%s err=%s", cid, exc)
        return ForecastMemoryStore(customer_id=cid)


def save_memory(store: ForecastMemoryStore) -> None:
    """Persist the memory store. Creates directory if needed."""
    path = _store_path(store.customer_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(store.model_dump_json(indent=2))
    logger.debug(
        "context.memory.saved customer_id=%s episodes=%d",
        store.customer_id,
        len(store.episodic),
    )


# ── Update helpers ─────────────────────────────────────────────────────────────


def append_episodic(store: ForecastMemoryStore, entry: EpisodicEntry) -> None:
    """Append a run entry; prune oldest when over EPISODIC_MAX_ENTRIES."""
    store.episodic.append(entry)
    if len(store.episodic) > EPISODIC_MAX_ENTRIES:
        store.episodic = store.episodic[-EPISODIC_MAX_ENTRIES:]


def update_semantic(
    store: ForecastMemoryStore,
    horizon: str,
    model_variant: str,
    mase: float,
    drift_detected: bool,
) -> None:
    """Update semantic facts after a completed run."""
    facts = store.semantic
    current_best = facts.best_mase_by_horizon.get(horizon, float("inf"))
    if mase < current_best:
        facts.best_mase_by_horizon[horizon] = mase
        facts.best_strategy_by_horizon[horizon] = model_variant
    if drift_detected:
        facts.total_drift_events += 1
    facts.total_runs += 1
    facts.last_updated = datetime.now(tz=UTC).isoformat()


# ── Context formatting ─────────────────────────────────────────────────────────


def format_memory_context(store: ForecastMemoryStore, max_episodes: int = 3) -> dict[str, Any]:
    """
    Format memory into a context block for injection into the agent state.

    Returns a dict with:
      - "episodic_summary": recent run summaries as text
      - "semantic_summary": aggregated facts as text
      - "best_strategy_hint": model variant that worked best for the most common horizon
    """
    # Episodic: most recent N entries, newest first
    recent = store.episodic[-max_episodes:][::-1]
    if recent:
        lines = ["<past_runs>"]
        for e in recent:
            status = "PASSED" if e.all_passed else "FAILED"
            drift = " [DRIFT]" if e.drift_detected else ""
            lines.append(
                f"  {e.timestamp[:10]} | {e.strategy_variant} | {e.horizon} | "
                f"MASE={e.overall_mase:.3f} | {status}{drift}"
            )
            if e.reflection:
                lines.append(f"    note: {e.reflection[:120]}")
        lines.append("</past_runs>")
        episodic_summary = "\n".join(lines)
    else:
        episodic_summary = ""

    # Semantic: known-good strategies per horizon
    facts = store.semantic
    if facts.best_strategy_by_horizon:
        sem_lines = ["<known_best_strategies>"]
        for h, variant in facts.best_strategy_by_horizon.items():
            best_mase = facts.best_mase_by_horizon.get(h, float("nan"))
            sem_lines.append(f"  {h}: {variant} (best MASE={best_mase:.3f})")
        sem_lines.append(
            f"  total_runs={facts.total_runs}  drift_events={facts.total_drift_events}"
        )
        sem_lines.append("</known_best_strategies>")
        semantic_summary = "\n".join(sem_lines)
    else:
        semantic_summary = ""

    # Best strategy hint: pick the horizon with lowest absolute MASE seen
    best_hint: str | None = None
    if facts.best_strategy_by_horizon:
        best_h = min(facts.best_mase_by_horizon, key=lambda h: facts.best_mase_by_horizon[h])
        best_hint = facts.best_strategy_by_horizon.get(best_h)

    return {
        "episodic_summary": episodic_summary,
        "semantic_summary": semantic_summary,
        "best_strategy_hint": best_hint,
        "total_runs": facts.total_runs,
    }
