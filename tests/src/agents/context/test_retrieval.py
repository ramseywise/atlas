"""
Tests for src/agents/context/retrieval.py
"""

from __future__ import annotations

import pytest

from src.agents.context.memory import (
    EpisodicEntry,
    ForecastMemoryStore,
    append_episodic,
    save_memory,
    update_semantic,
)
from src.agents.context.retrieval import (
    format_retrieval_context,
    retrieve_best_strategy,
    retrieve_historical_forecasts,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _entry(
    run_id: str,
    horizon: str = "30d",
    passed: bool = True,
    mase: float = 0.7,
    timestamp: str = "2026-01-01T00:00:00+00:00",
) -> EpisodicEntry:
    return EpisodicEntry(
        run_id=run_id,
        timestamp=timestamp,
        cycle_count=2,
        overall_mase=mase,
        overall_smape=12.0,
        drift_detected=False,
        strategy_variant="amazon/chronos-t5-tiny",
        horizon=horizon,
        all_passed=passed,
    )


@pytest.fixture()
def populated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
    store = ForecastMemoryStore(customer_id="default")
    append_episodic(store, _entry("r1", horizon="30d", passed=True, mase=0.7))
    append_episodic(store, _entry("r2", horizon="30d", passed=False, mase=0.95))
    append_episodic(store, _entry("r3", horizon="7d", passed=True, mase=0.65))
    update_semantic(store, "30d", "amazon/chronos-t5-tiny", mase=0.7, drift_detected=False)
    update_semantic(store, "7d", "amazon/chronos-t5-small", mase=0.65, drift_detected=False)
    save_memory(store)
    return store


# ── retrieve_historical_forecasts ─────────────────────────────────────────────


class TestRetrieveHistoricalForecasts:
    def test_returns_all_entries_no_filter(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        entries = retrieve_historical_forecasts()
        assert len(entries) == 3

    def test_horizon_filter(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        entries = retrieve_historical_forecasts(horizon="7d")
        assert len(entries) == 1
        assert entries[0].run_id == "r3"

    def test_passed_only_filter(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        entries = retrieve_historical_forecasts(passed_only=True)
        assert all(e.all_passed for e in entries)
        assert len(entries) == 2

    def test_limit_respected(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        entries = retrieve_historical_forecasts(limit=2)
        assert len(entries) <= 2

    def test_empty_store_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        entries = retrieve_historical_forecasts(customer_id="nobody")
        assert entries == []

    def test_newest_first_ordering(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        store = ForecastMemoryStore(customer_id="default")
        append_episodic(store, _entry("older", timestamp="2025-01-01T00:00:00+00:00"))
        append_episodic(store, _entry("newer", timestamp="2026-06-01T00:00:00+00:00"))
        save_memory(store)
        entries = retrieve_historical_forecasts()
        assert entries[0].run_id == "newer"


# ── retrieve_best_strategy ────────────────────────────────────────────────────


class TestRetrieveBestStrategy:
    def test_returns_best_for_horizon(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        best = retrieve_best_strategy(horizon="7d")
        assert best == "amazon/chronos-t5-small"

    def test_returns_none_for_unknown_horizon(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        best = retrieve_best_strategy(horizon="90d")
        assert best is None

    def test_returns_global_best_when_no_horizon(self, populated_store, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        best = retrieve_best_strategy()
        # 7d has lower MASE (0.65) → chronos-small wins
        assert best == "amazon/chronos-t5-small"

    def test_empty_store_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        best = retrieve_best_strategy(customer_id="nobody")
        assert best is None


# ── format_retrieval_context ──────────────────────────────────────────────────


class TestFormatRetrievalContext:
    def test_empty_list_returns_empty_string(self):
        assert format_retrieval_context([]) == ""

    def test_formats_entries(self):
        entries = [_entry("r1", mase=0.75, passed=True)]
        text = format_retrieval_context(entries)
        assert "<retrieved_forecasts>" in text
        assert "MASE=0.750" in text
        assert "PASSED" in text

    def test_marks_failed_entries(self):
        entries = [_entry("r2", passed=False)]
        text = format_retrieval_context(entries)
        assert "FAILED" in text
