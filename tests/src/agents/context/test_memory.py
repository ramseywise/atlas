"""
Tests for src/agents/context/memory.py

All tests use a tmp_path-scoped memory dir to avoid touching the real data/agent_memory/.
"""

from __future__ import annotations

from src.agents.context.memory import (
    EPISODIC_MAX_ENTRIES,
    EpisodicEntry,
    ForecastMemoryStore,
    append_episodic,
    format_memory_context,
    load_memory,
    save_memory,
    update_semantic,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _entry(run_id: str = "abc123", mase: float = 0.7, passed: bool = True) -> EpisodicEntry:
    return EpisodicEntry(
        run_id=run_id,
        timestamp="2026-01-01T00:00:00+00:00",
        cycle_count=2,
        overall_mase=mase,
        overall_smape=12.5,
        drift_detected=False,
        strategy_variant="amazon/chronos-t5-tiny",
        horizon="30d",
        all_passed=passed,
    )


# ── ForecastMemoryStore construction ──────────────────────────────────────────


class TestForecastMemoryStore:
    def test_default_empty(self):
        store = ForecastMemoryStore(customer_id="x")
        assert store.episodic == []
        assert store.semantic.total_runs == 0

    def test_pydantic_validation(self):
        store = ForecastMemoryStore(customer_id="x", episodic=[_entry()])
        assert len(store.episodic) == 1


# ── append_episodic ───────────────────────────────────────────────────────────


class TestAppendEpisodic:
    def test_appends(self):
        store = ForecastMemoryStore(customer_id="x")
        append_episodic(store, _entry("r1"))
        assert len(store.episodic) == 1

    def test_prunes_when_over_max(self):
        store = ForecastMemoryStore(customer_id="x")
        for i in range(EPISODIC_MAX_ENTRIES + 5):
            append_episodic(store, _entry(f"r{i}"))
        assert len(store.episodic) == EPISODIC_MAX_ENTRIES
        # Most recent must survive
        assert store.episodic[-1].run_id == f"r{EPISODIC_MAX_ENTRIES + 4}"


# ── update_semantic ───────────────────────────────────────────────────────────


class TestUpdateSemantic:
    def test_records_first_run(self):
        store = ForecastMemoryStore(customer_id="x")
        update_semantic(store, "30d", "amazon/chronos-t5-tiny", mase=0.8, drift_detected=False)
        assert store.semantic.best_mase_by_horizon["30d"] == 0.8
        assert store.semantic.best_strategy_by_horizon["30d"] == "amazon/chronos-t5-tiny"
        assert store.semantic.total_runs == 1

    def test_updates_on_improvement(self):
        store = ForecastMemoryStore(customer_id="x")
        update_semantic(store, "30d", "amazon/chronos-t5-tiny", mase=0.8, drift_detected=False)
        update_semantic(store, "30d", "amazon/chronos-t5-small", mase=0.6, drift_detected=False)
        assert store.semantic.best_mase_by_horizon["30d"] == 0.6
        assert store.semantic.best_strategy_by_horizon["30d"] == "amazon/chronos-t5-small"

    def test_does_not_update_on_regression(self):
        store = ForecastMemoryStore(customer_id="x")
        update_semantic(store, "30d", "amazon/chronos-t5-small", mase=0.6, drift_detected=False)
        update_semantic(store, "30d", "amazon/chronos-t5-tiny", mase=0.9, drift_detected=False)
        # Small stays — tiny is worse
        assert store.semantic.best_strategy_by_horizon["30d"] == "amazon/chronos-t5-small"

    def test_drift_counter(self):
        store = ForecastMemoryStore(customer_id="x")
        update_semantic(store, "30d", "x", mase=0.8, drift_detected=True)
        update_semantic(store, "30d", "x", mase=0.8, drift_detected=False)
        assert store.semantic.total_drift_events == 1


# ── load_memory / save_memory ─────────────────────────────────────────────────


class TestPersistence:
    def test_save_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        store = ForecastMemoryStore(customer_id="cust-1")
        append_episodic(store, _entry("r1", mase=0.75))
        update_semantic(store, "30d", "amazon/chronos-t5-tiny", mase=0.75, drift_detected=False)
        save_memory(store)

        loaded = load_memory("cust-1")
        assert len(loaded.episodic) == 1
        assert loaded.episodic[0].run_id == "r1"
        assert loaded.semantic.best_mase_by_horizon["30d"] == 0.75

    def test_load_nonexistent_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        store = load_memory("nobody")
        assert store.episodic == []

    def test_load_corrupted_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        bad_file = tmp_path / "corrupted.json"
        bad_file.write_text("not json {{{{")
        store = load_memory("corrupted")
        assert store.episodic == []

    def test_file_path_uses_customer_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        store = ForecastMemoryStore(customer_id="cust-99")
        save_memory(store)
        assert (tmp_path / "cust-99.json").exists()


# ── format_memory_context ─────────────────────────────────────────────────────


class TestFormatMemoryContext:
    def test_empty_store_returns_empty_strings(self):
        store = ForecastMemoryStore(customer_id="x")
        ctx = format_memory_context(store)
        assert ctx["episodic_summary"] == ""
        assert ctx["semantic_summary"] == ""
        assert ctx["best_strategy_hint"] is None

    def test_with_episodes(self):
        store = ForecastMemoryStore(customer_id="x")
        append_episodic(store, _entry("r1", mase=0.7, passed=True))
        ctx = format_memory_context(store)
        assert "<past_runs>" in ctx["episodic_summary"]
        assert "MASE=0.700" in ctx["episodic_summary"]

    def test_best_strategy_hint_populated(self):
        store = ForecastMemoryStore(customer_id="x")
        update_semantic(store, "30d", "amazon/chronos-t5-small", mase=0.6, drift_detected=False)
        ctx = format_memory_context(store)
        assert ctx["best_strategy_hint"] == "amazon/chronos-t5-small"

    def test_max_episodes_respected(self):
        store = ForecastMemoryStore(customer_id="x")
        for i in range(10):
            append_episodic(store, _entry(f"r{i}"))
        ctx = format_memory_context(store, max_episodes=3)
        # Count occurrences of MASE= in output (one per entry)
        assert ctx["episodic_summary"].count("MASE=") == 3
