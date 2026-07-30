"""
Integration smoke test: context_load_node and context_save_node in the full graph.

Verifies that:
  1. The agent runs end-to-end with memory enabled
  2. After a run, an episodic entry is written to disk
  3. A second run loads the memory and reflects it in memory_context
  4. No context window exceeds 80% of ATLAS_CONTEXT_LIMIT chars (proxy check)
"""

from __future__ import annotations

import pytest

from core.preprocessing.synthetic import generate_sequence_dataset

ATLAS_CONTEXT_LIMIT = 200_000  # chars — rough proxy for token budget


@pytest.fixture(scope="module")
def small_df():
    return generate_sequence_dataset(n_days=365, seed=42)


class TestContextIntegration:
    def test_run_with_memory_creates_episodic_entry(self, small_df, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        from src.agents.graph import run_forecasting_agent

        run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="smoke-test",
            verbose=False,
        )

        memory_file = tmp_path / "smoke-test.json"
        assert memory_file.exists(), "memory file not written after run"

        import json

        data = json.loads(memory_file.read_text())
        assert len(data["episodic"]) == 1
        assert data["semantic"]["total_runs"] == 1

    def test_second_run_loads_memory_context(self, small_df, tmp_path, monkeypatch):
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        from src.agents.graph import run_forecasting_agent

        # First run — seeds memory
        run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="ctx-test",
            verbose=False,
        )

        # Second run — should load memory and reflect it in memory_context
        final = run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="ctx-test",
            verbose=False,
        )

        ctx = final.get("memory_context")
        assert ctx is not None, "memory_context not in final state"
        assert ctx.get("total_runs", 0) >= 1, "memory_context should reflect at least 1 prior run"

    def test_memory_context_size_within_budget(self, small_df, tmp_path, monkeypatch):
        """Proxy check: formatted context strings don't bloat the window."""
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        from src.agents.graph import run_forecasting_agent

        # Seed a run first
        run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="budget-test",
            verbose=False,
        )

        final = run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="budget-test",
            verbose=False,
        )

        ctx = final.get("memory_context") or {}
        total_chars = sum(len(str(v)) for v in ctx.values() if isinstance(v, str))
        limit_80pct = int(ATLAS_CONTEXT_LIMIT * 0.8)
        assert total_chars < limit_80pct, (
            f"memory_context ({total_chars} chars) exceeds 80% of context limit ({limit_80pct})"
        )

    def test_state_fields_present(self, small_df, tmp_path, monkeypatch):
        """New state fields exist and are correctly initialized."""
        monkeypatch.setenv("ATLAS_MEMORY_DIR", str(tmp_path))
        from src.agents.graph import run_forecasting_agent

        final = run_forecasting_agent(
            series_df=small_df,
            max_cycles=1,
            customer_id="fields-test",
            verbose=False,
        )

        # memory_context is set by context_load_node
        assert "memory_context" in final
        # customer_id passes through
        assert final.get("customer_id") == "fields-test"
