"""
Smoke tests for the segmentation agent loop.

No GPU or tsfresh required — profile embedding (manual feature extraction)
is used automatically. Tests verify node wiring, state transitions, and
quality metric plumbing without external dependencies.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest

from core.preprocessing.synthetic import generate_sequence_dataset


def _make_customer_parquet(n_customers: int = 8, n_days: int = 200, seed: int = 0) -> str:
    """Write a multi-customer Parquet file to a temp path and return the path."""
    frames = []
    for i in range(n_customers):
        df = generate_sequence_dataset(n_days=n_days, seed=seed + i)
        frames.append(
            df.with_columns(
                [
                    pl.lit(f"cust-{i:03d}").alias("customer_id"),
                    pl.col("source").alias("source"),
                    pl.col("sign").alias("sign"),
                    pl.col("value").alias("amount"),
                ]
            )
        )
    combined = pl.concat(frames)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        combined.write_parquet(tmp.name)
    return tmp.name


@pytest.fixture(scope="module")
def customer_parquet():
    path = _make_customer_parquet()
    yield path
    Path(path).unlink(missing_ok=True)


class TestSegmentationAgentSmoke:
    def test_single_cycle_completes(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        assert final.get("error") is None
        assert final.get("result") is not None

    def test_result_has_all_fields(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        result = final["result"]
        assert "customer_ids" in result
        assert "labels" in result
        assert "segment_names" in result
        assert "n_segments" in result
        assert result["n_segments"] >= 1

    def test_eval_report_populated(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        report = final["eval_report"]
        assert report is not None
        assert hasattr(report, "silhouette")
        assert hasattr(report, "all_passed")
        assert report.silhouette >= 0.0

    def test_customer_ids_match_input(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        result = final["result"]
        expected_ids = {f"cust-{i:03d}" for i in range(8)}
        assert set(result["customer_ids"]) == expected_ids

    def test_labels_align_with_customer_ids(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        result = final["result"]
        assert len(result["labels"]) == len(result["customer_ids"])

    def test_max_cycles_respected(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=2, verbose=False)
        assert final["cycle"] <= 2

    def test_eval_history_accumulates(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=2, verbose=False)
        assert len(final["eval_history"]) >= 1

    def test_segment_names_are_strings(self, customer_parquet):
        from src.agents.segment.graph import run_segmentation_agent

        final = run_segmentation_agent(customer_parquet, max_cycles=1, verbose=False)
        for _cid, info in final["result"]["segment_names"].items():
            assert isinstance(info["label"], str)
            assert len(info["label"]) > 0
