"""
LangGraph StateGraph for the segmentation agent.

Topology:
  START → profiler → embedder → clusterer → evaluator → labeler → (loop or END)

The evaluator sets converged=True when all quality thresholds pass.
Loop guard: max_cycles hard stop.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.agents.segment.nodes import (
    clusterer_node,
    embedder_node,
    evaluator_node,
    labeler_node,
    profiler_node,
)
from src.agents.segment.state import SegmentationState, SegmentationStrategy, SegmentResult

# ── Routing ───────────────────────────────────────────────────────────────────


def should_continue(state: SegmentationState) -> Literal["clusterer", "__end__"]:
    if state.get("error"):
        return END
    if state.get("converged", False):
        return END
    max_cycles = state.get("max_cycles", 3)
    if state.get("cycle", 0) >= max_cycles:
        return END
    return "clusterer"


# ── Graph factory ─────────────────────────────────────────────────────────────


def build_segmentation_graph() -> StateGraph:
    graph = StateGraph(SegmentationState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("embedder", embedder_node)
    graph.add_node("clusterer", clusterer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("labeler", labeler_node)

    graph.add_edge(START, "profiler")
    graph.add_edge("profiler", "embedder")
    graph.add_edge("embedder", "clusterer")
    graph.add_edge("clusterer", "evaluator")
    graph.add_edge("evaluator", "labeler")
    graph.add_conditional_edges("labeler", should_continue)

    return graph.compile()


# ── Runner ────────────────────────────────────────────────────────────────────


def run_segmentation_agent(
    customer_df_path: str,
    strategy: SegmentationStrategy | None = None,
    max_cycles: int = 3,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the segmentation agent loop.

    Args:
        customer_df_path: Path to a Parquet file with canonical CashFlowRecord rows.
        strategy:         Initial clustering strategy (defaults to HDBSCAN + profile embedding).
        max_cycles:       Maximum clustering iterations before forced termination.
        verbose:          Print a summary on completion.

    Returns:
        Final SegmentationState as a dict.
    """
    if strategy is None:
        strategy = SegmentationStrategy(
            algorithm="hdbscan",
            embedding="profile",
            n_clusters=None,
            umap_n_components=2,
            min_cluster_size=3,
        )

    graph = build_segmentation_graph()

    initial: dict[str, Any] = {
        "customer_df_ref": customer_df_path,
        "strategy": strategy,
        "result": None,
        "eval_report": None,
        "strategy_history": [],
        "eval_history": [],
        "cycle": 0,
        "max_cycles": max_cycles,
        "converged": False,
        "error": None,
        # Intermediate fields — will be populated by nodes
        "profile_vectors": {},
        "embedding_matrix": [],
        "embedding_customer_ids": [],
        "cluster_labels": [],
        "cluster_algorithm": "",
        "cluster_n": 0,
        "cluster_noise_fraction": 0.0,
        "cluster_metadata": {},
        "_cluster_result_labels": [],
        "_cluster_result_algo": "",
        "_cluster_result_n": 0,
        "_cluster_result_noise": 0.0,
    }

    final = graph.invoke(initial)

    if verbose:
        _print_summary(final)

    return final


def _print_summary(state: dict[str, Any]) -> None:
    try:
        from rich.console import Console

        console = Console()
        result: SegmentResult | None = state.get("result")
        report = state.get("eval_report")

        console.print(
            f"\n[bold green]✓ Segmentation agent completed — {state.get('cycle', 0)} cycle(s)[/bold green]"
        )

        if result:
            console.print(
                f"  Segments: {result['n_segments']} | Customers: {len(result['customer_ids'])}"
            )
            for cid, info in result["segment_names"].items():
                console.print(f"    [{cid}] {info['label']} — {info['description']}")

        if report:
            status = "[green]PASS[/green]" if report.all_passed else "[red]FAIL[/red]"
            console.print(
                f"\n  Quality {status}: sil={report.silhouette:.3f} "
                f"db={report.davies_bouldin:.3f} min_size={report.min_cluster_size}"
            )
    except ImportError:
        result = state.get("result")
        if result:
            print(
                f"\nSegmentation done: {result['n_segments']} segments, {len(result['customer_ids'])} customers"
            )
