"""
Segmentation agent nodes.

Five nodes, each receives SegmentationState and returns a partial state update:
  profiler_node   → builds CustomerProfile feature vectors from the input DataFrame
  embedder_node   → converts profiles to float32 embedding matrix
  clusterer_node  → runs clustering (HDBSCAN → KMeans fallback)
  evaluator_node  → scores with silhouette / DB / min-size
  labeler_node    → calls Haiku to generate human-readable segment names
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from src.agents.segment.state import (
    SegmentResult,
    SegmentationState,
    SegmentationStrategy,
)
from core.preprocessing.customer import CustomerProfile, build_customer_profiles
from core.preprocessing.embeddings import embed_tsfresh, embed_chronos
from core.segmentation.algorithms import (
    ClusterResult,
    fit_hdbscan,
    fit_kmeans,
    fit_gmm,
    fit_agglomerative,
    select_best,
)
from core.segmentation.evaluation import evaluate_clusters
from core.segmentation.naming import compute_centroids, name_segments


# ── Profiler ──────────────────────────────────────────────────────────────────


def profiler_node(state: SegmentationState) -> dict[str, Any]:
    """Load the customer DataFrame and build per-customer feature profiles."""
    df = pl.read_parquet(state["customer_df_ref"])
    profiles = build_customer_profiles(df)

    # Store as serialisable arrays keyed by customer_id
    profile_vectors = {
        cid: p.to_feature_vector().tolist()
        for cid, p in profiles.items()
    }
    return {"profile_vectors": profile_vectors}


# ── Embedder ──────────────────────────────────────────────────────────────────


def embedder_node(state: SegmentationState) -> dict[str, Any]:
    """
    Convert profiles to a float32 embedding matrix.
    Uses tsfresh features on profile vectors (no raw time-series needed here).
    Switch to chronos embedding by setting strategy.embedding = "chronos".
    """
    strategy: SegmentationStrategy = state["strategy"]
    profile_vectors: dict[str, list[float]] = state["profile_vectors"]

    # Profile vectors are already feature vectors — wrap as "time series" for tsfresh
    series_map = {cid: np.array(v, dtype=np.float32) for cid, v in profile_vectors.items()}

    if strategy["embedding"] == "chronos":
        matrix, customer_ids = embed_chronos(series_map)
    else:
        # "tsfresh" or "profile" — use profile vectors directly (already dense features)
        customer_ids = list(series_map.keys())
        matrix = np.vstack([series_map[cid] for cid in customer_ids]).astype(np.float32)

    # Normalise to zero mean / unit variance per feature
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0) + 1e-8
    matrix_norm = (matrix - mean) / std

    # Optional UMAP reduction — only applied when n_components < current feature dim
    # and the sample count is large enough (UMAP requires n_samples > n_components + 1)
    n_components = strategy.get("umap_n_components", 0)
    if n_components and n_components < matrix_norm.shape[1] and matrix_norm.shape[0] > n_components + 1:
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(
                n_components=n_components,
                n_neighbors=min(15, matrix_norm.shape[0] - 1),
                random_state=42,
                metric="cosine",
            )
            matrix_norm = reducer.fit_transform(matrix_norm).astype(np.float32)
        except ImportError:
            pass  # umap-learn not installed — skip reduction

    return {
        "embedding_matrix": matrix_norm.tolist(),
        "embedding_customer_ids": customer_ids,
    }


# ── Clusterer ─────────────────────────────────────────────────────────────────


def clusterer_node(state: SegmentationState) -> dict[str, Any]:
    """Run the clustering algorithm specified in strategy."""
    strategy: SegmentationStrategy = state["strategy"]
    X = np.array(state["embedding_matrix"], dtype=np.float32)
    customer_ids: list[str] = state["embedding_customer_ids"]

    algo = strategy["algorithm"]
    k = strategy.get("n_clusters")
    min_size = strategy.get("min_cluster_size", 3)

    if algo == "hdbscan":
        result = fit_hdbscan(X, min_cluster_size=min_size)
    elif algo == "kmeans":
        result = fit_kmeans(X, n_clusters=k or 5)
    elif algo == "gmm":
        result = fit_gmm(X, n_components=k or 5)
    elif algo == "agglomerative":
        result = fit_agglomerative(X, n_clusters=k or 5)
    else:
        result = select_best(X, min_cluster_size=min_size)

    return {
        "cluster_labels": result.labels.tolist(),
        "cluster_algorithm": result.algorithm,
        "cluster_n": result.n_clusters,
        "cluster_noise_fraction": result.noise_fraction,
        "cluster_metadata": result.metadata,
        # Stash result object fields for evaluator
        "_cluster_result_labels": result.labels.tolist(),
        "_cluster_result_algo": result.algorithm,
        "_cluster_result_n": result.n_clusters,
        "_cluster_result_noise": result.noise_fraction,
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────


def evaluator_node(state: SegmentationState) -> dict[str, Any]:
    """Score the current clustering with silhouette / DB / CH / min-size."""
    X = np.array(state["embedding_matrix"], dtype=np.float32)
    labels = np.array(state["_cluster_result_labels"], dtype=int)

    result = ClusterResult(
        algorithm=state["_cluster_result_algo"],
        labels=labels,
        n_clusters=state["_cluster_result_n"],
        noise_fraction=state["_cluster_result_noise"],
        metadata=state.get("cluster_metadata", {}),
    )

    report = evaluate_clusters(X, result, cycle=state.get("cycle", 0))

    # Strategy adaptation: if quality failed, adjust for next cycle
    next_strategy = dict(state["strategy"])
    if not report.all_passed:
        if result.algorithm == "hdbscan":
            # Switch to KMeans; if HDBSCAN found many clusters, try fewer
            next_strategy["algorithm"] = "kmeans"
            next_strategy["n_clusters"] = max(2, min(report.n_clusters, max(2, report.n_customers // 6)))
        elif result.algorithm == "kmeans" and report.n_clusters > 2:
            # KMeans also failed — try one fewer cluster
            next_strategy["n_clusters"] = report.n_clusters - 1

    return {
        "eval_report": report,
        "eval_history": [report],
        "strategy": next_strategy,
        "converged": report.all_passed,
    }


# ── Labeler ───────────────────────────────────────────────────────────────────


def labeler_node(state: SegmentationState) -> dict[str, Any]:
    """Call Haiku with cluster centroids → human-readable segment names."""
    X = np.array(state["embedding_matrix"], dtype=np.float32)
    labels = np.array(state["_cluster_result_labels"], dtype=int)
    customer_ids: list[str] = state["embedding_customer_ids"]

    centroids = compute_centroids(X, labels)
    segment_labels = name_segments(centroids, feature_names=CustomerProfile.feature_names())

    segment_names = {
        cid: {"label": sl.label, "description": sl.description}
        for cid, sl in segment_labels.items()
    }

    # If everything was noise (HDBSCAN all-noise edge case), treat as one segment
    if not segment_names:
        segment_names = {-1: {"label": "Unclustered", "description": "All customers in noise cluster — try KMeans."}}

    result = SegmentResult(
        customer_ids=customer_ids,
        labels=labels.tolist(),
        segment_names=segment_names,
        n_segments=max(1, len(segment_names)),
    )

    return {
        "result": result,
        "cycle": state.get("cycle", 0) + 1,
    }
