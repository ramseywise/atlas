"""SegmentationState — typed state for the segmentation agent loop."""
from __future__ import annotations

from typing import Annotated
import operator

from typing_extensions import TypedDict

from core.segmentation.evaluation import SegmentEvalReport


class SegmentationStrategy(TypedDict):
    algorithm: str        # "hdbscan" | "kmeans" | "gmm" | "agglomerative"
    embedding: str        # "tsfresh" | "chronos" | "profile"
    n_clusters: int | None  # None = auto (HDBSCAN)
    umap_n_components: int
    min_cluster_size: int


class SegmentResult(TypedDict):
    customer_ids: list[str]
    labels: list[int]
    segment_names: dict[int, dict[str, str]]  # {cluster_id: {label, description}}
    n_segments: int


class SegmentationState(TypedDict):
    customer_df_ref: str          # path or key to the input DataFrame
    strategy: SegmentationStrategy
    result: SegmentResult | None
    eval_report: SegmentEvalReport | None
    strategy_history: Annotated[list[SegmentationStrategy], operator.add]
    eval_history: Annotated[list[SegmentEvalReport], operator.add]
    cycle: int
    max_cycles: int
    converged: bool
    error: str | None

    # Intermediate node outputs (not persisted across cycles)
    profile_vectors: dict               # {customer_id: list[float]}
    embedding_matrix: list              # list of lists (n_customers × n_features)
    embedding_customer_ids: list[str]
    cluster_labels: list[int]
    cluster_algorithm: str
    cluster_n: int
    cluster_noise_fraction: float
    cluster_metadata: dict
    _cluster_result_labels: list[int]   # raw labels for evaluator
    _cluster_result_algo: str
    _cluster_result_n: int
    _cluster_result_noise: float
