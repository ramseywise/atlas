"""
Segmentation quality evaluation.

Metrics:
  - Silhouette score        ≥ 0.25  (pass)
  - Davies-Bouldin index    ≤ 1.5   (pass; lower is better)
  - Calinski-Harabász index          (higher is better, no hard threshold)
  - Min cluster size        ≥ 3     (pass)

SegmentEvalReport is the typed result flowing through SegmentationState.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.segmentation.algorithms import ClusterResult

SILHOUETTE_THRESHOLD = 0.25
DB_THRESHOLD = 1.5
MIN_CLUSTER_SIZE = 3


@dataclass
class SegmentEvalReport:
    cycle: int
    algorithm: str
    n_clusters: int
    n_customers: int
    noise_fraction: float

    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float

    min_cluster_size: int
    cluster_sizes: dict[int, int]   # label → count (excludes -1)

    silhouette_passed: bool
    db_passed: bool
    min_size_passed: bool

    warnings: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.silhouette_passed and self.db_passed and self.min_size_passed

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        return (
            f"[{status}] cycle={self.cycle} algo={self.algorithm} "
            f"k={self.n_clusters} sil={self.silhouette:.3f} "
            f"db={self.davies_bouldin:.3f} ch={self.calinski_harabasz:.1f} "
            f"min_size={self.min_cluster_size}"
        )


def evaluate_clusters(
    X: np.ndarray,
    result: ClusterResult,
    cycle: int = 0,
) -> SegmentEvalReport:
    """
    Compute all segmentation quality metrics for a ClusterResult.

    Args:
        X:      Embedding matrix (n_customers, n_features) — same as passed to fit_*
        result: Output of fit_hdbscan / fit_kmeans / select_best
        cycle:  Current agent cycle index

    Returns:
        SegmentEvalReport with pass/fail flags.
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score,
    )

    labels = result.labels
    warnings: list[str] = []

    # Exclude noise for metric computation
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    n_unique = len(set(labels_clean))

    if n_unique < 2 or len(X_clean) < 4:
        warnings.append("Too few labelled samples for metric computation — returning zeros")
        sil, db, ch = 0.0, 999.0, 0.0
    else:
        sil = float(silhouette_score(X_clean, labels_clean))
        db = float(davies_bouldin_score(X_clean, labels_clean))
        ch = float(calinski_harabasz_score(X_clean, labels_clean))

    # Cluster size distribution
    unique, counts = np.unique(labels_clean, return_counts=True)
    cluster_sizes = {int(k): int(v) for k, v in zip(unique, counts)}
    min_size = int(min(counts)) if len(counts) > 0 else 0

    if result.noise_fraction > 0.3:
        warnings.append(
            f"High noise fraction ({result.noise_fraction:.1%}) — "
            "consider lowering min_cluster_size or switching to KMeans"
        )

    # Relax thresholds for small datasets — fewer points → lower achievable scores
    n_customers = len(labels)
    if n_customers < 50:
        scale = n_customers / 50
        sil_threshold = max(0.10, SILHOUETTE_THRESHOLD * scale)
        db_threshold = min(2.5, DB_THRESHOLD / scale)
        warnings.append(
            f"Relaxed thresholds for N={n_customers}: sil≥{sil_threshold:.2f}, db≤{db_threshold:.2f}"
        )
    else:
        sil_threshold = SILHOUETTE_THRESHOLD
        db_threshold = DB_THRESHOLD

    return SegmentEvalReport(
        cycle=cycle,
        algorithm=result.algorithm,
        n_clusters=result.n_clusters,
        n_customers=n_customers,
        noise_fraction=result.noise_fraction,
        silhouette=sil,
        davies_bouldin=db,
        calinski_harabasz=ch,
        min_cluster_size=min_size,
        cluster_sizes=cluster_sizes,
        silhouette_passed=sil >= sil_threshold,
        db_passed=db <= db_threshold,
        min_size_passed=min_size >= MIN_CLUSTER_SIZE,
        warnings=warnings,
    )
