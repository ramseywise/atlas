"""
Clustering algorithms for customer segmentation.

All functions take a float32 embedding matrix and return (labels, metadata).
labels is an int array of shape (n_customers,); -1 = noise (HDBSCAN only).

Algorithm selection:
  - HDBSCAN first: handles variable-density B2B clusters, no k required
  - KMeans fallback: if silhouette < 0.25 or HDBSCAN produces >50% noise
  - GMM / agglomerative: available for comparison / ablation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    algorithm: str
    labels: np.ndarray  # shape (n_samples,); -1 = noise
    n_clusters: int  # excludes noise cluster
    noise_fraction: float  # fraction labelled -1
    metadata: dict  # algorithm-specific diagnostics


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> ClusterResult:
    try:
        import hdbscan

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X)
        n_noise = int(np.sum(labels == -1))
        n_clusters = len(set(labels) - {-1})
        return ClusterResult(
            algorithm="hdbscan",
            labels=labels,
            n_clusters=n_clusters,
            noise_fraction=n_noise / len(labels),
            metadata={
                "min_cluster_size": min_cluster_size,
                "min_samples": clusterer.min_samples,
                "probabilities": clusterer.probabilities_.tolist(),
            },
        )
    except ImportError:
        # fall back to KMeans if hdbscan not installed
        k = max(2, len(X) // max(min_cluster_size, 1))
        return fit_kmeans(X, n_clusters=min(k, 8))


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 5,
    n_init: int = 10,
    random_state: int = 42,
) -> ClusterResult:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X)
    return ClusterResult(
        algorithm="kmeans",
        labels=labels,
        n_clusters=n_clusters,
        noise_fraction=0.0,
        metadata={
            "inertia": float(km.inertia_),
            "n_iter": int(km.n_iter_),
        },
    )


def fit_gmm(
    X: np.ndarray,
    n_components: int = 5,
    covariance_type: str = "full",
    random_state: int = 42,
) -> ClusterResult:
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = gmm.fit_predict(X)
    return ClusterResult(
        algorithm="gmm",
        labels=labels,
        n_clusters=n_components,
        noise_fraction=0.0,
        metadata={
            "bic": float(gmm.bic(X)),
            "aic": float(gmm.aic(X)),
            "converged": bool(gmm.converged_),
        },
    )


def fit_agglomerative(
    X: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
) -> ClusterResult:
    from sklearn.cluster import AgglomerativeClustering

    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = agg.fit_predict(X)
    return ClusterResult(
        algorithm="agglomerative",
        labels=labels,
        n_clusters=n_clusters,
        noise_fraction=0.0,
        metadata={"linkage": linkage},
    )


def select_best(
    X: np.ndarray,
    min_cluster_size: int = 3,
    silhouette_threshold: float = 0.25,
    noise_threshold: float = 0.5,
    k_range: tuple[int, int] = (2, 8),
) -> ClusterResult:
    """
    Try HDBSCAN first. Fall back to KMeans if silhouette < threshold or
    noise fraction > threshold. KMeans k is selected by silhouette scan.
    """
    from sklearn.metrics import silhouette_score

    result = fit_hdbscan(X, min_cluster_size=min_cluster_size)

    # Filter noise for silhouette scoring
    mask = result.labels != -1
    if mask.sum() >= 2 and result.n_clusters >= 2:
        sil = silhouette_score(X[mask], result.labels[mask])
    else:
        sil = -1.0

    if sil >= silhouette_threshold and result.noise_fraction <= noise_threshold:
        return result

    # KMeans sweep
    best_sil, best_km = -1.0, None
    for k in range(k_range[0], k_range[1] + 1):
        if k >= len(X):
            break
        km = fit_kmeans(X, n_clusters=k)
        if len(set(km.labels)) < 2:
            continue
        s = silhouette_score(X, km.labels)
        if s > best_sil:
            best_sil, best_km = s, km

    return best_km if best_km is not None else fit_kmeans(X, n_clusters=min(3, len(X)))
