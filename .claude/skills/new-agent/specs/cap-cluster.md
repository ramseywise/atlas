# cap-cluster — Templates for cluster-builder subagent

## Design notes

### All-noise HDBSCAN fallback
On small datasets (< ~20 samples) or when series are too similar, HDBSCAN will
label every point as noise (-1), producing zero non-noise clusters. The labeler
node must handle this explicitly — an empty `segment_names` dict will cause
downstream code to break silently:

```python
# In labeler_node, after building segment_names:
if not segment_names:
    segment_names = {-1: {"label": "Unclustered", "description": "All customers in noise cluster — try KMeans."}}
result = SegmentResult(..., n_segments=max(1, len(segment_names)))
```

The evaluator should also detect this and set `next_strategy["algorithm"] = "kmeans"`
so the next cycle switches to KMeans automatically.

### Strategy fields must match what nodes actually read
If `SegmentationStrategy` declares `umap_n_components: int` but no node reads
it, the field is dead state that will confuse contributors. Either wire UMAP
reduction into `embedder_node` (after normalisation, before clustering), or
remove the field. Do not declare strategy fields speculatively.

### UMAP reduction (when `umap_n_components` is declared)
If you include UMAP in the strategy, apply it in the embedder node after
normalisation and before returning the matrix:

```python
if strategy.get("umap_n_components") and matrix_norm.shape[0] > strategy["umap_n_components"] + 1:
    from core.preprocessing.embeddings import reduce_dimensions
    matrix_norm = reduce_dimensions(matrix_norm, n_components=strategy["umap_n_components"])
```

Always guard with a minimum sample count — UMAP requires n_samples > n_components + 1.

### Deduplicating helper functions
`_autocorr` / `_autocorr_scalar` is commonly needed in both `customer.py` and
`embeddings.py`. Extract shared math helpers to `core/preprocessing/_math.py`
rather than duplicating.

---

## File: {OUTPUT_DIR}/embeddings.py

```python
from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_EMBED_MODEL = "amazon.titan-embed-text-v2:0"
_DEFAULT_EMBED_DIM = 1024  # Titan v2 output dimension


# ---------------------------------------------------------------------------
# Bedrock Titan Embeddings
# ---------------------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model_id: str = _DEFAULT_EMBED_MODEL,
    region: str | None = None,
    batch_size: int = 25,
) -> np.ndarray:
    """Embed a list of texts via Amazon Titan Embeddings (Bedrock).

    Args:
        texts: Input strings to embed.
        model_id: Bedrock model ID.
        region: AWS region. Falls back to AWS_REGION env var.
        batch_size: Texts are embedded one-by-one (Titan doesn't support batching),
            but this param controls progress logging frequency.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, _DEFAULT_EMBED_DIM))

    region = region or os.environ.get("AWS_REGION", "eu-west-1")
    client = boto3.client("bedrock-runtime", region_name=region)

    embeddings: list[list[float]] = []

    for i, text in enumerate(texts):
        if i % batch_size == 0:
            logger.info("Embedding texts %d–%d of %d", i, min(i + batch_size, len(texts)), len(texts))

        body = json.dumps({"inputText": text[:8192]})  # Titan max input length
        try:
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed for text index %d: %s", i, exc)
            # Append a zero vector to maintain alignment
            embeddings.append([0.0] * _DEFAULT_EMBED_DIM)

    arr = np.array(embeddings, dtype=np.float32)

    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ---------------------------------------------------------------------------
# UMAP dimensionality reduction
# ---------------------------------------------------------------------------


def reduce_dimensions(
    embeddings: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D (or n_components) via UMAP.

    Args:
        embeddings: Input array of shape (n_samples, embedding_dim).
        n_components: Target number of dimensions.
        n_neighbors: UMAP n_neighbors parameter (local structure).
        min_dist: UMAP min_dist (cluster density).
        random_state: Reproducibility seed.

    Returns:
        Reduced array of shape (n_samples, n_components).
    """
    try:
        import umap  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for dimensionality reduction. Run: pip install umap-learn"
        ) from exc

    if embeddings.shape[0] == 0:
        return np.empty((0, n_components))

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(n_neighbors, embeddings.shape[0] - 1),
        min_dist=min_dist,
        random_state=random_state,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)
```

## File: {OUTPUT_DIR}/cluster.py

```python
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    labels: np.ndarray          # Shape (n_samples,) — -1 means noise (HDBSCAN)
    centroids: np.ndarray       # Shape (n_clusters, embedding_dim)
    n_clusters: int
    silhouette_score: float
    noise_fraction: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "hdbscan",
    n_clusters: int | None = None,
    min_cluster_size: int = 5,
    random_state: int = 42,
) -> ClusterResult:
    """Cluster embeddings using HDBSCAN or KMeans.

    Args:
        embeddings: Array of shape (n_samples, dim).
        method: 'hdbscan' (density-based, discovers k) or 'kmeans' (requires n_clusters).
        n_clusters: Required for KMeans. Ignored for HDBSCAN.
        min_cluster_size: HDBSCAN min_cluster_size (ignored for KMeans).
        random_state: Seed for reproducibility.

    Returns:
        ClusterResult with labels, centroids, n_clusters, silhouette_score.
    """
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    if embeddings.shape[0] < 2:
        raise ValueError("Need at least 2 samples to cluster")

    if method == "hdbscan":
        labels = _hdbscan_cluster(embeddings, min_cluster_size=min_cluster_size)
    elif method == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters is required for KMeans clustering")
        labels = _kmeans_cluster(embeddings, n_clusters=n_clusters, random_state=random_state)
    else:
        raise ValueError(f"Unknown clustering method: {method!r}. Choose 'hdbscan' or 'kmeans'.")

    # Compute centroids per cluster (exclude noise label -1)
    unique_labels = sorted(set(labels) - {-1})
    n_clusters_found = len(unique_labels)

    if n_clusters_found == 0:
        logger.warning("All points labelled as noise (HDBSCAN). Try reducing min_cluster_size.")
        centroids = np.empty((0, embeddings.shape[1]))
        sil_score = 0.0
    else:
        centroids = np.stack(
            [embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels]
        )
        # Silhouette score requires ≥2 clusters and no single-cluster case
        mask = labels != -1
        if mask.sum() >= 2 and n_clusters_found >= 2:
            sil_score = float(silhouette_score(embeddings[mask], labels[mask], metric="cosine"))
        else:
            sil_score = 0.0

    noise_fraction = float((labels == -1).mean()) if -1 in labels else 0.0

    logger.info(
        "Clustering complete — method=%s  n_clusters=%d  silhouette=%.3f  noise=%.1f%%",
        method, n_clusters_found, sil_score, noise_fraction * 100,
    )

    return ClusterResult(
        labels=labels,
        centroids=centroids,
        n_clusters=n_clusters_found,
        silhouette_score=sil_score,
        noise_fraction=noise_fraction,
    )


def _hdbscan_cluster(embeddings: np.ndarray, min_cluster_size: int) -> np.ndarray:
    try:
        import hdbscan as hdbscan_lib  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("hdbscan is required. Run: pip install hdbscan") from exc

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    return clusterer.fit_predict(embeddings)


def _kmeans_cluster(
    embeddings: np.ndarray, n_clusters: int, random_state: int
) -> np.ndarray:
    from sklearn.cluster import KMeans  # noqa: PLC0415

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    return km.fit_predict(embeddings)


# ---------------------------------------------------------------------------
# LLM cluster labelling
# ---------------------------------------------------------------------------


async def label_clusters(
    texts_by_cluster: dict[int, list[str]],
    llm: Any,
    n_samples: int = 5,
) -> dict[int, str]:
    """Ask an LLM to name each cluster from a sample of its member texts.

    Args:
        texts_by_cluster: Mapping of cluster_id → list of member texts.
        llm: Async LLM client with .ainvoke().
        n_samples: Number of sample texts to include in each prompt.

    Returns:
        Mapping of cluster_id → human-readable cluster name (2–5 words).
    """
    import asyncio  # noqa: PLC0415

    async def _label_one(cluster_id: int, texts: list[str]) -> tuple[int, str]:
        samples = texts[:n_samples]
        sample_str = "\n".join(f"  - {t[:200]}" for t in samples)
        prompt = (
            f"You are labelling a cluster of related texts. Based on the following {len(samples)} "
            f"sample texts, provide a short (2–5 word) descriptive label for this cluster.\n\n"
            f"Samples:\n{sample_str}\n\n"
            f"Label (2–5 words, no quotes):"
        )
        try:
            result = await llm.ainvoke(prompt)
            text = result.content if hasattr(result, "content") else str(result)
            label = text.strip().rstrip(".")
            return cluster_id, label
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to label cluster %d: %s", cluster_id, exc)
            return cluster_id, f"Cluster {cluster_id}"

    tasks = [_label_one(cid, texts) for cid, texts in texts_by_cluster.items()]
    results = await asyncio.gather(*tasks)
    return dict(results)
```
