"""
Time-series embedding for clustering.

Two strategies — both return a float32 matrix of shape (n_customers, n_features):

  embed_tsfresh(series_map)
    Extract ~50 statistical features per series via tsfresh (or a manual subset
    if tsfresh is not installed). Fast, interpretable, no GPU needed.

  embed_chronos(series_map)
    Use the Chronos T5-Tiny encoder to produce a context-averaged embedding.
    Requires `chronos` + `torch`. Falls back to embed_tsfresh on import error.

Both functions accept a dict of {customer_id: np.ndarray} where each array is
a 1-D time series of daily net cash flow values (float32 or float64).
"""

from __future__ import annotations

import numpy as np


# ── tsfresh embedding ─────────────────────────────────────────────────────────


def embed_tsfresh(
    series_map: dict[str, np.ndarray],
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract tsfresh features for each customer time series.

    Returns:
        matrix: float32 array of shape (n_customers, n_features)
        customer_ids: list in the same row order as matrix
    """
    customer_ids = list(series_map.keys())

    try:
        return _embed_tsfresh_full(series_map, customer_ids, n_jobs)
    except ImportError:
        return _embed_manual(series_map, customer_ids)


def _embed_tsfresh_full(
    series_map: dict[str, np.ndarray],
    customer_ids: list[str],
    n_jobs: int,
) -> tuple[np.ndarray, list[str]]:
    import pandas as pd
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters

    rows = []
    for cid, arr in series_map.items():
        for t, v in enumerate(arr):
            rows.append({"id": cid, "time": t, "value": float(v)})
    df = pd.DataFrame(rows)

    features = extract_features(
        df,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=MinimalFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True,
        impute_function=None,
    )
    features = features.fillna(0.0)
    # preserve customer_ids row order
    features = features.loc[customer_ids]
    return features.values.astype(np.float32), customer_ids


def _embed_manual(
    series_map: dict[str, np.ndarray],
    customer_ids: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Manual ~20-feature subset used when tsfresh is unavailable."""
    rows = []
    for cid in customer_ids:
        arr = series_map[cid].astype(float)
        n = len(arr)
        if n == 0:
            rows.append(np.zeros(20, dtype=np.float32))
            continue

        mean = np.mean(arr)
        std = np.std(arr) + 1e-8
        feat = [
            mean,
            std,
            float(np.median(arr)),
            float(np.min(arr)),
            float(np.max(arr)),
            float(np.sum(arr > 0)) / n,           # fraction positive
            float(np.sum(arr < 0)) / n,           # fraction negative
            float(np.mean(np.abs(arr))),           # mean absolute
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
            float(np.percentile(arr, 75) - np.percentile(arr, 25)),  # IQR
            float(np.mean(np.abs(np.diff(arr)))) if n > 1 else 0.0,  # mean abs change
            float(np.sum(np.abs(np.diff(arr)))) if n > 1 else 0.0,   # total variation
            _autocorr_scalar(arr, lag=7),
            _autocorr_scalar(arr, lag=14),
            _autocorr_scalar(arr, lag=30),
            float(np.sum(arr > mean + 2 * std)) / n,  # fraction outliers high
            float(np.sum(arr < mean - 2 * std)) / n,  # fraction outliers low
            _linear_trend(arr),
            float(std / (abs(mean) + 1e-8)),       # CV
        ]
        rows.append(np.array(feat, dtype=np.float32))

    return np.vstack(rows), customer_ids


# ── Chronos embedding ─────────────────────────────────────────────────────────


def embed_chronos(
    series_map: dict[str, np.ndarray],
    model_id: str = "amazon/chronos-t5-tiny",
) -> tuple[np.ndarray, list[str]]:
    """
    Encode each customer series through the Chronos T5 encoder.
    Context is mean-pooled across time steps to produce one vector per customer.

    Falls back to embed_tsfresh on any import error.

    Returns:
        matrix: float32 array of shape (n_customers, encoder_dim)
        customer_ids: list in the same row order as matrix
    """
    customer_ids = list(series_map.keys())
    try:
        return _embed_chronos_encoder(series_map, customer_ids, model_id)
    except Exception:
        return embed_tsfresh(series_map)


def _embed_chronos_encoder(
    series_map: dict[str, np.ndarray],
    customer_ids: list[str],
    model_id: str,
) -> tuple[np.ndarray, list[str]]:
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    embeddings = []
    for cid in customer_ids:
        arr = series_map[cid]
        ctx = torch.tensor(arr[np.newaxis, :], dtype=torch.float32)
        with torch.no_grad():
            # encoder_last_hidden_state: (1, seq_len, hidden_dim)
            enc = pipeline.model.encoder(
                input_ids=pipeline._prepare_context(ctx)[0],
                attention_mask=torch.ones(1, len(arr), dtype=torch.long),
            ).last_hidden_state
        # mean pool over time → (hidden_dim,)
        emb = enc[0].mean(dim=0).numpy()
        embeddings.append(emb.astype(np.float32))

    return np.vstack(embeddings), customer_ids


# ── Helpers ───────────────────────────────────────────────────────────────────


def _autocorr_scalar(arr: np.ndarray, lag: int) -> float:
    if len(arr) <= lag:
        return 0.0
    a = arr[:-lag] - np.mean(arr)
    b = arr[lag:] - np.mean(arr)
    denom = np.std(arr) ** 2 * len(a) + 1e-8
    return float(np.sum(a * b) / denom)


def _linear_trend(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    x -= x.mean()
    y = arr - arr.mean()
    slope = float(np.dot(x, y) / (np.dot(x, x) + 1e-8))
    return slope / (np.mean(np.abs(arr)) + 1e-8)
