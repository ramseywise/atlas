"""Tests for crypto_forecaster_node division-by-zero guard."""

from __future__ import annotations

import polars as pl

from src.agents.crypto.nodes import crypto_forecaster_node
from src.agents.crypto.state import CryptoPlannerStrategy, PredictionType


def test_forecaster_skips_direction_on_zero_price():
    """When current_price==0, DIRECTION predictions must be skipped (not emit inf/nan confidence)."""
    strategy = CryptoPlannerStrategy(
        symbols=["BTC/USDT"],
        prediction_types=[PredictionType.DIRECTION],
        rationale="test",
    )
    df = pl.DataFrame({"close": [0.0] * 20})
    state = {
        "strategy": strategy,
        "ohlcv_data": {"BTC_USDT": df.to_dict(as_series=False)},
    }
    result = crypto_forecaster_node(state)
    direction_preds = [
        p for p in result["predictions"] if p.prediction_type == PredictionType.DIRECTION
    ]
    assert direction_preds == []
