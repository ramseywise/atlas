"""
Atlas FastAPI backend.

Routes:
  POST /ask                 — AtlasAgent conversational query
  POST /forecast/{id}       — run forecast for a customer
  GET  /segments            — get current segment assignments
  POST /segments/refresh    — re-run segmentation
  GET  /knowledge/metric    — look up a metric definition
  GET  /health
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Atlas", version="0.1.0")


# ── Schemas ───────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    query: str
    customer_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    tool_calls_made: list[str]


class ForecastResponse(BaseModel):
    customer_id: str
    horizon_days: int
    forecast: list[float]
    passed: bool


class SegmentResponse(BaseModel):
    segments: list[dict]


class MetricResponse(BaseModel):
    name: str
    definition: str
    formula: str | None


class CryptoPredictRequest(BaseModel):
    symbol: str = "BTC/USDT"
    exchange: str = "binance"
    timeframe: str = "1d"
    prediction_types: list[str] = ["direction", "absolute"]
    max_cycles: int = 2
    learner_policy: str = "bandit"


class CryptoPredictResponse(BaseModel):
    predictions: list[dict]
    eval_summary: str
    cycles_run: int


class CryptoStatsResponse(BaseModel):
    total_trades: int
    win_rate: float
    cumulative_pnl: float
    sharpe: float
    open_positions: int


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    from src.graph import build_atlas_graph
    from src.state import AtlasState

    graph = build_atlas_graph()
    result: AtlasState = graph.invoke(
        {
            "query": req.query,
            "customer_id": req.customer_id,
            "tool_calls": [],
            "synthesis": None,
            "error": None,
        }
    )
    return AskResponse(
        answer=result.get("synthesis") or "",
        tool_calls_made=[c["tool"] for c in result.get("tool_calls", [])],
    )


@app.post("/forecast/{customer_id}", response_model=ForecastResponse)
def forecast(customer_id: str, horizon_days: int = 30):
    try:
        from core.preprocessing.synthetic import generate_sequence_dataset
        from src.agents.graph import run_forecasting_agent

        # Load series data for this customer — replace with real data source in production
        df = generate_sequence_dataset(n_days=365, seed=42)
        result = run_forecasting_agent(series_df=df, max_cycles=2, verbose=False)

        forecasts = result.get("forecasts") or []
        point = forecasts[0].point_forecast if forecasts else []
        passed = result.get("eval_report").all_passed if result.get("eval_report") else False

        return ForecastResponse(
            customer_id=customer_id,
            horizon_days=horizon_days,
            forecast=point[:horizon_days],
            passed=passed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/segments", response_model=SegmentResponse)
def get_segments():
    try:
        from core.knowledge.graph import AtlasGraph

        g = AtlasGraph()
        with g.session() as s:
            result = s.run(
                "MATCH (s:Segment) RETURN s.id AS id, s.label AS label, s.description AS description"
            )
            segments = [dict(r) for r in result]
        g.close()
        return SegmentResponse(segments=segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/segments/refresh")
def refresh_segments():
    return {"status": "queued", "message": "Segmentation refresh not yet wired to async worker."}


@app.get("/knowledge/metric", response_model=MetricResponse)
def get_metric(name: str):
    from core.knowledge.graph import AtlasGraph

    g = AtlasGraph()
    result = g.lookup_metric(name)
    g.close()
    if not result:
        raise HTTPException(status_code=404, detail=f"Metric '{name}' not found.")
    return MetricResponse(name=name, definition=result["definition"], formula=result.get("formula"))


# ── Crypto Routes ────────────────────────────────────────────────────────────


@app.post("/crypto/predict", response_model=CryptoPredictResponse)
def crypto_predict(req: CryptoPredictRequest):
    """Run a crypto prediction cycle."""
    try:
        import asyncio

        from core.preprocessing.crypto.fetcher import CryptoFetcher
        from core.preprocessing.crypto.indicators import add_all_indicators
        from src.agents.crypto.graph import run_crypto_agent

        async def _fetch():
            async with CryptoFetcher(exchange=req.exchange) as fetcher:
                return await fetcher.fetch_ohlcv(
                    symbol=req.symbol, timeframe=req.timeframe, limit=200
                )

        df = asyncio.run(_fetch())
        df = add_all_indicators(df)
        sym_key = req.symbol.replace("/", "_")

        result = run_crypto_agent(
            ohlcv_data={sym_key: df},
            symbols=[req.symbol],
            max_cycles=req.max_cycles,
            learner_policy=req.learner_policy,
            verbose=False,
        )

        predictions = result.get("predictions", [])
        pred_dicts = [
            {
                "symbol": p.symbol,
                "type": p.prediction_type.value,
                "direction": p.direction.value if p.direction else None,
                "confidence": p.direction_confidence,
                "point_forecast": p.point_forecast[:5] if p.point_forecast else None,
                "spread_value": p.spread_value,
            }
            for p in predictions
        ]

        eval_report = result.get("eval_report")
        summary = eval_report.summary if eval_report else "No evaluation"

        return CryptoPredictResponse(
            predictions=pred_dicts,
            eval_summary=summary,
            cycles_run=result.get("cycle_count", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/crypto/predictions")
def crypto_predictions(limit: int = 50):
    """List recent crypto predictions from the log."""
    import json
    from pathlib import Path

    log_path = Path("data/crypto/predictions.jsonl")
    if not log_path.exists():
        return {"predictions": []}

    lines = log_path.read_text().strip().split("\n")
    predictions = [json.loads(line) for line in lines[-limit:]]
    return {"predictions": predictions}


@app.get("/crypto/stats", response_model=CryptoStatsResponse)
def crypto_stats():
    """Paper trading statistics."""
    from pathlib import Path

    from src.agents.crypto.paper_trading import PaperTrader

    state_path = Path("data/crypto/paper_trader.json")
    if not state_path.exists():
        return CryptoStatsResponse(
            total_trades=0, win_rate=0.0, cumulative_pnl=0.0, sharpe=0.0, open_positions=0
        )

    trader = PaperTrader.load(state_path)
    stats = trader.stats()
    return CryptoStatsResponse(**stats)
