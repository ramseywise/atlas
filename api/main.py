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
