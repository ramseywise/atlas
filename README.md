# atlas

Agentic financial intelligence for B2B companies. Forecast cash flows, segment customers, explain financial metrics — all from one backend.

---

## What it does

Atlas is a backend service with three agentic capabilities orchestrated by a single conversational agent:

| Agent | What it does |
|---|---|
| **Forecast** | Self-learning cash flow predictions per customer/segment. LangGraph loop: Planner → Forecaster → Evaluator → Learner. |
| **Segment** | Customer cohort discovery via HDBSCAN + UMAP. Claude Haiku reads cluster centroids and writes human-readable segment names. |
| **Knowledge** | Neo4j knowledge graph of financial metrics, customers, and segments. LLM explains metrics in plain English. |

```
User query
    │
    ▼
AtlasAgent (Haiku router)
    ├── forecast_tool ──→ ForecastAgent loop
    ├── segment_tool  ──→ SegmentationAgent loop
    └── knowledge_tool ─→ Neo4j + LLM explanation
    │
    ▼
Haiku synthesizes answer → FastAPI → Next.js dashboard
```

---

## Quick start

```bash
# Python backend
uv sync
uv run uvicorn api.main:app --reload

# Seed knowledge graph (requires Neo4j running)
docker run -p 7474:7474 -p 7687:7687 neo4j:latest
uv run python -m core.knowledge.seeds

# Next.js frontend
cd web && npm install && npm run dev
```

Requires `ANTHROPIC_API_KEY`. Falls back to rule-based strategy if unset.

---

## Repo layout

```
core/           domain logic — data, preprocessing, models, segmentation, knowledge graph
src/            LangGraph agents — forecast, segment, atlas orchestrator
api/            FastAPI backend
web/            Next.js + Tremor dashboard
evals/          graders, harness, model comparison
tests/          pytest suite
nbks/           Jupyter notebooks
data/           raw/processed/synthetic data (gitignored)
```

---

## Stack

**Python**: LangGraph · Chronos · statsforecast · statsmodels · HDBSCAN · tsfresh · polars · neo4j · pydantic v2 · fastapi · anthropic SDK · uv · ruff · pytest

**Frontend**: Next.js 15 · Tremor v3 · Recharts · Tailwind CSS · TypeScript

---

## Make targets

```
make test           all tests
make test-fast      core + evals only
make test-segment   segmentation unit tests
make test-smoke     agent loop smoke tests
make compare        ARIMA vs Chronos comparison
make lint           ruff check + format check
make format         ruff format + fix
make run            python main.py
make clean          remove __pycache__, drift logs
```
