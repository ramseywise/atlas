# infra — Templates for infra-builder subagent

## File: {OUTPUT_DIR}/Makefile

```makefile
AGENT := {AGENT_NAME}
UV    := uv run
PORT  ?= 8080

.PHONY: install run dev test test-quick lint format grade grade-ablation-top grade-dry clean

install:
	uv sync

run:
	$(UV) uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload

dev:
	$(UV) uvicorn app:app --host 0.0.0.0 --port $(PORT) --reload --log-level debug

test:
	$(UV) pytest tests/ -v

test-quick:
	$(UV) pytest tests/ -x -q

lint:
	$(UV) ruff check . --exclude .venv

format:
	$(UV) ruff check . --fix --exclude .venv
	$(UV) ruff format . --exclude .venv

grade:
	$(UV) python -m evals.runner --limit 20

grade-ablation-top:
	$(UV) python -m evals.runner --limit 50 --top

grade-dry:
	$(UV) python -m evals.runner --dry-run

clean:
	find . -type d -name __pycache__ | xargs rm -rf
	find . -name "*.pyc" -delete
```

## File: {OUTPUT_DIR}/pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{AGENT_NAME}"
version = "0.1.0"
description = "{AGENT_NAME} agent — {FRAMEWORK}"
readme = "README.md"
requires-python = "==3.12.*"
dependencies = [
    "fastapi>=0.115.0",
    "httpx>=0.28.0",
    "google-adk>=1.28.0",
    "google-genai>=1.69.0",
    "langchain-core>=0.3.0",
    "langchain-google-genai>=2.0.0",
    "langgraph>=0.3.0",
    "langsmith>=0.2.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.2.1",
    "uvicorn[standard]>=0.30.0",
    "boto3>=1.35.0",
]

[project.optional-dependencies]
rag = [
    "langchain-aws>=0.2",
]
forecast = [
    "scikit-learn>=1.4",
    "shap>=0.45",
    "polars>=0.20",
    "numpy>=1.26",
]
cluster = [
    "scikit-learn>=1.4",
    "hdbscan",
    "umap-learn",
    "polars>=0.20",
]
search = [
    "httpx>=0.27",
]
kg = [
    "neo4j>=5.0",
]
streaming = [
    "sse-starlette>=1.6",
]
batch = [
    "tqdm",
]
vision = [
    "Pillow>=10.0",
]
a2a = [
    "websockets>=12.0",
]
finetune = [
    "datasets>=2.0",
    "transformers>=4.40",
    "torch>=2.3",
    "accelerate",
    "peft",
]
rlhf = [
    "trl>=0.9",
    "torch>=2.3",
    "accelerate",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "ruff>=0.8.0",
    "moto[s3,bedrock]>=5.0",
]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
asyncio_mode = "auto"
```

## File: {OUTPUT_DIR}/Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install dependencies (no dev extras)
RUN uv sync --no-dev --no-editable

# Copy source
COPY . .

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

## File: {OUTPUT_DIR}/.env.example

```bash
# {AGENT_NAME} environment variables
# Copy to .env and fill in values. Never commit .env.

# --- Observability ---
LOG_LEVEL=INFO
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_PROJECT={AGENT_NAME}

# --- Model ---
GEMINI_MODEL=gemini-2.5-flash
THINKING_BUDGET=0            # 0 = off; 1024 = low thinking; 8192 = high

# --- AWS / Bedrock ---
AWS_REGION=eu-west-1
AWS_PROFILE=
BEDROCK_KB_ID=
RETRIEVAL_BACKEND=bedrock     # bedrock | rag | custom

# --- Feature flags ---
GROUNDING_ENABLED=true
LLM_PLANNER=false
ROUTING_CONFIDENCE_THRESHOLD=0.2
HITL_GATES_ENABLED=false
HITL_CONFIDENCE_THRESHOLD=0.3
POST_ANSWER_EVAL_ENABLED=false
CRAG_ENABLED=true
CRAG_HIGH_CONFIDENCE=0.7
LG_MEMORY_TURNS=3

# --- Retrieval backends ---
HC_RAG_AGENT_URL=http://localhost:8013

# --- Prompt versioning (auto-set from schema.py PROMPT_VERSION) ---
PROMPT_VERSION=
```
