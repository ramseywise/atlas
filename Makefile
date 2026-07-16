.PHONY: test test-fast test-core test-graders test-arima test-smoke test-features test-segment test-segment-smoke test-crypto test-learner lint format run forecast segment crypto crypto-monitor compare compare-learner clean

# ── Test targets ──────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

test-fast:
	uv run pytest tests/core/ tests/evals/ -v

test-core:
	uv run pytest tests/core/ -v

test-graders:
	uv run pytest tests/evals/test_graders.py -v

test-arima:
	uv run pytest tests/core/test_arima.py -v

test-data:
	uv run pytest tests/core/test_synthetic.py tests/core/test_preprocessing.py -v

test-features:
	uv run pytest tests/core/test_features.py -v

test-smoke:
	uv run pytest tests/src/test_agent_smoke.py -v

test-segment:
	uv run pytest tests/core/segmentation/ -v

test-segment-smoke:
	uv run pytest tests/src/agents/test_segment_smoke.py -v

test-crypto:
	uv run pytest tests/src/agents/crypto/ tests/core/crypto/ tests/evals/test_crypto_graders.py -v

test-learner:
	uv run pytest tests/src/agents/learner/ -v

# ── Pipelines ─────────────────────────────────────────────────────────────────

forecast:
	uv run python -m pipelines.forecast

segment:
	uv run python -m pipelines.segment

crypto:
	uv run python -m pipelines.crypto

crypto-monitor:
	uv run python -m pipelines.crypto_monitor

# ── Dev servers ───────────────────────────────────────────────────────────────

api:
	uv run uvicorn api.main:app --reload

seed-knowledge:
	uv run python -m core.knowledge.seeds

# ── Code quality ──────────────────────────────────────────────────────────────

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	uv run python main.py

compare:
	uv run python -c "from core.preprocessing.synthetic import generate_sequence_dataset, temporal_split; from evals.comparison import run_model_comparison; df = generate_sequence_dataset(n_days=730, seed=42); split = temporal_split(df); run_model_comparison(split.train, horizon_days=30, max_folds=3, models=['arima', 'chronos'])"

compare-learner:
	uv run python -c "from evals.comparison_learner import run_learner_comparison; run_learner_comparison()"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .ruff_cache evals/reports/*.jsonl

.PHONY: precommit
precommit:  ## run all pre-commit hooks (ruff, format, gitleaks, eslint where wired) on all files
	pre-commit run --all-files
