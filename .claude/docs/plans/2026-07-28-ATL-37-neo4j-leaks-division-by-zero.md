Status: COMPLETE
Evidence: all 8 steps implemented and merged (PR #39); 189 tests pass, lint clean.

# ATL-37: Neo4j connection leaks in API routes + division-by-zero in indicators

GitHub issue: #37
Source: `/akira all` sweep 2026-07-27

## Plan
Date: 2026-07-28
Based on: direct codebase inspection (issue #37 body + read of all cited files)

### Goal
Close two AtlasGraph resource leaks in `api/main.py`, guard three division/zero paths in
crypto indicators and forecasting, and clean up the three noted warnings (hardcoded model
name, hardcoded Neo4j localhost fallback, bare `print()`), without changing route contracts.

### Open Questions
1. **`AtlasGraph` fix shape — context manager on the class, or `try/finally` at each call
   site?** Assumed default: make `AtlasGraph` an `__enter__`/`__exit__` context manager
   (Step 1) and use `with AtlasGraph() as g:` at both call sites (Step 2). This is more
   reusable than duplicating `try/finally` twice and matches the class's existing
   `@contextmanager` style for `.session()`. If a future call site needs the graph object
   without `with`, `try/finally` remains available as a fallback — not blocking.
2. **RSI zero-loss semantics**: when `loss == 0` (price only went up over the window), RSI
   should mathematically be 100 (maximally overbought), not `NaN`/`inf`. Assumed default:
   special-case `loss == 0` → `rsi = 100.0` (and if `gain == 0` too, i.e. flat price, → `50.0`
   neutral). Confirm this matches the intended trading semantics — no test currently locks
   this in, so it's a free choice, but it should be a *documented* choice, not silent.
3. **`volume_ratio` zero/NaN denominator**: when `rolling_mean(volume) == 0` (or is `null`
   during warmup), assumed default is `volume_ratio = null` (preserve existing NaN/null
   warmup behavior for the first `period` rows) but replace an actual `0.0` divisor with
   `null` rather than `inf`. Downstream consumers (indicators are read by `crypto_predict`
   route and `crypto_forecaster_node`) already tolerate nulls at warmup, so this is
   consistent with current behavior, just closing the `inf` case specifically.
4. **`current_price == 0` guard in `crypto_forecaster_node`**: assumed default is to skip
   emitting a `DIRECTION` prediction for that symbol when `current_price <= 0` (treat as bad
   data upstream, log and continue) rather than emitting `Direction.NEUTRAL` with a fake
   confidence — a zero/negative price is a data-quality problem, not a real "no signal"
   case, and silently emitting NEUTRAL could mask a fetcher bug. If the reviewer prefers
   NEUTRAL-with-zero-confidence instead of a skip, that's a one-line change in Step 5.
5. **Env var name for the model constant** (warning cleanup, not in original bug list but
   4 call sites share the same hardcoded string): assumed default `ATLAS_LLM_MODEL`,
   defined once in a new small module `src/config.py` per repo's "named constants or env
   vars" style rule, defaulting to the current literal so behavior is unchanged.
6. **Neo4j localhost fallback** (`core/knowledge/graph.py:22`): the issue flags this as a
   warning, not something to remove — localhost is a reasonable *local-dev default* for
   `NEO4J_URI`. Assumed default: leave the `os.environ.get(..., "bolt://localhost:7687")`
   pattern as-is; it already reads from env first. If Ramsey wants a hard-fail in
   production (no fallback), that's a bigger config-strategy change out of scope here —
   flagged as NOTE, not fixed in this plan.

### Approach
Fix the two blocker-level resource leaks first (Steps 1–2) since they're independent of
everything else and highest severity. Then fix the three division-by-zero warnings
(Steps 3–5), each isolated to one function with existing or new unit tests. Finish with
the three code-quality warnings (Steps 6–8), each mechanical and low-risk. Steps are
ordered by risk (blocker → data-correctness → style) and each is independently testable
and revertable.

### Out of Scope
- Making `AtlasGraph` connection-pooled/cached across requests (a longer-lived driver
  instance shared via FastAPI `lifespan`/dependency injection) — that's a performance
  improvement, not a leak fix, and changes the app's startup/shutdown contract. Flagged
  as a follow-up NOTE.
- Removing the Neo4j localhost fallback / enforcing required env vars in prod — separate
  config-hardening decision (Open Question 6).
- Any change to `/ask`, `/forecast/{id}`, `/segments/refresh`, `/crypto/*` routes — they
  don't touch `AtlasGraph` and are unaffected.
- Adding retry/backoff to Neo4j driver construction — not requested, no evidence of
  transient-connection issues in the issue.
- `nodes.py:342`'s sibling occurrences are being fixed too (Open Question 5) since they're
  the *same* literal, same file family, and are already caught by the pre-existing
  "no hardcoded model strings" hook rule — but no other pyproject/hook config is touched.

### Steps

#### Step 1: Make `AtlasGraph` a context manager
**Files**: `core/knowledge/graph.py` (lines 27-42)
**What**: Add `__enter__`/`__exit__` to `AtlasGraph` so callers can use
`with AtlasGraph() as g:` and `close()` is guaranteed even on exception. Keep the existing
`close()` method and `.session()` contextmanager untouched — this is additive.
**Snippet**:
```python
# before (lines 27-37)
class AtlasGraph:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

# after
class AtlasGraph:
    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def __enter__(self) -> AtlasGraph:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```
**Test**: `uv run pytest tests/core/knowledge/test_graph.py::test_context_manager_closes_on_exception -v`
(new test file — see Step 1a below)

#### Step 1a: Add unit test for `AtlasGraph` context manager
**Files**: `tests/core/knowledge/test_graph.py` (new file; create `tests/core/knowledge/`
dir if it doesn't exist)
**What**: Unit test that `__exit__` calls `close()` even when the `with` body raises,
using a stub/mock driver so no real Neo4j connection is required (repo has no Neo4j test
fixture today — confirmed by `find tests -iname "*graph*"` returning nothing under
`tests/core/knowledge/`).
**Snippet**:
```python
"""Tests for AtlasGraph context manager behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.knowledge.graph import AtlasGraph


def _make_graph_with_mock_driver() -> tuple[AtlasGraph, MagicMock]:
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        g = AtlasGraph(uri="bolt://fake", user="u", password="p")
    return g, mock_driver


def test_context_manager_closes_on_success():
    g, mock_driver = _make_graph_with_mock_driver()
    with g:
        pass
    mock_driver.close.assert_called_once()


def test_context_manager_closes_on_exception():
    g, mock_driver = _make_graph_with_mock_driver()
    with pytest.raises(ValueError):
        with g:
            raise ValueError("boom")
    mock_driver.close.assert_called_once()
```
**Test**: `uv run pytest tests/core/knowledge/test_graph.py -v`
**Done when**: both tests pass; `mock_driver.close.assert_called_once()` proves `close()`
fires on the exception path, which is the exact bug being fixed.

#### Step 2: Use `with AtlasGraph()` at both leak sites in `api/main.py`
**Files**: `api/main.py` (lines 127-141 `get_segments`, lines 149-158 `get_metric`)
**What**: Replace manual `g = AtlasGraph(); ...; g.close()` with `with AtlasGraph() as g:`
so the driver closes even if `s.run(...)` / `g.lookup_metric(...)` raises. For
`get_segments`, the existing `try/except` around the whole block stays (it converts
exceptions to HTTP 500) — only the resource lifecycle changes. `get_metric` currently has
**no** try/except at all (a bare exception would 500 with a raw traceback AND leak the
driver); wrap it the same way `get_segments` and `forecast` already are, per the file's
existing convention.
**Snippet**:
```python
# before (lines 127-141)
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

# after
@app.get("/segments", response_model=SegmentResponse)
def get_segments():
    try:
        from core.knowledge.graph import AtlasGraph

        with AtlasGraph() as g, g.session() as s:
            result = s.run(
                "MATCH (s:Segment) RETURN s.id AS id, s.label AS label, s.description AS description"
            )
            segments = [dict(r) for r in result]
        return SegmentResponse(segments=segments)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
```
```python
# before (lines 149-158)
@app.get("/knowledge/metric", response_model=MetricResponse)
def get_metric(name: str):
    from core.knowledge.graph import AtlasGraph

    g = AtlasGraph()
    result = g.lookup_metric(name)
    g.close()
    if not result:
        raise HTTPException(status_code=404, detail=f"Metric '{name}' not found.")
    return MetricResponse(name=name, definition=result["definition"], formula=result.get("formula"))

# after
@app.get("/knowledge/metric", response_model=MetricResponse)
def get_metric(name: str):
    from core.knowledge.graph import AtlasGraph

    try:
        with AtlasGraph() as g:
            result = g.lookup_metric(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if not result:
        raise HTTPException(status_code=404, detail=f"Metric '{name}' not found.")
    return MetricResponse(name=name, definition=result["definition"], formula=result.get("formula"))
```
**Test**: `uv run pytest tests/api/test_main.py -v` (new file — see Step 2a)
**Done when**: neither route path can leave a call without `close()` executing — verified
via the mocked test in Step 2a rather than a live Neo4j instance.

#### Step 2a: Add route-level test proving `close()` fires on exception
**Files**: `tests/api/test_main.py` (new file; create `tests/api/__init__.py` if the repo's
test layout needs it — check sibling dirs under `tests/core/` for whether `__init__.py`
files exist before adding one)
**What**: Use `fastapi.testclient.TestClient` with `core.knowledge.graph.AtlasGraph`
patched so `.lookup_metric()` / `s.run()` raises, and assert the underlying driver's
`close()` was still called, and the route returns 500 (not an unhandled exception).
**Snippet**:
```python
"""Tests for api/main.py route resource-lifecycle and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_metric_closes_driver_on_exception():
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        with patch(
            "core.knowledge.graph.AtlasGraph.lookup_metric",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/knowledge/metric", params={"name": "burn_ratio"})
    assert resp.status_code == 500
    mock_driver.close.assert_called_once()


def test_get_segments_closes_driver_on_exception():
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.side_effect = RuntimeError("boom")
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_gdb.driver.return_value = mock_driver
        resp = client.get("/segments")
    assert resp.status_code == 500
    mock_driver.close.assert_called_once()
```
**Test**: `uv run pytest tests/api/test_main.py -v`
**Done when**: both tests pass, proving the fix (not just the happy path) — a regression
back to `g.close()` placed after a risky call would make `mock_driver.close.assert_called_once()`
fail.

#### Step 3: Guard RSI division by zero
**Files**: `core/preprocessing/crypto/indicators.py` (lines 12-20, `compute_rsi`)
**What**: When `loss == 0`, `gain / loss` is `inf` (or `NaN` if `gain` is also 0),
producing `NaN`/`inf` in the `rsi` column instead of the mathematically correct 100
(all gains, no losses) or 50 (flat/no movement). Use `pl.when` to special-case both.
**Snippet**:
```python
# before (lines 12-20)
def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Relative Strength Index (0-100). Oversold < 30, Overbought > 70."""
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=period)

    return df.with_columns(
        (100.0 - 100.0 / (1.0 + gain / loss)).alias("rsi"),
    )

# after
def compute_rsi(df: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Relative Strength Index (0-100). Oversold < 30, Overbought > 70."""
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).rolling_mean(window_size=period)
    loss = (-delta.clip(upper_bound=0)).rolling_mean(window_size=period)

    rsi = (
        pl.when(loss == 0)
        .then(pl.when(gain == 0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + gain / loss))
    )
    return df.with_columns(rsi.alias("rsi"))
```
**Test**: `uv run pytest tests/core/crypto/test_indicators.py::TestRSI -v` plus new test
below.
**Snippet (new test)**:
```python
# add to tests/core/crypto/test_indicators.py, class TestRSI
def test_rsi_all_gains_is_100(self):
    """Monotonically increasing close -> loss==0 -> RSI should be 100, not NaN/inf."""
    df = pl.DataFrame({"close": [float(i) for i in range(1, 20)]})
    result = compute_rsi(df, period=5)
    rsi_values = result["rsi"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(rsi_values))
    assert np.allclose(rsi_values, 100.0)

def test_rsi_flat_price_is_50(self):
    """Constant close -> gain==0 and loss==0 -> RSI should be 50 (neutral), not NaN."""
    df = pl.DataFrame({"close": [100.0] * 20})
    result = compute_rsi(df, period=5)
    rsi_values = result["rsi"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(rsi_values))
    assert np.allclose(rsi_values, 50.0)
```
**Done when**: `test_rsi_all_gains_is_100` and `test_rsi_flat_price_is_50` pass, and
existing `TestRSI::test_rsi_bounded` still passes (values stay within `[0, 100]` inclusive
of the new fixed points).

#### Step 4: Guard `volume_ratio` division by zero
**Files**: `core/preprocessing/crypto/indicators.py` (lines 82-89, `compute_volume_sma`)
**What**: When the rolling mean of volume is `0` (illiquid/zero-volume window),
`volume / 0` produces `inf`. Replace with `null` in that case so downstream consumers
(which already null-check/drop-null per Step 5 discussion and existing test patterns)
don't see `inf`. `NaN` from an actual `null` rolling mean during warmup already
propagates to `null` under Polars division and is unaffected by this change.
**Snippet**:
```python
# before (lines 82-89)
def compute_volume_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Volume simple moving average for volume spike detection."""
    return df.with_columns(
        pl.col("volume").rolling_mean(window_size=period).alias("volume_sma"),
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=period)).alias(
            "volume_ratio"
        ),
    )

# after
def compute_volume_sma(df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
    """Volume simple moving average for volume spike detection."""
    volume_sma = pl.col("volume").rolling_mean(window_size=period)
    volume_ratio = (
        pl.when(volume_sma == 0)
        .then(None)
        .otherwise(pl.col("volume") / volume_sma)
    )
    return df.with_columns(
        volume_sma.alias("volume_sma"),
        volume_ratio.alias("volume_ratio"),
    )
```
**Test**: `uv run pytest tests/core/crypto/test_indicators.py::TestVolumeSMA -v` plus new
test below.
**Snippet (new test)**:
```python
# add to tests/core/crypto/test_indicators.py, class TestVolumeSMA
def test_volume_ratio_zero_volume_window_is_null_not_inf(self):
    """A window of all-zero volume must yield null volume_ratio, not inf."""
    df = pl.DataFrame({"volume": [0.0] * 25})
    result = compute_volume_sma(df, period=5)
    ratio = result["volume_ratio"].drop_nulls().to_numpy()
    assert np.all(np.isfinite(ratio)) if len(ratio) else True
    assert not result["volume_ratio"].is_infinite().any()
```
**Done when**: new test passes and `result["volume_ratio"].is_infinite().any()` is
`False` for the zero-volume fixture; existing `TestVolumeSMA::test_volume_columns_exist`
and `TestAddAllIndicators` tests still pass unmodified.

#### Step 5: Guard `current_price == 0` in `crypto_forecaster_node`
**Files**: `src/agents/crypto/nodes.py` (lines 64-115, inside `crypto_forecaster_node`)
**What**: `pct_change = (forecast_mean - current_price) / current_price` divides by
`current_price`, which can be `0` (or negative) on bad/missing fetcher data, producing
`inf`/`nan` confidence propagated into `CryptoPrediction.direction_confidence`. Per Open
Question 4's default: skip emitting the `DIRECTION` prediction for that symbol, without
affecting the `ABSOLUTE` prediction (appended earlier in the same loop iteration, lines
79-90) or later symbols/the `SPREAD` block after the loop. Use a guard on the `if`
condition — not `continue` — since `continue` would also skip anything added after the
`DIRECTION` block for this symbol in the future. Hoist `current_price` above the
`ABSOLUTE` block so both blocks share one computation.
**Snippet**:
```python
# before (lines 70-115)
        df = pl.from_dict(series_dict)
        close_values = df["close"].tail(strategy.context_bars).to_numpy()

        if len(close_values) < 10:
            continue

        model_id = strategy.model_variant.value
        point, lower, upper = _forecast_series(close_values, strategy.forecast_bars, model_id)

        if PredictionType.ABSOLUTE in strategy.prediction_types:
            predictions.append(
                CryptoPrediction(
                    symbol=symbol,
                    prediction_type=PredictionType.ABSOLUTE,
                    timestamp=now,
                    point_forecast=point,
                    lower_80=lower,
                    upper_80=upper,
                    model_used=strategy.model_variant,
                )
            )

        if PredictionType.DIRECTION in strategy.prediction_types:
            current_price = float(close_values[-1])
            forecast_mean = np.mean(point)
            pct_change = (forecast_mean - current_price) / current_price
            ...

# after
        df = pl.from_dict(series_dict)
        close_values = df["close"].tail(strategy.context_bars).to_numpy()

        if len(close_values) < 10:
            continue

        current_price = float(close_values[-1])
        model_id = strategy.model_variant.value
        point, lower, upper = _forecast_series(close_values, strategy.forecast_bars, model_id)

        if PredictionType.ABSOLUTE in strategy.prediction_types:
            predictions.append(
                CryptoPrediction(
                    symbol=symbol,
                    prediction_type=PredictionType.ABSOLUTE,
                    timestamp=now,
                    point_forecast=point,
                    lower_80=lower,
                    upper_80=upper,
                    model_used=strategy.model_variant,
                )
            )

        if PredictionType.DIRECTION in strategy.prediction_types and current_price > 0:
            forecast_mean = np.mean(point)
            pct_change = (forecast_mean - current_price) / current_price
            ...
```
(the `pct_change`/`direction`/`confidence`/`predictions.append(...)` body inside the
`DIRECTION` block is otherwise unchanged — only the `if` condition and the removal of
the now-redundant `current_price = float(close_values[-1])` line change.)
**Test**: `uv run pytest tests/agents/crypto/test_nodes.py::test_forecaster_skips_direction_on_zero_price -v`
(new file)
**Snippet (new test)**:
```python
"""Tests for crypto_forecaster_node division-by-zero guard."""

from __future__ import annotations

import polars as pl

from src.agents.crypto.nodes import crypto_forecaster_node
from src.agents.crypto.state import CryptoPlannerStrategy, PredictionType


def test_forecaster_skips_direction_on_zero_price():
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
```
**Done when**: new test passes; confirm via `grep -n "current_price" src/agents/crypto/nodes.py`
that no other read of `current_price` divides without the same guard.

#### Step 6: Centralize hardcoded model name behind an env var
**Files**: new `src/config.py`; edits to `src/graph.py` (lines 66, 180),
`src/agents/nodes.py` (line 342), `core/segmentation/naming.py` (line 105)
**What**: All four call sites hardcode the literal `"claude-haiku-4-5-20251001"`. Add one
module-level constant read from env (repo style: "No magic numbers — named constants or
env vars"), default preserves current behavior exactly.
**Snippet**:
```python
# new file: src/config.py
"""Shared configuration constants."""

from __future__ import annotations

import os

ATLAS_LLM_MODEL = os.environ.get("ATLAS_LLM_MODEL", "claude-haiku-4-5-20251001")
```
```python
# each of the 4 call sites, e.g. src/agents/nodes.py:342
# before
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            ...

# after
        from src.config import ATLAS_LLM_MODEL
        response = client.messages.create(
            model=ATLAS_LLM_MODEL,
            ...
```
(Import at module top-level instead of inline where the surrounding file already imports
at top-level; match each file's existing import style — `src/graph.py` and
`src/agents/nodes.py` import `anthropic` lazily inside functions per current code, so
follow that lazy-import convention there; `core/segmentation/naming.py` — check its
existing import placement before deciding.)
**Test**: `uv run pytest --tb=no -q` (no behavior change — full suite must stay green)
plus `grep -rn "claude-haiku-4-5-20251001" src/ core/` should return zero matches outside
`src/config.py`.
**Done when**: `grep -rn "claude-haiku-4-5-20251001" src/ core/` shows exactly one hit
(in `src/config.py`), full test suite passes, and the pre-existing "no hardcoded model
strings" hook does not flag any of the four edited files on save.

#### Step 7: Replace bare `print()` in `segment/graph.py`
**Files**: `src/agents/segment/graph.py` (line 156, inside the `except ImportError:`
fallback branch)
**What**: Repo hook rule bans bare `print()` in `src/`. The surrounding `try` block uses
`rich`'s `console.print(...)` when available; the `except ImportError:` fallback uses bare
`print()` when `rich` isn't installed. Replace with the standard `logging` module so it's
consistent with the rest of the codebase's logging convention (confirmed via
`logger.warning` usage in `src/agents/nodes.py`) instead of stdout.
**Snippet**:
```python
# before (lines 154-158)
    except ImportError:
        result = state.get("result")
        if result:
            print(
                f"\nSegmentation done: {result['n_segments']} segments, {len(result['customer_ids'])} customers"
            )

# after
    except ImportError:
        result = state.get("result")
        if result:
            logger.info(
                "Segmentation done: %d segments, %d customers",
                result["n_segments"],
                len(result["customer_ids"]),
            )
```
Add `import logging` and `logger = logging.getLogger(__name__)` near the top of
`src/agents/segment/graph.py` if not already present — check the file's current imports
first (read lines 1-30 at execute time; this plan hasn't confirmed whether a module
logger already exists there).
**Test**: `uv run pytest --tb=no -q` (no test currently exercises the ImportError fallback
branch directly — this is a style-only change with no behavioral test to add; verify no
regression by running the segmentation test suite: `uv run pytest tests/ -k segment -v`)
**Done when**: `grep -n "print(" src/agents/segment/graph.py` returns no matches, and the
PostToolUse hook's "no print() in src/" check passes on save.

#### Step 8: Verify Neo4j localhost fallback and no-op confirmation
**Files**: `core/knowledge/graph.py` (lines 22-24) — read-only verification step, no edit
**What**: Per Open Question 6, this is a warning about a *pattern*, not a bug: the
`NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` fallbacks already read from env first via
`os.environ.get(...)` and only fall back to `localhost`/`neo4j`/`password` for local dev.
No code change — document the decision in this plan and close the warning as
"acknowledged, no action" unless Ramsey overrides in review.
**Test**: none (no-op)
**Done when**: Open Question 6 is either confirmed as-is by reviewer, or reopened as a
separate issue for prod config hardening (out of scope here either way).

### Test Plan
1. Run full baseline before touching anything: `uv run pytest --tb=no -q` (currently 181
   passed — confirmed 2026-07-28).
2. After each step, run that step's scoped test command (listed per-step above).
3. After all steps: `uv run pytest --tb=no -q` — expect 181 + new tests (2 in Step 1a, 2
   in Step 2a, 2 in Step 3, 1 in Step 4, 1 in Step 5 = +8 new tests, 189 total), all
   green, zero regressions.
4. `grep -rn "claude-haiku-4-5-20251001" src/ core/` → exactly 1 match (`src/config.py`).
5. `grep -n "print(" src/agents/segment/graph.py` → 0 matches.
6. Manual smoke (optional, requires local Neo4j): hit `GET /knowledge/metric?name=x` and
   `GET /segments` against a running Neo4j instance, confirm driver connections don't
   accumulate (`neo4j` server-side connection count stable across repeated calls) —
   defer to reviewer if no local Neo4j available; the mocked unit tests in Step 2a are
   the primary proof and don't require a live database.

### Risks & Rollback
- **Step 1/2 risk**: `with AtlasGraph() as g, g.session() as s:` combines two context
  managers on one line — low risk, standard Python, but verify Python version support
  (repo uses 3.12 per `.venv/lib/python3.12` — parenthesized/comma-separated `with`
  statements are supported since 3.10, fine).
- **Step 3 risk**: changing RSI's zero-loss behavior from `NaN` to `100.0`/`50.0` is a
  **behavior change**, not just a bug fix, if any downstream code currently relies on
  `NaN` to detect "insufficient data" vs. "no losses". Mitigated by Step 3's test suite
  and by keeping the *warmup* `null` behavior (insufficient window) completely
  unchanged — only the true divide-by-zero case changes. Rollback: revert
  `compute_rsi` to the plain `gain / loss` expression.
- **Step 4 risk**: low — `inf → null` is strictly safer for any consumer doing
  arithmetic or comparisons downstream.
- **Step 5 risk**: skipping DIRECTION predictions on zero price could reduce prediction
  coverage in `crypto_predict` route responses. Acceptable per Open Question 4's
  reasoning (bad data shouldn't produce fake signals) but flag to reviewer as a
  behavior change worth a changelog note.
- **Step 6 risk**: purely mechanical; rollback is trivial (revert `src/config.py` and
  the 4 call-site edits).
- **Step 7 risk**: none — dead code path (ImportError fallback), unlikely to be hit in
  CI since `rich` is presumably an installed dependency; verify with
  `grep -n "rich" pyproject.toml` before executing, since if `rich` isn't a dependency
  at all this fallback branch might actually be the *primary* path in some environments.
- **General rollback**: each step is one file (or one new file), independently
  revertible via `git checkout -- <file>` before commit; no step depends on a later
  step's code existing.

### Open Questions
(See "Open Questions" section above, placed first per plan template — six items listed:
AtlasGraph fix shape, RSI zero-loss semantics, volume_ratio null-vs-inf, current_price
skip-vs-neutral, model-name env var naming, Neo4j localhost fallback disposition.)
