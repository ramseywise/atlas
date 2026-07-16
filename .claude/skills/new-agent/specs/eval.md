# eval — Templates for eval-harness-builder subagent

## Design notes

### Warning-only graders must be excluded from `all_passed`
Graders like `DriftGrader` signal degradation trends but should never block the
agent loop from completing. Exclude them explicitly in `all_passed` by name:

```python
all_passed = all(
    s.passed
    for scores in series_scores.values()
    for s in scores
    if s.grader_name != "DriftDetection"   # warning-only — not a hard gate
)
```

### DriftGrader.score should not accept None
`DriftGrader` is stateful (rolling history) and its `score()` method doesn't
use the `forecast` parameter at all. Drop the parameter rather than passing
`None` and adding `# type: ignore`:

```python
# Wrong — passes None, misleads type checkers
drift_score = self._drift_grader.score(np.array([]), forecasts[0] if forecasts else None)

# Correct — DriftGrader.score takes no forecast arg
def score(self) -> GraderScore:
    ...
drift_score = self._drift_grader.score()
```

### Stateless graders should be instance attributes, not method-local
`SMAPEGrader`, `DirectionalGrader`, and `CoverageGrader` are stateless and can
be instantiated once in `EvalHarness.__init__` rather than on every `run()` call:

```python
class EvalHarness:
    def __init__(self, ...):
        self._smape_g = SMAPEGrader()
        self._dir_g = DirectionalGrader()
        self._cov_g = CoverageGrader()
        self._drift_grader = DriftGrader(baseline_mase=baseline_mase)
```

---

## File: {OUTPUT_DIR}/evals/datasets/eval.jsonl

```jsonl
{"query": "How does this work?", "expected_intent": "answerable", "golden_answer": "Here is how it works...", "expected_sources": [], "domain": "{DOMAIN}"}
{"query": "I need to speak to a human", "expected_intent": "escalation", "golden_answer": "", "expected_sources": [], "domain": "{DOMAIN}"}
{"query": "help", "expected_intent": "clarification", "golden_answer": "", "expected_sources": [], "domain": "{DOMAIN}"}
```

## File: {OUTPUT_DIR}/evals/runner.py

```python
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Adjust import path based on where the agent module lives
# ---------------------------------------------------------------------------
try:
    from {AGENT_NAME}.main import AgentRunner
    from {AGENT_NAME}.schema import AssistantResponse
except ImportError as e:
    raise ImportError(
        f"Could not import {AGENT_NAME} — is the agent installed or on PYTHONPATH? {e}"
    )

DATASET_PATH = Path(__file__).parent / "datasets" / "eval.jsonl"
DEFAULT_CONCURRENCY = 5


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------


class EvalRunner:
    def __init__(
        self,
        limit: int | None = None,
        dry_run: bool = False,
        top: bool = False,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        self.limit = limit
        self.dry_run = dry_run
        self.top = top
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> list[dict[str, Any]]:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        items: list[dict[str, Any]] = []
        with DATASET_PATH.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        if self.top:
            # top flag: sort by expected_intent so escalation/clarification come last
            items.sort(key=lambda x: x.get("expected_intent", ""))

        if self.limit is not None:
            items = items[: self.limit]

        return items

    # ------------------------------------------------------------------
    # Single-item execution
    # ------------------------------------------------------------------

    async def run_agent(self, item: dict[str, Any]) -> dict[str, Any]:
        """Call the agent and return a result dict with timing."""
        import uuid

        query = item["query"]
        start = time.monotonic()

        if self.dry_run:
            # Simulate a response without calling the agent
            await asyncio.sleep(0.01)
            return {
                "query": query,
                "expected_intent": item.get("expected_intent"),
                "response": None,
                "latency_s": time.monotonic() - start,
                "error": None,
                "dry_run": True,
            }

        try:
            async with self._semaphore:
                runner = AgentRunner(session_id=str(uuid.uuid4()))
                response: AssistantResponse = await runner.run(query)
            return {
                "query": query,
                "expected_intent": item.get("expected_intent"),
                "response": response,
                "latency_s": time.monotonic() - start,
                "error": None,
                "dry_run": False,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "query": query,
                "expected_intent": item.get("expected_intent"),
                "response": None,
                "latency_s": time.monotonic() - start,
                "error": str(exc),
                "dry_run": False,
            }

    # ------------------------------------------------------------------
    # Evaluation logic
    # ------------------------------------------------------------------

    def evaluate(self, result: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
        """Check intent match, non-empty message, contact_support flag."""
        if result["error"]:
            return {**result, "pass": False, "fail_reason": f"agent_error: {result['error']}"}

        if result["dry_run"]:
            return {**result, "pass": True, "fail_reason": None, "note": "dry_run_skipped"}

        response: AssistantResponse = result["response"]
        expected_intent = item.get("expected_intent", "")

        # 1. Non-empty message
        if not response.message or not response.message.strip():
            return {**result, "pass": False, "fail_reason": "empty_message"}

        # 2. Intent-specific checks
        if expected_intent == "escalation":
            if not response.contact_support:
                return {
                    **result,
                    "pass": False,
                    "fail_reason": "escalation_not_flagged",
                }

        if expected_intent == "answerable":
            if response.contact_support:
                return {
                    **result,
                    "pass": False,
                    "fail_reason": "unexpected_escalation",
                }

        # 3. Sources must come from retrieved set (grounding check)
        if response.sources:
            retrieved_urls = {
                p["location"]["webLocation"]["url"]
                for p in getattr(response, "_retrieved_passages", [])
            }
            if retrieved_urls:
                for src in response.sources:
                    if str(src.url) not in retrieved_urls:
                        return {
                            **result,
                            "pass": False,
                            "fail_reason": f"ungrounded_source: {src.url}",
                        }

        return {**result, "pass": True, "fail_reason": None}

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def run_all(self) -> list[dict[str, Any]]:
        items = self.load_dataset()
        print(f"[{'{AGENT_NAME}'}] Running eval on {len(items)} item(s) "
              f"(dry_run={self.dry_run}, concurrency={self.concurrency})")

        raw_results = await asyncio.gather(*(self.run_agent(item) for item in items))
        evaluated = [self.evaluate(result, item) for result, item in zip(raw_results, items)]

        # ------ Print results ------
        passed = sum(1 for r in evaluated if r["pass"])
        failed = len(evaluated) - passed

        print("\n--- Results ---")
        for r in evaluated:
            status = "PASS" if r["pass"] else "FAIL"
            reason = f" ({r.get('fail_reason')})" if not r["pass"] else ""
            latency = f"{r['latency_s']:.2f}s"
            print(f"  [{status}] {r['query']!r:<50} {latency}{reason}")

        print(f"\nSummary: {passed}/{len(evaluated)} passed  |  {failed} failed")
        if evaluated:
            avg_latency = sum(r["latency_s"] for r in evaluated) / len(evaluated)
            print(f"Average latency: {avg_latency:.2f}s")

        return evaluated


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run {AGENT_NAME} eval pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Max items to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Skip agent calls")
    parser.add_argument("--top", action="store_true", help="Sort by intent before limiting")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Regenerate HTML report from last results (not yet implemented)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max parallel agent calls",
    )
    args = parser.parse_args()

    runner = EvalRunner(
        limit=args.limit,
        dry_run=args.dry_run,
        top=args.top,
        concurrency=args.concurrency,
    )
    asyncio.run(runner.run_all())


if __name__ == "__main__":
    main()
```
