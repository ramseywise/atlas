# cap-batch — Templates for batch-runner-builder subagent

## File: {OUTPUT_DIR}/batch_runner.py

```python
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BatchRunner:
    """Process a JSONL input file through the agent with concurrency control.

    Each line of the input file must be a JSON object with at minimum a "query" key.
    Results are written as JSONL with the original item plus agent output and timing.

    Usage:
        runner = BatchRunner("data/queries.jsonl", "data/results.jsonl", concurrency=10)
        asyncio.run(runner.run_all())

    Or via CLI:
        uv run python -m {AGENT_NAME}.batch_runner --input data/queries.jsonl \\
            --output data/results.jsonl --concurrency 10
    """

    def __init__(
        self,
        input_path: str,
        output_path: str,
        concurrency: int = 10,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_items(self) -> list[dict[str, Any]]:
        """Read JSONL input file and return a list of item dicts."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        items: list[dict[str, Any]] = []
        with self.input_path.open() as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if "query" not in item:
                        logger.warning("Line %d missing 'query' key — skipping", line_no)
                        continue
                    items.append(item)
                except json.JSONDecodeError as exc:
                    logger.warning("Line %d JSON parse error: %s — skipping", line_no, exc)

        logger.info("Loaded %d items from %s", len(items), self.input_path)
        return items

    # ------------------------------------------------------------------
    # Single-item execution
    # ------------------------------------------------------------------

    async def run_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Call the agent for one item and return result dict with timing.

        Errors are captured and surfaced in the result rather than raised,
        so a single failure doesn't abort the batch.
        """
        from {AGENT_NAME}.main import AgentRunner  # noqa: PLC0415

        session_id = item.get("session_id") or str(uuid.uuid4())
        query = item["query"]
        start = time.monotonic()

        async with self._semaphore:
            try:
                runner = AgentRunner(session_id=session_id)
                response = await runner.run(query)

                return {
                    **item,
                    "session_id": session_id,
                    "response": response.model_dump(),
                    "latency_s": round(time.monotonic() - start, 3),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("Agent failed for query %r: %s", query[:80], exc)
                return {
                    **item,
                    "session_id": session_id,
                    "response": None,
                    "latency_s": round(time.monotonic() - start, 3),
                    "error": str(exc),
                }

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def run_all(self) -> list[dict[str, Any]]:
        """Process all items concurrently and write results to output JSONL.

        Returns the list of result dicts (also written to disk).
        """
        try:
            from tqdm.asyncio import tqdm_asyncio as tqdm_gather  # type: ignore[import]

            use_tqdm = True
        except ImportError:
            use_tqdm = False

        items = self.load_items()

        if not items:
            logger.warning("No items to process")
            return []

        logger.info(
            "Starting batch: %d items, concurrency=%d, output=%s",
            len(items),
            self.concurrency,
            self.output_path,
        )
        batch_start = time.monotonic()

        tasks = [self.run_item(item) for item in items]

        if use_tqdm:
            results = await tqdm_gather.gather(*tasks, desc="{AGENT_NAME} batch")
        else:
            results = await asyncio.gather(*tasks)

        elapsed = time.monotonic() - batch_start
        self.write_results(results)

        # Summary stats
        errors = sum(1 for r in results if r["error"])
        latencies = [r["latency_s"] for r in results if r["error"] is None]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        logger.info(
            "Batch complete: %d/%d succeeded in %.1fs (avg latency %.2fs per item)",
            len(results) - errors,
            len(results),
            elapsed,
            avg_latency,
        )

        if errors > 0:
            logger.warning("%d item(s) failed — check 'error' field in output", errors)

        return list(results)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_results(self, results: list[dict[str, Any]]) -> None:
        """Write results list to the output JSONL path (overwrite if exists)."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w") as fh:
            for result in results:
                fh.write(json.dumps(result, default=str) + "\n")

        logger.info("Wrote %d results to %s", len(results), self.output_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Run {AGENT_NAME} over a JSONL input file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL input file (each line: {\"query\": \"...\", ...})",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSONL results",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Maximum parallel agent calls",
    )
    args = parser.parse_args()

    runner = BatchRunner(
        input_path=args.input,
        output_path=args.output,
        concurrency=args.concurrency,
    )
    asyncio.run(runner.run_all())


if __name__ == "__main__":
    main()
```
