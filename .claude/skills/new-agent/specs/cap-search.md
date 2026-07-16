# cap-search — Templates for search-builder subagent

## File: {OUTPUT_DIR}/search_tool.py

```python
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from {AGENT_NAME}.schema import Source

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"
_TAVILY_URL = "https://api.tavily.com/search"
_RATE_LIMIT_DELAY = 1.0  # seconds between requests
_REQUEST_TIMEOUT = 15.0  # seconds


# ---------------------------------------------------------------------------
# Search tool
# ---------------------------------------------------------------------------


class SearchTool:
    """Web search via Serper or Tavily, normalised to list[Source].

    Provider is controlled by the SEARCH_PROVIDER env var:
        - "serper"  (default) — uses SERPER_API_KEY
        - "tavily"             — uses TAVILY_API_KEY

    Usage:
        tool = SearchTool()
        sources = await tool.search("sample search query", n=5)
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._provider = (provider or os.environ.get("SEARCH_PROVIDER", "serper")).lower()
        self._api_key = api_key or self._load_api_key()
        self._last_request_time: float = 0.0

    def _load_api_key(self) -> str:
        if self._provider == "serper":
            key = os.environ.get("SERPER_API_KEY", "")
        elif self._provider == "tavily":
            key = os.environ.get("TAVILY_API_KEY", "")
        else:
            raise ValueError(f"Unknown search provider: {self._provider!r}")

        if not key:
            logger.warning(
                "No API key found for provider %r — searches will fail", self._provider
            )
        return key

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def search(self, query: str, n: int = 5) -> list[Source]:
        """Run a web search and return up to n Source objects.

        Enforces 1 req/sec rate limiting.
        """
        await self._rate_limit()

        if self._provider == "serper":
            raw = await self._serper_search(query, n)
            return self._normalise_serper(raw, n)
        elif self._provider == "tavily":
            raw = await self._tavily_search(query, n)
            return self._normalise_tavily(raw, n)
        else:
            raise ValueError(f"Unknown provider: {self._provider!r}")

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    async def _rate_limit(self) -> None:
        """Enforce a minimum gap of _RATE_LIMIT_DELAY seconds between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < _RATE_LIMIT_DELAY:
            await asyncio.sleep(_RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Serper backend
    # ------------------------------------------------------------------

    async def _serper_search(self, query: str, n: int) -> dict[str, Any]:
        """POST to Serper API and return raw JSON."""
        payload = {"q": query, "num": min(n, 10), "gl": "de", "hl": "de"}
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(_SERPER_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

    def _normalise_serper(self, raw: dict[str, Any], n: int) -> list[Source]:
        """Map Serper organic results to Source objects."""
        sources: list[Source] = []

        for item in raw.get("organic", [])[:n]:
            title = item.get("title", "")
            url = item.get("link", "")
            snippet = item.get("snippet", "")

            if not url:
                continue

            try:
                sources.append(
                    Source(
                        title=title or url,
                        url=url,
                        snippet=snippet or None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed Serper result: %s — %s", url, exc)

        return sources

    # ------------------------------------------------------------------
    # Tavily backend
    # ------------------------------------------------------------------

    async def _tavily_search(self, query: str, n: int) -> dict[str, Any]:
        """POST to Tavily API and return raw JSON."""
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": min(n, 10),
            "search_depth": "basic",
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.post(_TAVILY_URL, json=payload)
            response.raise_for_status()
            return response.json()

    def _normalise_tavily(self, raw: dict[str, Any], n: int) -> list[Source]:
        """Map Tavily results to Source objects."""
        sources: list[Source] = []

        for item in raw.get("results", [])[:n]:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("content", "")

            if not url:
                continue

            try:
                sources.append(
                    Source(
                        title=title or url,
                        url=url,
                        snippet=snippet[:300] if snippet else None,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping malformed Tavily result: %s — %s", url, exc)

        return sources
```
