# cap-a2a — Templates for agent-to-agent-builder subagent

## File: {OUTPUT_DIR}/a2a_client.py

```python
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_MAX_RETRIES = 3
_BACKOFF_BASE = 1.0  # seconds; retry delay = _BACKOFF_BASE * 2^attempt


class A2AClient:
    """HTTP client for agent-to-agent communication.

    Agents expose a POST /a2a/{agent_name} endpoint (see a2a_router.py).
    This client handles:
    - Request dispatch with shared-secret auth header
    - Parallel broadcast to multiple agents
    - Exponential backoff retry (max 3 attempts)
    - 30-second timeout per request

    Usage:
        client = A2AClient(base_url="http://localhost:8080")
        response = await client.send("billing_agent", {"query": "What is my balance?"})
        responses = await client.broadcast(["billing_agent", "tax_agent"], payload)
    """

    def __init__(
        self,
        base_url: str,
        secret: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._secret = secret or os.environ.get("A2A_SECRET", "")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Single send
    # ------------------------------------------------------------------

    async def send(
        self,
        target_agent: str,
        payload: dict[str, Any],
        *,
        retries: int = _MAX_RETRIES,
    ) -> dict[str, Any]:
        """POST payload to /a2a/{target_agent} with retry and backoff.

        Args:
            target_agent: Name of the destination agent.
            payload: Dict to send as JSON body.
            retries: Number of retry attempts on transient failures.

        Returns:
            Parsed JSON response dict from the target agent.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors (4xx).
            RuntimeError: If all retry attempts are exhausted.
        """
        url = f"{self._base_url}/a2a/{target_agent}"
        headers = self._build_headers()
        last_exc: Exception | None = None

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    # 4xx errors are not retried (caller error)
                    if 400 <= response.status_code < 500:
                        response.raise_for_status()

                    # 5xx are retried
                    if response.status_code >= 500:
                        raise httpx.HTTPStatusError(
                            f"Server error {response.status_code}",
                            request=response.request,
                            response=response,
                        )

                    return response.json()

            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < retries - 1:
                    delay = _BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "A2A send to %r failed (attempt %d/%d): %s — retrying in %.1fs",
                        target_agent,
                        attempt + 1,
                        retries,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"A2A send to {target_agent!r} failed after {retries} attempts: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Broadcast
    # ------------------------------------------------------------------

    async def broadcast(
        self,
        agents: list[str],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Send the same payload to multiple agents in parallel.

        Args:
            agents: List of agent names to broadcast to.
            payload: Payload to send to each agent.

        Returns:
            List of response dicts in the same order as agents.
            Failed agents return {"error": "<message>", "agent": "<name>"}.
        """
        tasks = [self.send(agent, payload) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        responses: list[dict[str, Any]] = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                logger.error("Broadcast to %r failed: %s", agent, result)
                responses.append({"error": str(result), "agent": agent})
            else:
                responses.append(result)  # type: ignore[arg-type]

        return responses

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._secret:
            headers["X-A2A-Secret"] = self._secret
        return headers
```

## File: {OUTPUT_DIR}/a2a_router.py

```python
from __future__ import annotations

import logging
import os
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from {AGENT_NAME}.schema import AssistantResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_SHARED_SECRET = os.environ.get("A2A_SECRET", "")

security = HTTPBearer(auto_error=False)


def _verify_secret(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Validate the shared secret from X-A2A-Secret or Bearer token header."""
    if not _SHARED_SECRET:
        # Secret not configured — skip auth (development mode)
        logger.debug("A2A auth: no secret configured, skipping verification")
        return

    token: str | None = None

    if credentials:
        token = credentials.credentials
    else:
        return  # No credentials provided and secret not set — allow

    if token != _SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid A2A secret")


# ---------------------------------------------------------------------------
# Request / response schema
# ---------------------------------------------------------------------------


class A2ARequest(BaseModel):
    query: str
    session_id: str | None = None
    context: dict[str, Any] | None = None
    source_agent: str | None = None


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Callable] = {}


def register_handler(agent_name: str, handler: Callable) -> Callable:
    """Decorator to register a handler for incoming A2A messages.

    Usage:
        @register_handler("billing_agent")
        async def handle_billing(request: A2ARequest) -> AssistantResponse:
            ...
    """
    _HANDLERS[agent_name] = handler
    logger.debug("Registered A2A handler for agent: %s", agent_name)
    return handler


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/a2a", tags=["a2a"])


@router.post("/{agent_name}")
async def receive_a2a_message(
    agent_name: str,
    request: A2ARequest,
    _auth: None = Depends(_verify_secret),
) -> AssistantResponse:
    """Receive an A2A message and route it to the registered handler.

    Authentication: pass the shared secret in one of:
    - X-A2A-Secret header
    - Authorization: Bearer <secret> header

    If no handler is registered for agent_name, returns 404.
    If the handler raises, returns 500 with error detail.
    """
    handler = _HANDLERS.get(agent_name)

    if handler is None:
        available = list(_HANDLERS)
        logger.warning(
            "No A2A handler for %r. Registered: %s", agent_name, available
        )
        raise HTTPException(
            status_code=404,
            detail=f"No handler registered for agent {agent_name!r}. "
                   f"Available: {available}",
        )

    logger.info(
        "A2A message received for %r from %r: %r",
        agent_name,
        request.source_agent,
        request.query[:80],
    )

    try:
        result = await handler(request)
    except Exception as exc:  # noqa: BLE001
        logger.error("A2A handler for %r raised: %s", agent_name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not isinstance(result, AssistantResponse):
        raise TypeError(
            f"A2A handler for {agent_name!r} must return AssistantResponse, "
            f"got {type(result).__name__}"
        )

    return result


# ---------------------------------------------------------------------------
# Mount example — add this to your FastAPI app in app.py:
#
#   from {AGENT_NAME}.a2a_router import router as a2a_router, register_handler
#   app.include_router(a2a_router)
#
#   @register_handler("{AGENT_NAME}")
#   async def handle_incoming(request: A2ARequest) -> AssistantResponse:
#       runner = AgentRunner(session_id=request.session_id or str(uuid.uuid4()))
#       return await runner.run(request.query)
# ---------------------------------------------------------------------------
```
