# core.md — schema-builder spec

Write the following files for the `{AGENT_NAME}` agent.
Each file path is relative to `{OUTPUT_DIR}`.

---

## File: {OUTPUT_DIR}/schema.py

```python
"""Output schema for {AGENT_NAME} agent.

AssistantResponse is the eval pipeline contract — callers read
message, sources, suggestions, contact_support. All agents must return
a dict serialised from this model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

PROMPT_VERSION = "{AGENT_NAME}-v1"


class FailureReason:
    """Structured escalation/failure taxonomy.

    Set failure_reason whenever contact_support=True or the response is a fallback.
    Enables the EscalationGrader to distinguish infrastructure errors, content gaps,
    and user-initiated escalations rather than treating all as equivalent.
    """

    # Layer 1 — input blocked by injection or unicode sanitizer
    INJECTION_BLOCKED = "injection_blocked"

    # Infrastructure — retrieval backend raised an exception
    RETRIEVAL_BACKEND_ERROR = "retrieval_backend_error"

    # Retrieval quality — KB / vector search returned nothing
    NO_RETRIEVAL_RESULTS = "no_retrieval_results"

    # Retrieval quality — results returned but scores below confidence threshold
    LOW_RETRIEVAL_SCORES = "low_retrieval_scores"

    # Retrieval quality — reranker returned nothing after filtering
    NO_RERANK_RESULTS = "no_rerank_results"

    # Quality gate — overall confidence too low after retrieval + rerank
    LOW_CONFIDENCE = "low_confidence"

    # Content gap — docs retrieved but LLM judged topic not covered
    DOCUMENTATION_NOT_COVERED = "documentation_not_covered"

    # User intent — regex / LLM classified query as explicit human-handoff request
    USER_REQUESTED_HUMAN = "user_requested_human"

    # Post-answer quality — LLM post-answer evaluator rejected the generated answer
    POST_ANSWER_FAILED = "post_answer_evaluator"

    # Catch-all — agent returned unclassified fallback
    UNKNOWN = "unknown"


class Source(BaseModel):
    title: str = Field(description="Article or page title.")
    url: str = Field(description="Full URL to the source article.")


class ClaimItem(BaseModel):
    """A single cited claim with its verbatim supporting quote. Used for Tier 3 grounding."""

    supporting_quote: str = Field(description="Verbatim excerpt from the cited passage.")
    citations: list[str] = Field(
        default_factory=list,
        description="Source URLs cited for this specific claim.",
    )


class PassageItem(BaseModel):
    text: str
    url: str = ""
    title: str = ""
    score: float = 0.0


class AssistantResponse(BaseModel):
    message: str = Field(
        description=(
            "Main response as markdown. This is what the user reads. "
            "Never include raw JSON or schema markers here."
        )
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="2-4 suggested follow-up questions shown as clickable chips.",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Source article links. Populate from retrieved passage URLs.",
    )
    contact_support: bool = Field(
        default=False,
        description="True when the user needs to speak to a human support agent.",
    )
    relevance_score: float | None = Field(
        default=None,
        description=(
            "0.0-1.0 self-assessed relevance of the retrieved passages to the question. "
            "Omit when not using the search tool (greetings, escalations). "
            "Below 0.5 → prefer contact_support=true over a speculative answer."
        ),
    )
    citations: list[str] = Field(
        default_factory=list,
        description=(
            "All source URLs cited across this response. "
            "Populated when the agent uses verbatim citation mode (Tier 3 grounding)."
        ),
    )
    claims: list[ClaimItem] = Field(
        default_factory=list,
        description=(
            "Per-claim verbatim quote extractions. "
            "Each claim maps a supporting_quote to the source URLs it came from. "
            "Enables Tier 3 quote-coverage grounding check."
        ),
    )
    passages: list[PassageItem] = Field(
        default_factory=list,
        description=(
            "Retrieved passage chunks used to generate this response. "
            "Populated for eval runs — enables RAGAS faithfulness + context precision grading. "
            "Empty in production to reduce payload size."
        ),
    )
    failure_reason: str | None = Field(
        default=None,
        description=(
            "Machine-readable failure reason when contact_support=True or response is a fallback. "
            "One of the FailureReason constants. Null when the agent answered normally."
        ),
    )
```

---

## File: {OUTPUT_DIR}/observability.py

```python
"""Logging and LangSmith / LangChain tracing setup.

Call configure_runtime() once at process start before invoking the graph
so traces and log levels apply consistently.
"""

from __future__ import annotations

import logging
import os
import sys

from config import (
    LANGCHAIN_API_KEY,
    LANGCHAIN_ENDPOINT,
    LANGCHAIN_PROJECT,
    LANGCHAIN_TRACING_V2,
    LOG_LEVEL,
)

_configured = False


def _configure_root_logging(level: int, *, force: bool = False) -> None:
    """One stdout handler on the root logger; plain format."""
    root = logging.getLogger()
    if not force and root.handlers:
        return
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )
    root.addHandler(handler)


def configure_runtime() -> None:
    """Configure root logging and ensure LangSmith env vars are visible.

    Safe to call multiple times; only the first call has effect.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    if not isinstance(level, int):
        level = logging.INFO

    _configure_root_logging(level, force=True)

    # Suppress HF Hub noise before any lazy import of sentence-transformers.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.ERROR)

    log = logging.getLogger(__name__)

    if LANGCHAIN_TRACING_V2:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        if LANGCHAIN_ENDPOINT:
            os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGCHAIN_ENDPOINT)
        if LANGCHAIN_API_KEY:
            os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
        if LANGCHAIN_PROJECT:
            os.environ.setdefault("LANGCHAIN_PROJECT", LANGCHAIN_PROJECT)

        if LANGCHAIN_API_KEY:
            log.info(
                "LangSmith tracing enabled (project=%s)",
                LANGCHAIN_PROJECT or "(default project)",
            )
        else:
            log.warning(
                "LANGCHAIN_TRACING_V2 is enabled but LANGCHAIN_API_KEY is not set — "
                "LangSmith runs may fail. Set LANGCHAIN_API_KEY in your environment.",
            )
    else:
        log.debug(
            "LangSmith tracing off (set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable)",
        )
```

---

## File: {OUTPUT_DIR}/memory.py

```python
"""In-memory conversation history for {AGENT_NAME}.

Stores turn-level history per session_id. Not persisted across restarts —
eval sessions are short-lived and independent. To add persistence, swap
_store for a SQLite or Redis backend using the same interface.
"""

from __future__ import annotations

from collections import defaultdict

_store: dict[str, list[dict]] = defaultdict(list)


def append(session_id: str, role: str, content: str) -> None:
    _store[session_id].append({"role": role, "content": content})


def get_history(session_id: str) -> list[dict]:
    return list(_store[session_id])


def clear(session_id: str) -> None:
    _store.pop(session_id, None)
```

---

## File: {OUTPUT_DIR}/config.py

```python
"""Environment-variable config for {AGENT_NAME}.

All settings are read at import time so they are visible immediately to
submodules that import them. Add agent-specific vars below the shared block.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Shared observability / tracing
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY: str | None = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT: str | None = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT: str | None = os.getenv("LANGCHAIN_PROJECT", "{AGENT_NAME}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
THINKING_BUDGET: int = int(os.getenv("THINKING_BUDGET", "0"))

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

GROUNDING_ENABLED: bool = os.getenv("GROUNDING_ENABLED", "true").lower() == "true"
RETRIEVAL_BACKEND: str = os.getenv("RETRIEVAL_BACKEND", "bedrock")  # bedrock | rag | custom

# ---------------------------------------------------------------------------
# Agent-specific — add {AGENT_NAME} vars here
# ---------------------------------------------------------------------------
```

---

## File: {OUTPUT_DIR}/prompts/answer.txt

```
You are a helpful {DOMAIN} assistant. Answer the user's question clearly and concisely
using ONLY the documentation passages provided below.

Rules:
1. Base your answer exclusively on the provided passages. Do not invent facts.
2. If the passages do not contain enough information, set contact_support=true.
3. Include 2-4 relevant follow-up suggestions in the "suggestions" field.
4. Populate "sources" with the URLs from passages you actually used.
5. Set relevance_score between 0.0 and 1.0 reflecting how well the passages match.

{history_block}
```

---

## File: {OUTPUT_DIR}/prompts/clarify.txt

```
You are a helpful {DOMAIN} assistant. The user's message is too vague to search for
a precise answer.

Ask exactly ONE targeted clarifying question to understand what they need.
- Be specific: ask about the concrete action, feature, or error they are dealing with.
- Do not ask multiple questions in one message.
- Set contact_support=false and suggestions=[] when clarifying.

{history_block}
```

---

## File: {OUTPUT_DIR}/prompts/escalate.txt

```
You are a helpful {DOMAIN} assistant. The user needs to speak with a human support agent.

Respond empathetically. Acknowledge their frustration or need without dismissing it.
Keep your message under 3 sentences.

Set contact_support=true and failure_reason to the appropriate FailureReason constant.
Do not include sources or suggestions.
```
