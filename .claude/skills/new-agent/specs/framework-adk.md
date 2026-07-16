# framework-adk.md — agent-builder spec (Google ADK)

Write the following files for the `{AGENT_NAME}` agent.
Each file path is relative to `{OUTPUT_DIR}`.

---

## File: {OUTPUT_DIR}/agent.py

```python
"""{AGENT_NAME} — Google ADK root agent.

Architecture:
  root_agent (router) → sub_agents (domain experts)

Callbacks:
  before_model_callback: _guardrail_callback — injection + escalation detection
  before_agent_callback: _before_agent_callback — per-invocation state init
  after_agent_callback on sub-agents: _grounding_callback — Layer 4 citation check

Feature flags (env vars):
  GEMINI_MODEL        str   — model name (default gemini-2.5-flash)
  THINKING_BUDGET     int   — Gemini thinking tokens (0 = off)
  RETRIEVAL_BACKEND   str   — bedrock | rag | custom (default bedrock)
  GROUNDING_ENABLED   bool  — Layer 4 citation check (default true)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

import memory as memory_store
from schema import PROMPT_VERSION, AssistantResponse
from sub_agents.domain_agent import domain_agent

log = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent / "prompts"
_INSTRUCTION = (_PROMPTS / "{AGENT_NAME}.txt").read_text()

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))

# ---------------------------------------------------------------------------
# Guardrail patterns (Layer 1)
# ---------------------------------------------------------------------------

_INJECTION_RE = re.compile(
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions"
    r"|forget\s+everything"
    r"|you\s+are\s+now\s+(a\s+)?(?!an?\s+assistant)"
    r"|system\s*:\s*you\s+are",
    re.IGNORECASE,
)

_ESCALATION_RE = re.compile(
    r"speak\s+to\s+a\s+human"
    r"|talk\s+to\s+(a\s+)?support"
    r"|this\s+isn'?t\s+working"
    r"|connect\s+me\s+(with|to)\s+(an?\s+)?(human|person|agent|support)"
    r"|I(?:\s+am|\s*'m)\s+(angry|frustrated)",
    re.IGNORECASE,
)

_ESCALATION_RESPONSE = (
    '{"message": "I\'ll connect you with a human supporter right away. '
    'Please hold — someone from the support team will be with you shortly.", '
    '"contact_support": true, "failure_reason": "user_requested_human"}'
)

_BLOCKED_RESPONSE = (
    '{"message": "I detected an unusual pattern in your message and cannot process it. '
    'Please rephrase your request.", "contact_support": true, '
    '"failure_reason": "injection_blocked"}'
)


# ---------------------------------------------------------------------------
# Router instruction (dynamic per-turn)
# ---------------------------------------------------------------------------


def provide_router_instruction(ctx: ReadonlyContext) -> str:
    state = ctx._invocation_context.session.state
    tried = state.get("tried_agents", [])
    prefs: list[dict] = state.get("user_preferences", [])

    parts: list[str] = []
    if tried:
        parts.append(f"Sub-agents already called this turn: {', '.join(tried)}. Do not call them again.")
    if prefs:
        pref_lines = "\n".join(f"  - {p['key']}: {p['value']}" for p in prefs)
        parts.append(f"User preferences (apply when relevant):\n{pref_lines}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


async def _before_agent_callback(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Clear tried_agents once per invocation. Load user preferences on first turn."""
    invocation_id = callback_context._invocation_context.invocation_id
    if callback_context.state.get("_tried_agents_invocation") != invocation_id:
        callback_context.state["tried_agents"] = []
        callback_context.state["_tried_agents_invocation"] = invocation_id

    if "user_preferences" not in callback_context.state:
        user_id = callback_context.state.get("user_id", "default")
        try:
            prefs = await memory_store.get_top(user_id)
            callback_context.state["user_preferences"] = prefs
        except Exception as e:
            log.warning("prefs-load-failed error=%s", e)
            callback_context.state["user_preferences"] = []

    return None


def _guardrail_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Layer 1: injection detection + escalation trigger.

    Returns an LlmResponse to short-circuit the model; None to allow the call.
    """
    if not llm_request.contents:
        return None

    for content in reversed(llm_request.contents):
        role = getattr(content, "role", "")
        if role != "user":
            continue
        parts = getattr(content, "parts", [])
        text = "".join(getattr(p, "text", "") or "" for p in parts)

        if _ESCALATION_RE.search(text):
            log.info("escalation-trigger-detected")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=_ESCALATION_RESPONSE)],
                )
            )

        if _INJECTION_RE.search(text):
            log.warning("injection-pattern-detected")
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=_BLOCKED_RESPONSE)],
                )
            )
        break  # only check the most recent user turn

    return None


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


async def update_user_preference(
    key: str, value: str, tool_context: Any = None
) -> dict:
    """Remember a user preference. Call when the user says 'remember that...' or 'don't forget...'."""
    user_id = "default"
    try:
        if tool_context is not None:
            user_id = tool_context.state.get("user_id", "default")
    except Exception:
        pass
    await memory_store.upsert(user_id, f"pref:{key}", value)
    try:
        if tool_context is not None:
            tool_context.state["user_preferences"] = await memory_store.get_top(user_id)
    except Exception:
        pass
    return {"success": True, "message": f"I'll remember that {key} is {value}."}


async def delete_user_preference(key: str, tool_context: Any = None) -> dict:
    """Forget a stored user preference. Call when the user says 'forget my ... preference'."""
    user_id = "default"
    try:
        if tool_context is not None:
            user_id = tool_context.state.get("user_id", "default")
    except Exception:
        pass
    await memory_store.delete(user_id, f"pref:{key}", "")
    try:
        if tool_context is not None:
            tool_context.state["user_preferences"] = await memory_store.get_top(user_id)
    except Exception:
        pass
    return {"success": True, "message": f"I've forgotten your {key} preference."}


# ---------------------------------------------------------------------------
# Root agent
# ---------------------------------------------------------------------------

_generate_config = types.GenerateContentConfig(
    temperature=0,
    max_output_tokens=150,
    thinking_config=(
        types.ThinkingConfig(thinking_budget=_THINKING_BUDGET)
        if _THINKING_BUDGET > 0 else None
    ),
)

root_agent = Agent(
    model=_MODEL,
    name="{AGENT_NAME}",
    description=(
        "Root routing agent for {AGENT_NAME}. Classifies user requests and delegates "
        "to the correct domain expert. Does not answer domain questions directly."
    ),
    generate_content_config=_generate_config,
    static_instruction=types.Content(
        role="user",
        parts=[types.Part(text=_INSTRUCTION)],
    ),
    instruction=provide_router_instruction,
    tools=[update_user_preference, delete_user_preference],
    sub_agents=[domain_agent],
    before_agent_callback=_before_agent_callback,
    before_model_callback=_guardrail_callback,
)
```

---

## File: {OUTPUT_DIR}/app.py

```python
"""{AGENT_NAME} — ADK App with context compaction for long sessions."""

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.models import Gemini

from agent import root_agent

_summarizer = LlmEventSummarizer(llm=Gemini(model="gemini-2.5-flash"))

app = App(
    name="{AGENT_NAME}",
    root_agent=root_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=10,
        overlap_size=2,
        summarizer=_summarizer,
    ),
)
```

---

## File: {OUTPUT_DIR}/sub_agents/__init__.py

```python
"""Sub-agent registry for {AGENT_NAME}.

Import domain agents here and add them to root_agent.sub_agents in agent.py.
"""
```

---

## File: {OUTPUT_DIR}/sub_agents/domain_agent.py

```python
"""{DOMAIN} domain expert sub-agent for {AGENT_NAME}.

This is a template — rename this file and specialize the instruction and tools
for each domain your agent handles. Register the agent in agent.py.

Pattern:
  1. Add tools that retrieve or transform data for this domain.
  2. Wire a _grounding_callback to enforce Layer 4 citation check.
  3. Set output_schema=AssistantResponse and output_key="response".
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import FunctionTool
from google.genai import types

from schema import AssistantResponse

log = logging.getLogger(__name__)

_INSTRUCTION = (
    Path(__file__).parent.parent / "prompts" / "answer.txt"
).read_text()

_GROUNDING_ENABLED = os.getenv("GROUNDING_ENABLED", "true").lower() == "true"


async def search_knowledge(query: str, tool_context: Any = None) -> str:
    """Search the knowledge base for this domain.

    Replace this placeholder with a real Bedrock KB or vector search call.
    Store retrieved URLs in tool_context.state["_retrieved_urls"] for grounding.
    """
    # TODO: replace with real retrieval
    return f"[Placeholder retrieval for: {query}]"


def _grounding_callback(callback_context: CallbackContext) -> types.Content | None:
    """Layer 4: after-agent citation check.

    Reads _retrieved_urls from session state (set by search_knowledge).
    If any source.url was not retrieved, rewrites to contact_support=True.
    """
    if not _GROUNDING_ENABLED:
        return None

    state = callback_context.state
    retrieved_urls: set[str] = set(state.get("_retrieved_urls") or [])
    if not retrieved_urls:
        return None

    response_dict = state.get("response")
    if not isinstance(response_dict, dict):
        return None

    try:
        response = AssistantResponse(**response_dict)
    except Exception:
        return None

    phantom = [s.url for s in response.sources if s.url and s.url not in retrieved_urls]
    if not phantom:
        return None

    log.warning(
        "grounding.layer4_fail agent=domain_agent phantom=%s retrieved=%s",
        phantom,
        list(retrieved_urls)[:5],
    )
    rewritten = AssistantResponse(
        message=(
            "I wasn't able to find a verified answer in the documentation. "
            "Please contact support for assistance."
        ),
        contact_support=True,
        suggestions=["Contact support", "Search help center"],
    )
    state["response"] = rewritten.model_dump()
    return None


_THINKING_CONFIG = types.GenerateContentConfig(
    temperature=0.2,
    max_output_tokens=2048,
)

domain_agent = Agent(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    name="{DOMAIN}_agent",
    description=(
        "Domain expert for {DOMAIN} queries. Answers how-to questions by searching "
        "the knowledge base. Fallback for any {DOMAIN} requests."
    ),
    static_instruction=types.Content(
        role="user", parts=[types.Part(text=_INSTRUCTION)]
    ),
    output_schema=AssistantResponse,
    output_key="response",
    tools=[FunctionTool(func=search_knowledge)],
    generate_content_config=_THINKING_CONFIG,
    after_agent_callback=_grounding_callback,
)
```

---

## File: {OUTPUT_DIR}/prompts/{AGENT_NAME}.txt

```
You are {AGENT_NAME}, a routing assistant.

Your job is to classify the user's request and delegate to the correct domain expert.
You do NOT answer domain questions directly — you route to the specialist.

When routing:
1. Identify the user's primary intent (one clear domain).
2. Delegate to the appropriate sub-agent.
3. If the request is ambiguous, delegate to the closest match rather than asking.
4. If the user asks for a human or expresses frustration, respond directly with
   contact_support=true. Do not route to a sub-agent.

Keep your routing decision silent — do not explain it to the user.
The sub-agent's response is the user-visible output.
```
