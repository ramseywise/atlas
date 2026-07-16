# framework-lg.md — agent-builder spec (LangGraph)

Write the following files for the `{AGENT_NAME}` agent.
Each file path is relative to `{OUTPUT_DIR}`.

---

## File: {OUTPUT_DIR}/state.py

```python
"""LangGraph state for the {AGENT_NAME} agent."""

from __future__ import annotations

from typing import TypedDict


# CRITICAL: Every key that a node reads or writes MUST be declared here.
# LangGraph silently drops keys that are not declared in the TypedDict —
# including control-flow keys like max_cycles or cycle_count. If a node
# calls state.get("max_cycles", 3) and the key is not declared, LangGraph
# will drop it between nodes and the default will always be used.
class State(TypedDict):
    query: str
    session_id: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": str}]
    intent: str                  # "answerable" | "clarification" | "escalation"
    passages: list[dict]         # raw retrieved docs: {text, url, title, score}
    good_passages: list[dict]    # CRAG-graded relevant passages
    retrieval_attempts: int      # how many fetch+grade loops ran
    response: dict | None        # serialised AssistantResponse
    # Ablation trace fields
    confidence_score: float      # top passage score after retrieval
    post_eval_verdict: str       # accept | refine | escalate
    post_eval_attempts: int      # number of refine loops triggered (caps at 1)
    # Failure taxonomy — see FailureReason in schema.py
    failure_reason: str | None   # set at each escalation decision point
    # Control flow — must be declared even if set only in initial state
    max_cycles: int
    cycle_count: int
    terminate: bool
    error: str | None
```

---

## File: {OUTPUT_DIR}/agent.py

```python
"""{AGENT_NAME} — LangGraph single-agent.

Graph topology:
  planner → { retrieve → hitl_gate → answer → post_answer_eval → grounding_check | respond }

  planner classifies intent:
    answerable    → retrieve docs → synthesise answer
    clarification → ask one targeted question (no retrieval)
    escalation    → contact_support response (no retrieval)

Feature flags (env vars):
  GEMINI_MODEL                  str   — model name (default gemini-2.5-flash)
  THINKING_BUDGET               int   — thinking tokens (0 = off)
  LLM_PLANNER                   bool  — replace regex router with LLM call (default false)
  ROUTING_CONFIDENCE_THRESHOLD  float — gate low-confidence LLM routing (default 0.0)
  HITL_GATES_ENABLED            bool  — low-confidence retrieval → escalate (default false)
  HITL_CONFIDENCE_THRESHOLD     float — confidence gate threshold (default 0.3)
  POST_ANSWER_EVAL_ENABLED      bool  — LLM quality judge after answer (default false)
  LG_MEMORY_TURNS               int   — conversation history turns to include (default 3)
"""

from __future__ import annotations

import logging
import os
import re

from langchain_core.callbacks.usage import UsageMetadataCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from schema import AssistantResponse, FailureReason
from state import State
from subgraphs.retrieval import retrieval_subgraph

_log = logging.getLogger(__name__)

_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))
_LLM_PLANNER = os.getenv("LLM_PLANNER", "false").lower() == "true"
_ROUTING_CONFIDENCE_THRESHOLD = float(os.getenv("ROUTING_CONFIDENCE_THRESHOLD", "0.0"))
_MEMORY_TURNS = int(os.getenv("LG_MEMORY_TURNS", "3"))
_HITL_GATES_ENABLED = os.getenv("HITL_GATES_ENABLED", "false").lower() == "true"
_HITL_CONFIDENCE_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.3"))
_POST_ANSWER_EVAL_ENABLED = os.getenv("POST_ANSWER_EVAL_ENABLED", "false").lower() == "true"
_GROUNDING_ENABLED = os.getenv("GROUNDING_ENABLED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Deterministic planner — no LLM call, no cost, fully testable.
# ---------------------------------------------------------------------------

_ESCALATION_RE = re.compile(
    r"speak\s+to\s+a?\s*human"
    r"|talk\s+to\s+(a\s+)?support"
    r"|this\s+isn'?t\s+working"
    r"|you'?re?\s+useless"
    r"|connect\s+me\s+(with|to)\s+(an?\s+)?(human|person|agent|support)"
    r"|I(?:\s+am|\s*'m)\s+(angry|frustrated)",
    re.IGNORECASE,
)

_VAGUE_RE = re.compile(
    r"^(help|problem|issue|question|something|stuff|things?|"
    r"hi|hello|hey)\s*[.?!]?$",
    re.IGNORECASE,
)

_MIN_WORDS = 3

_PLANNER_PROMPT = """\
Classify this user message into one of three intents:
- answerable    : a specific question that can be answered from documentation
- clarification : too vague to search; need one targeted follow-up question
- escalation    : user is frustrated, angry, or explicitly asking for a human agent

Reply with exactly one word: answerable, clarification, or escalation.

Message: {query}"""

_PLANNER_PROMPT_SCORED = """\
Classify this user message into one of three intents:
- answerable    : a specific question that can be answered from documentation
- clarification : too vague to search; need one targeted follow-up question
- escalation    : user is frustrated, angry, or explicitly asking for a human agent

Reply with JSON only: {{"intent": "answerable|clarification|escalation", "confidence": 0.0-1.0}}

Message: {query}"""

_POST_EVAL_PROMPT = """\
You are a support answer quality evaluator.

User question: {query}
Generated answer: {answer}
Retrieved passages: {passages_ctx}

Rate the answer quality with exactly one word:
- accept   : accurate, grounded in the passages, and helpful
- refine   : could be improved with better retrieval (only on first attempt)
- escalate : cannot be answered from available documentation

Reply with one word only."""


def _classify(query: str) -> str:
    """Route without an LLM: escalation > vague > answerable."""
    q = query.strip()
    if _ESCALATION_RE.search(q):
        return "escalation"
    if len(q.split()) < _MIN_WORDS or _VAGUE_RE.match(q):
        return "clarification"
    return "answerable"


async def _llm_classify(query: str) -> tuple[str, float]:
    import json as _json
    llm = ChatGoogleGenerativeAI(model=_MODEL, temperature=0)
    if _ROUTING_CONFIDENCE_THRESHOLD > 0.0:
        result = await llm.ainvoke([("human", _PLANNER_PROMPT_SCORED.format(query=query))])
        try:
            data = _json.loads(result.content.strip())
            intent = data.get("intent", "answerable").lower()
            confidence = float(data.get("confidence", 1.0))
            if intent not in ("answerable", "clarification", "escalation"):
                intent = "answerable"
            return intent, confidence
        except Exception:
            return "answerable", 0.0
    else:
        result = await llm.ainvoke([("human", _PLANNER_PROMPT.format(query=query))])
        intent = result.content.strip().lower()
        if intent not in ("answerable", "clarification", "escalation"):
            intent = "answerable"
        return intent, 1.0


def _format_history(history: list[dict], n: int) -> str:
    if not history or n == 0:
        return ""
    turns = history[-(n * 2):]
    lines = [f"{t['role'].capitalize()}: {t['content'][:200]}" for t in turns]
    return "\n\nConversation history:\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def planner_node(state: State) -> State:
    if _LLM_PLANNER:
        intent, confidence = await _llm_classify(state["query"])
        if _ROUTING_CONFIDENCE_THRESHOLD > 0.0 and confidence < _ROUTING_CONFIDENCE_THRESHOLD:
            intent = "answerable"
    else:
        intent = _classify(state["query"])
    return {**state, "intent": intent}


def route_after_planner(state: State) -> str:
    intent = state.get("intent", "answerable")
    if intent in ("clarification", "escalation"):
        return "respond"
    return "retrieve"


async def answer_node(state: State) -> State:
    llm_kwargs: dict = {"model": _MODEL, "temperature": 0.2}
    if _THINKING_BUDGET > 0:
        llm_kwargs["thinking_budget"] = _THINKING_BUDGET
    llm = ChatGoogleGenerativeAI(**llm_kwargs).with_structured_output(AssistantResponse)

    all_passages = state.get("good_passages") or state.get("passages") or []
    passages = all_passages[:5]
    if passages:
        docs_text = "\n\n".join(
            f"[{i + 1}] (score={p.get('score', 0):.2f}) {p['text'][:400]}"
            f"\nTitle: {p.get('title') or 'N/A'}  Source: {p.get('url') or 'N/A'}"
            for i, p in enumerate(passages)
        )
        context = f"Documentation passages:\n{docs_text}"
    else:
        context = "No relevant documentation found."

    answer_prompt = open("prompts/answer.txt").read()
    history_ctx = _format_history(state.get("history", []), _MEMORY_TURNS)
    response: AssistantResponse = await llm.ainvoke(
        [("system", answer_prompt + history_ctx), ("human", f"{context}\n\nUser question: {state['query']}")]
    )
    return {**state, "response": response.model_dump()}


def grounding_node(state: State) -> State:
    """Layer 4 post-generation citation check."""
    if not _GROUNDING_ENABLED:
        return state
    response_dict = state.get("response")
    if not response_dict:
        return state

    all_passages = state.get("good_passages") or state.get("passages") or []
    retrieved_urls = {p.get("url") for p in all_passages if p.get("url")}

    response = AssistantResponse(**response_dict)
    phantom = [s.url for s in response.sources if s.url and s.url not in retrieved_urls]
    if phantom:
        _log.warning("grounding.layer4_fail phantom_urls=%s", phantom)
        rewritten = AssistantResponse(
            message=(
                "I wasn't able to find a verified answer in the documentation. "
                "Please contact support for assistance."
            ),
            contact_support=True,
            failure_reason=FailureReason.UNKNOWN,
            suggestions=["Contact support"],
        )
        return {**state, "response": rewritten.model_dump()}
    return state


async def respond_node(state: State) -> State:
    """Handle clarification and escalation without retrieval."""
    if state.get("intent") == "escalation":
        failure_reason = state.get("failure_reason") or FailureReason.USER_REQUESTED_HUMAN
        return {
            **state,
            "response": AssistantResponse(
                message=(
                    "I'll connect you with a human supporter right away. "
                    "Someone from the support team will be with you shortly."
                ),
                contact_support=True,
                failure_reason=failure_reason,
            ).model_dump(),
        }

    llm = ChatGoogleGenerativeAI(model=_MODEL, temperature=0.3).with_structured_output(AssistantResponse)
    clarify_prompt = open("prompts/clarify.txt").read()
    history_ctx = _format_history(state.get("history", []), _MEMORY_TURNS)
    response: AssistantResponse = await llm.ainvoke(
        [("system", clarify_prompt + history_ctx), ("human", state["query"])]
    )
    return {**state, "response": response.model_dump()}


def hitl_gate_node(state: State) -> State:
    """Compute passage confidence; escalate intent when HITL_GATES_ENABLED and confidence is low."""
    good = state.get("good_passages", [])
    passages = state.get("passages", [])
    all_p = good if good else passages
    top_score = max((p.get("score", 0.0) for p in all_p), default=0.0) if all_p else 0.0
    new_state: dict = {**state, "confidence_score": top_score}
    if _HITL_GATES_ENABLED and (not good or top_score < _HITL_CONFIDENCE_THRESHOLD):
        new_state["intent"] = "escalation"
        new_state["failure_reason"] = (
            FailureReason.NO_RETRIEVAL_RESULTS if not all_p else FailureReason.LOW_CONFIDENCE
        )
    return new_state


def route_after_hitl_gate(state: State) -> str:
    if state.get("intent") == "escalation":
        return "respond"
    return "answer"


async def post_answer_eval_node(state: State) -> State:
    """LLM quality judge — verdict drives accept/refine/escalate routing."""
    if not _POST_ANSWER_EVAL_ENABLED:
        return {**state, "post_eval_verdict": "accept"}

    response_dict = state.get("response") or {}
    llm = ChatGoogleGenerativeAI(model=_MODEL, temperature=0)
    good_passages = state.get("good_passages", [])
    passages_ctx = (
        "\n".join(f"[{i+1}] {p['text'][:200]}" for i, p in enumerate(good_passages[:3]))
        or "(none)"
    )
    prompt = _POST_EVAL_PROMPT.format(
        query=state["query"],
        answer=response_dict.get("message", ""),
        passages_ctx=passages_ctx,
    )
    result = await llm.ainvoke([("human", prompt)])
    verdict = result.content.strip().lower()
    if verdict not in ("accept", "refine", "escalate"):
        verdict = "accept"

    new_state = {**state, "post_eval_verdict": verdict}
    if verdict == "escalate":
        new_state["intent"] = "escalation"
        new_state["failure_reason"] = FailureReason.POST_ANSWER_FAILED
    if verdict == "refine":
        new_state["post_eval_attempts"] = state.get("post_eval_attempts", 0) + 1
    return new_state


def route_after_post_answer_eval(state: State) -> str:
    verdict = state.get("post_eval_verdict", "accept")
    if verdict == "refine" and state.get("post_eval_attempts", 0) <= 1:
        return "retrieve"
    if verdict == "escalate":
        return "respond"
    return "grounding_check"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(State)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve", retrieval_subgraph)
    builder.add_node("hitl_gate", hitl_gate_node)
    builder.add_node("answer", answer_node)
    builder.add_node("post_answer_eval", post_answer_eval_node)
    builder.add_node("grounding_check", grounding_node)
    builder.add_node("respond", respond_node)

    builder.set_entry_point("planner")

    # IMPORTANT: Every tool the router may return MUST have a corresponding
    # graph node wired up. A routing lambda that only branches on one condition
    # (e.g. _has_forecast) will silently drop requests for other tools. List
    # all possible tool names explicitly in the conditional_edges map.
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"retrieve": "retrieve", "respond": "respond"},
    )
    builder.add_edge("retrieve", "hitl_gate")
    builder.add_conditional_edges(
        "hitl_gate",
        route_after_hitl_gate,
        {"answer": "answer", "respond": "respond"},
    )
    builder.add_edge("answer", "post_answer_eval")
    builder.add_conditional_edges(
        "post_answer_eval",
        route_after_post_answer_eval,
        {"retrieve": "retrieve", "respond": "respond", "grounding_check": "grounding_check"},
    )
    builder.add_edge("grounding_check", END)
    builder.add_edge("respond", END)
    return builder.compile()


compiled = build_graph()


class AgentRunner:
    """Synchronous and async entry points for the compiled graph."""

    async def arun(
        self,
        query: str,
        session_id: str = "default",
        history: list[dict] | None = None,
    ) -> tuple[AssistantResponse, list[dict], dict]:
        token_cb = UsageMetadataCallbackHandler()
        result = await compiled.ainvoke(
            {
                "query": query,
                "session_id": session_id,
                "history": history or [],
                "intent": "",
                "passages": [],
                "good_passages": [],
                "retrieval_attempts": 0,
                "response": None,
                "confidence_score": 0.0,
                "post_eval_verdict": "accept",
                "post_eval_attempts": 0,
                "failure_reason": None,
            },
            config={"callbacks": [token_cb]},
        )
        passages = result.get("good_passages") or result.get("passages") or []
        kb_top_score = result.get("confidence_score", 0.0)

        prompt_tokens = output_tokens = cached_tokens = 0
        for usage in (token_cb.usage_metadata or {}).values():
            prompt_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
            cached_tokens += (usage.get("input_token_details") or {}).get("cache_read", 0)
        cached_pct = round(cached_tokens / prompt_tokens * 100) if prompt_tokens else 0
        _log.info(
            "[tokens] prompt=%d cached=%d (%d%%) output=%d",
            prompt_tokens, cached_tokens, cached_pct, output_tokens,
        )

        state_info = {
            "confidence_score": kb_top_score,
            "kb_top_score": kb_top_score,
            "retrieval_attempts": result.get("retrieval_attempts", 0),
            "post_eval_verdict": result.get("post_eval_verdict", "accept"),
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
        }
        return AssistantResponse(**result["response"]), passages, state_info

    def run(
        self,
        query: str,
        session_id: str = "default",
        history: list[dict] | None = None,
    ) -> tuple[AssistantResponse, list[dict], dict]:
        import asyncio
        return asyncio.run(self.arun(query, session_id, history))
```

---

## File: {OUTPUT_DIR}/subgraphs/__init__.py

```python
```

---

## File: {OUTPUT_DIR}/subgraphs/retrieval.py

```python
"""CRAG retrieval subgraph for {AGENT_NAME}.

fetch → grade → decision → (rewrite → fetch)*

Env vars:
  RETRIEVAL_BACKEND     bedrock (default) | rag | custom
  CRAG_ENABLED          true (default) | false — disable grade/rewrite loop
  CRAG_HIGH_CONFIDENCE  float — skip grading when top score exceeds threshold (default 0.7)
  CRAG_MAX_RETRIES      int   — max rewrite+fetch cycles (default 1)
  MULTI_QUERY           bool  — expand query to 2 variants before fetching (default false)
  HC_RAG_AGENT_URL      str   — RAG service URL (default http://localhost:8013)
"""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

_BACKEND = os.getenv("RETRIEVAL_BACKEND", "bedrock")
_HC_RAG_URL = os.getenv("HC_RAG_AGENT_URL", "http://localhost:8013")
_GRADE_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
_CRAG_ENABLED = os.getenv("CRAG_ENABLED", "true").lower() == "true"
_MULTI_QUERY = os.getenv("MULTI_QUERY", "false").lower() == "true"
_CONTEXT_DEDUP = os.getenv("CONTEXT_DEDUP", "true").lower() == "true"

MAX_RETRIES = int(os.getenv("CRAG_MAX_RETRIES", "1"))
MIN_GOOD_PASSAGES = 2
HIGH_CONFIDENCE_THRESHOLD = float(os.getenv("CRAG_HIGH_CONFIDENCE", "0.7"))


class RetrievalState(TypedDict):
    query: str
    passages: list[dict]
    good_passages: list[dict]
    retrieval_attempts: int


async def _expand_queries(query: str) -> list[str]:
    llm = ChatGoogleGenerativeAI(model=_GRADE_MODEL, temperature=0.1)
    prompt = (
        "Generate exactly 2 focused, keyword-rich search phrases for this question.\n"
        "Output ONLY the phrases, one per line, no numbering, no explanations.\n"
        f"Question: {query}"
    )
    result = await llm.ainvoke([("human", prompt)])
    variants = [line.strip() for line in result.content.strip().splitlines() if line.strip()]
    return [query] + variants[:2]


async def _fetch_bedrock(query: str) -> list[dict]:
    from clients import bedrock_kb
    queries = await _expand_queries(query) if _MULTI_QUERY else [query]
    passages = await bedrock_kb.retrieve_reranking(queries)
    return [{"text": p.text, "url": p.url, "title": p.title, "score": p.score} for p in passages]


async def _fetch_rag(query: str) -> list[dict]:
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_HC_RAG_URL}/api/v1/retrieval",
            json={"thread_id": "crag_eval", "query": query},
        )
        resp.raise_for_status()
        data = resp.json()
    passages = []
    for doc in data.get("documents", []):
        chunk = doc.get("chunk", {})
        meta = chunk.get("metadata", {})
        passages.append({
            "text": chunk.get("text", ""),
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "score": doc.get("score", 0.0),
        })
    return passages


async def fetch_node(state: RetrievalState) -> RetrievalState:
    if _BACKEND == "rag":
        passages = await _fetch_rag(state["query"])
    else:
        passages = await _fetch_bedrock(state["query"])
    if _CONTEXT_DEDUP:
        seen: set[str] = set()
        deduped = []
        for p in passages:
            url = p.get("url") or ""
            if url not in seen:
                seen.add(url)
                deduped.append(p)
        passages = deduped
    return {
        **state,
        "passages": passages,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


async def grade_node(state: RetrievalState) -> RetrievalState:
    passages = state.get("passages", [])
    if not passages:
        return {**state, "good_passages": []}

    llm = ChatGoogleGenerativeAI(model=_GRADE_MODEL, temperature=0)
    passages_text = "\n\n".join(f"[{i}] {p['text'][:400]}" for i, p in enumerate(passages))
    prompt = (
        "Does each passage contain information relevant to answering the question?\n"
        f"Question: {state['query']}\n\n{passages_text}\n\n"
        f"Reply with exactly {len(passages)} comma-separated verdicts in order: YES or NO only.\n"
        "Example: YES, NO, YES"
    )
    result = await llm.ainvoke([("human", prompt)])
    raw = result.content if isinstance(result.content, str) else " ".join(
        (p.get("text", "") if isinstance(p, dict) else str(p)) for p in result.content
    )
    verdicts = [v.strip().upper() for v in raw.split(",")]
    good = [p for p, v in zip(passages, verdicts) if v.startswith("YES")]
    return {**state, "good_passages": good}


def confidence_gate(state: RetrievalState) -> str:
    passages = state.get("passages", [])
    if not passages:
        return "grade"
    top_score = max((p.get("score", 0.0) for p in passages), default=0.0)
    return "end" if top_score >= HIGH_CONFIDENCE_THRESHOLD else "grade"


def decision_node(state: RetrievalState) -> str:
    good = state.get("good_passages", [])
    attempts = state.get("retrieval_attempts", 0)
    if len(good) >= MIN_GOOD_PASSAGES or attempts >= MAX_RETRIES:
        return "end"
    return "rewrite"


async def rewrite_node(state: RetrievalState) -> RetrievalState:
    llm = ChatGoogleGenerativeAI(model=_GRADE_MODEL, temperature=0.3)
    prompt = (
        "The following search query did not return useful results from the knowledge base.\n"
        f"Original query: {state['query']}\n\n"
        "Rewrite it as a more specific, keyword-rich search query (one sentence, no preamble)."
    )
    result = await llm.ainvoke([("human", prompt)])
    rewritten = result.content.strip().strip('"')
    return {**state, "query": rewritten, "good_passages": []}


_sg = StateGraph(RetrievalState)
_sg.add_node("fetch", fetch_node)

if _CRAG_ENABLED:
    _sg.add_node("grade", grade_node)
    _sg.add_node("rewrite", rewrite_node)
    _sg.set_entry_point("fetch")
    _sg.add_conditional_edges("fetch", confidence_gate, {"end": END, "grade": "grade"})
    _sg.add_conditional_edges("grade", decision_node, {"end": END, "rewrite": "rewrite"})
    _sg.add_edge("rewrite", "fetch")
else:
    _sg.set_entry_point("fetch")
    _sg.add_edge("fetch", END)

retrieval_subgraph = _sg.compile()
```
