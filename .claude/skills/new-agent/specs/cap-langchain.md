# cap-langchain — Templates for langchain-builder subagent

LangChain LCEL composition patterns: chains, output parsers, retrieval chains,
structured output, memory, streaming, and parallel/conditional runnable routing.
Use when the agent needs composable chain logic without a full LangGraph state machine.

---

## File: {OUTPUT_DIR}/chains/__init__.py

```python
```

---

## File: {OUTPUT_DIR}/chains/base.py

Core LCEL primitives: prompt templates, output parsers, and the base chain factory.

```python
"""Base LCEL chain components for {AGENT_NAME}."""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from {AGENT_NAME}.schema import AssistantResponse

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant for the {DOMAIN} domain. "
        "Answer the user's question using only the context provided. "
        "If the context does not contain the answer, say so clearly. "
        "Always cite your sources by URL."
    )),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

CLARIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. The user's question is too vague to answer directly."),
    ("human", "{question}"),
    ("ai", "Could you please clarify what you mean? Specifically: "),
])

STRUCTURED_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant for the {DOMAIN} domain. "
        "Respond with valid JSON matching the schema provided. "
        "Use only information from the context."
    )),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nSchema: {schema}"),
])


# ---------------------------------------------------------------------------
# Output parsers
# ---------------------------------------------------------------------------

str_parser = StrOutputParser()
json_parser = JsonOutputParser()
response_parser = PydanticOutputParser(pydantic_object=AssistantResponse)


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def make_answer_chain(llm: Runnable) -> Runnable:
    """LCEL answer chain: prompt | llm | str_parser."""
    return ANSWER_PROMPT | llm | str_parser


def make_structured_chain(llm: Runnable, output_schema: type) -> Runnable:
    """Chain that returns structured Pydantic output via .with_structured_output()."""
    structured_llm = llm.with_structured_output(output_schema)
    return STRUCTURED_ANSWER_PROMPT | structured_llm


def make_parallel_chain(llm: Runnable) -> Runnable:
    """Run answer + follow-up suggestions in parallel, merge results."""
    answer_chain = make_answer_chain(llm)
    suggestion_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "Generate 3 short follow-up questions the user might ask next. Return as JSON list."),
            ("human", "Original question: {question}\nAnswer: {answer}"),
        ])
        | llm
        | json_parser
    )

    return (
        RunnableParallel(answer=answer_chain, context=RunnablePassthrough())
        | RunnableLambda(lambda x: {
            **x["context"],
            "answer": x["answer"],
        })
    )


def make_conditional_chain(llm: Runnable) -> Runnable:
    """Route to answer or clarification chain based on query length heuristic."""
    answer_chain = make_answer_chain(llm)
    clarify_chain = CLARIFY_PROMPT | llm | str_parser

    return RunnableBranch(
        (lambda x: len(x.get("question", "").split()) < 3, clarify_chain),
        answer_chain,
    )
```

---

## File: {OUTPUT_DIR}/chains/retrieval.py

RAG-style retrieval chain using LCEL and LangChain's document chain helpers.

```python
"""LCEL retrieval chain for {AGENT_NAME}."""
from __future__ import annotations

import logging
from typing import Any

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

_log = logging.getLogger(__name__)

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful assistant for the {DOMAIN} domain. "
        "Answer using only the retrieved context below. "
        "Cite sources by URL. If context is insufficient, say so.\n\n"
        "Context:\n{context}"
    )),
    ("human", "{input}"),
])


def _passages_to_documents(passages: list[dict]) -> list[Document]:
    """Convert raw passage dicts to LangChain Document objects."""
    return [
        Document(
            page_content=p.get("text", p.get("content", {}).get("text", "")),
            metadata={"url": p.get("url", ""), "title": p.get("title", ""), "score": p.get("score", 0.0)},
        )
        for p in passages
    ]


def make_rag_chain(llm: Runnable, retriever: Any) -> Runnable:
    """Full RAG chain: retriever → stuff documents → LLM → answer.

    retriever must implement .invoke(query) -> list[Document] or
    be a LangChain BaseRetriever.
    """
    doc_chain = create_stuff_documents_chain(llm, _RAG_PROMPT)
    return create_retrieval_chain(retriever, doc_chain)


def make_passage_chain(llm: Runnable) -> Runnable:
    """RAG chain that accepts pre-retrieved passage dicts (no live retriever needed).

    Input: {"input": str, "passages": list[dict]}
    Output: {"answer": str, "sources": list[str]}
    """
    def _inject_context(inputs: dict) -> dict:
        docs = _passages_to_documents(inputs.get("passages", []))
        return {
            "input": inputs["input"],
            "context": docs,
        }

    def _extract_sources(inputs: dict) -> dict:
        docs = _passages_to_documents(inputs.get("passages", []))
        return {
            "answer": inputs["answer"],
            "sources": [d.metadata["url"] for d in docs if d.metadata.get("url")],
        }

    doc_chain = create_stuff_documents_chain(llm, _RAG_PROMPT)

    return (
        RunnableLambda(_inject_context)
        | doc_chain
        | RunnableLambda(lambda ans: {"answer": ans})
    )
```

---

## File: {OUTPUT_DIR}/chains/memory.py

Session memory wired into LCEL via `RunnableWithMessageHistory`.

```python
"""LCEL memory integration for {AGENT_NAME}."""
from __future__ import annotations

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory

# Session store — swap for Redis/DynamoDB in production
_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


def with_history(chain: Runnable, input_key: str = "question") -> Runnable:
    """Wrap any LCEL chain with per-session message history.

    Usage:
        chain = make_answer_chain(llm)
        chain_with_mem = with_history(chain)
        result = chain_with_mem.invoke(
            {"question": "...", "context": "..."},
            config={"configurable": {"session_id": "abc123"}},
        )
    """
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key=input_key,
        history_messages_key="history",
    )


def clear_session(session_id: str) -> None:
    _store.pop(session_id, None)
```

---

## File: {OUTPUT_DIR}/chains/streaming.py

Async streaming helpers for LCEL chains.

```python
"""Streaming utilities for LCEL chains in {AGENT_NAME}."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import Runnable


async def astream_tokens(chain: Runnable, inputs: dict, config: dict | None = None) -> AsyncIterator[str]:
    """Yield string tokens from a streaming LCEL chain.

    Usage:
        async for token in astream_tokens(chain, {"question": "...", "context": "..."}):
            print(token, end="", flush=True)
    """
    async for chunk in chain.astream(inputs, config=config or {}):
        if isinstance(chunk, str):
            yield chunk
        elif isinstance(chunk, dict) and "answer" in chunk:
            # create_retrieval_chain yields {"answer": str} chunks
            yield chunk["answer"]
        elif hasattr(chunk, "content"):
            # AIMessageChunk
            yield chunk.content


async def collect_stream(chain: Runnable, inputs: dict, config: dict | None = None) -> str:
    """Run a streaming chain and collect the full output as a string."""
    parts: list[str] = []
    async for token in astream_tokens(chain, inputs, config):
        parts.append(token)
    return "".join(parts)
```

---

## File: {OUTPUT_DIR}/chains/callbacks.py

Custom LangChain callback handler for token counting and LangSmith-compatible tracing.

```python
"""LangChain callback handler for {AGENT_NAME}."""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

_log = logging.getLogger(__name__)


class TokenCountingCallback(BaseCallbackHandler):
    """Accumulates token usage across LLM calls in a chain run."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self._start: float = 0.0

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs: Any) -> None:
        self._start = time.monotonic()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        elapsed = time.monotonic() - self._start
        for generations in response.generations:
            for gen in generations:
                usage = getattr(gen, "generation_info", {}) or {}
                self.prompt_tokens += usage.get("prompt_tokens", 0)
                self.completion_tokens += usage.get("completion_tokens", 0)
                self.total_tokens += usage.get("total_tokens", 0)
        _log.debug(
            "LLM call finished in %.2fs | prompt=%d completion=%d total=%d",
            elapsed,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
        )

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        _log.error("LLM error: %s", error)

    @property
    def summary(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
```

---

## SKILL.md additions

Add `langchain` to the capability table in SKILL.md:

```
| `langchain` | LCEL chains, output parsers, retrieval chain, memory, streaming callbacks |
```

And the corresponding spec load in the dispatch logic:

```
~/.claude/skills/new-agent/specs/cap-langchain.md   # if langchain in capabilities
```

Spawns **langchain-builder** subagent with this spec.

**Note on framework interaction:**
- `--framework langgraph` + `--capabilities langchain` → LCEL chains used *inside* LangGraph nodes (e.g., a node calls `make_answer_chain(llm).invoke(...)`)
- `--framework adk` + `--capabilities langchain` → LCEL chains called from ADK tool handlers
- Without `--framework` + `--capabilities langchain` only → generates a standalone chain-based agent (no graph, no ADK) using `chains/base.py` as the entry point
