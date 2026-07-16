# cap-rag — Templates for rag-builder subagent

## File: {OUTPUT_DIR}/retrieval.py

```python
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass
class PassageItem:
    text: str
    url: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bedrock(cls, raw: dict[str, Any]) -> "PassageItem":
        text = raw.get("content", {}).get("text", "")
        url = (
            raw.get("location", {})
            .get("webLocation", {})
            .get("url", "")
        )
        score = float(raw.get("score", 0.0))
        metadata = {k: v for k, v in raw.items() if k not in ("content", "location", "score")}
        return cls(text=text, url=url, score=score, metadata=metadata)


# ---------------------------------------------------------------------------
# Bedrock KB client
# ---------------------------------------------------------------------------


class BedrockKBClient:
    """Thin wrapper around boto3 bedrock-agent-runtime for knowledge base retrieval."""

    def __init__(self, region: str | None = None) -> None:
        self._region = region or os.environ.get("AWS_REGION", "eu-west-1")
        self._client = boto3.client("bedrock-agent-runtime", region_name=self._region)

    def retrieve(
        self,
        query: str,
        kb_id: str,
        n_results: int = 5,
    ) -> list[PassageItem]:
        """Call Bedrock KB and return a list of PassageItem."""
        try:
            response = self._client.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": n_results}
                },
            )
        except ClientError as exc:
            logger.error("Bedrock retrieve failed: %s", exc)
            return []

        raw_results = response.get("retrievalResults", [])
        return [PassageItem.from_bedrock(r) for r in raw_results]


# ---------------------------------------------------------------------------
# Passage grading
# ---------------------------------------------------------------------------


async def grade_passage(passage: PassageItem, query: str, llm: Any) -> float:
    """Ask the LLM to score passage relevance to the query. Returns 0.0–1.0."""
    prompt = (
        f"On a scale from 0 to 1, how relevant is the following passage to the query?\n\n"
        f"Query: {query}\n\n"
        f"Passage: {passage.text[:500]}\n\n"
        f"Reply with a single decimal number between 0 and 1, nothing else."
    )
    try:
        result = await llm.ainvoke(prompt)
        # Handle both string and AIMessage responses
        text = result.content if hasattr(result, "content") else str(result)
        score = float(text.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, AttributeError) as exc:
        logger.warning("Grade passage failed to parse score: %s", exc)
        return 0.5  # Neutral fallback


def confidence_gate(passages: list[PassageItem], threshold: float = 0.7) -> bool:
    """Return True if the top passage score meets the confidence threshold."""
    if not passages:
        return False
    return passages[0].score >= threshold


# ---------------------------------------------------------------------------
# Query rewriting
# ---------------------------------------------------------------------------


async def rewrite_query(query: str, llm: Any) -> str:
    """Rewrite the query to improve retrieval coverage."""
    prompt = (
        f"Rewrite the following search query to improve retrieval from a help center knowledge base. "
        f"Make it more specific, use alternative phrasings, and expand abbreviations.\n\n"
        f"Original query: {query}\n\n"
        f"Rewritten query (one sentence only):"
    )
    try:
        result = await llm.ainvoke(prompt)
        text = result.content if hasattr(result, "content") else str(result)
        return text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query rewrite failed: %s", exc)
        return query


# ---------------------------------------------------------------------------
# CRAG retrieve loop
# ---------------------------------------------------------------------------


async def crag_retrieve(
    query: str,
    llm: Any,
    kb_id: str,
    n_results: int = 5,
    confidence_threshold: float | None = None,
    max_attempts: int = 2,
    client: BedrockKBClient | None = None,
) -> list[PassageItem]:
    """Retrieve passages with CRAG loop: retrieve → grade → if low confidence, rewrite and retry.

    Args:
        query: The user's query.
        llm: An async LLM client with .ainvoke().
        kb_id: Bedrock Knowledge Base ID.
        n_results: Number of results to retrieve per attempt.
        confidence_threshold: Score threshold for the confidence gate.
            Defaults to CRAG_HIGH_CONFIDENCE env var or 0.7.
        max_attempts: Maximum number of retrieve–grade iterations.
        client: Optional pre-built BedrockKBClient (useful for testing).

    Returns:
        List of PassageItem sorted by score descending.
    """
    crag_enabled = os.environ.get("CRAG_ENABLED", "true").lower() == "true"
    if confidence_threshold is None:
        confidence_threshold = float(os.environ.get("CRAG_HIGH_CONFIDENCE", "0.7"))

    kb = client or BedrockKBClient()
    current_query = query

    for attempt in range(max_attempts):
        passages = kb.retrieve(current_query, kb_id=kb_id, n_results=n_results)
        passages.sort(key=lambda p: p.score, reverse=True)

        if not crag_enabled:
            logger.debug("CRAG disabled — returning raw passages")
            return passages

        if confidence_gate(passages, threshold=confidence_threshold):
            logger.debug("Confidence gate passed on attempt %d", attempt + 1)
            return passages

        # Low confidence — rewrite and retry (unless this is the last attempt)
        if attempt < max_attempts - 1:
            logger.info(
                "Low confidence (top score=%.3f) on attempt %d — rewriting query",
                passages[0].score if passages else 0.0,
                attempt + 1,
            )
            # Grade passages concurrently to pick the best rewrite signal
            if passages:
                scores = await asyncio.gather(
                    *(grade_passage(p, current_query, llm) for p in passages[:3])
                )
                logger.debug("LLM grades: %s", scores)

            current_query = await rewrite_query(current_query, llm)
            logger.info("Rewritten query: %r", current_query)

    logger.warning("CRAG max attempts reached — returning best available passages")
    return passages
```

## File: {OUTPUT_DIR}/chunking.py

```python
from __future__ import annotations

import re
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


def fixed_chunker(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[str]:
    """Split text into fixed-size chunks with character-level overlap.

    Args:
        text: The source text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters to repeat at the start of the next chunk.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to break on a sentence boundary near the end
        if end < text_len:
            boundary = _find_sentence_boundary(text, end, lookback=100)
            if boundary:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= end:
            # Safety: prevent infinite loop if overlap >= chunk_size
            start = end

    return chunks


def _find_sentence_boundary(text: str, pos: int, lookback: int = 100) -> int | None:
    """Find the nearest sentence-ending punctuation before `pos`."""
    window = text[max(0, pos - lookback) : pos]
    # Find the last '. ', '! ', or '? ' in the window
    matches = list(re.finditer(r"[.!?]\s", window))
    if matches:
        last_match = matches[-1]
        return max(0, pos - lookback) + last_match.end()
    return None


# ---------------------------------------------------------------------------
# Semantic chunker stub
# ---------------------------------------------------------------------------


def semantic_chunker(
    text: str,
    embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    similarity_threshold: float = 0.85,
) -> list[str]:
    """Semantic chunker: splits on sentence boundaries where embedding similarity drops.

    This is a stub implementation. For production use, provide an embedding_fn
    (e.g., a Titan Embeddings wrapper) that returns normalized vectors.

    Args:
        text: The source text.
        embedding_fn: Callable that takes a list of sentences and returns embeddings.
        similarity_threshold: Cosine similarity below this value triggers a split.

    Returns:
        List of semantically coherent text chunks.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if embedding_fn is None:
        # Fall back to fixed chunking when no embedding function is available
        return fixed_chunker(text)

    embeddings = embedding_fn(sentences)
    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = _cosine_similarity(embeddings[i - 1], embeddings[i])
        if similarity >= similarity_threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]

    if current:
        chunks.append(" ".join(current))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x**2 for x in a) ** 0.5
    norm_b = sum(x**2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_chunker(strategy: ChunkingStrategy | str) -> Callable[[str], list[str]]:
    """Return a chunker callable for the given strategy.

    Usage:
        chunker = get_chunker(ChunkingStrategy.FIXED)
        chunks = chunker("some long document text...")
    """
    strategy = ChunkingStrategy(strategy)

    if strategy == ChunkingStrategy.FIXED:
        return fixed_chunker
    if strategy == ChunkingStrategy.SEMANTIC:
        return semantic_chunker
    if strategy == ChunkingStrategy.HIERARCHICAL:
        # Hierarchical: first pass fixed-size, second pass within each chunk semantic
        def hierarchical_chunker(text: str) -> list[str]:
            broad_chunks = fixed_chunker(text, chunk_size=2000, overlap=200)
            fine: list[str] = []
            for broad in broad_chunks:
                fine.extend(fixed_chunker(broad, chunk_size=500, overlap=100))
            return fine

        return hierarchical_chunker

    raise ValueError(f"Unknown chunking strategy: {strategy!r}")
```
