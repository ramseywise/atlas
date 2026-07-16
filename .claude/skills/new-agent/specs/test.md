# test — Templates for test-scaffolder subagent

## Design notes

### Always smoke-test max_cycles enforcement
The most common LangGraph state bug is control-flow keys (cycle_count, max_cycles,
terminate) being silently dropped between nodes because they are not declared in
the TypedDict. Include this test in every agent smoke suite:

```python
def test_max_cycles_respected(self, agent_fixture):
    result = run_agent(..., max_cycles=2)
    assert result.get("cycle_count", 0) <= 2, (
        "Agent exceeded max_cycles — likely max_cycles was dropped from TypedDict state"
    )
```

### Smoke-test the column-availability assumption
Agent nodes that read DataFrame columns derived from synthetic data must be
tested against the actual generated schema — not assumed. Include a test that
runs the full agent on the synthetic dataset and checks it completes without
ColumnNotFoundError:

```python
def test_synthetic_data_columns_compatible(self, df_fixture):
    # Fails if any node reads a column that generate_*_dataset() doesn't produce
    result = run_agent(series_df=df_fixture)
    assert result.get("error") is None
```

### Test all-noise HDBSCAN on small data (cluster agents)
Small fixtures (< 15 samples) reliably trigger the HDBSCAN all-noise path.
Verify the agent handles it gracefully rather than crashing or returning empty:

```python
def test_all_noise_hdbscan_fallback(self, small_fixture):
    result = run_segmentation_agent(small_fixture, max_cycles=1)
    seg = result.get("result")
    assert seg is not None
    assert seg["n_segments"] >= 1   # never zero — fallback sentinel required
```

---

## File: {OUTPUT_DIR}/tests/conftest.py

```python
from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake passage helper
# ---------------------------------------------------------------------------

def _make_passage(text: str, url: str, score: float = 0.9) -> dict[str, Any]:
    return {
        "content": {"text": text},
        "location": {"webLocation": {"url": url}},
        "score": score,
    }


# ---------------------------------------------------------------------------
# Bedrock KB mock
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_bedrock_kb():
    """Returns a mocked boto3 bedrock-agent-runtime client that yields 3 fake passages."""
    passages = [
        _make_passage(
            "This is sample passage one about the {DOMAIN} domain.",
            "https://docs.example.com/{AGENT_NAME}/topic-one",
            score=0.95,
        ),
        _make_passage(
            "This is sample passage two covering a related concept.",
            "https://docs.example.com/{AGENT_NAME}/topic-two",
            score=0.82,
        ),
        _make_passage(
            "This is sample passage three with supplementary information.",
            "https://docs.example.com/{AGENT_NAME}/topic-three",
            score=0.74,
        ),
    ]

    mock_client = MagicMock()
    mock_client.retrieve.return_value = {
        "retrievalResults": passages,
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }

    with patch("boto3.client", return_value=mock_client):
        yield mock_client


# ---------------------------------------------------------------------------
# Scalar fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_query() -> str:
    return "How does this work?"


@pytest.fixture()
def mock_session_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Golden Q&A pairs (5 items, domain-appropriate for {DOMAIN})
# ---------------------------------------------------------------------------

@pytest.fixture()
def golden_qa() -> list[dict[str, Any]]:
    return [
        {
            "query": "How do I get started?",
            "expected_intent": "answerable",
            "golden_answer": "Here is how to get started...",
            "domain": "{DOMAIN}",
        },
        {
            "query": "I need to speak to a human agent right now",
            "expected_intent": "escalation",
            "golden_answer": "",
            "domain": "{DOMAIN}",
        },
        {
            "query": "help",
            "expected_intent": "clarification",
            "golden_answer": "",
            "domain": "{DOMAIN}",
        },
        {
            "query": "What are the main features?",
            "expected_intent": "answerable",
            "golden_answer": "The main features include...",
            "domain": "{DOMAIN}",
        },
        {
            "query": "Can I automate this process?",
            "expected_intent": "answerable",
            "golden_answer": "Yes, automation is available via...",
            "domain": "{DOMAIN}",
        },
    ]


# ---------------------------------------------------------------------------
# Environment mock
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def mock_env(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatches env vars so tests never need real credentials."""
    env_vars = {
        "AWS_REGION": "eu-west-1",
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "BEDROCK_KB_ID": "test-kb-id",
        "GROUNDING_ENABLED": "true",
        "CRAG_ENABLED": "false",
        "LOG_LEVEL": "WARNING",
        "LANGCHAIN_TRACING_V2": "false",
        "HITL_GATES_ENABLED": "false",
        "ROUTING_CONFIDENCE_THRESHOLD": "0.2",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
```

## File: {OUTPUT_DIR}/tests/test_schema.py

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from {AGENT_NAME}.schema import AssistantResponse, FailureReason, Source


class TestAssistantResponse:
    def test_message_field_required(self):
        with pytest.raises(ValidationError):
            AssistantResponse()  # type: ignore[call-arg]

    def test_message_field_accepted(self):
        r = AssistantResponse(message="Hello!")
        assert r.message == "Hello!"

    def test_sources_default_empty(self):
        r = AssistantResponse(message="ok")
        assert r.sources == []

    def test_suggestions_default_empty(self):
        r = AssistantResponse(message="ok")
        assert r.suggestions == []

    def test_contact_support_defaults_false(self):
        r = AssistantResponse(message="ok")
        assert r.contact_support is False

    def test_contact_support_can_be_true(self):
        r = AssistantResponse(message="ok", contact_support=True)
        assert r.contact_support is True

    def test_failure_reason_accepts_none(self):
        r = AssistantResponse(message="ok", failure_reason=None)
        assert r.failure_reason is None

    def test_failure_reason_accepts_constants(self):
        _reasons = [v for k, v in vars(FailureReason).items() if not k.startswith("_")]
        for reason in _reasons:
            r = AssistantResponse(message="ok", failure_reason=reason)
            assert r.failure_reason == reason

    def test_failure_reason_accepts_arbitrary_string(self):
        # FailureReason constants are plain strings — schema accepts any str
        r = AssistantResponse(message="ok", failure_reason="custom_reason")
        assert r.failure_reason == "custom_reason"


class TestSource:
    def test_source_requires_title_and_url(self):
        with pytest.raises(ValidationError):
            Source(title="Only title")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            Source(url="https://example.com")  # type: ignore[call-arg]

    def test_source_valid(self):
        s = Source(title="Help article", url="https://docs.example.com/topic-one")
        assert s.title == "Help article"
        assert s.url == "https://docs.example.com/topic-one"

    def test_source_accepts_plain_url_string(self):
        # Source.url is a plain str — no URL validation, any string is accepted
        s = Source(title="Test", url="not-a-url")
        assert s.url == "not-a-url"

    def test_full_response_with_sources(self):
        r = AssistantResponse(
            message="Here is how.",
            sources=[
                Source(title="Topic One", url="https://docs.example.com/topic-one")
            ],
            suggestions=["Tell me more", "Show me an example"],
        )
        assert len(r.sources) == 1
        assert len(r.suggestions) == 2
```

## File: {OUTPUT_DIR}/tests/test_agent.py

```python
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from {AGENT_NAME}.schema import AssistantResponse


# Mark all tests in this module as integration tests — skipped in CI unless
# RUN_INTEGRATION_TESTS=1 is set.
pytestmark = pytest.mark.integration


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: slow tests that call real or mocked agents")


@pytest.fixture(autouse=True)
def skip_without_flag(request: pytest.FixtureRequest):
    import os

    if request.node.get_closest_marker("integration"):
        if not os.getenv("RUN_INTEGRATION_TESTS"):
            pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run integration tests")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_passages() -> list[dict]:
    return [
        {
            "content": {"text": "Sample passage text for the {DOMAIN} domain."},
            "location": {"webLocation": {"url": "https://docs.example.com/{AGENT_NAME}/topic-one"}},
            "score": 0.95,
        }
    ]


def _mock_kb_client(passages=None):
    mock = MagicMock()
    mock.retrieve.return_value = {
        "retrievalResults": passages or _make_passages(),
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    """Integration stubs — each test mocks the KB and runs the full agent."""

    @pytest.mark.integration
    async def test_run_answerable_query(self, mock_env):
        """Agent returns a non-empty AssistantResponse for a factual query."""
        from {AGENT_NAME}.main import AgentRunner

        with patch("boto3.client", return_value=_mock_kb_client()):
            runner = AgentRunner(session_id=str(uuid.uuid4()))
            response = await runner.run("How do I get started?")

        assert isinstance(response, AssistantResponse)
        assert response.message, "Expected a non-empty message"
        assert not response.contact_support

    @pytest.mark.integration
    async def test_run_clarification_query(self, mock_env):
        """Agent returns a clarifying response for an ambiguous query."""
        from {AGENT_NAME}.main import AgentRunner

        with patch("boto3.client", return_value=_mock_kb_client(passages=[])):
            runner = AgentRunner(session_id=str(uuid.uuid4()))
            response = await runner.run("help")

        assert isinstance(response, AssistantResponse)
        assert response.message, "Expected a non-empty clarification message"
        # Clarification should not trigger escalation
        assert not response.contact_support

    @pytest.mark.integration
    async def test_run_escalation_query(self, mock_env):
        """Agent sets contact_support=True for explicit escalation requests."""
        from {AGENT_NAME}.main import AgentRunner

        with patch("boto3.client", return_value=_mock_kb_client()):
            runner = AgentRunner(session_id=str(uuid.uuid4()))
            response = await runner.run("I need to speak to a human agent right now")

        assert isinstance(response, AssistantResponse)
        assert response.contact_support is True, "Expected escalation to set contact_support=True"
```

## File: {OUTPUT_DIR}/tests/test_grounding.py

```python
from __future__ import annotations

import pytest

from {AGENT_NAME}.schema import AssistantResponse, Source


# ---------------------------------------------------------------------------
# Import grounding enforcer — adjust path if the agent lives in support-agents
# ---------------------------------------------------------------------------

try:
    from {AGENT_NAME}.grounding import enforce_grounding
except ImportError:
    from guardrails.grounding import enforce_grounding  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RETRIEVED_URLS = [
    "https://docs.example.com/{AGENT_NAME}/topic-one",
    "https://docs.example.com/{AGENT_NAME}/topic-two",
]


def _response_with_sources(urls: list[str]) -> AssistantResponse:
    return AssistantResponse(
        message="Here is how this works.",
        sources=[Source(title=f"Article {i}", url=url) for i, url in enumerate(urls)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnforceGrounding:
    def test_passes_when_all_sources_in_retrieved(self):
        response = _response_with_sources(_RETRIEVED_URLS)
        result = enforce_grounding(response, _RETRIEVED_URLS)
        # enforce_grounding returns the original response when grounding passes
        assert result is not None
        assert result.message == response.message

    def test_fails_when_url_not_in_passages(self):
        hallucinated_url = "https://example.com/not-retrieved"
        response = _response_with_sources([hallucinated_url])
        result = enforce_grounding(response, _RETRIEVED_URLS)
        # enforce_grounding returns None when grounding fails
        assert result is None

    def test_passes_when_sources_empty(self):
        response = AssistantResponse(message="I don't have information on that.", sources=[])
        result = enforce_grounding(response, _RETRIEVED_URLS)
        assert result is not None

    def test_passes_when_retrieved_urls_empty_and_sources_empty(self):
        response = AssistantResponse(message="No sources found.", sources=[])
        result = enforce_grounding(response, [])
        assert result is not None

    def test_fails_when_retrieved_empty_but_sources_present(self):
        response = _response_with_sources(["https://example.com/some-article"])
        result = enforce_grounding(response, [])
        assert result is None

    def test_partial_mismatch_fails(self):
        """All sources must be grounded — even one bad URL should fail."""
        mixed_urls = [_RETRIEVED_URLS[0], "https://example.com/hallucinated"]
        response = _response_with_sources(mixed_urls)
        result = enforce_grounding(response, _RETRIEVED_URLS)
        assert result is None

    def test_grounding_disabled_skips_check(self, monkeypatch: pytest.MonkeyPatch):
        """When GROUNDING_ENABLED=false the function should return response unchanged."""
        monkeypatch.setenv("GROUNDING_ENABLED", "false")
        hallucinated_url = "https://example.com/not-retrieved"
        response = _response_with_sources([hallucinated_url])
        result = enforce_grounding(response, _RETRIEVED_URLS)
        # Should pass through without checking
        assert result is not None
```
