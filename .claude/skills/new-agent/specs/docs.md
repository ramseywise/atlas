# docs — Templates for docs-builder subagent

## File: {OUTPUT_DIR}/README.md

````markdown
# {AGENT_NAME}

## Overview

{AGENT_NAME} is a retrieval-augmented agent for the {DOMAIN} domain. It answers user questions by retrieving relevant passages from the knowledge base, generating a grounded response, and — when confidence is low or the user explicitly requests it — escalating to a human.

## Architecture

The agent implements five protection layers:

| Layer | Name | Location | What it does |
|---|---|---|---|
| 1 | Input guardrail | `main.py` / `agent.py` | Blocks injection, PII extraction, and prompt escalation patterns |
| 2 | Routing confidence | `main.py` | Low-confidence intent → direct clarification node, never a guess |
| 3 | Retrieval quality gate | `retrieval.py` | CRAG confidence gate — rewrites and re-retrieves if score is low |
| 4 | Post-gen grounding | `grounding.py` | Enforces that all `sources[].url` values come from the retrieved passage set |
| 5 | Escalation path | `main.py` | Friction signals routed to `contact_support=True` |

## Setup

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Configure environment
cp .env.example .env
# Edit .env — at minimum set BEDROCK_KB_ID and AWS_REGION
```

## Running

```bash
make run
# Or directly:
uv run python -m {AGENT_NAME}.main
```

## Testing

```bash
make test
# Integration tests (requires real or mocked AWS):
RUN_INTEGRATION_TESTS=1 make test
```

## Eval

```bash
# Quick eval — first 20 items, safe for cost
make grade

# Broader ablation — 50 items
make grade-ablation-top

# Dry run (no agent calls, validates dataset loading)
make grade-dry
```

## Capabilities

- Retrieval-augmented Q&A against Bedrock Knowledge Base
- CRAG re-ranking loop for low-confidence retrievals
- Structured response schema (`AssistantResponse`) with grounded sources
- Escalation detection and routing
- LangSmith tracing (set `LANGCHAIN_TRACING_V2=true`)
- Optional HITL interrupt gates
- Optional streaming via FastAPI SSE

## Feature Flags

| Env var | Default | What it controls |
|---|---|---|
| `GROUNDING_ENABLED` | `true` | Layer 4 citation enforcement |
| `CRAG_ENABLED` | `true` | CRAG retrieve–grade–rewrite loop |
| `CRAG_HIGH_CONFIDENCE` | `0.7` | Skip re-grading above this threshold |
| `LLM_PLANNER` | `false` | Use LLM-based intent routing (slower) |
| `ROUTING_CONFIDENCE_THRESHOLD` | `0.2` | Below this confidence → clarification node |
| `HITL_GATES_ENABLED` | `false` | Enable LangGraph interrupt gates |
| `HITL_CONFIDENCE_THRESHOLD` | `0.3` | Interrupt below this confidence score |
| `THINKING_BUDGET` | `0` | Gemini extended thinking tokens (ADK only) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name (ADK only) |
| `RETRIEVAL_BACKEND` | `bedrock` | `bedrock`, `rag`, or `custom` |

## Adding a new node / tool

1. **Create the node function** in `graph/nodes/<name>.py` (LangGraph) or register it as a tool in `tool_registry.py` (ADK). The function must accept `state: State` and return `dict`.
2. **Wire it into the graph** in `graph/builder.py`: add `builder.add_node("<name>", <name>_node)` and connect edges. Run `make test` to verify the graph compiles.
3. **Bump `PROMPT_VERSION`** in `schema.py` if you change any prompts — this ensures LangSmith experiments remain comparable. Re-run `make grade` to confirm pass rates hold.
````

## File: {OUTPUT_DIR}/HACKING.md

````markdown
# HACKING — {AGENT_NAME}

Developer guide for working inside this agent.

## Adding a node (LangGraph)

1. Create `graph/nodes/<name>.py`:
   ```python
   from {AGENT_NAME}.state import State

   async def <name>_node(state: State) -> dict:
       # Read from state, return partial state dict
       return {"<key>": <value>}
   ```
2. Register in `graph/builder.py`:
   ```python
   builder.add_node("<name>", <name>_node)
   builder.add_edge("<predecessor>", "<name>")
   ```
3. Run `make test` — if the graph fails to compile, check for missing edges or typos in node names.

## Adding a tool (ADK / ToolRegistry)

```python
from {AGENT_NAME}.tool_registry import registry

@registry.tool(name="my_tool", description="Does X given Y")
def my_tool(param: str) -> str:
    return f"Result for {param}"
```

The tool is automatically included in `registry.all_tools()` and callable via `registry.dispatch("my_tool", {"param": "value"})`.

## Changing prompts — PROMPT_VERSION bump required

Every prompt change **must** increment `PROMPT_VERSION` in `schema.py`:

```python
# schema.py
PROMPT_VERSION = "1.2.0"  # bump this
```

Why: LangSmith experiments tag runs with PROMPT_VERSION. Without a bump, old and new prompt results are mixed in the same experiment, making ablation results unreadable.

After bumping: run `make grade --limit 20` and compare against the previous experiment in LangSmith.

## Running evals locally

```bash
# Safe first run — 20 items, low cost
make grade

# Full ablation on top-50
make grade-ablation-top

# No agent calls — validates dataset loading only
make grade-dry
```

Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY=<your-key>` in `.env` to see traces in LangSmith.

## Common issues

### AWS credentials not found

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

Fix: set `AWS_PROFILE` in `.env` to your configured profile name, or export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` directly.

Verify with: `aws sts get-caller-identity --profile <your-profile>`

### LangSmith 403 Forbidden

Your `LANGCHAIN_API_KEY` is wrong or expired. Regenerate at https://smith.langchain.com → Settings → API Keys. The key must have write access to the project named `{AGENT_NAME}`.

### Bedrock throttling (429 / ThrottlingException)

Lower `--concurrency` in the eval runner or add `--limit 10`. In production, configure Bedrock provisioned throughput.

Alternatively, set `CRAG_ENABLED=false` to reduce the number of Bedrock calls per query (removes the rewrite loop).

### Test isolation — moto not intercepting boto3 calls

Ensure `mock_env` fixture is applied (`autouse=False` by default — add it explicitly to the test function signature). Also confirm the `moto[bedrock]` extra is installed:

```bash
uv sync --extra dev
```
````
