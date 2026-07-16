---
name: langgraph-scaffold
description: >
  Scaffold a new LangGraph agent's core files — state schema, graph builder, nodes,
  checkpointer wiring, tests, and infra. Use when the user wants to build a new
  LangGraph agent (e.g. "build me a langgraph agent for X") or when /new-agent selects
  the langgraph framework. Unlike ADK, there is no official scaffolder CLI — this
  skill hand-writes the file set directly.
---

# LangGraph Project Scaffolding Guide

Read `.agents/skills/langgraph-fundamentals/SKILL.md` before writing graph code if you
haven't already — it covers `StateGraph`, nodes, edges, `Command`, and `Send`. Read
`.agents/skills/langgraph-persistence/SKILL.md` and
`.agents/skills/langgraph-human-in-the-loop/SKILL.md` if the agent needs cross-turn
memory or approval steps.

---

## Step 1: Gather Requirements

Start with the use case, then ask follow-ups based on answers.

**Always ask:**

1. **What problem will the agent solve?** — Core purpose and capabilities
2. **What are the discrete steps?** — Sketch the flowchart; each step becomes a node
3. **External APIs or data sources needed?** — Tools, integrations, auth requirements
4. **Persistence needs?** — None (stateless), thread-scoped only (checkpointer), or cross-thread memory too (checkpointer + `Store`)? See `langgraph-persistence` for the tradeoffs.
5. **Human-in-the-loop steps?** — Any point where the graph must pause for approval or missing input? See `langgraph-human-in-the-loop`.
6. **Deployment shape?** — A FastAPI service (sync + optionally streaming), a batch job, or a library called from elsewhere? Ask — do not default silently.

If human-in-the-loop is requested, a checkpointer is mandatory even if nothing else needs persistence — interrupts fail without one.

---

## Step 2: Write DESIGN_SPEC.md

Compose a **detailed** spec with these sections. Present the full spec for user approval before scaffolding.

```markdown
# DESIGN_SPEC.md

## Overview
2-3 paragraphs describing the agent's purpose and how it works.

## Graph Flow
The discrete steps from Step 1, as a node-by-node flowchart (text is fine).

## Example Use Cases
3-5 concrete examples with expected inputs and outputs.

## Tools Required
Each tool with its purpose, API details, and authentication needs.

## Persistence & HITL
Checkpointer choice and rationale; any interrupt points and what they ask for.

## Constraints & Safety Rules
Specific rules — not just generic statements.

## Success Criteria
Measurable outcomes for evaluation.

## Edge Cases to Handle
At least 3-5 scenarios the agent must handle gracefully.
```

The spec should be thorough enough for another developer to implement the agent without additional context.

---

## Step 3: Scaffold the File Set

Substitute `{AGENT_NAME}` → snake_case agent name, `{DOMAIN}` → the domain label from
Step 1, `{OUTPUT_DIR}` → the output path (default: `{source_root}/agents/{AGENT_NAME}`,
confirm with the user if `source_root` isn't obvious from the project layout).

```
{OUTPUT_DIR}/
  schema.py             # Pydantic request/response models — API boundary types
  state.py              # TypedDict graph state + reducers (see langgraph-fundamentals)
  graph.py              # StateGraph builder: add_node/add_edge/compile
  nodes/
    __init__.py
    <node_name>.py       # one file per node from the Step 1 flowchart; pure function(state) -> dict
  checkpointer.py         # factory: InMemorySaver (dev) / PostgresSaver (prod), per Step 1's choice
  main.py                 # FastAPI app: POST /chat, GET /health — only if deployment shape is "service"
  tests/
    __init__.py
    test_graph.py         # compiles + invokes the graph with a fixture state
  pyproject.toml
  Makefile
  .env.example
  README.md
```

Generation rules:
- **`state.py`**: define `State(TypedDict)` with the fields identified in Step 1. Add `Annotated[list, operator.add]` (or a custom reducer) for any field that should accumulate rather than overwrite — see `langgraph-fundamentals`'s state-update-strategies table.
- **`graph.py`**: wire nodes per the Step 1 flowchart. Use `add_conditional_edges` for branching, `Command` where a node needs to update state and route in one return, `Send` for fan-out. Compile with the checkpointer from `checkpointer.py`.
- **`checkpointer.py`**: `InMemorySaver` for dev/tests; if the spec calls for production persistence, wire `PostgresSaver` behind an environment-variable switch — never leave `InMemorySaver` as the only option for anything the spec describes as needing to survive a restart.
- **Human-in-the-loop**: if requested, add an approval node using `interrupt()` and route around it with `Command`, per `langgraph-human-in-the-loop`. Remember: code before `interrupt()` re-runs on every resume — keep it idempotent.
- **`main.py`**: only generate if the deployment shape is "service." Always pass `thread_id` through from the request; never invoke the graph without one if a checkpointer is configured.
- **`tests/test_graph.py`**: at minimum, compile the graph and `invoke()` it once with a representative input, asserting on output shape. Add one test per conditional branch identified in Step 1.

---

## Step 4: Validate

```bash
uv run ruff check . --fix
uv run pytest tests/ -x -q
```

Report: files written, lint/test results, and — explicitly — anything the spec asked
for that this pass didn't implement. Do not silently drop requirements.

---

## Step 5: Save DESIGN_SPEC.md

Save the approved spec from Step 2 to the project root (or `{OUTPUT_DIR}/DESIGN_SPEC.md`
if this agent lives in a multi-agent monorepo) as `DESIGN_SPEC.md`.

---

## Critical Rules

- **Always `compile()` before `invoke()`** — see `langgraph-fundamentals`.
- **Never leave `InMemorySaver` as the checkpointer for anything the spec describes as needing production persistence** — use `PostgresSaver` behind a config switch. See `langgraph-persistence`.
- **A checkpointer is mandatory if human-in-the-loop is requested** — `interrupt()` fails without one.
- **Ask before choosing the deployment shape** (FastAPI service vs. library vs. batch job) — don't default silently.
- **NEVER change an existing model/provider string** in code you're modifying unless explicitly asked.

---

## Scaffold as Reference

Unlike ADK, there's no CLI to spin up a throwaway reference project. If you need to see
a fuller example of a particular pattern (e.g. multi-node fan-out, Postgres checkpointer
wiring), read the worked examples inside `langgraph-fundamentals`, `langgraph-persistence`,
and `langgraph-human-in-the-loop` rather than generating a scratch project.
