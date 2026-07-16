---
name: framework-selection
description: >
  INVOKE THIS SKILL before scaffolding a new agent (e.g. via /new-agent) or writing
  agent code from scratch, whenever the framework hasn't already been decided.
  Decides between Google ADK and LangGraph based on deployment target, control-flow
  needs, and ecosystem fit.
---

# Framework Selection — ADK vs LangGraph

This project supports two agent frameworks. They are not layered the way ADK/LangGraph
sit relative to LangChain elsewhere — pick one per agent based on its deployment target
and control-flow needs.

---

## Decision Guide

Answer in order; stop at the first "yes":

| Question | Yes → |
|---|---|
| Deploying on GCP / Vertex AI, want a managed session service and a maintained scaffolder CLI (`agent-starter-pack`)? | **ADK** |
| Need fine-grained custom control flow — branching, loops, parallel fan-out/fan-in, or human-in-the-loop gates at arbitrary points in the graph? | **LangGraph** |
| Already have LangChain tools, retrievers, or chains to reuse, or need multi-provider LLM support beyond Gemini? | **LangGraph** |
| Straightforward single-agent (or agent + sub-agents) assistant, Gemini-first, no exotic control flow? | **ADK** |

If none clearly apply, default to **ADK** for anything deploying to GCP, **LangGraph**
for anything that needs to run anywhere else or that already has non-Gemini LLM calls
in the codebase.

---

## Framework Profiles

### Google ADK

**Best for:**
- Single-agent or sub-agent-hierarchy assistants
- GCP/Vertex AI deployment (Agent Engine or Cloud Run)
- Projects that want a maintained scaffolder CLI rather than hand-rolled infra

**Not ideal when:**
- You need a graph with arbitrary branching/loops/parallel workers
- You need LLM providers other than Gemini

**Skills to invoke next:** `.agents/skills/adk-scaffold/SKILL.md`, then `.agents/skills/adk-dev-guide/SKILL.md`

### LangGraph

**Best for:**
- Agents with branching logic, loops, or reflection (retry-until-correct)
- Multi-step workflows where different paths depend on intermediate results
- Human-in-the-loop approval at specific graph nodes
- Parallel fan-out / fan-in (map-reduce patterns)
- Any LLM provider via LangChain integrations

**Not ideal when:**
- A simple single-agent assistant would do — the extra graph-authoring effort isn't worth it
- There's no need for anything beyond what ADK's built-in agent/sub-agent hierarchy gives you for free

**Skills to invoke next:** `.agents/skills/langgraph-scaffold/SKILL.md`, then `.agents/skills/langgraph-fundamentals/SKILL.md`; also `.agents/skills/langgraph-persistence/SKILL.md` and `.agents/skills/langgraph-human-in-the-loop/SKILL.md` as needed.

---

## Quick Reference

| | ADK | LangGraph |
|---|---|---|
| Scaffolder | `agent-starter-pack` (official Google CLI) | none — hand-written via `langgraph-scaffold` |
| Control flow | Agent / sub-agent hierarchy, callbacks | Full graph control (nodes, edges, `Command`, `Send`) |
| Managed deployment | Agent Engine (built-in) | Manual (Cloud Run, ECS, etc.) |
| Persistence | Session service (managed or Cloud SQL) | Checkpointer (`InMemorySaver`/`SqliteSaver`/`PostgresSaver`) |
| Multi-provider LLM | Gemini-first (Vertex AI or AI Studio) | Any provider via LangChain integrations |
| Human-in-the-loop | Callback-based approval | `interrupt()` / `Command(resume=...)` |
