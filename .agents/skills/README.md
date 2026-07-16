# `.agents/skills/` — Agent Framework Reference Library

This directory is **not Claude-specific**. It's a tool-agnostic reference library for
building agents with Google ADK or LangGraph — readable by Claude Code, by ADK's own
tooling, or by any other agent harness this project ends up using. `.claude/skills/`
holds workflow commands scoped to Claude Code; this directory holds durable framework
knowledge that outlives any one tool.

`.claude/skills/new-agent/SKILL.md` is the entry point most sessions will use — it
reads `framework-selection` to pick ADK or LangGraph, then hands off to the matching
scaffold skill below.

## Quick map

| Skill | Use when | What it does |
|---|---|---|
| `framework-selection` | Starting a new agent, framework not yet decided | Decision guide: ADK vs LangGraph |
| `adk-scaffold` | Building or enhancing an ADK agent project | Wraps Google's `agent-starter-pack` CLI to create/enhance ADK projects |
| `adk-dev-guide` | Any ADK development session | Spec-driven workflow, code-preservation rules, model-selection rules, troubleshooting |
| `langgraph-scaffold` | Building a new LangGraph agent | Hand-written file scaffold: state, graph, nodes, checkpointer, tests |
| `langgraph-fundamentals` | Writing any LangGraph code | `StateGraph`, nodes, edges, `Command`, `Send`, streaming, error handling |
| `langgraph-persistence` | Adding memory or cross-turn state | Checkpointers, thread IDs, time travel, `Store`, subgraph checkpointer scoping |
| `langgraph-human-in-the-loop` | Adding approval/pause-for-input steps | `interrupt()`, `Command(resume=...)`, idempotency rules for pre-interrupt side effects |

## Attribution

`adk-scaffold` and `adk-dev-guide` are ported from Google's Apache-2.0-licensed ADK
skill set — see the `LICENSE.txt` in each of those directories. The LangGraph skills
carry no external license — they're project-authored reference material.

## Not yet included

`adk-cheatsheet`, `adk-eval-guide`, `adk-deploy-guide`, and `adk-observability-guide`
exist upstream but aren't ported here yet — `adk-dev-guide` falls back to official docs
where it would otherwise point to one of them. Port the missing piece into this
directory (matching the format above) if a gap comes up repeatedly, rather than
duplicating the knowledge inline in a workflow skill.

## Claude Code relevance

These are useful reading for Claude even outside of `/new-agent`:

- `framework-selection` — before choosing which framework to build a new agent on
- `adk-scaffold` / `adk-dev-guide` — before writing or modifying any ADK agent code
- `langgraph-scaffold` / `langgraph-fundamentals` — before writing or modifying any LangGraph code
- `langgraph-persistence` / `langgraph-human-in-the-loop` — before touching checkpointing, memory, or approval-gate logic
