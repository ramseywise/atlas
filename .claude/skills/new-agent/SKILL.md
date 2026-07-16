---
name: new-agent
description: >
  Scaffold a new AI agent — Google ADK or LangGraph. Decides the framework (via
  framework-selection, if not specified), then hands off to the matching scaffold
  skill in .agents/skills/. Triggers on: "scaffold a new agent", "build me an agent
  for X", "new ADK agent", "new LangGraph agent", "/new-agent <name>".
---

# /new-agent

Thin orchestrator — all framework-specific knowledge lives in `.agents/skills/`,
which is shared with any other agent harness in this project (it's not Claude-specific).
If you're extending this skill, add capability reference material there, not here.

## Usage

```
/new-agent <name> [--framework adk|langgraph] [--domain <string>] [--output path/to/dir]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `<name>` | required | snake_case agent name, e.g. `order_support`, `insights_agent` |
| `--framework` | ask via `framework-selection` | `adk` or `langgraph` |
| `--domain` | `general` | Free-form label injected into prompts/README |
| `--output` | framework default (see below) | Override the output directory |

## Steps

### Step 1 — Parse arguments

- Validate `name` matches `^[a-z][a-z0-9_]*$` — error if not.
- If `--framework` is missing, don't guess — go to Step 2.
- If `--output` is missing, let the framework-specific scaffold skill pick its own
  default (ADK: `agent-starter-pack`'s project-directory convention; LangGraph:
  typically `{source_root}/agents/{name}` — confirm the source directory name from
  the project layout if it isn't obvious).

### Step 2 — Decide the framework (if not given)

Read `.agents/skills/framework-selection/SKILL.md` and walk its decision table with
the user, or infer from project context (existing agent code, deployment target
mentioned) and confirm before proceeding. Do not silently default.

### Step 3 — Hand off to the scaffold skill

**If framework == `adk`:**
1. Read `.agents/skills/adk-scaffold/SKILL.md` and follow it exactly — it drives
   requirement-gathering, `DESIGN_SPEC.md`, and the `agent-starter-pack` CLI invocation.
2. Then read `.agents/skills/adk-dev-guide/SKILL.md` for the development workflow
   before writing any agent logic.

**If framework == `langgraph`:**
1. Read `.agents/skills/langgraph-scaffold/SKILL.md` and follow it — it drives
   requirement-gathering, `DESIGN_SPEC.md`, and the file scaffold.
2. Read `.agents/skills/langgraph-fundamentals/SKILL.md` before writing graph code.
3. If the spec calls for cross-turn memory, also read
   `.agents/skills/langgraph-persistence/SKILL.md`.
4. If the spec calls for approval/pause-for-input steps, also read
   `.agents/skills/langgraph-human-in-the-loop/SKILL.md`.

### Step 4 — Report

Report what was created (files written, commands run) and call out anything the
requirements-gathering step surfaced that the scaffold doesn't yet support — be
explicit, don't silently drop requirements.

## Notes

- There's no fixed capability-token list (no `--capabilities rag,hitl,...` flag).
  Capabilities are handled by pointing to the relevant `.agents/skills/*` reference
  during Step 3 (e.g. a memory requirement → `langgraph-persistence`). Add new
  `.agents/skills/` entries as the project needs them, rather than pre-declaring a
  fixed set that may not have real reference material behind it.
- ADK scaffolding delegates to Google's `agent-starter-pack` CLI (real, maintained
  tooling). LangGraph scaffolding is hand-written by `langgraph-scaffold` since no
  equivalent official CLI exists — expect to iterate on the generated files more than
  you would with ADK's `enhance` command.
