---
name: akira
description: Proactive codebase quality agent. Three modes — kiyoko (yin wanderer, mid-session questions), kaneda (yang scanner, 5 parallel domain subagents, findings doc), dao (the path, LLM triage per finding: auto-fix low-blast-radius changes, surface complex ones for review, discard false positives). Trigger on "akira", "quality check", "what did we miss", or any code quality request.
allowed-tools: Bash
---

# /akira

Delegates to `{source_root}/agents/akira/` — a LangGraph agent with three subgraphs.

## Parse arguments

- `wander`, `kiyoko`, `?` → `make akira-kiyoko`
- no args, `scan`, `kaneda` → `make akira-kaneda`
- `dao`, `fix` → `make akira-dao`
- path glob (e.g. `{source_root}/agents/rag_agent/`) → `PYTHONPATH={source_root} uv run python -m agents.akira kaneda <path>`

## Run

```bash
make akira-kiyoko   # yin: reads delta, asks questions in chat
make akira-kaneda   # yang: 5 subagents, writes {source_root}/agents/akira/findings/findings-{date}.md
make akira-dao      # the path: triages findings, auto-fixes, reverts on test failure, writes summary
```

Findings live at `{source_root}/agents/akira/findings/findings-{date}.md` — review, then `make akira-dao`.
Dao applies everything, reverts what breaks tests, and writes a run summary at the top.

## The 5 kaneda subagents

- **SafeguardAgent** — guardrail coverage across `lg_agent`/`rag_agent`/`adk_agent`
- **SchemaAgent** — `settings.py`/`schema.py` field-name collisions and drift across the three example agents (the same bug class caught twice while building this template — see the puffin-integration plan doc)
- **EvalAgent** — eval-suite conventions in `{eval_root}/`, including the known rag_agent/adk_agent coverage gap
- **CodeQualityAgent** — long functions, missing tests, hardcoded values, dead code (repo-agnostic)
- **DocsAgent** — CLAUDE.md/README claims vs. actual repo structure

Requires `ANTHROPIC_API_KEY` set (`kiyoko` and `dao` make real LLM calls; `kaneda`'s 5 subagents do too).
