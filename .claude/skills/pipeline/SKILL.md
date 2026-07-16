---
name: pipeline
description: "Map the full phased workflow; start from a chosen phase with human gates between artifacts."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Write
---

Map the full phased workflow and start from the chosen phase.

## Phases (in order)

| Phase | Skill | Artifact | Gate |
|-------|-------|----------|------|
| 1. Research | `/research <name>` | `.claude/docs/in-progress/<name>/research.md` | Human reviews before continuing |
| 2. Plan | `/plan <name>` | `.claude/docs/in-progress/<name>/plan.md` | Human reviews before continuing |
| 2.5. Plan Review | `/plan-review` | plan file (iterated) | Blockers resolved, questions answered |
| — | `/compact` | — | **Run before execute** |
| 3. Execute | `/execute` | `CHANGELOG.md` | Human confirms each step |
| 4. Review | `/review <name>` | `.claude/docs/in-progress/<name>/review.md` | Verdict: go / no-go |

All phase artifacts live in `.claude/docs/in-progress/<name>/`. `SESSION.md` tracks the active plan and research under `## Active docs`.

Ad-hoc (skip pipeline): `/debug`, `/review`, `/refactor`.

## On invoke

1. If the user asked to **run the full workflow**: run only the first applicable phase. Do not auto-chain — gates matter.
2. If the user named a **specific phase** (e.g. "just plan"): run that phase only.
3. If unclear: show the table above and ask.

## Rules

- Do not spawn subagents — run all phases directly in the main conversation
- Each phase is a separate skill invocation; phases are not chained automatically
- `/compact` before `/execute` is mandatory, not optional
