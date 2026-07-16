# Skills Guide

This directory holds the command-style workflows used in this repo.

## Quick map

| Skill | Use when | What it does | Hook automation? |
|---|---|---|---|
| `research-review` | You need to understand a codebase area, bug, or comparison | Writes a research artifact with evidence, gaps, and recommendations | No |
| `plan-review` | You need an implementation plan from research | Writes or refines a step-by-step plan | No |
| `execute-plan` | A plan is approved and ready to build | Implements one plan step at a time and records progress | Partial |
| `execute-tasks` | Working through a TASKS.md milestone | Implements planned tasks one at a time, marks them done | Partial |
| `code-review` | Implementation is complete and needs review | Reviews diff against the plan and writes the review artifact | Partial |
| `review-pr` | An open PR needs review | Reviews a PR against project standards and acceptance criteria | No |
| `plan-refactor` | You want an opportunistic quality refactor | Proposes improvements, then applies them one at a time | Partial |
| `code-debug` | You have a specific traceback or failing test | Focused diagnose/fix/verify loop | No |
| `design-sprint` | Starting a new product or platform initiative | Produces a structured sprint backlog | No |
| `scope-initiative` | An initiative is already named and agreed on | Produces a ticket-ready backlog and hierarchy | No |
| `define-milestones` | Starting a planning cycle | Defines milestones — goal, success metrics, initiative list | No |
| `doc-to-linear-tickets` | You have a planning doc to turn into tickets | Parses a doc and creates structured Linear issues | No |
| `github-projects` | Syncing status/iteration with GitHub Projects V2 | Provides GraphQL templates for common sync operations | No |
| `quick-commit` | You want a branch + commit only | Creates a feature branch and commits safely | Partial |
| `quick-pr` | You want commit + push + PR flow | Handles staging, commit, push, PR creation, optional merge | Partial |
| `mcp-builder` | Building or extending an MCP server | Guides Python/FastMCP or Node/TS MCP server design, tools, and evals | No |
| `new-agent` | Scaffolding a new ADK or LangGraph agent | Picks a framework via `.agents/skills/framework-selection`, then hands off to the matching scaffold skill | No |
| `claude-insights` | You want workflow signals from session logs | Summarizes friction patterns, attribution, and skill candidates | No |
| `compact-session` | Mid-session or end of session | Saves artifacts, writes session note, commits + pushes + PR; mid-session also compacts context | No |
| `prototype` | Quickly validating an idea or API before committing | Spike/exploration mode — explicitly not for production code | No |
| `skill-creator` | Capturing a workflow as a reusable skill | Creates or improves a SKILL.md, iterates from test results | No |
| `sanyi` | Enforcing an architectural change-contract across many diffs | Classifies components into 变易/简易/不易 layers, catches decay `code-review` structurally can't see | No |
## Agent framework reference

`new-agent` is a thin orchestrator. The actual ADK/LangGraph framework knowledge lives
in `.agents/skills/` — a tool-agnostic library also readable by ADK's own tooling or
any other agent harness in this project. See `.agents/skills/README.md` for the map.

## Stale assumptions to watch

- `CHANGELOG.md` should be treated as optional unless a workflow explicitly uses it.
- Compact reminders are a single workflow hint, not a repeated instruction everywhere.
- Hooks should own enforcement; skills should describe intent and manual steps.
- `doc-to-linear-tickets` and the `LIN-{id}` branch convention assume Linear — swap or remove if this project uses Jira/GitHub Issues instead.
