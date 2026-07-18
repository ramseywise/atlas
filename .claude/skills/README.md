# Skills Guide

Generic workflows (research/plan/execute/code-review phases, quick-commit, quick-pr,
compact-session, insights, sanyi, …) are **global** — they live in `~/.claude/skills/`
and load in every session. Never copy them here (`~/.claude/CLAUDE.md` → Config Layering).

Phase artifacts: one doc per work item at `.claude/docs/plans/YYYY-MM-DD-<slug>.md`
with a `Status:` line — no SESSION.md, no in-progress/ (convention: `~/.claude/rules/docs.md`).
In atlas, `.claude/docs/plans/` is scratch/local — never committed (hook-enforced).

This directory holds **atlas-specific** skills only:

| Skill | What it does |
|---|---|
| `claude-insights` | Workflow signals from session logs |
| `debug` | Focused diagnose/fix/verify loop |
| `define-epic` / `plan-epic` | ROADMAP.md epic definition and task breakdown |
| `dream` | Companion-doc and memory maintenance pass |
| `eval-report` | Eval reporting |
| `grow-companion` | Update collaboration companion doc |
| `document.md` / `memory_write.md` | Doc + memory helpers |
| `ml-experiment` | ML experiment workflow |
| `new-agent` | Scaffold ADK/LangGraph agent (framework knowledge in `.agents/skills/`) |
| `refactor` | Atlas refactor workflow |
| `segment-analysis` | Segment analysis workflow |
| `sprint-balance` | Review/rebalance Linear sprint milestone assignments |
