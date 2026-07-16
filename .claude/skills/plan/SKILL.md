---
name: plan
description: "Phase 2. Reads research from SESSION.md active docs and produces a concrete, step-by-step implementation plan. Writes to .claude/docs/in-progress/<name>/plan.md."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Write
---

You are a principal engineer writing an implementation plan. Do not write production code. Do not implement anything.

`$ARGUMENTS` — plan name (snake_case). If omitted, derive from SESSION.md active research.

## Before planning

1. Read `SESSION.md` → find `## Active docs` → load the active `research.md`
2. If no research exists and the task is small and well-scoped: use `specs/from-scratch.md` protocol
3. Run `specs/scope.md` — declare Out of Scope before writing any steps
4. Run `specs/check.md` at the end to verify the plan will achieve the goal

## PLAN.md template

```markdown
## Plan: [name]
Date: [today]
Based on: research/<name>.md | direct codebase inspection

## Goal
[One sentence — what will be true when this plan is executed successfully]

## Out of scope
- [specific named thing] — [one-line reason]

## Open questions (resolved before planning)
- Q: [question]  A: [answer or "deferred to Step N"]

## Steps

### Step N: [step name]
**Files**: `src/path/file.py` (lines M-N if relevant)
**What**: [plain-language description]
**Snippet**:
\`\`\`python
# before
[existing code]
# after
[new code]
\`\`\`
**Test**: `uv run pytest tests/path/test_file.py::test_name -xvs`
**Done when**: [observable/testable condition — not "dependency installed"]

## Risks & Rollback
[Per-step failure modes, blast radius, rollback commands — see specs/risk.md]
```

## Plan quality rules

- Every step must have: files, what, snippet or equivalent, test command, done-when
- "Done when" must be observable: "pytest passes" not "module updated"
- Out of Scope must name concrete things — not "anything not mentioned above"
- 2–4 steps: good. 5–6: warning. 7+: split required (see specs/scope.md)
- Steps must be sequenced with no forward dependencies
- Run `specs/check.md` before declaring the plan ready

## Iterating the plan

Use `specs/iterate.md` to update PLAN.md from feedback. Surgical edits only — preserve numbering, test commands, and done-when conditions.

## Risk section

Use `specs/risk.md` for the Risks & Rollback section. Generic risks are useless. Every step with Data or higher blast radius gets its own entry with a concrete rollback command.
