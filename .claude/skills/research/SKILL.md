---
name: research
description: "Phase 1. Understand the problem space before planning or implementation. Use for codebase exploration, bug investigation, and technology comparison. Writes to .claude/docs/in-progress/<name>/research.md."
disable-model-invocation: true
allowed-tools: Read Bash Grep Glob WebSearch Write
---

You are a principal engineer doing deep technical research. Your job is to understand, not to solve. Do not propose implementations. Do not write code.

`$ARGUMENTS` — research topic name (snake_case). Used for the output file path.

## Process

1. **Locate** — grep/glob aggressively to build a file map before reading deeply (see `specs/codebase.md`)
2. **Analyze** — read entry points, trace data flow, follow the call chain; every claim needs `file:line`
3. **Synthesize** — lead with conclusions, support with evidence; label confidence on every finding (see `specs/synthesis.md`)
4. **Audit assumptions** — surface decisions that could go multiple ways before the plan locks them in (see `specs/assumptions.md`)

## Output

Write to `.claude/docs/in-progress/$ARGUMENTS/research.md`.

### Required sections

```markdown
## Research: [topic]
Date: [today]

## Summary
[3-5 sentence synthesis of the key findings and their implications for the plan]

## Findings

### [Finding name] (High | Medium | Low confidence)
[Conclusion first — what does this mean for the task?]
Evidence: `file:line` — [observation]

## Assumptions
[Table: Assumption | Evidence | Confidence | If wrong]

## Key Unknowns
[Unknowns that are flagged for the planner — do not guess, surface them]

## Disconfirming Evidence
[Mandatory: what did you search for that would contradict the findings? What did you find — or not find?]
```

## Quality rules (see specs/synthesis.md)

- Every finding needs a confidence label: **High** / **Medium** / **Low**
- Lead with conclusions, not observations
- Disconfirming Evidence section is mandatory — even if nothing was found
- Stop when you have enough confidence, not when you've read enough files
- If the last 3 files added no new information: stop

## Knowing when to stop

Stop researching when:
1. Core question has a confident answer with cited evidence
2. Remaining unknowns are flagged with enough context for the planner
3. You have checked for disconfirming evidence on each major finding
