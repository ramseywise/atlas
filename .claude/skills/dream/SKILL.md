---
name: dream
description: Use when user says 'dream', 'tidy up', 'maintenance', 'clean up', or when the companion doc or memory index feel stale between sessions. Lightweight maintenance pass — audits companion-doc staleness, session-note backlog, and memory bloat; does light consolidation. Not for feature work.
---

# Dream

A reflective pass over the project's accumulated context. Survey what exists, find what's drifted, tidy what's grown messy. Maintenance, not transformation.

## Phase 1: Orient

Survey the landscape. This IS the audit — if nothing needs attention, report that and stop.

### Discover Structure

Don't assume paths — discover them:

```
Glob: .claude/docs/companion/*.md
Glob: ~/.claude/sessions/*.md
```

For the memory index (`MEMORY.md`), location varies by environment — discover it rather than assuming a fixed path; it's typically under a per-project directory alongside other memory files.

### Check State

| Check | What to look for |
|-------|-------------------|
| Companion doc freshness | "Last Transformed" date in `collaboration.md` — older than 4 weeks? |
| Session-note backlog | How many session notes since the last `/dream`? Piling up unprocessed? |
| Memory index size | Is `MEMORY.md` under 200 lines? |
| Memory duplication | Do multiple memory files cover the same ground? |
| Contradictions | Does the companion doc say one thing while recent session notes show another? |

Report findings before proceeding. If everything is clean, say so and stop.

## Phase 2: Gather Signal

Read what matters based on Phase 1 findings — don't read exhaustively:
- If the companion doc is stale → read it plus the 3-5 most recent session notes
- If the memory index is bloated → read `MEMORY.md`
- If contradictions are suspected → read the companion doc plus the specific conflicting session note

## Phase 3: Consolidate

Act on what you found — any combination of:

### Refresh the Companion Doc (if stale or contradicted)
Same discipline as `/grow-companion`: transform, don't append. Update "Last Transformed".

### Tidy the Memory Index (if over 200 lines)
Compress or merge redundant entries. Never delete content that's still true — compress it.

### Resolve Contradictions
Trust recent session notes over a stale companion doc — rewrite the stale section, don't just note the conflict.

## Phase 4: Report

```
Dream complete.

State: [clean / tidied / needs deeper work]

Found:
- [what was checked]

Done:
- [what was consolidated or tidied]

Flagged for attention:
- [anything needing manual attention]
```

## Critical Rules

- **Discover, never assume.** Paths vary by project and environment.
- **Light touch.** Maintenance, not transformation — a real rewrite belongs to `/grow-companion`, not this skill.
- **Never delete true content.** Compress, consolidate — never lose it.
- **Stop early if clean.** That's a valid outcome, not a failure to find something.
