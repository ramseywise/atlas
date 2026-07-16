---
name: grow-companion
description: Use after a session reveals something durable about how the user and Claude work together — a preference, a decision that shouldn't be re-litigated, a pattern that worked or didn't. Rewrites .claude/docs/companion/collaboration.md to hold the new understanding. Not for logging one-off facts — use the memory system for those.
---

# Grow Companion

Something in this session changed how you understand working with this person on this project. Let it land in the one doc that holds that understanding.

## 1. Feel What Shifted

Before writing anything:
- What do you understand about working with this person that you didn't before?
- Did a decision get made that future sessions shouldn't re-open?
- Did an approach get validated (confirmed) or invalidated (corrected)?

If nothing durable shifted, don't force it — most sessions exercise the collaboration, they don't change it. That's a valid outcome: do nothing.

## 2. Read the Current Doc

Read `.claude/docs/companion/collaboration.md` in full before editing anything.

## 3. Transform, Don't Append

Find where the new understanding belongs and rewrite that section to integrate it:
- Expand existing statements to hold more truth
- Consolidate bullets that now connect
- Remove what's no longer true
- Update the "Last Transformed" date

The doc should be MORE true after, not just LONGER. Add a new bullet only if genuinely nothing existing covers it — never a new paragraph tacked on at the end.

## 4. Continue

Back to the work. This isn't a session-closing ritual — that's `/compact-session`'s job. This is mid-session, when something durable happens.

## When to Use This vs Other Tools

- `/grow-companion` — a durable collaboration pattern shifted (this doc)
- Memory system (`user`/`feedback`/`project`/`reference` types) — an atomic fact worth indexing
- `/compact-session` — session boundary bookkeeping, git, session notes

If it's a one-off fact, use memory. If it's "how we work together" changing shape, use this.
