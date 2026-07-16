#!/usr/bin/env bash
# PreToolUse(Bash) — .claude/docs/plans/ and .claude/docs/reviews/ are scratch/local
# workspace (research-in-progress, review drafts), not durable project knowledge.
# Block them from ever being staged or committed; promote durable findings into
# project docs instead.

set -euo pipefail

source "$(dirname "$0")/lib.sh"

cmd=$(claude_command)
echo "$cmd" | grep -qE 'git (add|commit)' || exit 0

# Direct reference in the command itself (e.g. `git add .claude/docs/plans/foo.md`)
direct=$(echo "$cmd" | grep -oE '\.claude/docs/(plans|reviews)/[^[:space:];|&]*' || true)

# Already-staged scratch docs (e.g. `git add -A` earlier, then `git commit`)
staged=""
if echo "$cmd" | grep -qE 'git commit'; then
  staged=$(git diff --cached --name-only 2>/dev/null | grep -E '^\.claude/docs/(plans|reviews)/' || true)
fi

matches=$(printf '%s\n%s' "$direct" "$staged" | sed '/^$/d' | sort -u || true)
[ -z "$matches" ] && exit 0

block "BLOCKED: .claude/docs/plans/ and .claude/docs/reviews/ are scratch/local-only, not for git.
Matched: $(echo "$matches" | tr '\n' ' ')
Promote durable findings into project docs instead, or unstage these paths."
