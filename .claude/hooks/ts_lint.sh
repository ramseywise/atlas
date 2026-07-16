#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — runs eslint on / after any TS file edit.
# Skips gracefully when node_modules/.bin/eslint is absent (deps not installed).
# Blocks (exit 2) on lint errors so the agent must fix them before proceeding.

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '\.(tsx?|ts)$' || exit 0

# Derive the  project root from the file path
project_root=$(echo "$path" | grep -oE '.*/' | head -1)
[ -z "$project_root" ] && exit 0

eslint_bin="$project_root/node_modules/.bin/eslint"
[ -x "$eslint_bin" ] || exit 0  # skip if deps not installed

result=$(cd "$project_root" && "$eslint_bin" "$path" 2>&1)
rc=$?

if [ $rc -ne 0 ]; then
  printf "ESLint errors in %s:\n%s\n" "$path" "$result" >&2
  exit 2
fi

exit 0
