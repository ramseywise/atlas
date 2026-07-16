#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — syntax check + auto-format Python files

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '\.py$' || exit 0

python3 -m py_compile "$path" 2>&1
uv run ruff format --quiet "$path" 2>/dev/null
uv run ruff check --quiet --fix "$path" 2>/dev/null
exit 0
