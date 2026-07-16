#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — detect circular imports in src/

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '/src/.*\.py$' || exit 0

uv run python -c "import agents" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo 'Import cycle detected — fix circular imports before proceeding' >&2; exit 2; }
exit 0
