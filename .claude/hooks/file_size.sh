#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — warn when Python files exceed 400 lines

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '\.py$' || exit 0

lines=$(wc -l < "$path" 2>/dev/null || echo 0)
[ "$lines" -gt 400 ] && echo "Warning: $path is $lines lines (>400) — consider splitting" >&2
exit 0
