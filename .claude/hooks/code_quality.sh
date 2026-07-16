#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — enforces baseline Python code standards in src/.
# Blocks (exit 2) on violations. Agent must fix before proceeding.

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '/src/.*\.py$' || exit 0

issues=""

# [no-print] No print() in source — use a logger instead
line=$(grep -n 'print(' "$path" 2>/dev/null | grep -v '# noqa' | head -1 || true)
[ -n "$line" ] && issues="$issues  [no-print] use a logger, not print(): $line\n"

# [bare-except] No bare except — catch specific exceptions
line=$(grep -nE '^[[:space:]]*except:' "$path" 2>/dev/null | head -1 || true)
[ -n "$line" ] && issues="$issues  [bare-except] catch specific exceptions: $line\n"

# [mutable-default] No mutable default arguments
line=$(grep -nE 'def \w+\([^)]*=\s*(\[\]|\{\})' "$path" 2>/dev/null | head -1 || true)
[ -n "$line" ] && issues="$issues  [mutable-default] avoid mutable default arguments: $line\n"

if [ -n "$issues" ]; then
  printf "Code standard violations in %s:\n%b" "$path" "$issues" >&2
  exit 2
fi

exit 0
