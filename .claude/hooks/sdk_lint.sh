#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — enforces AI SDK best practices in src/
# Covers: Anthropic SDK, Google Gemini SDK

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '/src/.*\.py$' || exit 0

issues=""

# --- B1: No bare SDK client instantiation ---
# Anthropic: only the client factory file may call anthropic.Anthropic(
if ! echo "$path" | grep -qE '(utils/client|clients/llm)\.py$'; then
  line=$(grep -n 'anthropic\.Anthropic(' "$path" 2>/dev/null | grep -v '# noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [sdk-factory] use create_client(), not bare anthropic.Anthropic(): $line\n"
fi

# Google: only the factory file may instantiate genai/generativeai clients
if ! echo "$path" | grep -qE '(utils/client|clients/llm)\.py$'; then
  line=$(grep -nE 'genai\.Client\(|generativeai\.GenerativeModel\(|GenerativeModel\(' "$path" 2>/dev/null | grep -v '# noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [sdk-factory] use factory function, not bare Google AI client: $line\n"
fi

# --- B2: No hardcoded model strings ---
# Allow in config.py/settings.py (pydantic-settings default values) and test files
if ! echo "$path" | grep -qE 'config\.py$|settings\.py$|/tests/'; then
  line=$(grep -nE 'model\s*=\s*"claude-|model\s*=\s*"gemini-|model\s*=\s*"gpt-' "$path" 2>/dev/null | grep -v '# noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [sdk-model] use settings for model names, not hardcoded strings: $line\n"
fi

# --- B3: No bare observability client instantiation ---
# Only a factory file named observability.py (anywhere in src/) may
# instantiate a tracing provider client — keeps the provider swappable in one place.
if ! echo "$path" | grep -qE '(^|/)observability\.py$'; then
  line=$(grep -nE 'langfuse\.Langfuse\(|Langfuse\(|langsmith\.Client\(' "$path" 2>/dev/null | grep -v '# noqa' | head -1 || true)
  [ -n "$line" ] && issues="$issues  [observability-factory] instantiate tracing clients only in observability.py: $line\n"
fi

# --- B4: Token usage logging (advisory) ---
has_api_call=$(grep -cE 'client\.messages\.create|\.generate_content\(' "$path" 2>/dev/null)
has_api_call=${has_api_call:-0}
if [ "$has_api_call" -gt 0 ]; then
  has_usage=$(grep -cE 'usage|token' "$path" 2>/dev/null)
  has_usage=${has_usage:-0}
  if [ "$has_usage" -eq 0 ]; then
    echo "Advisory: $path makes API calls but has no token usage logging" >&2
  fi
fi

if [ -n "$issues" ]; then
  printf "SDK lint violations in %s:\n%b" "$path" "$issues" >&2
  exit 2
fi

exit 0
