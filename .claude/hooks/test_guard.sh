#!/usr/bin/env bash
# PostToolUse hook for Write|Edit — advisory test coverage warnings
# Two checks in one pass:
#   1. Public functions in src/ without matching tests (all Python)
#   2. Newly added public API in src/agents/ without test file coverage
# Advisory only (exit 0) — does not block edits

path=$(echo "$CLAUDE_TOOL_INPUT" | jq -r '.file_path // empty')
[ -z "$path" ] && exit 0
echo "$path" | grep -qE '/src/.*\.py$' || exit 0

basename=$(basename "$path")
echo "$basename" | grep -qE '^__' && exit 0
echo "$basename" | grep -qE '^(config|settings|paths|constants|exceptions)\.py$' && exit 0

# --- Check 1: public function coverage (all src/) ---
rel=${path#*src/}
pkg_dir=$(dirname "$rel")
mod=$(basename "$rel" .py)
test_file="tests/${pkg_dir}/test_${mod}.py"

src_funcs=$(uv run python3 -c "
import ast, sys
with open('$path') as f:
    tree = ast.parse(f.read())
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not node.name.startswith('_'):
            print(node.name)
" 2>/dev/null | sort -u)

if [ -n "$src_funcs" ]; then
  if [ ! -f "$test_file" ]; then
    count=$(echo "$src_funcs" | wc -l | tr -d ' ')
    echo "Coverage: $test_file does not exist — $count public function(s) untested" >&2
  else
    missing=""
    while IFS= read -r func; do
      [ -z "$func" ] && continue
      if ! grep -qE "def test_.*${func}|def test_${func}" "$test_file" 2>/dev/null; then
        missing="$missing  $func\n"
      fi
    done <<< "$src_funcs"
    if [ -n "$missing" ]; then
      printf "Coverage: untested public functions in %s:\n%b" "$path" "$missing" >&2
    fi
  fi
fi

# --- Check 2: newly added public API in src/agents/ ---
echo "$path" | grep -qE '/src/agents/.*\.py$' || exit 0

python3 - "$path" <<'PY'
import ast, os, re, subprocess, sys

path = sys.argv[1]
repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
abs_path = os.path.abspath(path)
rel = os.path.relpath(abs_path, repo)
test_rel = rel.replace("src/agents/", "tests/")
test_rel = os.path.join(os.path.dirname(test_rel), f"test_{os.path.basename(test_rel)}")
test_path = os.path.join(repo, test_rel)

diff = subprocess.run(["git", "diff", "--unified=0", "--", rel], capture_output=True, text=True, check=False).stdout
added = set()
for line in diff.splitlines():
    if not line.startswith("+") or line.startswith("+++"):
        continue
    text = line[1:].lstrip()
    m = re.match(r"(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
    if m and not m.group(1).startswith("_"):
        added.add(m.group(1))
    m = re.match(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\b", text)
    if m and not m.group(1).startswith("_"):
        added.add(m.group(1))

status = subprocess.run(["git", "status", "--porcelain", "--", rel], capture_output=True, text=True, check=False).stdout.strip()
if not diff.strip() and status.startswith("??"):
    with open(abs_path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=abs_path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            added.add(node.name)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            added.add(node.name)

if not added:
    raise SystemExit(0)
if not os.path.exists(test_path):
    print(f"Coverage: {test_rel} missing — new public API needs tests: {', '.join(sorted(added))}", file=sys.stderr)
    raise SystemExit(0)

with open(test_path, encoding="utf-8") as fh:
    test_text = fh.read()
missing = [n for n in sorted(added) if f"def test_{n}" not in test_text and n not in test_text]
if missing:
    print(f"Coverage: add tests for new public API in {path}: {', '.join(missing)}", file=sys.stderr)
PY

exit 0
