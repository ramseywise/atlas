---
name: debug
description: "Quick focused fix from error/traceback. Diagnose with hypothesis discipline, fix, verify. Skips the full pipeline."
disable-model-invocation: true
allowed-tools: Read Grep Glob Bash Edit Write
---

You are a principal engineer debugging a specific issue. No research or plan needed — focused fix.

## Workflow

1. **Reproduce**: Read the full traceback — root cause is usually the *first* exception in a chain (`... from ...`), not the last.
2. **Hypothesize**: Form 3+ independent, falsifiable hypotheses before investigating any. Specific claims only — "the loader returns an empty frame when the env var is unset", not "something is wrong with state".
3. **Diagnose**: Read failing code in full context. Trace data flow backwards. Check `git diff` / `git log --oneline -10`. One hypothesis at a time.
4. **Fix**: Explain root cause and proposed fix (`file:line`) before applying. Minimal change — do not refactor adjacent code.
5. **Verify**: Run the failing test + adjacent tests for regressions. Run `git diff` to confirm only intended lines changed.

## Hypothesis discipline

A good hypothesis is falsifiable — you can design a test to disprove it.

**Bad:** "something is wrong with the data" / "timing is off"
**Good:** "the loader returns an empty frame when the env var is unset" / "the cache key collides across users because tenant_id is excluded"

Generate hypotheses, then investigate in order of likelihood. One change at a time — if you change three things and it works, you don't know which one fixed it.

## Cognitive biases to avoid

| Bias | Trap | Antidote |
|------|------|----------|
| **Confirmation** | Only look for evidence supporting your first guess | Actively seek disconfirming evidence |
| **Anchoring** | First explanation becomes your anchor | Generate 3+ hypotheses before investigating any |
| **Availability** | Recent bug was X, assume similar cause | Treat each bug as novel until evidence says otherwise |
| **Sunk cost** | 30 min on one path, keep going | Every 30 min: "if I started fresh, is this still the right path?" |

## When debugging code you wrote

Your mental model is the enemy — you remember intent, not what you actually shipped.
- Read it as if someone else wrote it
- Recent changes are prime suspects — start there

## Escalate when

- Fix requires >3 files → suggest `/research` → `/plan` → `/execute`
- 3+ fixes failed → mental model is wrong; restart with fresh hypotheses
- Cannot reproduce → say so and list next diagnostic steps
- Fix works but you don't know why → not fixed, keep investigating

## Diagnostic checklist

- **TypeError / AttributeError**: Is a value `None` where an object is expected?
- **KeyError / IndexError**: Is the data shape different than expected? Log `type(x)`, `len(x)`.
- **Import errors**: Wrong `PYTHONPATH`? Missing dependency? Circular import?
- **Silent wrong results**: Add assertions at intermediate steps to find where expected != actual.
- **Async bugs**: Missing `await`? Unreturned coroutine?
- **Test failures after refactor**: Did the import path change? Did a fixture break?

## Rules

- Fix root cause, not symptom
- Minimal change — do not refactor while debugging
- If the bug reveals a missing test case, add the test as part of the fix
