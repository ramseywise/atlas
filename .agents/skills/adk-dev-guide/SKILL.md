---
name: adk-dev-guide
description: >
  ALWAYS ACTIVE — read at the start of any ADK agent development session.
  ADK development lifecycle and mandatory coding guidelines — spec-driven
  workflow, code preservation rules, model selection, and troubleshooting.
metadata:
  license: Apache-2.0
  author: Google
  source: ported into this project's .agents/skills/ reference library — see LICENSE.txt in this directory
---

# ADK Development Workflow & Guidelines

> **Note on companion skills:** the upstream version of this guide references
> `adk-cheatsheet`, `adk-eval-guide`, `adk-deploy-guide`, and `adk-observability-guide`.
> Only `adk-scaffold` and this guide are included in this project's `.agents/skills/`
> library so far. Where this doc points to one of the others, fall back to the
> official ADK documentation or a targeted web search — port the missing skill into
> `.agents/skills/` if the gap comes up repeatedly.

## Session Continuity

If this is a long session, re-read this guide before each phase — the guidance below
covers scaffolding, implementation, evaluation, and deployment. Context compaction may
have dropped earlier content.

---

## DESIGN_SPEC.md — Your Primary Reference

**IMPORTANT**: If `DESIGN_SPEC.md` exists in this project, it is your primary source of truth.

Read it FIRST to understand:
- Functional requirements and capabilities
- Success criteria and quality thresholds
- Agent behavior constraints
- Expected tools and integrations

**The spec is your contract.** All implementation decisions should align with it. When in doubt, refer back to DESIGN_SPEC.md.

---

## Phase 1: Understand the Spec

Before writing any code:
1. Read `DESIGN_SPEC.md` thoroughly
2. Identify the core capabilities required
3. Note any constraints or things the agent should NOT do
4. Understand success criteria for evaluation

## Phase 2: Build and Implement

Implement the agent logic:

1. Write/modify code in the agent directory (check the agent guidance file, e.g. GEMINI.md or CLAUDE.md, for directory name)
2. Use `make playground` (or `adk web .`) for interactive testing during development
3. Iterate on the implementation based on user feedback

For ADK API patterns and code examples, consult the official ADK documentation (a dedicated `adk-cheatsheet` reference is not yet ported into this project).

## Phase 3: Evaluate

**This is the most important phase.** Evaluation validates agent behavior end-to-end using evalsets and scoring metrics.

**Tests (`pytest`) are NOT evaluation.** They test code correctness but say nothing about whether the agent behaves correctly. Always run `adk eval`.

1. **Start small**: Begin with 1-2 sample eval cases, not a full suite
2. Run evaluations: `adk eval` (or `make eval` if the project has a Makefile)
3. Discuss results with the user
4. Fix issues and iterate on the core cases first
5. Only after core cases pass, add edge cases and new scenarios
6. Repeat until quality thresholds are met

**Expect 5-10+ iterations here.**

## Phase 4: Deploy

Once evaluation thresholds are met:

1. Deploy when ready — see `.agents/skills/adk-scaffold/SKILL.md` for `agent-starter-pack enhance` deployment commands (Agent Engine, Cloud Run)

**IMPORTANT**: Never deploy without explicit human approval.

---

# Operational Guidelines for Coding Agents

## Principle 1: Code Preservation & Isolation

When executing code modifications, your paramount objective is surgical precision. You **must alter only the code segments directly targeted** by the user's request, while **strictly preserving all surrounding and unrelated code.**

**Mandatory Pre-Execution Verification:**

Before finalizing any code replacement, verify:

1.  **Target Identification:** Clearly define the exact lines or expressions to be changed, based *solely* on the user's explicit instructions.
2.  **Preservation Check:** Ensure all code, configuration values (e.g., `model`, `version`, `api_key`), comments, and formatting *outside* the identified target remain identical.

**Example:**

*   **User Request:** "Change the agent's instruction to be a recipe suggester."
*   **Incorrect (VIOLATION):**
    ```python
    root_agent = Agent(
        name="recipe_suggester",
        model="gemini-1.5-flash",  # UNINTENDED - model was not requested to change
        instruction="You are a recipe suggester."
    )
    ```
*   **Correct (COMPLIANT):**
    ```python
    root_agent = Agent(
        name="recipe_suggester",  # OK, related to new purpose
        model="gemini-3-flash-preview",  # PRESERVED
        instruction="You are a recipe suggester."  # OK, the direct target
    )
    ```

## Principle 2: Execution Best Practices

*   **Model Selection — CRITICAL:**
    *   **NEVER change the model unless explicitly asked.** If the code uses `gemini-3-flash-preview`, keep it as `gemini-3-flash-preview`. Do NOT "upgrade" or "fix" model names.
    *   When creating NEW agents (not modifying existing), use Gemini 3 series: `gemini-3-flash-preview`, `gemini-3-pro-preview`.
    *   Do NOT use older models (`gemini-2.0-flash`, `gemini-1.5-flash`, etc.) unless the user explicitly requests them.

*   **Location Matters More Than Model:**
    *   If a model returns a 404, it's almost always a `GOOGLE_CLOUD_LOCATION` issue (e.g., needing `global` instead of `us-central1`).
    *   Changing the model name to "fix" a 404 is a violation — fix the location instead.
    *   Some models (like `gemini-3-flash-preview`) require specific locations. Check the error message for hints.

*   **ADK Built-in Tool Imports (Precision Required):**
    ```python
    # CORRECT - imports the tool instance
    from google.adk.tools.load_web_page import load_web_page

    # WRONG - imports the module, not the tool
    from google.adk.tools import load_web_page
    ```
    Pass the imported tool directly to `tools=[load_web_page]`, not `tools=[load_web_page.load_web_page]`.

*   **Running Python Commands:**
    *   Always use `uv` to execute Python commands (e.g., `uv run python script.py`)
    *   Run `make install` (or `uv sync`) before executing scripts
    *   Consult `Makefile` and `README.md` for available commands (if present)

*   **Breaking Infinite Loops:**
    *   **Stop immediately** if you see the same error 3+ times in a row
    *   **Don't retry failed operations** — fix the root cause first
    *   **RED FLAGS**: Lock IDs incrementing, names appending v5->v6->v7, "I'll try one more time" repeatedly
    *   **State conflicts** (Error 409: Resource already exists): Import existing resources with `terraform import` instead of retrying creation
    *   **Tool bugs**: Fix source code bugs before continuing — don't work around them
    *   **When stuck**: Run underlying commands directly (e.g., `terraform` CLI) instead of calling problematic tools

*   **Troubleshooting:**
    *   Search the installed ADK package with Glob/Grep/Read (find it with `python -c "import google.adk; print(google.adk.__path__[0])"` — use `uv run python` if using uv)
    *   For framework questions or GCP products, check official documentation
    *   When encountering persistent errors, a targeted web search often finds solutions faster

---

## Scaffold as Reference

When you need specific infrastructure files but don't want to scaffold the current project directly, use `.agents/skills/adk-scaffold/SKILL.md` to create a temporary reference project in `/tmp/` and copy over what you need.

---

## Development Commands

Projects created via the Agent Starter Pack CLI include a Makefile with these commands.
To create or enhance a project, see `.agents/skills/adk-scaffold/SKILL.md` for full instructions.
For non-scaffolded projects, use the ADK CLI equivalents.

| Make (scaffolded) | ADK CLI equivalent | Purpose |
|------------|-------------------|---------|
| `make playground` | `adk web .` | Interactive local testing |
| `make test` | `pytest` | Run unit and integration tests |
| `make eval` | `adk eval <agent_dir> <evalset>` | Run evaluation against evalsets |
| `make lint` | `ruff check .` | Check code quality |
| `make setup-dev-env` | — (scaffolded only) | Set up dev infrastructure (Terraform) |
| `make deploy` | — (scaffolded only) | Deploy to dev (requires human approval) |
