# SANYI.md — change contract

project: atlas
version: 1
last-audit: 2026-07-17

<!-- Drafted 2026-07-17 from CLAUDE.md hard rules, hook-enforced standards, and
     src/agents/ structure. Buyi entries below passed the consequence test on
     paper; the owner's init interview (confirm/extend Buyi, challenge budgets)
     is still owed. -->

## 不易 Buyi

### Secrets never committed

- paths: .env, .env.example
- contract: LLM/tracing keys live only in .env; no secrets in source or logs.
- evidence: secrets_scan hook (PostToolUse), hardcoded-model-string hook

### Forecast interval integrity

- paths: src/agents/state.py#ForecastResult
- contract: A served forecast must satisfy lower_80 ≤ point_forecast ≤
  upper_80 for every step — enforced by the model validator, not by caller
  discipline. Serving inverted intervals is a financial/trust failure.
- evidence: ForecastResult validator (raises on violation)

### Internal data classification

- paths: data/
- contract: Repo data is classified internal (CLAUDE.md hard rule 4) — never
  uploaded to external services or committed to public remotes.

## 简易 Jianyi

### Agent state schemas

- paths: src/agents/state.py
- contract: Shared state shapes for the domain agents. Note: feature_flags is
  a dict[str, bool] catch-all — value-typed, but new flags enter without
  schema visibility; treat flag additions as growth needing justification
  (JY-3-adjacent).
- budget: new fields and new feature flags need justification in the PR
- current: baseline at first audit (2026-07-17)

### Graph topology

- paths: src/agents/graph.py, src/agents/nodes.py
- contract: The multi-agent graph (crypto, forecast, knowledge, learner,
  segment) is the dominant complexity carrier. New domain agents, edges, or
  cycles need justification; topology changes bump the contract version.
- budget: 5 domain agents; qualitative until scripted flow metrics
- current: 5 agents (2026-07-17)

### API boundary models

- paths: api/
- contract: Pydantic models at all API boundaries (style rule); raw dicts
  crossing the boundary are escape hatches.
- budget: new endpoints/models need justification
- current: baseline at first audit (2026-07-17)

## 变易 Bianyi

### Model selection

- paths: src/agents/state.py#ModelVariant
- contract: Model choice is an enum + factory concern (hook-enforced: no bare
  client instantiation outside factory files, no hardcoded model strings).
  A literal model id in business logic is BN-1.

### Feature flags and thresholds

- paths: src/agents/state.py#ForecastConfig
- contract: Horizon, context multiplier, and feature flags are tunables —
  changing a value must not require touching node/graph logic.

## Migrations

<!-- Empty at init. Format: - YYYY-MM-DD: <from> → <to> / <entry> — <rationale>. (author: <who>) -->

## Pending

### .claude/docs/plans and reviews never committed

- paths: .claude/docs/plans/, .claude/docs/reviews/
- contract: Scratch workspace is never staged or committed (hook-enforced,
  PreToolUse). Disputed layer: hygiene rule vs. confidentiality invariant
  (internal data classification). Enforced as Buyi until the owner resolves.

## Debt

- [BY-4] Internal data classification (data/) — declared invariant has
  docs-only enforcement; no deterministic guard against external upload
  (recorded 2026-07-17)
