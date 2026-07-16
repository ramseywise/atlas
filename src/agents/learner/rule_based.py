from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from src.agents.learner.base import LearnerPolicy
from src.agents.state import (
    EvalReport,
    ModelVariant,
    PlannerStrategy,
)

MODEL_LADDER = [
    ModelVariant.CHRONOS_TINY,
    ModelVariant.CHRONOS_MINI,
    ModelVariant.CHRONOS_SMALL,
]


class RuleBasedPolicy(LearnerPolicy):
    """Deterministic if-then strategy adaptation — the original learner logic."""

    def select_strategy(
        self,
        eval_report: EvalReport,
        current_strategy: PlannerStrategy,
        cycle: int,
    ) -> PlannerStrategy:
        current_idx = (
            MODEL_LADDER.index(current_strategy.model_variant)
            if current_strategy.model_variant in MODEL_LADDER
            else 0
        )

        new_idx = current_idx
        ctx_mult = current_strategy.context_multiplier
        flags = dict(current_strategy.feature_flags)
        changes: list[str] = []

        if eval_report.overall_mase > 0.9:
            new_idx = min(current_idx + 1, len(MODEL_LADDER) - 1)
            changes.append(f"Upgraded model: {MODEL_LADDER[current_idx]} → {MODEL_LADDER[new_idx]}")

        if eval_report.overall_smape > 20.0:
            flags["decompose_by_category"] = True
            changes.append("Enabled decompose_by_category due to high SMAPE")

        if eval_report.directional_accuracy < 55.0:
            ctx_mult = min(ctx_mult + 1.0, 8.0)
            changes.append(f"Increased context_multiplier to {ctx_mult}")

        if eval_report.coverage_80 < 70.0:
            new_idx = min(new_idx + 1, len(MODEL_LADDER) - 1)
            changes.append("Upgraded model for better interval calibration")

        rationale = " | ".join(changes) if changes else "No changes — metrics within thresholds"

        return PlannerStrategy(
            horizon=current_strategy.horizon,
            model_variant=MODEL_LADDER[new_idx],
            context_multiplier=ctx_mult,
            feature_flags=flags,
            rationale=rationale,
        )

    def update(self, eval_report: EvalReport, strategy_used: PlannerStrategy) -> None:
        pass

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"policy": "rule_based", "model_ladder": [m.value for m in MODEL_LADDER]})
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls()
