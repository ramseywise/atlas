from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from src.agents.learner.base import LearnerPolicy, RewardFunction
from src.agents.state import (
    EvalReport,
    ModelVariant,
    PlannerStrategy,
)

CONTEXT_MULTIPLIER_OPTIONS = [2.0, 3.0, 5.0, 8.0]

MODEL_OPTIONS = [
    ModelVariant.CHRONOS_TINY,
    ModelVariant.CHRONOS_MINI,
    ModelVariant.CHRONOS_SMALL,
    ModelVariant.STATSFORECAST_ARIMA,
    ModelVariant.STATSFORECAST_ETS,
]


@dataclass
class ArmState:
    """Beta-distribution posterior for one arm."""

    alpha: float = 1.0
    beta: float = 1.0
    total_reward: float = 0.0
    pulls: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


@dataclass
class StrategyArm:
    """Maps an arm index to a concrete PlannerStrategy configuration."""

    model_variant: ModelVariant
    context_multiplier: float
    arm_id: int = 0


def build_arm_space() -> list[StrategyArm]:
    """Generate the combinatorial strategy space (models × context multipliers)."""
    arms: list[StrategyArm] = []
    arm_id = 0
    for model in MODEL_OPTIONS:
        for ctx in CONTEXT_MULTIPLIER_OPTIONS:
            arms.append(StrategyArm(model_variant=model, context_multiplier=ctx, arm_id=arm_id))
            arm_id += 1
    return arms


class BanditPolicy(LearnerPolicy):
    """Thompson Sampling multi-armed bandit over strategy configurations."""

    def __init__(
        self,
        reward_fn: RewardFunction | None = None,
        reward_threshold: float = -0.5,
        seed: int | None = None,
    ) -> None:
        self.reward_fn = reward_fn or RewardFunction()
        self.reward_threshold = reward_threshold
        self.arms = build_arm_space()
        self.arm_states: list[ArmState] = [ArmState() for _ in self.arms]
        self.rng = np.random.default_rng(seed)
        self._last_arm_idx: int | None = None

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def select_strategy(
        self,
        eval_report: EvalReport,
        current_strategy: PlannerStrategy,
        cycle: int,
    ) -> PlannerStrategy:
        samples = np.array([self.rng.beta(s.alpha, s.beta) for s in self.arm_states])
        best_arm_idx = int(np.argmax(samples))
        self._last_arm_idx = best_arm_idx

        arm = self.arms[best_arm_idx]
        return PlannerStrategy(
            horizon=current_strategy.horizon,
            model_variant=arm.model_variant,
            context_multiplier=arm.context_multiplier,
            feature_flags=current_strategy.feature_flags,
            rationale=f"Bandit selected arm {best_arm_idx}: {arm.model_variant.value} ctx×{arm.context_multiplier}",
        )

    def update(self, eval_report: EvalReport, strategy_used: PlannerStrategy) -> None:
        arm_idx = self._find_arm_idx(strategy_used)
        if arm_idx is None:
            return

        reward = self.reward_fn.compute(eval_report)
        state = self.arm_states[arm_idx]
        state.pulls += 1
        state.total_reward += reward

        if reward > self.reward_threshold:
            state.alpha += 1.0
        else:
            state.beta += 1.0

    def _find_arm_idx(self, strategy: PlannerStrategy) -> int | None:
        if self._last_arm_idx is not None:
            return self._last_arm_idx
        for i, arm in enumerate(self.arms):
            if (
                arm.model_variant == strategy.model_variant
                and abs(arm.context_multiplier - strategy.context_multiplier) < 0.01
            ):
                return i
        return None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "policy": "bandit",
            "reward_threshold": self.reward_threshold,
            "reward_fn": self.reward_fn.model_dump(),
            "arm_states": [
                {"alpha": s.alpha, "beta": s.beta, "total_reward": s.total_reward, "pulls": s.pulls}
                for s in self.arm_states
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> Self:
        data = json.loads(path.read_text())
        policy = cls(
            reward_fn=RewardFunction(**data["reward_fn"]),
            reward_threshold=data["reward_threshold"],
        )
        for i, arm_data in enumerate(data["arm_states"]):
            policy.arm_states[i] = ArmState(**arm_data)
        return policy
