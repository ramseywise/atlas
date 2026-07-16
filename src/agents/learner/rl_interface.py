"""
RL policy scaffold for future PPO/DQN implementation.

Observation space (per cycle):
  - overall_mase: float [0, inf)
  - overall_smape: float [0, 200]
  - directional_accuracy: float [0, 100]
  - coverage_80: float [0, 100]
  - drift_ratio: float [0, inf)
  - current_model_idx: int [0, 4]
  - context_multiplier: float [1, 10]
  - cycle: int [0, max_cycles]

Action space (discrete):
  - model_variant: 5 options
  - context_multiplier: 4 options
  - Total: 20 discrete actions (same as bandit arms)

Reward:
  - Scalar from RewardFunction.compute()
  - Episode terminates when all_passed or max_cycles reached

To implement: subclass LearnerPolicy, wrap in gymnasium.Env,
train with stable-baselines3 PPO or CleanRL DQN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from src.agents.learner.base import LearnerPolicy
from src.agents.state import EvalReport, PlannerStrategy


class RLPolicy(LearnerPolicy):
    """Placeholder for deep RL policy — not yet implemented."""

    def select_strategy(
        self,
        eval_report: EvalReport,
        current_strategy: PlannerStrategy,
        cycle: int,
    ) -> PlannerStrategy:
        raise NotImplementedError(
            "RLPolicy requires training via stable-baselines3 or CleanRL. "
            "See module docstring for observation/action/reward space definitions."
        )

    def update(self, eval_report: EvalReport, strategy_used: PlannerStrategy) -> None:
        raise NotImplementedError("RLPolicy.update() requires a trained model checkpoint.")

    def save(self, path: Path) -> None:
        raise NotImplementedError("RLPolicy.save() requires a trained model.")

    @classmethod
    def load(cls, path: Path) -> Self:
        raise NotImplementedError("RLPolicy.load() requires a model checkpoint at the given path.")
