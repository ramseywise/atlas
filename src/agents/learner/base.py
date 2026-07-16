from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field

from src.agents.state import EvalReport, PlannerStrategy


class RewardFunction(BaseModel):
    """Converts multi-metric EvalReport into a scalar reward signal."""

    w_mase: float = Field(default=0.4, ge=0.0, le=1.0)
    w_smape: float = Field(default=0.3, ge=0.0, le=1.0)
    w_directional: float = Field(default=0.3, ge=0.0, le=1.0)

    def compute(self, report: EvalReport) -> float:
        mase_penalty = self.w_mase * report.overall_mase
        smape_penalty = self.w_smape * (report.overall_smape / 100.0)
        dir_penalty = self.w_directional * (1.0 - report.directional_accuracy / 100.0)
        return -(mase_penalty + smape_penalty + dir_penalty)


class LearnerPolicy(ABC):
    """Abstract base for strategy selection policies (rule-based, bandit, RL)."""

    @abstractmethod
    def select_strategy(
        self,
        eval_report: EvalReport,
        current_strategy: PlannerStrategy,
        cycle: int,
    ) -> PlannerStrategy:
        """Choose the next strategy given current evaluation metrics."""
        ...

    @abstractmethod
    def update(self, eval_report: EvalReport, strategy_used: PlannerStrategy) -> None:
        """Update internal state after observing a reward."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist learned parameters to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Load a previously saved policy."""
        ...
