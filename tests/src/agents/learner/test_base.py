"""Tests for RewardFunction and LearnerPolicy base."""

from __future__ import annotations

from datetime import date

from src.agents.learner.base import RewardFunction
from src.agents.state import EvalReport


def _make_report(
    mase: float = 0.8,
    smape: float = 12.0,
    dir_acc: float = 60.0,
    cov: float = 80.0,
    drift: float = 1.0,
) -> EvalReport:
    return EvalReport(
        cycle_id="test",
        forecast_date=date(2024, 1, 1),
        series_scores={},
        overall_mase=mase,
        overall_smape=smape,
        directional_accuracy=dir_acc,
        coverage_80=cov,
        drift_ratio=drift,
        all_passed=True,
    )


class TestRewardFunction:
    def test_perfect_report_gives_negative_reward(self):
        rf = RewardFunction()
        report = _make_report(mase=0.0, smape=0.0, dir_acc=100.0)
        reward = rf.compute(report)
        assert reward == 0.0

    def test_bad_report_gives_low_reward(self):
        rf = RewardFunction()
        report = _make_report(mase=2.0, smape=50.0, dir_acc=30.0)
        reward = rf.compute(report)
        assert reward < -0.5

    def test_custom_weights(self):
        rf = RewardFunction(w_mase=1.0, w_smape=0.0, w_directional=0.0)
        report = _make_report(mase=1.5)
        reward = rf.compute(report)
        assert abs(reward - (-1.5)) < 0.001

    def test_reward_monotonic_in_mase(self):
        rf = RewardFunction()
        r_good = rf.compute(_make_report(mase=0.5))
        r_bad = rf.compute(_make_report(mase=1.5))
        assert r_good > r_bad

    def test_reward_monotonic_in_directional(self):
        rf = RewardFunction()
        r_good = rf.compute(_make_report(dir_acc=90.0))
        r_bad = rf.compute(_make_report(dir_acc=40.0))
        assert r_good > r_bad
