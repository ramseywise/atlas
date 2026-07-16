"""Tests for BanditPolicy Thompson Sampling."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from src.agents.learner.bandit import BanditPolicy, build_arm_space
from src.agents.state import EvalReport, ForecastHorizon, ModelVariant, PlannerStrategy


def _make_report(mase: float = 0.8, dir_acc: float = 60.0) -> EvalReport:
    return EvalReport(
        cycle_id="test",
        forecast_date=date(2024, 1, 1),
        series_scores={},
        overall_mase=mase,
        overall_smape=12.0,
        directional_accuracy=dir_acc,
        coverage_80=80.0,
        drift_ratio=1.0,
        all_passed=True,
    )


def _default_strategy() -> PlannerStrategy:
    return PlannerStrategy(
        horizon=ForecastHorizon.MONTH,
        model_variant=ModelVariant.CHRONOS_TINY,
        context_multiplier=3.0,
    )


class TestBanditPolicy:
    def test_arm_space_has_expected_size(self):
        arms = build_arm_space()
        assert len(arms) == 20  # 5 models × 4 ctx multipliers

    def test_select_strategy_returns_valid_strategy(self):
        policy = BanditPolicy(seed=42)
        report = _make_report()
        strategy = _default_strategy()

        new_strategy = policy.select_strategy(report, strategy, cycle=0)
        assert isinstance(new_strategy, PlannerStrategy)
        assert new_strategy.model_variant in ModelVariant
        assert 1.0 <= new_strategy.context_multiplier <= 10.0

    def test_update_modifies_arm_state(self):
        policy = BanditPolicy(seed=42)
        report = _make_report(mase=0.3, dir_acc=80.0)
        strategy = _default_strategy()

        policy.select_strategy(report, strategy, cycle=0)
        initial_alpha = policy.arm_states[policy._last_arm_idx].alpha

        policy.update(report, strategy)
        final_alpha = policy.arm_states[policy._last_arm_idx].alpha

        assert final_alpha > initial_alpha or policy.arm_states[policy._last_arm_idx].beta > 1.0

    def test_save_and_load_roundtrip(self):
        policy = BanditPolicy(seed=42)
        report = _make_report()
        strategy = _default_strategy()

        policy.select_strategy(report, strategy, cycle=0)
        policy.update(report, strategy)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bandit.json"
            policy.save(path)

            loaded = BanditPolicy.load(path)
            assert loaded.n_arms == policy.n_arms
            assert loaded.arm_states[0].alpha == policy.arm_states[0].alpha

    def test_deterministic_with_seed(self):
        report = _make_report()
        strategy = _default_strategy()

        s1 = BanditPolicy(seed=123).select_strategy(report, strategy, cycle=0)
        s2 = BanditPolicy(seed=123).select_strategy(report, strategy, cycle=0)

        assert s1.model_variant == s2.model_variant
        assert s1.context_multiplier == s2.context_multiplier

    def test_exploration_varies_with_different_seeds(self):
        report = _make_report()
        strategy = _default_strategy()

        strategies = set()
        for seed in range(20):
            s = BanditPolicy(seed=seed).select_strategy(report, strategy, cycle=0)
            strategies.add((s.model_variant, s.context_multiplier))

        assert len(strategies) > 1
