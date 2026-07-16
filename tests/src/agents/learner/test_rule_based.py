"""Tests for RuleBasedPolicy — verifies parity with original inline logic."""

from __future__ import annotations

from datetime import date

from src.agents.learner.rule_based import RuleBasedPolicy
from src.agents.state import EvalReport, ForecastHorizon, ModelVariant, PlannerStrategy


def _make_report(**kwargs) -> EvalReport:
    defaults = {
        "cycle_id": "test",
        "forecast_date": date(2024, 1, 1),
        "series_scores": {},
        "overall_mase": 0.7,
        "overall_smape": 10.0,
        "directional_accuracy": 65.0,
        "coverage_80": 80.0,
        "drift_ratio": 1.0,
        "all_passed": True,
    }
    defaults.update(kwargs)
    return EvalReport(**defaults)


def _default_strategy() -> PlannerStrategy:
    return PlannerStrategy(
        horizon=ForecastHorizon.MONTH,
        model_variant=ModelVariant.CHRONOS_TINY,
        context_multiplier=3.0,
    )


class TestRuleBasedPolicy:
    def test_no_changes_when_all_good(self):
        policy = RuleBasedPolicy()
        report = _make_report()
        strategy = _default_strategy()

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.model_variant == ModelVariant.CHRONOS_TINY
        assert result.context_multiplier == 3.0

    def test_upgrades_model_on_high_mase(self):
        policy = RuleBasedPolicy()
        report = _make_report(overall_mase=1.2)
        strategy = _default_strategy()

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.model_variant == ModelVariant.CHRONOS_MINI

    def test_enables_decompose_on_high_smape(self):
        policy = RuleBasedPolicy()
        report = _make_report(overall_smape=25.0)
        strategy = _default_strategy()

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.feature_flags.get("decompose_by_category") is True

    def test_increases_context_on_low_directional(self):
        policy = RuleBasedPolicy()
        report = _make_report(directional_accuracy=40.0)
        strategy = _default_strategy()

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.context_multiplier == 4.0

    def test_context_multiplier_capped_at_8(self):
        policy = RuleBasedPolicy()
        report = _make_report(directional_accuracy=40.0)
        strategy = PlannerStrategy(
            horizon=ForecastHorizon.MONTH,
            model_variant=ModelVariant.CHRONOS_TINY,
            context_multiplier=8.0,
        )

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.context_multiplier == 8.0

    def test_upgrades_model_on_low_coverage(self):
        policy = RuleBasedPolicy()
        report = _make_report(coverage_80=60.0)
        strategy = _default_strategy()

        result = policy.select_strategy(report, strategy, cycle=0)
        assert result.model_variant in (ModelVariant.CHRONOS_MINI, ModelVariant.CHRONOS_SMALL)

    def test_update_is_noop(self):
        policy = RuleBasedPolicy()
        report = _make_report()
        strategy = _default_strategy()
        policy.update(report, strategy)
