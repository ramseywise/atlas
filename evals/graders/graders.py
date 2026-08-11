"""
Eval grader suite for the cash flow forecasting agent.

Each grader is:
  - Independently testable (takes arrays, returns GraderScore)
  - Stateless (no side effects), except DriftGrader which maintains a rolling window
  - Composable via EvalHarness

Graders follow the legacy eval pattern: one module, harness orchestrates.

Pass thresholds imported from evals/metrics/constants.py — change thresholds there,
not here, so graders stay decoupled from tier policy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from evals.metrics.constants import (
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_DIRECTIONAL_THRESHOLD,
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_INTERVAL_WIDTH_THRESHOLD,
    DEFAULT_MASE_THRESHOLD,
    DEFAULT_SMAPE_THRESHOLD,
)
from src.agents.state import EvalReport, ForecastResult, GraderScore

# ── Base ──────────────────────────────────────────────────────────────────────


class BaseGrader(ABC):
    name: str
    threshold: float

    @abstractmethod
    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore: ...

    #: Direction of the threshold comparison; overridden by higher-is-better graders.
    lower_is_better: bool = True

    def _make_score(
        self,
        value: float,
        detail: str = "",
        per_step: list[float] | None = None,
    ) -> GraderScore:
        return GraderScore(
            grader_name=self.name,
            metric_value=round(value, 4),
            threshold=self.threshold,
            passed=self._passes(value),
            detail=detail,
            lower_is_better=self.lower_is_better,
            per_step=[round(v, 4) for v in (per_step or [])],
        )

    def _passes(self, value: float) -> bool:
        return value < self.threshold  # lower is better by default


# ── MASE ──────────────────────────────────────────────────────────────────────


class MASEGrader(BaseGrader):
    """
    Mean Absolute Scaled Error.
    Scale = MAE of naïve seasonal forecast (lag-s on training data).
    MASE < 1.0 means the model beats the naïve baseline.
    """

    name = "MASE"
    threshold = DEFAULT_MASE_THRESHOLD

    def __init__(self, train_actuals: np.ndarray, seasonal_period: int = 7):
        self.seasonal_period = seasonal_period
        diff = train_actuals[seasonal_period:] - train_actuals[:-seasonal_period]
        self.naive_mae = np.mean(np.abs(diff))
        if self.naive_mae < 1e-8:
            self.naive_mae = 1e-8

    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore:
        preds = np.array(forecast.point_forecast[: len(actuals)])
        abs_err = np.abs(actuals - preds)
        mae = np.mean(abs_err)
        mase = mae / self.naive_mae
        # Scaled error at each horizon step — reveals degradation with distance.
        per_step = (abs_err / self.naive_mae).tolist()
        return self._make_score(
            mase,
            f"MAE={mae:.2f}, naive_MAE={self.naive_mae:.2f}",
            per_step=per_step,
        )


# ── SMAPE ─────────────────────────────────────────────────────────────────────


class SMAPEGrader(BaseGrader):
    """Symmetric MAPE. Bounded [0, 200%]. Threshold < 15%."""

    name = "SMAPE"
    threshold = DEFAULT_SMAPE_THRESHOLD

    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore:
        preds = np.array(forecast.point_forecast[: len(actuals)])
        denom = np.where(
            (np.abs(actuals) + np.abs(preds)) / 2 < 1e-8,
            1e-8,
            (np.abs(actuals) + np.abs(preds)) / 2,
        )
        per_step_err = 100.0 * (np.abs(actuals - preds) / denom)
        smape = float(np.mean(per_step_err))
        return self._make_score(smape, f"SMAPE={smape:.2f}%", per_step=per_step_err.tolist())


# ── Directional Accuracy ──────────────────────────────────────────────────────


class DirectionalGrader(BaseGrader):
    """% correct direction of change. Threshold > 55% (better than coin flip)."""

    name = "DirectionalAccuracy"
    threshold = DEFAULT_DIRECTIONAL_THRESHOLD
    lower_is_better = False

    def _passes(self, value: float) -> bool:
        return value > self.threshold

    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore:
        preds = np.array(forecast.point_forecast[: len(actuals)])
        if len(actuals) < 2:
            return self._make_score(50.0, "Too few steps")
        hits = np.sign(np.diff(actuals)) == np.sign(np.diff(preds))
        accuracy = 100.0 * float(np.mean(hits))
        # 100/0 per step; averaged across series this becomes an accuracy-by-horizon
        # curve, which is where mean-reversion at long horizons shows up.
        return self._make_score(
            accuracy,
            f"Correct direction on {accuracy:.1f}% of steps",
            per_step=(100.0 * hits.astype(float)).tolist(),
        )


# ── Coverage ──────────────────────────────────────────────────────────────────


class CoverageGrader(BaseGrader):
    """% actuals inside 80% PI. Well-calibrated: 75–85%. Threshold ≥ 75%."""

    name = "Coverage80"
    threshold = DEFAULT_COVERAGE_THRESHOLD
    lower_is_better = False

    def _passes(self, value: float) -> bool:
        return value >= self.threshold

    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore:
        lower = np.array(forecast.lower_80[: len(actuals)])
        upper = np.array(forecast.upper_80[: len(actuals)])
        in_interval = (actuals >= lower) & (actuals <= upper)
        coverage = 100.0 * float(np.mean(in_interval))
        return self._make_score(
            coverage,
            f"{int(np.sum(in_interval))}/{len(actuals)} in 80% PI",
            per_step=(100.0 * in_interval.astype(float)).tolist(),
        )


# ── Interval width ────────────────────────────────────────────────────────────


class IntervalWidthGrader(BaseGrader):
    """
    Mean 80% PI width as a % of the actual level (a.k.a. normalised interval score
    component). Coverage alone is trivially gamed by widening intervals — a model
    predicting ±∞ scores 100% coverage. This makes that trade-off visible.

    Warning-only, like drift: a wide interval is not a hard failure if the point
    forecast is accurate, but it should never widen silently.
    """

    name = "IntervalWidth"
    threshold = DEFAULT_INTERVAL_WIDTH_THRESHOLD

    def score(self, actuals: np.ndarray, forecast: ForecastResult) -> GraderScore:
        lower = np.array(forecast.lower_80[: len(actuals)])
        upper = np.array(forecast.upper_80[: len(actuals)])
        denom = np.where(np.abs(actuals) < 1e-8, 1e-8, np.abs(actuals))
        per_step_width = 100.0 * ((upper - lower) / denom)
        width = float(np.mean(per_step_width))
        return self._make_score(
            width,
            f"mean 80% PI width = {width:.1f}% of level",
            per_step=per_step_width.tolist(),
        )


# ── Drift ─────────────────────────────────────────────────────────────────────


class DriftGrader(BaseGrader):
    """
    Detects performance drift: rolling MASE / baseline MASE.
    Drift ratio > 1.2 flags degradation. Warning only — not a hard gate.
    """

    name = "DriftDetection"
    threshold = DEFAULT_DRIFT_THRESHOLD
    window = 12

    def __init__(self, baseline_mase: float = 0.85):
        self.baseline_mase = max(baseline_mase, 1e-8)
        self._mase_history: list[float] = []

    def update(self, mase: float) -> None:
        self._mase_history.append(mase)
        if len(self._mase_history) > self.window:
            self._mase_history.pop(0)

    def score(self) -> GraderScore:
        if not self._mase_history:
            return GraderScore(
                grader_name=self.name,
                metric_value=1.0,
                threshold=self.threshold,
                passed=True,
                detail="No history — drift check skipped",
            )
        rolling_mase = float(np.mean(self._mase_history))
        drift_ratio = rolling_mase / self.baseline_mase
        return self._make_score(
            drift_ratio,
            f"rolling_mase={rolling_mase:.3f}, baseline={self.baseline_mase:.3f}",
        )


# ── Harness ───────────────────────────────────────────────────────────────────


class EvalHarness:
    """
    Orchestrates all graders for a single forecast cycle.
    DriftGrader is stateful (rolling history). All others are stateless.
    """

    def __init__(
        self,
        train_data_by_series: dict[str, np.ndarray],
        baseline_mase: float = 0.85,
    ):
        self._drift_grader = DriftGrader(baseline_mase=baseline_mase)
        self._mase_graders = {sid: MASEGrader(arr) for sid, arr in train_data_by_series.items()}
        self._smape_g = SMAPEGrader()
        self._dir_g = DirectionalGrader()
        self._cov_g = CoverageGrader()
        self._width_g = IntervalWidthGrader()

    def run(
        self,
        cycle_id: str,
        forecast_date: date,
        forecasts: list[ForecastResult],
        actuals_by_series: dict[str, np.ndarray],
    ) -> EvalReport:
        series_scores: dict[str, list[GraderScore]] = {}
        all_mase, all_smape, all_dir, all_cov, all_width = [], [], [], [], []

        for fc in forecasts:
            sid = fc.series_id
            actuals = actuals_by_series.get(sid)
            if actuals is None or len(actuals) == 0:
                continue

            scores: list[GraderScore] = []
            mase_g = self._mase_graders.get(sid)

            if mase_g:
                mase_score = mase_g.score(actuals, fc)
                scores.append(mase_score)
                all_mase.append(mase_score.metric_value)

            smape_score = self._smape_g.score(actuals, fc)
            dir_score = self._dir_g.score(actuals, fc)
            cov_score = self._cov_g.score(actuals, fc)
            width_score = self._width_g.score(actuals, fc)
            scores.extend([smape_score, dir_score, cov_score, width_score])

            all_smape.append(smape_score.metric_value)
            all_dir.append(dir_score.metric_value)
            all_cov.append(cov_score.metric_value)
            all_width.append(width_score.metric_value)
            series_scores[sid] = scores

        overall_mase = float(np.mean(all_mase)) if all_mase else float("nan")
        overall_smape = float(np.mean(all_smape)) if all_smape else float("nan")
        dir_acc = float(np.mean(all_dir)) if all_dir else float("nan")
        cov = float(np.mean(all_cov)) if all_cov else float("nan")
        width = float(np.mean(all_width)) if all_width else float("nan")

        if not math.isnan(overall_mase):
            self._drift_grader.update(overall_mase)
        drift_score = self._drift_grader.score()

        for sid in series_scores:
            series_scores[sid].append(drift_score)

        # Drift and interval width are warning-only signals, not hard gates.
        advisory = {"DriftDetection", "IntervalWidth"}
        all_passed = all(
            s.passed
            for scores in series_scores.values()
            for s in scores
            if s.grader_name not in advisory
        )

        return EvalReport(
            cycle_id=cycle_id,
            forecast_date=forecast_date,
            series_scores=series_scores,
            overall_mase=round(overall_mase, 4),
            overall_smape=round(overall_smape, 4),
            directional_accuracy=round(dir_acc, 4),
            coverage_80=round(cov, 4),
            drift_ratio=round(drift_score.metric_value, 4),
            all_passed=all_passed,
            interval_width=round(width, 4),
            summary=_summarise(overall_mase, overall_smape, dir_acc, cov, width, drift_score),
        )


def _summarise(mase, smape, dir_acc, cov, width, drift_score) -> str:
    drift_flag = " ⚠ DRIFT" if not drift_score.passed else ""
    width_str = "" if math.isnan(width) else f" | PIw={width:.1f}%"
    return (
        f"MASE={mase:.3f} | SMAPE={smape:.1f}% | Dir={dir_acc:.1f}% "
        f"| Cov80={cov:.1f}%{width_str}{drift_flag}"
    )
