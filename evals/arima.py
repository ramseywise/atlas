"""
ARIMA / SARIMA model with explicit assumption checking and diagnostics.

ARIMA vs Chronos — the key differences:

  ARIMA (AutoRegressive Integrated Moving Average)
  ─────────────────────────────────────────────────
  Parametric model: y_t = φ₁y_{t-1} + ... + φ_p y_{t-p}
                        - θ₁ε_{t-1} - ... - θ_q ε_{t-q} + ε_t
                    after d-th order differencing to achieve stationarity.

  ASSUMPTIONS (all must hold — violation degrades or invalidates the model):
    1. Stationarity after d differences (unit root removed — ADF test)
    2. No autocorrelation in residuals (Ljung-Box test on ε_t)
    3. Homoskedastic residuals (constant variance — Breusch-Pagan or ARCH test)
    4. Normally distributed residuals (Jarque-Bera; needed for valid PIs)
    5. No structural breaks in the series (Chow test / visual inspection)
    6. Sufficient history: at least 4× the seasonal period (so 52+ obs for weekly)

  SARIMA adds seasonal AR/MA/I orders: SARIMA(p,d,q)(P,D,Q)[s]
    - s = seasonal period (7 for weekly, 12 for monthly annual, 365 for daily annual)
    - Seasonal differencing D handles stable periodic patterns

  ARIMAX adds exogenous regressors X_t (e.g., calendar features, Fourier terms)
    - X_t must also be available for the forecast horizon (no future leakage)
    - Fourier terms from features.py are ideal exogenous regressors

  Chronos (foundation model)
  ─────────────────────────────────────────────────
  - Zero-shot: pretrained on 100k+ real series, no per-series fitting
  - No stationarity assumption — handles trend and seasonality implicitly
  - Probabilistic: samples from learned distribution over future values
  - Cannot incorporate external regressors in the base model
  - Requires ~seconds of CPU inference (vs ~ms for ARIMA after fitting)
  - No interpretable coefficients (black box)

  When to prefer ARIMA:
    - Interpretability required (coefficient inspection, residual diagnosis)
    - Very short series (< 2 years daily) where Chronos may overfit priors
    - Strict stationarity assumptions can be verified and met
    - Exogenous regressors are available and predictable

  When to prefer Chronos:
    - Cold start (new customer, little history)
    - Multiple seasonalities without manual SARIMA order search
    - Probabilistic forecasts needed without distributional assumptions
    - Speed of deployment over interpretability

Usage:
    model = ARIMAForecaster(series_id="payroll")
    model.check_assumptions(train_values)   # prints diagnostic report
    model.fit(train_values)
    result = model.predict(horizon=30)
    print(result.comparison_vs_naive())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.preprocessing.preprocessing import (
    StationarityReport,
    check_stationarity,
    difference_series,
)

# ── Assumption diagnostics ─────────────────────────────────────────────────────


@dataclass
class AssumptionReport:
    """Full diagnostic output from assumption checking."""

    series_id: str
    n_obs: int
    stationarity: StationarityReport
    ljung_box_passed: bool | None  # None if statsmodels unavailable
    ljung_box_pvalue: float | None
    heteroskedasticity_passed: bool | None
    arch_pvalue: float | None
    normality_passed: bool | None  # Jarque-Bera
    jb_pvalue: float | None
    min_obs_met: bool
    seasonal_period_detected: int | None
    violations: list[str]
    warnings: list[str]
    recommendation: str


def check_arima_assumptions(
    values: np.ndarray,
    series_id: str = "unknown",
    min_obs_multiplier: int = 4,
    seasonal_period: int = 7,
) -> AssumptionReport:
    """
    Run all ARIMA pre-fit assumption checks and return a structured report.

    Called before fitting to surface any violations early, with clear
    recommendations on transforms or model changes.
    """
    violations: list[str] = []
    warnings: list[str] = []
    n = len(values)

    # ── 1. Stationarity ───────────────────────────────────────────────────────
    stat = check_stationarity(values, series_id=series_id)
    if not stat.adf_stationary:
        violations.append(
            f"Non-stationary (ADF p={stat.adf_pvalue:.3f}). "
            f"Apply d={stat.recommended_d} differencing before fitting."
        )

    # ── 2. Minimum observations ───────────────────────────────────────────────
    min_obs = min_obs_multiplier * seasonal_period
    min_obs_met = n >= min_obs
    if not min_obs_met:
        violations.append(
            f"Insufficient history: {n} obs < {min_obs} (4 × seasonal period {seasonal_period}). "
            "ARIMA estimates will be unreliable."
        )

    # ── 3. Residual autocorrelation, homoskedasticity, normality ──────────────
    ljung_passed = ljung_pval = arch_pval = hetero_passed = jb_passed = jb_pval = None

    try:
        from scipy.stats import jarque_bera
        from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

        # Fit a simple AR(1) to get residuals for diagnostics
        from statsmodels.tsa.arima.model import ARIMA as StatsARIMA  # noqa: N811

        d = stat.recommended_d
        work = difference_series(values, d) if d > 0 else values
        work = work[~np.isnan(work)]

        try:
            fit = StatsARIMA(work, order=(1, 0, 0)).fit(disp=False)
            resid = fit.resid

            # Ljung-Box: test for autocorrelation in residuals
            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            ljung_pval = float(lb["lb_pvalue"].iloc[0])
            ljung_passed = ljung_pval > 0.05
            if not ljung_passed:
                violations.append(
                    f"Residual autocorrelation (Ljung-Box p={ljung_pval:.3f}). "
                    "Increase AR or MA order."
                )

            # ARCH test: test for heteroskedasticity in residuals
            arch_test = het_arch(resid, nlags=5)
            arch_pval = float(arch_test[1])
            hetero_passed = arch_pval > 0.05
            if not hetero_passed:
                warnings.append(
                    f"Heteroskedastic residuals (ARCH p={arch_pval:.3f}). "
                    "Consider log transform or GARCH. Prediction intervals may be unreliable."
                )

            # Jarque-Bera: normality of residuals
            jb_stat, jb_pval_raw = jarque_bera(resid)
            jb_pval = float(jb_pval_raw)
            jb_passed = jb_pval > 0.05
            if not jb_passed:
                warnings.append(
                    f"Non-normal residuals (JB p={jb_pval:.3f}). "
                    "Prediction intervals are approximate — use bootstrap PI."
                )

        except Exception:
            warnings.append("Could not fit diagnostic AR(1) — skip residual tests")

    except ImportError:
        warnings.append("statsmodels/scipy not available — skipping residual diagnostics")

    # ── 4. Seasonal period detection ─────────────────────────────────────────
    detected_period = _detect_seasonal_period(values)
    if detected_period and detected_period != seasonal_period:
        warnings.append(
            f"Detected seasonal period {detected_period} differs from specified {seasonal_period}. "
            "Consider SARIMA with correct s parameter."
        )

    # ── Recommendation ────────────────────────────────────────────────────────
    if not violations:
        rec = f"Assumptions met for ARIMA(p,{stat.recommended_d},q). Safe to fit."
    elif len(violations) == 1 and not stat.adf_stationary:
        rec = f"Apply d={stat.recommended_d} differencing, then fit ARIMA."
    else:
        rec = (
            "Multiple assumption violations. Consider: "
            "(1) log transform for heteroskedasticity, "
            "(2) higher d for non-stationarity, "
            "(3) Chronos as a non-parametric alternative."
        )

    return AssumptionReport(
        series_id=series_id,
        n_obs=n,
        stationarity=stat,
        ljung_box_passed=ljung_passed,
        ljung_box_pvalue=ljung_pval,
        heteroskedasticity_passed=hetero_passed,
        arch_pvalue=arch_pval,
        normality_passed=jb_passed,
        jb_pvalue=jb_pval,
        min_obs_met=min_obs_met,
        seasonal_period_detected=detected_period,
        violations=violations,
        warnings=warnings,
        recommendation=rec,
    )


def _detect_seasonal_period(values: np.ndarray, max_period: int = 52) -> int | None:
    """Heuristic: find dominant period via autocorrelation peak."""
    n = len(values)
    if n < 2 * max_period:
        return None
    work = values - np.mean(values)
    autocorr = np.correlate(work, work, mode="full")[n - 1 :]
    autocorr = autocorr[1 : max_period + 1] / (autocorr[0] + 1e-8)
    if len(autocorr) == 0:
        return None
    period = int(np.argmax(autocorr)) + 1
    return period if autocorr[period - 1] > 0.3 else None


# ── Forecaster ────────────────────────────────────────────────────────────────


@dataclass
class ARIMAForecastResult:
    series_id: str
    model_spec: str  # e.g. "ARIMA(2,1,1)" or "AutoARIMA"
    point_forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    aic: float | None
    bic: float | None
    horizon: int
    assumption_report: AssumptionReport | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def comparison_vs_naive(self, actuals: np.ndarray | None = None) -> str:
        """Quick text summary for comparison output."""
        lines = [
            f"Model: {self.model_spec}",
            f"Horizon: {self.horizon} steps",
            f"AIC: {self.aic:.1f}" if self.aic else "AIC: N/A",
            f"Forecast range: [{min(self.point_forecast):.0f}, {max(self.point_forecast):.0f}]",
        ]
        if actuals is not None and len(actuals) >= self.horizon:
            act = np.array(actuals[: self.horizon])
            preds = np.array(self.point_forecast)
            mae = np.mean(np.abs(act - preds))
            naive = np.mean(np.abs(np.diff(act)))
            mase = mae / (naive + 1e-8)
            lines.append(f"MASE vs naïve: {mase:.3f}")
        return "\n".join(lines)


class ARIMAForecaster:
    """
    ARIMA / SARIMA / AutoARIMA wrapper with assumption diagnostics.

    Fits one model per series_id. For multi-series forecasting, instantiate
    one ARIMAForecaster per series (stateful — holds fitted model).

    Auto mode (auto=True): uses statsforecast AutoARIMA, which searches
    (p, d, q)(P, D, Q)[s] space via information criterion — fastest path.

    Manual mode: specify (p, d, q) explicitly after running check_assumptions().
    """

    def __init__(
        self,
        series_id: str,
        auto: bool = True,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),  # (P,D,Q,s)
        exog_cols: list[str] | None = None,
    ):
        self.series_id = series_id
        self.auto = auto
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols or []
        self._fitted_model: Any = None
        self._last_train_values: np.ndarray | None = None

    def check_assumptions(
        self,
        values: np.ndarray,
        seasonal_period: int = 7,
    ) -> AssumptionReport:
        """Run full assumption check and print a human-readable report."""
        report = check_arima_assumptions(
            values, series_id=self.series_id, seasonal_period=seasonal_period
        )
        _print_assumption_report(report)
        return report

    def fit(
        self,
        values: np.ndarray,
        exog: np.ndarray | None = None,
        seasonal_period: int = 7,
    ) -> AssumptionReport | None:
        """
        Fit ARIMA or AutoARIMA on training values.
        Returns assumption report if auto=False (skipped for AutoARIMA speed).
        """
        self._last_train_values = values.copy()
        assumption_report = None

        if self.auto:
            self._fitted_model = self._fit_auto(values, exog, seasonal_period)
        else:
            assumption_report = check_arima_assumptions(
                values, self.series_id, seasonal_period=seasonal_period
            )
            self._fitted_model = self._fit_manual(values, exog)

        return assumption_report

    def predict(
        self,
        horizon: int = 30,
        exog_future: np.ndarray | None = None,
        level: int = 80,
    ) -> ARIMAForecastResult:
        """Generate point + interval forecasts."""
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before predict()")

        point, lower, upper, aic, bic, spec = self._generate_forecast(horizon, exog_future, level)

        return ARIMAForecastResult(
            series_id=self.series_id,
            model_spec=spec,
            point_forecast=[max(0.0, p) for p in point],  # cash flows non-negative
            lower_80=[max(0.0, lo) for lo in lower],
            upper_80=[max(0.0, u) for u in upper],
            aic=aic,
            bic=bic,
            horizon=horizon,
            assumption_report=None,
        )

    # ── Private fitting methods ───────────────────────────────────────────────

    def _fit_auto(
        self,
        values: np.ndarray,
        exog: np.ndarray | None,
        seasonal_period: int,
    ) -> Any:
        try:
            import pandas as pd
            from statsforecast import StatsForecast
            from statsforecast.models import AutoARIMA

            n = len(values)
            df_sf = pd.DataFrame(
                {
                    "unique_id": [self.series_id] * n,
                    "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                    "y": values.tolist(),
                }
            )
            sf = StatsForecast(
                models=[AutoARIMA(season_length=seasonal_period, approximation=True)],
                freq="D",
                n_jobs=1,
            )
            sf.fit(df_sf)
            return ("statsforecast", sf, df_sf)
        except ImportError:
            return self._fit_manual(values, exog)

    def _fit_manual(self, values: np.ndarray, exog: np.ndarray | None) -> Any:
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA  # noqa: N811

            p, d, q = self.order
            model = StatsARIMA(values, order=(p, d, q), exog=exog).fit(disp=False)
            return ("statsmodels", model)
        except Exception:
            # Fallback: naïve seasonal (lag-7)
            return ("naive", values.copy())

    def _generate_forecast(
        self,
        horizon: int,
        exog_future: np.ndarray | None,
        level: int,
    ) -> tuple[list[float], list[float], list[float], float | None, float | None, str]:
        backend, *rest = self._fitted_model

        if backend == "statsforecast":
            sf, df_sf = rest
            pred = sf.predict(h=horizon, level=[level])
            col = "AutoARIMA"
            point = pred[col].tolist()
            lower = pred[f"{col}-lo-{level}"].tolist()
            upper = pred[f"{col}-hi-{level}"].tolist()
            return point, lower, upper, None, None, "AutoARIMA(seasonal)"

        elif backend == "statsmodels":
            model = rest[0]
            fc = model.get_forecast(steps=horizon, exog=exog_future)
            ci = fc.conf_int(alpha=1 - level / 100)
            point = fc.predicted_mean.tolist()
            lower = ci.iloc[:, 0].tolist()
            upper = ci.iloc[:, 1].tolist()
            p, d, q = self.order
            return point, lower, upper, float(model.aic), float(model.bic), f"ARIMA({p},{d},{q})"

        else:
            # Naïve seasonal fallback
            vals = rest[0]
            last7 = vals[-7:] if len(vals) >= 7 else vals
            import math

            point = list(np.tile(last7, math.ceil(horizon / len(last7)))[:horizon])
            lower = [p * 0.85 for p in point]
            upper = [p * 1.15 for p in point]
            return point, lower, upper, None, None, "NaïveSeasonal(lag-7)"


# ── Pretty printer ─────────────────────────────────────────────────────────────


def _print_assumption_report(report: AssumptionReport) -> None:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()
        console.print(
            Panel.fit(
                f"[bold]ARIMA Assumption Report — {report.series_id}[/bold]\n"
                f"n={report.n_obs} obs | "
                f"Recommended d={report.stationarity.recommended_d}"
            )
        )

        table = Table(show_lines=True)
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Detail")

        def row(name, passed, detail):
            status = (
                "[green]PASS[/green]"
                if passed
                else "[red]FAIL[/red]"
                if passed is False
                else "[yellow]N/A[/yellow]"
            )
            table.add_row(name, status, detail)

        row(
            "Stationarity (ADF)",
            report.stationarity.adf_stationary,
            f"p={report.stationarity.adf_pvalue:.3f} — {report.stationarity.conclusion}",
        )
        row("Min observations", report.min_obs_met, f"{report.n_obs} obs")
        row(
            "Residual autocorrelation (Ljung-Box)",
            report.ljung_box_passed,
            f"p={report.ljung_box_pvalue:.3f}" if report.ljung_box_pvalue else "Not run",
        )
        row(
            "Homoskedasticity (ARCH)",
            report.heteroskedasticity_passed,
            f"p={report.arch_pvalue:.3f}" if report.arch_pvalue else "Not run",
        )
        row(
            "Residual normality (Jarque-Bera)",
            report.normality_passed,
            f"p={report.jb_pvalue:.3f}" if report.jb_pvalue else "Not run",
        )

        console.print(table)

        if report.violations:
            console.print("\n[bold red]Violations:[/bold red]")
            for v in report.violations:
                console.print(f"  ✗ {v}")

        if report.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for w in report.warnings:
                console.print(f"  ⚠ {w}")

        console.print(f"\n[bold]Recommendation:[/bold] {report.recommendation}")

    except ImportError:
        print(f"\n=== ARIMA Assumptions: {report.series_id} ===")
        print(
            f"Stationarity: {'PASS' if report.stationarity.adf_stationary else 'FAIL'} "
            f"(d={report.stationarity.recommended_d})"
        )
        for v in report.violations:
            print(f"  VIOLATION: {v}")
        for w in report.warnings:
            print(f"  WARNING: {w}")
        print(f"Recommendation: {report.recommendation}")
