"""
LangGraph nodes for the crypto forecasting agent.

Nodes: crypto_planner → crypto_forecaster → crypto_evaluator → crypto_learner
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from src.agents.crypto.state import (
    TIMEFRAME_HOURS,
    CryptoAgentState,
    CryptoEvalReport,
    CryptoLearnerFeedback,
    CryptoPlannerStrategy,
    CryptoPrediction,
    Direction,
    PredictionType,
)
from src.agents.learner.rule_based import RuleBasedPolicy
from src.agents.nodes import POLICY_REGISTRY
from src.agents.state import (
    EvalReport,
    ForecastHorizon,
    PlannerStrategy,
)


def crypto_planner_node(state: CryptoAgentState) -> dict[str, Any]:
    """Select strategy for this cycle based on prior learner feedback."""
    feedback: CryptoLearnerFeedback | None = state.get("learner_feedback")

    if feedback is not None:
        strategy = feedback.updated_strategy
    else:
        symbols = state.get("symbols", ["BTC/USDT"])
        pred_types = [PredictionType.DIRECTION, PredictionType.ABSOLUTE]
        if len(symbols) >= 2:
            pred_types.append(PredictionType.SPREAD)
        strategy = CryptoPlannerStrategy(
            symbols=symbols,
            prediction_types=pred_types,
            rationale="Initial default strategy",
        )

    return {"strategy": strategy, "strategy_history": [strategy]}


def crypto_forecaster_node(state: CryptoAgentState) -> dict[str, Any]:
    """Generate price forecasts for each symbol."""
    strategy: CryptoPlannerStrategy = state["strategy"]
    ohlcv_data = state["ohlcv_data"]

    predictions: list[CryptoPrediction] = []
    now = datetime.now(tz=UTC)

    for symbol in strategy.symbols:
        symbol_key = symbol.replace("/", "_")
        series_dict = ohlcv_data.get(symbol_key)
        if series_dict is None:
            continue

        df = pl.from_dict(series_dict)
        close_values = df["close"].tail(strategy.context_bars).to_numpy()

        if len(close_values) < 10:
            continue

        model_id = strategy.model_variant.value
        point, lower, upper = _forecast_series(close_values, strategy.forecast_bars, model_id)

        if PredictionType.ABSOLUTE in strategy.prediction_types:
            predictions.append(
                CryptoPrediction(
                    symbol=symbol,
                    prediction_type=PredictionType.ABSOLUTE,
                    timestamp=now,
                    point_forecast=point,
                    lower_80=lower,
                    upper_80=upper,
                    model_used=strategy.model_variant,
                )
            )

        if PredictionType.DIRECTION in strategy.prediction_types:
            current_price = float(close_values[-1])
            forecast_mean = np.mean(point)
            pct_change = (forecast_mean - current_price) / current_price

            if pct_change > 0.005:
                direction = Direction.UP
            elif pct_change < -0.005:
                direction = Direction.DOWN
            else:
                direction = Direction.NEUTRAL

            confidence = min(abs(pct_change) * 20, 1.0)

            predictions.append(
                CryptoPrediction(
                    symbol=symbol,
                    prediction_type=PredictionType.DIRECTION,
                    timestamp=now,
                    direction=direction,
                    direction_confidence=round(confidence, 3),
                    model_used=strategy.model_variant,
                )
            )

    if PredictionType.SPREAD in strategy.prediction_types and len(strategy.symbols) >= 2:
        _add_spread_predictions(predictions, state, strategy, now)

    return {"predictions": predictions}


def _add_spread_predictions(
    predictions: list[CryptoPrediction],
    state: CryptoAgentState,
    strategy: CryptoPlannerStrategy,
    now: datetime,
) -> None:
    """Compute spread predictions between symbol pairs."""
    ohlcv_data = state["ohlcv_data"]
    symbols = strategy.symbols

    for i in range(len(symbols) - 1):
        sym_a = symbols[i]
        sym_b = symbols[i + 1]
        key_a = sym_a.replace("/", "_")
        key_b = sym_b.replace("/", "_")

        if key_a not in ohlcv_data or key_b not in ohlcv_data:
            continue

        df_a = pl.from_dict(ohlcv_data[key_a])
        df_b = pl.from_dict(ohlcv_data[key_b])

        close_a = df_a["close"].tail(1).to_list()[0] if len(df_a) > 0 else None
        close_b = df_b["close"].tail(1).to_list()[0] if len(df_b) > 0 else None

        if close_a is None or close_b is None or close_b == 0:
            continue

        spread = close_a / close_b

        predictions.append(
            CryptoPrediction(
                symbol=sym_a,
                prediction_type=PredictionType.SPREAD,
                timestamp=now,
                spread_value=round(spread, 6),
                spread_pair=f"{sym_a}/{sym_b}",
                model_used=strategy.model_variant,
            )
        )


def _forecast_series(
    values: np.ndarray,
    horizon: int,
    model_id: str,
) -> tuple[list[float], list[float], list[float]]:
    """Run forecast model on a price series."""
    try:
        if model_id.startswith("amazon/"):
            import torch
            from chronos import ChronosPipeline

            pipeline = ChronosPipeline.from_pretrained(
                model_id, device_map="cpu", torch_dtype=torch.float32
            )
            context = torch.tensor(values[np.newaxis, :], dtype=torch.float32)
            quantiles, mean = pipeline.predict_quantiles(
                context,
                prediction_length=horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                num_samples=50,
            )
            point = mean[0].numpy().tolist()
            lower = quantiles[0, :, 0].numpy().tolist()
            upper = quantiles[0, :, 2].numpy().tolist()
            return point, lower, upper
    except Exception:
        pass

    return _statsforecast_fallback(values, horizon)


def _statsforecast_fallback(
    values: np.ndarray,
    horizon: int,
) -> tuple[list[float], list[float], list[float]]:
    """StatsForecast AutoETS fallback."""
    try:
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        n = len(values)
        df_sf = pd.DataFrame(
            {
                "unique_id": ["s1"] * n,
                "ds": pd.date_range("2020-01-01", periods=n, freq="D"),
                "y": values.tolist(),
            }
        )
        sf = StatsForecast(models=[AutoETS(season_length=7)], freq="D", n_jobs=1)
        sf.fit(df_sf)
        pred = sf.predict(h=horizon, level=[80])
        point = pred["AutoETS"].tolist()
        lower = pred["AutoETS-lo-80"].tolist()
        upper = pred["AutoETS-hi-80"].tolist()

        lower = [min(lo, pt) for lo, pt in zip(lower, point, strict=False)]
        upper = [max(hi, pt) for hi, pt in zip(upper, point, strict=False)]
        return point, lower, upper
    except Exception:
        last = values[-7:] if len(values) >= 7 else values
        point = np.tile(last, math.ceil(horizon / len(last)))[:horizon].tolist()
        lower = (np.array(point) * 0.92).tolist()
        upper = (np.array(point) * 1.08).tolist()
        return point, lower, upper


def crypto_evaluator_node(state: CryptoAgentState) -> dict[str, Any]:
    """Evaluate crypto predictions using financial metrics."""
    predictions = state["predictions"]
    ohlcv_data = state["ohlcv_data"]
    strategy: CryptoPlannerStrategy = state["strategy"]
    cycle_id = state.get("cycle_id") or str(uuid.uuid4())[:8]

    abs_predictions = [p for p in predictions if p.prediction_type == PredictionType.ABSOLUTE]
    dir_predictions = [p for p in predictions if p.prediction_type == PredictionType.DIRECTION]

    sharpe = _compute_sharpe(abs_predictions, ohlcv_data, strategy)
    sortino = _compute_sortino(abs_predictions, ohlcv_data, strategy)
    max_dd = _compute_max_drawdown(ohlcv_data, strategy)
    dir_acc = _compute_directional_accuracy(dir_predictions, ohlcv_data, strategy)

    all_passed = sharpe > 0.5 and sortino > 0.7 and max_dd < 0.15 and dir_acc > 55.0

    report = CryptoEvalReport(
        cycle_id=cycle_id,
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        directional_accuracy=round(dir_acc, 2),
        all_passed=all_passed,
        summary=f"Sharpe={sharpe:.2f} Sortino={sortino:.2f} DD={max_dd:.2%} Dir={dir_acc:.1f}%",
    )

    return {"eval_report": report, "eval_history": [report]}


def _compute_sharpe(
    predictions: list[CryptoPrediction],
    ohlcv_data: dict,
    strategy: CryptoPlannerStrategy,
) -> float:
    """Annualized Sharpe ratio from prediction-implied returns."""
    if not predictions:
        return 0.0

    returns = []
    for pred in predictions:
        key = pred.symbol.replace("/", "_")
        if key not in ohlcv_data or pred.point_forecast is None:
            continue
        df = pl.from_dict(ohlcv_data[key])
        if len(df) == 0:
            continue

        current = df["close"].to_list()[-1]
        forecast_end = pred.point_forecast[-1]
        ret = (forecast_end - current) / current if current != 0 else 0.0
        returns.append(ret)

    if not returns:
        return 0.0

    arr = np.array(returns)
    mean_ret = arr.mean()
    std_ret = arr.std()
    if std_ret == 0:
        return 0.0

    periods_per_year = 365 * 24 / TIMEFRAME_HOURS.get(strategy.timeframe, 24)
    return float(mean_ret / std_ret * np.sqrt(periods_per_year))


def _compute_sortino(
    predictions: list[CryptoPrediction],
    ohlcv_data: dict,
    strategy: CryptoPlannerStrategy,
) -> float:
    """Sortino ratio — penalizes only downside volatility."""
    if not predictions:
        return 0.0

    returns = []
    for pred in predictions:
        key = pred.symbol.replace("/", "_")
        if key not in ohlcv_data or pred.point_forecast is None:
            continue
        df = pl.from_dict(ohlcv_data[key])
        if len(df) == 0:
            continue

        current = df["close"].to_list()[-1]
        forecast_end = pred.point_forecast[-1]
        ret = (forecast_end - current) / current if current != 0 else 0.0
        returns.append(ret)

    if not returns:
        return 0.0

    arr = np.array(returns)
    mean_ret = arr.mean()
    downside = arr[arr < 0]
    downside_std = downside.std() if len(downside) > 0 else arr.std()
    if downside_std == 0:
        return 0.0

    periods_per_year = 365 * 24 / TIMEFRAME_HOURS.get(strategy.timeframe, 24)
    return float(mean_ret / downside_std * np.sqrt(periods_per_year))


def _compute_max_drawdown(ohlcv_data: dict, strategy: CryptoPlannerStrategy) -> float:
    """Max drawdown from the primary symbol's price series."""
    if not strategy.symbols:
        return 0.0

    key = strategy.symbols[0].replace("/", "_")
    if key not in ohlcv_data:
        return 0.0

    df = pl.from_dict(ohlcv_data[key])
    prices = df["close"].to_numpy()
    if len(prices) < 2:
        return 0.0

    peak = prices[0]
    max_dd = 0.0
    for price in prices[1:]:
        if price > peak:
            peak = price
        dd = (peak - price) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return float(max_dd)


def _compute_directional_accuracy(
    predictions: list[CryptoPrediction],
    ohlcv_data: dict,
    strategy: CryptoPlannerStrategy,
) -> float:
    """Evaluate direction predictions against recent price movement."""
    if not predictions:
        return 50.0

    correct = 0
    total = 0

    for pred in predictions:
        if pred.direction is None or pred.direction == Direction.NEUTRAL:
            continue

        key = pred.symbol.replace("/", "_")
        if key not in ohlcv_data:
            continue

        df = pl.from_dict(ohlcv_data[key])
        closes = df["close"].to_list()
        if len(closes) < 2:
            continue

        actual_direction = Direction.UP if closes[-1] > closes[-2] else Direction.DOWN
        if pred.direction == actual_direction:
            correct += 1
        total += 1

    return (correct / total * 100.0) if total > 0 else 50.0


def crypto_learner_node(state: CryptoAgentState) -> dict[str, Any]:
    """Adapt crypto strategy using shared learner policy."""
    report: CryptoEvalReport = state["eval_report"]
    strategy: CryptoPlannerStrategy = state["strategy"]
    cycle_count = state.get("cycle_count", 0)
    max_cycles = state.get("max_cycles", 5)
    policy_name = state.get("learner_policy_name", "rule_based")

    # Convert crypto eval to the standard EvalReport for policy compatibility
    compat_report = EvalReport(
        cycle_id=report.cycle_id,
        forecast_date=datetime.now(tz=UTC).date(),
        series_scores={},
        overall_mase=report.mase if report.mase is not None else 1.0 - report.sharpe_ratio,
        overall_smape=(1.0 - report.directional_accuracy / 100.0) * 30.0,
        directional_accuracy=report.directional_accuracy,
        coverage_80=report.coverage_80 if report.coverage_80 is not None else 75.0,
        drift_ratio=1.0 + report.max_drawdown,
        all_passed=report.all_passed,
    )

    compat_strategy = PlannerStrategy(
        horizon=ForecastHorizon.MONTH,
        model_variant=strategy.model_variant,
        context_multiplier=strategy.context_bars / 30.0,
        rationale=strategy.rationale,
    )

    policy_cls = POLICY_REGISTRY.get(policy_name, RuleBasedPolicy)
    policy = policy_cls()
    policy.update(compat_report, compat_strategy)
    new_compat = policy.select_strategy(compat_report, compat_strategy, cycle_count)

    new_strategy = CryptoPlannerStrategy(
        timeframe=strategy.timeframe,
        model_variant=new_compat.model_variant,
        prediction_types=strategy.prediction_types,
        context_bars=int(new_compat.context_multiplier * 30),
        forecast_bars=strategy.forecast_bars,
        symbols=strategy.symbols,
        use_indicators=strategy.use_indicators,
        rationale=new_compat.rationale,
    )

    feedback = CryptoLearnerFeedback(
        cycle_id=report.cycle_id,
        updated_strategy=new_strategy,
        drift_detected=report.max_drawdown > 0.15,
        reflection_text=f"Crypto cycle {cycle_count + 1}: {report.summary}",
        strategy_changes=[new_compat.rationale] if new_compat.rationale else [],
    )

    terminate = cycle_count + 1 >= max_cycles or (report.all_passed and cycle_count >= 1)

    return {
        "learner_feedback": feedback,
        "strategy": new_strategy,
        "cycle_count": cycle_count + 1,
        "terminate": terminate,
    }
