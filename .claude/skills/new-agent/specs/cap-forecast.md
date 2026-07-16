# cap-forecast — Templates for forecast-builder subagent

## Design notes

### Interval validator clamping
When constructing a forecast result with a Pydantic model validator that enforces
`lower <= point <= upper`, AutoETS and other statsforecast models can produce
asymmetric prediction intervals that violate this on short or skewed series.
Always clamp the arrays before constructing the model:

```python
lower = [min(lo, pt) for lo, pt in zip(lower, point)]
upper = [max(hi, pt) for hi, pt in zip(upper, point)]
```

### Naïve fallback `math` import guard
Any naïve/seasonal fallback that uses `math.ceil` must import `math` at module
level, not inside the Chronos try-block. The naïve path runs precisely when
Chronos is unavailable — which is when the import won't have happened yet.

```python
import math  # top of file, not inside try-block

def _run_statsforecast_fallback(series_values, horizon_days):
    ...
    # last resort naïve seasonal
    point = np.tile(last, math.ceil(horizon_days / len(last)))[:horizon_days].tolist()
```

---

## File: {OUTPUT_DIR}/features.py

```python
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard feature transforms for time-series forecasting.

    Expects a DataFrame with a 'date' column (or datetime index) and at least
    one numeric target column. Returns df with new feature columns appended.

    Transforms applied:
    - Date part extraction: year, month, day, day_of_week, quarter, week_of_year
    - Lag features: lag_1, lag_7, lag_14, lag_30 (for each numeric column)
    - Rolling means: rolling_mean_7, rolling_mean_14, rolling_mean_30
    - Rolling std: rolling_std_7
    - Pct change: pct_change_1, pct_change_7
    """
    df = df.copy()

    # Ensure datetime index
    if "date" in df.columns:
        df = df.set_index(pd.to_datetime(df["date"])).drop(columns=["date"])
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'date' column or DatetimeIndex")

    df = df.sort_index()

    # Date parts
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["day_of_week"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["week_of_year"] = df.index.isocalendar().week.astype(int)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    # Identify numeric columns (exclude the date parts we just added)
    date_cols = {"year", "month", "day", "day_of_week", "quarter", "week_of_year", "is_weekend"}
    numeric_cols = [c for c in df.columns if c not in date_cols and pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        # Lags
        for lag in [1, 7, 14, 30]:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

        # Rolling means
        for window in [7, 14, 30]:
            df[f"{col}_rolling_mean_{window}"] = (
                df[col].shift(1).rolling(window=window, min_periods=1).mean()
            )

        # Rolling std (volatility proxy)
        df[f"{col}_rolling_std_7"] = (
            df[col].shift(1).rolling(window=7, min_periods=2).std().fillna(0)
        )

        # Percentage change
        for period in [1, 7]:
            df[f"{col}_pct_change_{period}"] = df[col].pct_change(periods=period).replace(
                [np.inf, -np.inf], np.nan
            )

    return df


# ---------------------------------------------------------------------------
# SHAP feature importance
# ---------------------------------------------------------------------------


def compute_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model: Any,
) -> pd.DataFrame:
    """Compute SHAP-based feature importance for a fitted sklearn-compatible model.

    Args:
        X: Feature matrix (must already be fitted/transformed).
        y: Target series (used to fit model if not already fitted).
        model: A fitted sklearn-compatible estimator.

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for feature importance. Run: pip install shap") from exc

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer (slower)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception:  # noqa: BLE001
        logger.warning("TreeExplainer failed — falling back to LinearExplainer")
        try:
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        except Exception:  # noqa: BLE001
            logger.warning("LinearExplainer failed — falling back to KernelExplainer (slow)")
            explainer = shap.KernelExplainer(model.predict, shap.sample(X, 50))
            shap_values = explainer.shap_values(X)

    # Handle multi-output models (take mean across outputs)
    if isinstance(shap_values, list):
        sv = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        sv = np.abs(shap_values)

    importance = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "mean_abs_shap": sv.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return importance.reset_index(drop=True)


def feature_importance_report(importance_df: pd.DataFrame, top_n: int = 10) -> str:
    """Format a human-readable feature importance summary.

    Args:
        importance_df: Output of compute_importance().
        top_n: How many features to include in the report.

    Returns:
        Formatted string suitable for logging or display.
    """
    if importance_df.empty:
        return "No feature importance data available."

    top = importance_df.head(top_n)
    max_score = top["mean_abs_shap"].max()

    lines = [f"Top {min(top_n, len(top))} features by SHAP importance:"]
    lines.append("-" * 50)

    for _, row in top.iterrows():
        feature = row["feature"]
        score = row["mean_abs_shap"]
        bar_len = int((score / max_score) * 30) if max_score > 0 else 0
        bar = "#" * bar_len
        lines.append(f"  {feature:<35} {score:6.4f}  {bar}")

    lines.append("-" * 50)
    lines.append(
        f"Total features: {len(importance_df)} | "
        f"Top-{top_n} cumulative SHAP: {top['mean_abs_shap'].sum():.4f}"
    )

    return "\n".join(lines)
```

## File: {OUTPUT_DIR}/pipeline.py

```python
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ForecastPipeline:
    """Sklearn-based forecasting pipeline with fit/predict/evaluate/save/load.

    Usage:
        fp = ForecastPipeline()
        fp.fit(X_train, y_train)
        preds = fp.predict(X_test)
        metrics = fp.evaluate(X_test, y_test)
        fp.save("model.pkl")

        fp2 = ForecastPipeline.load("model.pkl")
    """

    DEFAULT_ESTIMATOR = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )

    def __init__(self, estimator: Any | None = None) -> None:
        self._estimator = estimator or self.DEFAULT_ESTIMATOR
        self._pipeline: Pipeline | None = None
        self._feature_names: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastPipeline":
        """Fit the pipeline on training data.

        Drops rows with NaN in either X or y before fitting.
        """
        df = X.copy()
        df["__target__"] = y.values

        # Drop NaN rows
        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            logger.info("Dropped %d rows with NaN before fitting", dropped)

        X_clean = df.drop(columns=["__target__"])
        y_clean = df["__target__"]

        self._feature_names = X_clean.columns.tolist()
        self._pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("estimator", self._estimator),
            ]
        )
        self._pipeline.fit(X_clean, y_clean)
        self._is_fitted = True
        logger.info("ForecastPipeline fitted on %d samples, %d features", len(X_clean), len(self._feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for X."""
        self._assert_fitted()
        X_aligned = X[self._feature_names].fillna(0)
        return self._pipeline.predict(X_aligned)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Return MAE, RMSE, and R2 on the given split."""
        self._assert_fitted()
        y_pred = self.predict(X)
        y_true = y.values

        # Align lengths
        n = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:n], y_pred[:n]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = r2_score(y_true, y_pred)

        metrics = {"mae": mae, "rmse": rmse, "r2": r2}
        logger.info("Evaluation — MAE=%.4f  RMSE=%.4f  R2=%.4f", mae, rmse, r2)
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the pipeline to disk."""
        self._assert_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "pipeline": self._pipeline,
                    "feature_names": self._feature_names,
                    "estimator_class": type(self._estimator).__name__,
                },
                fh,
            )
        logger.info("ForecastPipeline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ForecastPipeline":
        """Load a pickled pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with path.open("rb") as fh:
            state = pickle.load(fh)

        instance = cls()
        instance._pipeline = state["pipeline"]
        instance._feature_names = state["feature_names"]
        instance._is_fitted = True
        logger.info("ForecastPipeline loaded from %s (%s)", path, state.get("estimator_class"))
        return instance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_fitted(self) -> None:
        if not self._is_fitted or self._pipeline is None:
            raise RuntimeError("Pipeline is not fitted. Call .fit() first.")

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)
```
