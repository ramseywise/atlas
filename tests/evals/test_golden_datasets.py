"""
Golden dataset tests — run the 5 eval graders against real curated data.

These tests verify that:
  1. All 5 graders run without errors on every golden entry.
  2. Entries categorized as "good_forecast" pass the forecast graders.
  3. Entries categorized as "bad_forecast" fail at least MASE or SMAPE.
  4. Crypto bull-run entries pass all crypto graders.
  5. Bear-market entries fail Sharpe and/or MaxDrawdown.
  6. Segment clean-separation entries pass silhouette, Davies-Bouldin, min_size.
  7. No golden entry causes a crash regardless of category.

Graders tested:
  Forecast: MASEGrader, SMAPEGrader, DirectionalGrader, CoverageGrader, DriftGrader
  Crypto:   SharpeGrader, SortinoGrader, MaxDrawdownGrader
  Segment:  evaluate_clusters (silhouette, Davies-Bouldin, Calinski-Harabász, min_size)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.segmentation.algorithms import ClusterResult
from core.segmentation.evaluation import evaluate_clusters
from evals.graders.crypto_graders import CryptoEvalHarness, MaxDrawdownGrader, SharpeGrader
from evals.graders.graders import (
    CoverageGrader,
    DirectionalGrader,
    DriftGrader,
    EvalHarness,
    MASEGrader,
    SMAPEGrader,
)
from src.agents.state import CategoryType, ForecastHorizon, ForecastResult, ModelVariant

GOLDEN_DIR = Path(__file__).parent.parent.parent / "evals" / "golden"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_jsonl(name: str) -> list[dict]:
    path = GOLDEN_DIR / name
    if not path.exists():
        pytest.skip(
            f"Golden dataset missing: {path} — run: uv run python -m evals.golden._generate_golden"
        )
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _to_forecast_result(entry: dict) -> ForecastResult:
    """Reconstruct a ForecastResult from a golden entry."""
    horizon_map = {
        7: ForecastHorizon.WEEK,
        14: ForecastHorizon.FORTNIGHT,
        30: ForecastHorizon.MONTH,
        90: ForecastHorizon.QUARTER,
    }
    horizon = horizon_map.get(entry["horizon"], ForecastHorizon.MONTH)

    category_map = {
        "income_recurring": CategoryType.INCOME_RECURRING,
        "income_variable": CategoryType.INCOME_VARIABLE,
        "expense_fixed": CategoryType.EXPENSE_FIXED,
        "expense_discretionary": CategoryType.EXPENSE_DISCRETIONARY,
    }
    cat = category_map.get(
        entry.get("category_type", "income_recurring"), CategoryType.INCOME_RECURRING
    )

    from datetime import date

    return ForecastResult(
        series_id=entry["series_id"],
        category=cat,
        forecast_date=date(2024, 1, 1),
        horizon=horizon,
        point_forecast=entry["point_forecast"],
        lower_80=entry["lower_80"],
        upper_80=entry["upper_80"],
        model_used=ModelVariant.STATSFORECAST_ARIMA,
        forecast_steps=len(entry["point_forecast"]),
    )


# ── Forecast grader tests ──────────────────────────────────────────────────────


class TestForecastGolden:
    """All 5 forecast graders against the golden forecast dataset."""

    @pytest.fixture(scope="class")
    @classmethod
    def entries(cls) -> list[dict]:
        return _load_jsonl("forecast_golden_qa.jsonl")

    def test_dataset_has_minimum_entries(self, entries):
        assert len(entries) >= 30, f"Expected >=30 entries, got {len(entries)}"

    def test_all_required_fields_present(self, entries):
        required = {
            "id",
            "series_id",
            "train_values",
            "point_forecast",
            "lower_80",
            "upper_80",
            "actuals",
        }
        for entry in entries:
            missing = required - entry.keys()
            assert not missing, f"Entry {entry.get('id')} missing fields: {missing}"

    def test_graders_do_not_crash_on_any_entry(self, entries):
        """No golden entry should cause an exception — including edge cases."""
        for entry in entries:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)

            try:
                mase = MASEGrader(train)
                mase.score(actuals, fc)
            except Exception as exc:
                pytest.fail(f"MASEGrader raised on {entry['id']}: {exc}")

            try:
                SMAPEGrader().score(actuals, fc)
            except Exception as exc:
                pytest.fail(f"SMAPEGrader raised on {entry['id']}: {exc}")

            try:
                DirectionalGrader().score(actuals, fc)
            except Exception as exc:
                pytest.fail(f"DirectionalGrader raised on {entry['id']}: {exc}")

            try:
                CoverageGrader().score(actuals, fc)
            except Exception as exc:
                pytest.fail(f"CoverageGrader raised on {entry['id']}: {exc}")

    def test_mase_not_nan_or_inf_on_any_entry(self, entries):
        for entry in entries:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            score = MASEGrader(train).score(actuals, fc)
            assert not np.isnan(score.metric_value), f"NaN MASE on {entry['id']}"
            assert not np.isinf(score.metric_value), f"Inf MASE on {entry['id']}"

    def test_smape_bounded_on_all_entries(self, entries):
        for entry in entries:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            _ = train
            score = SMAPEGrader().score(actuals, fc)
            assert 0 <= score.metric_value <= 200, (
                f"SMAPE out of bounds [{score.metric_value}] on {entry['id']}"
            )

    def test_good_forecasts_pass_mase(self, entries):
        good = [e for e in entries if e.get("category") == "good_forecast"]
        assert len(good) >= 10, f"Expected >=10 good_forecast entries, got {len(good)}"
        failures = []
        for entry in good:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            score = MASEGrader(train).score(actuals, fc)
            if not score.passed:
                failures.append((entry["id"], score.metric_value))
        assert not failures, f"good_forecast entries failed MASE: {failures}"

    def test_good_forecasts_pass_smape(self, entries):
        good = [e for e in entries if e.get("category") == "good_forecast"]
        failures = []
        for entry in good:
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            score = SMAPEGrader().score(actuals, fc)
            if not score.passed:
                failures.append((entry["id"], score.metric_value))
        assert not failures, f"good_forecast entries failed SMAPE: {failures}"

    def test_bad_forecasts_fail_at_least_one_grader(self, entries):
        bad = [e for e in entries if e.get("category") == "bad_forecast"]
        assert len(bad) >= 5, f"Expected >=5 bad_forecast entries, got {len(bad)}"
        for entry in bad:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            mase_score = MASEGrader(train).score(actuals, fc)
            smape_score = SMAPEGrader().score(actuals, fc)
            assert not mase_score.passed or not smape_score.passed, (
                f"bad_forecast entry {entry['id']} unexpectedly passed all graders"
            )

    def test_directional_grader_returns_0_to_100(self, entries):
        for entry in entries:
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            score = DirectionalGrader().score(actuals, fc)
            assert 0 <= score.metric_value <= 100, (
                f"DirectionalAccuracy out of range on {entry['id']}: {score.metric_value}"
            )

    def test_coverage_grader_returns_0_to_100(self, entries):
        for entry in entries:
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            score = CoverageGrader().score(actuals, fc)
            assert 0 <= score.metric_value <= 100, (
                f"Coverage80 out of range on {entry['id']}: {score.metric_value}"
            )

    def test_drift_grader_no_crash_on_all_entries(self, entries):
        """DriftGrader maintains rolling state; accumulate all MASE values."""
        grader = DriftGrader(baseline_mase=0.85)
        for entry in entries:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            mase_score = MASEGrader(train).score(actuals, fc)
            grader.update(mase_score.metric_value)
        drift_score = grader.score()
        assert not np.isnan(drift_score.metric_value)
        assert not np.isinf(drift_score.metric_value)

    def test_eval_harness_runs_on_good_entries(self, entries):
        """EvalHarness end-to-end on good_forecast entries grouped by series_id."""
        from datetime import date

        good = [e for e in entries if e.get("category") == "good_forecast"]
        train_by_series: dict[str, np.ndarray] = {}
        forecasts: list[ForecastResult] = []
        actuals_by_series: dict[str, np.ndarray] = {}

        for entry in good:
            sid = entry["series_id"]
            train_by_series[sid] = np.array(entry["train_values"])
            forecasts.append(_to_forecast_result(entry))
            actuals_by_series[sid] = np.array(entry["actuals"])

        harness = EvalHarness(train_data_by_series=train_by_series)
        report = harness.run(
            cycle_id="golden-forecast-001",
            forecast_date=date(2024, 1, 1),
            forecasts=forecasts,
            actuals_by_series=actuals_by_series,
        )
        assert report.overall_mase >= 0
        assert report.overall_smape >= 0
        assert isinstance(report.all_passed, bool)

    def test_categories_covered(self, entries):
        """Verify the dataset covers all required scenario categories."""
        cats = {e.get("category") for e in entries}
        required_categories = {
            "good_forecast",
            "bad_forecast",
            "high_variance",
            "anomaly_in_actuals",
            "short_history_edge_case",
        }
        missing = required_categories - cats
        assert not missing, f"Golden dataset missing categories: {missing}"

    def test_series_ids_match_real_pipeline_sources(self, entries):
        """Series IDs should match PipelineSource values or archetype-derived names."""
        from core.preprocessing.synthetic import PipelineSource

        valid_sources = {s.value for s in PipelineSource} | {"erp_revenue_retail"}
        for entry in entries:
            sid = entry["series_id"]
            assert sid in valid_sources, f"Entry {entry['id']} has unknown series_id '{sid}'"

    def test_horizon_values_match_forecast_horizon_enum(self, entries):
        """Horizon days must be one of the ForecastHorizon values."""
        valid_horizons = {7, 14, 30, 90}
        for entry in entries:
            h = entry.get("horizon")
            assert h in valid_horizons, f"Entry {entry['id']} has unsupported horizon {h}"


# ── Segment grader tests ───────────────────────────────────────────────────────


class TestSegmentGolden:
    """evaluate_clusters against the golden segment dataset."""

    @pytest.fixture(scope="class")
    @classmethod
    def entries(cls) -> list[dict]:
        return _load_jsonl("segment_golden_qa.jsonl")

    def test_dataset_has_minimum_entries(self, entries):
        assert len(entries) >= 30, f"Expected >=30 entries, got {len(entries)}"

    def test_all_required_fields_present(self, entries):
        for entry in entries:
            # drift entries have nested round_1/round_2 structure
            if entry.get("category") == "segment_drift_multi_step":
                assert "round_1" in entry and "round_2" in entry
            else:
                required = {
                    "id",
                    "embedding_matrix",
                    "labels",
                    "algorithm",
                    "n_clusters",
                    "noise_fraction",
                    "n_customers",
                }
                missing = required - entry.keys()
                assert not missing, f"Entry {entry.get('id')} missing: {missing}"

    def test_evaluate_clusters_does_not_crash_on_any_entry(self, entries):
        for entry in entries:
            if entry.get("category") == "segment_drift_multi_step":
                for rnd_key in ("round_1", "round_2"):
                    rnd = entry[rnd_key]
                    X = np.array(rnd["embedding_matrix"], dtype=np.float32)
                    labels = np.array(rnd["labels"])
                    result = ClusterResult(
                        algorithm=rnd["algorithm"],
                        labels=labels,
                        n_clusters=rnd["n_clusters"],
                        noise_fraction=rnd["noise_fraction"],
                        metadata={},
                    )
                    try:
                        evaluate_clusters(X, result)
                    except Exception as exc:
                        pytest.fail(f"evaluate_clusters raised on {entry['id']} {rnd_key}: {exc}")
            else:
                X = np.array(entry["embedding_matrix"], dtype=np.float32)
                labels = np.array(entry["labels"])
                result = ClusterResult(
                    algorithm=entry["algorithm"],
                    labels=labels,
                    n_clusters=entry["n_clusters"],
                    noise_fraction=entry["noise_fraction"],
                    metadata={},
                )
                try:
                    evaluate_clusters(X, result)
                except Exception as exc:
                    pytest.fail(f"evaluate_clusters raised on {entry['id']}: {exc}")

    def test_clean_separation_entries_pass_silhouette(self, entries):
        clean = [e for e in entries if e.get("category") == "clean_separation"]
        assert len(clean) >= 3, f"Expected >=3 clean_separation entries, got {len(clean)}"
        failures = []
        for entry in clean:
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            if not report.silhouette_passed:
                failures.append((entry["id"], report.silhouette))
        assert not failures, f"clean_separation failed silhouette: {failures}"

    def test_clean_separation_entries_pass_db(self, entries):
        clean = [e for e in entries if e.get("category") == "clean_separation"]
        failures = []
        for entry in clean:
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            if not report.db_passed:
                failures.append((entry["id"], report.davies_bouldin))
        assert not failures, f"clean_separation failed Davies-Bouldin: {failures}"

    def test_clean_separation_passes_all(self, entries):
        clean = [e for e in entries if e.get("category") == "clean_separation"]
        for entry in clean:
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            assert report.all_passed, (
                f"clean_separation entry {entry['id']} failed: {report.summary()}"
            )

    def test_noise_excluded_from_cluster_sizes(self, entries):
        """For HDBSCAN entries, noise label -1 must not appear in cluster_sizes."""
        hdb = [e for e in entries if e.get("category") == "noise_fraction"]
        for entry in hdb:
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            assert -1 not in report.cluster_sizes, (
                f"Noise label -1 found in cluster_sizes for {entry['id']}"
            )

    def test_silhouette_in_valid_range(self, entries):
        for entry in entries:
            if entry.get("category") == "segment_drift_multi_step":
                continue
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            assert -1.0 <= report.silhouette <= 1.0, (
                f"Silhouette out of range on {entry['id']}: {report.silhouette}"
            )

    def test_drift_entries_have_two_rounds(self, entries):
        drift = [e for e in entries if e.get("category") == "segment_drift_multi_step"]
        assert len(drift) >= 4, f"Expected >=4 drift entries, got {len(drift)}"
        for entry in drift:
            r1 = entry["round_1"]
            r2 = entry["round_2"]
            assert r1["n_clusters"] != r2["n_clusters"], (
                f"Drift entry {entry['id']} should have different n_clusters per round"
            )

    def test_categories_covered(self, entries):
        cats = {e.get("category") for e in entries}
        required = {
            "clean_separation",
            "full_archetype_mix",
            "overlapping_archetypes",
            "noise_fraction",
            "small_dataset_edge_case",
            "segment_drift_multi_step",
        }
        missing = required - cats
        assert not missing, f"Segment golden missing categories: {missing}"

    def test_archetype_centroids_are_distinct(self, entries):
        """Validate that the 7 archetype centroids used in generation are distinct."""
        from evals.golden._generate_golden import _archetype_centroid

        archetypes = [
            "early_stage_founder",
            "smb_services",
            "saas_growth",
            "manufacturing",
            "retail_seasonal",
            "professional_services",
            "marketplace",
        ]
        centroids = np.array([_archetype_centroid(a) for a in archetypes])
        for i in range(len(archetypes)):
            for j in range(i + 1, len(archetypes)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                assert dist > 0, (
                    f"Archetypes {archetypes[i]} and {archetypes[j]} have identical centroids"
                )


# ── Crypto grader tests ────────────────────────────────────────────────────────


class TestCryptoGolden:
    """CryptoEvalHarness (Sharpe, Sortino, MaxDrawdown) against golden crypto dataset."""

    @pytest.fixture(scope="class")
    @classmethod
    def entries(cls) -> list[dict]:
        return _load_jsonl("crypto_golden_qa.jsonl")

    def test_dataset_has_minimum_entries(self, entries):
        assert len(entries) >= 30, f"Expected >=30 entries, got {len(entries)}"

    def test_all_required_fields_present(self, entries):
        required = {"id", "returns", "ohlcv", "n_bars"}
        for entry in entries:
            missing = required - entry.keys()
            assert not missing, f"Entry {entry.get('id')} missing: {missing}"

    def test_harness_does_not_crash_on_any_entry(self, entries):
        harness = CryptoEvalHarness()
        for entry in entries:
            returns = np.array(entry["returns"])
            try:
                results = harness.run(returns)
            except Exception as exc:
                pytest.fail(f"CryptoEvalHarness raised on {entry['id']}: {exc}")
            assert len(results) == 3, f"Expected 3 grader results on {entry['id']}"

    def test_all_grader_names_present(self, entries):
        harness = CryptoEvalHarness()
        expected_names = {"sharpe_ratio", "sortino_ratio", "max_drawdown"}
        for entry in entries:
            returns = np.array(entry["returns"])
            results = harness.run(returns)
            names = {r["grader_name"] for r in results}
            assert names == expected_names, f"Entry {entry['id']}: unexpected grader names {names}"

    def test_bull_run_entries_pass_sharpe(self, entries):
        bull = [e for e in entries if e.get("category") == "bull_run"]
        assert len(bull) >= 6, f"Expected >=6 bull_run entries, got {len(bull)}"
        grader = SharpeGrader(threshold=0.5)
        failures = []
        for entry in bull:
            returns = np.array(entry["returns"])
            result = grader.score(returns)
            if not result["passed"]:
                failures.append((entry["id"], result["metric_value"]))
        assert not failures, f"bull_run entries failed Sharpe>0.5: {failures}"

    def test_bull_run_entries_pass_max_drawdown(self, entries):
        bull = [e for e in entries if e.get("category") == "bull_run"]
        grader = MaxDrawdownGrader(threshold=0.15)
        failures = []
        for entry in bull:
            returns = np.array(entry["returns"])
            result = grader.score(returns)
            if not result["passed"]:
                failures.append((entry["id"], result["metric_value"]))
        assert not failures, f"bull_run entries failed MaxDrawdown<0.15: {failures}"

    def test_bear_market_entries_fail_sharpe(self, entries):
        bear = [e for e in entries if e.get("category") == "bear_market"]
        assert len(bear) >= 5, f"Expected >=5 bear_market entries, got {len(bear)}"
        grader = SharpeGrader(threshold=0.5)
        failures = []
        for entry in bear:
            returns = np.array(entry["returns"])
            result = grader.score(returns)
            if result["passed"]:
                failures.append((entry["id"], result["metric_value"]))
        assert not failures, f"bear_market entries unexpectedly passed Sharpe: {failures}"

    def test_bear_market_entries_fail_max_drawdown(self, entries):
        bear = [e for e in entries if e.get("category") == "bear_market"]
        grader = MaxDrawdownGrader(threshold=0.15)
        failures = []
        for entry in bear:
            returns = np.array(entry["returns"])
            result = grader.score(returns)
            if result["passed"]:
                failures.append((entry["id"], result["metric_value"]))
        assert not failures, f"bear_market entries unexpectedly passed MaxDrawdown: {failures}"

    def test_metric_values_are_finite_on_all_entries(self, entries):
        harness = CryptoEvalHarness()
        for entry in entries:
            returns = np.array(entry["returns"])
            results = harness.run(returns)
            for r in results:
                assert np.isfinite(r["metric_value"]) or r["metric_value"] == 0.0, (
                    f"Non-finite metric {r['grader_name']}={r['metric_value']} on {entry['id']}"
                )

    def test_max_drawdown_in_zero_to_one_range(self, entries):
        grader = MaxDrawdownGrader(threshold=0.15)
        for entry in entries:
            returns = np.array(entry["returns"])
            result = grader.score(returns)
            assert 0.0 <= result["metric_value"] <= 1.0, (
                f"MaxDrawdown out of range on {entry['id']}: {result['metric_value']}"
            )

    def test_returns_sufficient_length(self, entries):
        """Each entry must have at least 2 returns (needed for std/Sharpe)."""
        for entry in entries:
            returns = entry["returns"]
            assert len(returns) >= 2, (
                f"Entry {entry['id']} has insufficient returns: {len(returns)}"
            )

    def test_categories_covered(self, entries):
        cats = {e.get("category") for e in entries}
        required = {
            "bull_run",
            "moderate_growth",
            "bear_market",
            "low_volatility_sideways",
            "multi_symbol_spread",
        }
        missing = required - cats
        assert not missing, f"Crypto golden missing categories: {missing}"

    def test_ohlcv_has_all_columns(self, entries):
        """OHLCV dicts must have open, high, low, close, volume for the primary key."""
        required_cols = {"open", "high", "low", "close", "volume"}
        for entry in entries:
            primary_key = entry.get("symbol_key", next(iter(entry["ohlcv"].keys())))
            ohlcv = entry["ohlcv"].get(primary_key, {})
            missing = required_cols - ohlcv.keys()
            assert not missing, (
                f"Entry {entry['id']} OHLCV key '{primary_key}' missing columns: {missing}"
            )

    def test_multi_symbol_entries_have_two_ohlcv_keys(self, entries):
        multi = [e for e in entries if e.get("category") == "multi_symbol_spread"]
        assert len(multi) >= 4, f"Expected >=4 multi_symbol_spread entries, got {len(multi)}"
        for entry in multi:
            assert len(entry["ohlcv"]) >= 2, (
                f"Multi-symbol entry {entry['id']} has only 1 OHLCV key"
            )


# ── Baseline score recording ───────────────────────────────────────────────────


class TestBaselineScores:
    """Record aggregate baseline scores for regression tracking.

    These do not assert pass/fail thresholds — they measure the current
    baseline so future runs can detect regressions (e.g. via recorded values
    in CI output or a baseline.json artifact).
    """

    def test_record_forecast_baseline(self, capsys):
        entries = _load_jsonl("forecast_golden_qa.jsonl")
        good = [e for e in entries if e.get("category") == "good_forecast"]

        mase_scores, smape_scores, dir_scores, cov_scores = [], [], [], []
        for entry in good:
            train = np.array(entry["train_values"])
            actuals = np.array(entry["actuals"])
            fc = _to_forecast_result(entry)
            mase_scores.append(MASEGrader(train).score(actuals, fc).metric_value)
            smape_scores.append(SMAPEGrader().score(actuals, fc).metric_value)
            dir_scores.append(DirectionalGrader().score(actuals, fc).metric_value)
            cov_scores.append(CoverageGrader().score(actuals, fc).metric_value)

        with capsys.disabled():
            print(f"\n[BASELINE] Forecast (good_forecast, n={len(good)})")
            print(f"  MASE:     mean={np.mean(mase_scores):.4f}  std={np.std(mase_scores):.4f}")
            print(f"  SMAPE:    mean={np.mean(smape_scores):.2f}%  std={np.std(smape_scores):.2f}%")
            print(f"  Dir:      mean={np.mean(dir_scores):.1f}%  std={np.std(dir_scores):.1f}%")
            print(f"  Cov80:    mean={np.mean(cov_scores):.1f}%  std={np.std(cov_scores):.1f}%")

        # Soft gate: baseline MASE on good entries must beat naïve (< 1.0)
        assert np.mean(mase_scores) < 1.0, (
            f"Baseline MASE regression: mean={np.mean(mase_scores):.4f} ≥ 1.0"
        )

    def test_record_crypto_baseline(self, capsys):
        entries = _load_jsonl("crypto_golden_qa.jsonl")
        harness = CryptoEvalHarness()

        sharpe_all, sortino_all, dd_all = [], [], []
        for entry in entries:
            returns = np.array(entry["returns"])
            results = {r["grader_name"]: r["metric_value"] for r in harness.run(returns)}
            sharpe_all.append(results["sharpe_ratio"])
            sortino_all.append(results["sortino_ratio"])
            dd_all.append(results["max_drawdown"])

        with capsys.disabled():
            print(f"\n[BASELINE] Crypto (all entries, n={len(entries)})")
            print(f"  Sharpe:   mean={np.mean(sharpe_all):.4f}  std={np.std(sharpe_all):.4f}")
            print(f"  Sortino:  mean={np.mean(sortino_all):.4f}  std={np.std(sortino_all):.4f}")
            print(f"  MaxDD:    mean={np.mean(dd_all):.4f}  std={np.std(dd_all):.4f}")

    def test_record_segment_baseline(self, capsys):
        entries = _load_jsonl("segment_golden_qa.jsonl")
        non_drift = [e for e in entries if e.get("category") != "segment_drift_multi_step"]

        sil_all, db_all = [], []
        for entry in non_drift:
            X = np.array(entry["embedding_matrix"], dtype=np.float32)
            labels = np.array(entry["labels"])
            result = ClusterResult(
                algorithm=entry["algorithm"],
                labels=labels,
                n_clusters=entry["n_clusters"],
                noise_fraction=entry["noise_fraction"],
                metadata={},
            )
            report = evaluate_clusters(X, result)
            sil_all.append(report.silhouette)
            db_all.append(report.davies_bouldin)

        with capsys.disabled():
            print(f"\n[BASELINE] Segment (non-drift entries, n={len(non_drift)})")
            print(f"  Silhouette:   mean={np.mean(sil_all):.4f}  std={np.std(sil_all):.4f}")
            print(f"  Davies-Bouldin: mean={np.mean(db_all):.4f}  std={np.std(db_all):.4f}")
