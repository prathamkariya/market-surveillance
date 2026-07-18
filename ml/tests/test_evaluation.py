import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.evaluation.metrics import (
    compare_models_time_series_cv,
    calibration_analysis,
    compute_learning_curve,
    evaluate_on_label,
    compare_combined_vs_per_pattern_auc,
)


@pytest.fixture
def time_series_classification_data():
    rng = np.random.RandomState(30)
    n = 600
    X = pd.DataFrame({
        "f0": rng.uniform(-2, 2, n),
        "f1": rng.uniform(-2, 2, n),
    })
    y = pd.Series((X["f0"] + X["f1"] * 0.5 + rng.normal(0, 0.5, n) > 0).astype(int))
    return X, y


class TestCompareModelsTimeSeriesCv:
    def test_returns_one_row_per_model(self, time_series_classification_data):
        X, y = time_series_classification_data
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        }
        result = compare_models_time_series_cv(X, y, models, n_splits=4, scale_for={"Logistic Regression"})
        assert len(result) == 2
        assert set(result["model"]) == {"Logistic Regression", "Random Forest"}

    def test_sorted_by_mean_auc_descending(self, time_series_classification_data):
        X, y = time_series_classification_data
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
        }
        result = compare_models_time_series_cv(X, y, models, n_splits=4, scale_for={"Logistic Regression"})
        aucs = result["mean_auc"].values
        assert (aucs[:-1] >= aucs[1:]).all()

    def test_folds_never_use_future_data_for_training(self, time_series_classification_data):
        """Indirect check: TimeSeriesSplit's fold boundaries should be
        strictly increasing (each successive fold's test set starts
        after the previous fold's test set ends)."""
        from sklearn.model_selection import TimeSeriesSplit
        X, y = time_series_classification_data
        tscv = TimeSeriesSplit(n_splits=4)
        last_test_start = -1
        for train_idx, test_idx in tscv.split(X):
            assert max(train_idx) < min(test_idx)
            assert min(test_idx) > last_test_start
            last_test_start = min(test_idx)

    def test_skips_folds_with_zero_positive_test_examples(self):
        """A fold with y_test_fold.sum() == 0 should be silently skipped,
        not crash on undefined AUC."""
        n = 100
        X = pd.DataFrame({"f0": np.arange(n, dtype=float)})
        y = pd.Series([0] * (n - 5) + [1] * 5)  # positives only at the very end
        models = {"RF": RandomForestClassifier(n_estimators=10, random_state=1)}
        result = compare_models_time_series_cv(X, y, models, n_splits=5)
        # should not raise, and should produce a valid (possibly nan) result
        assert len(result) == 1


class TestCalibrationAnalysis:
    def test_auc_barely_changes_after_calibration(self, time_series_classification_data):
        """File 25 Section 3's core claim: calibration only rescales
        probabilities, it doesn't change rank-ordering, so AUC should be
        nearly unchanged."""
        X, y = time_series_classification_data
        split = 450
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        result = calibration_analysis(model, X_test, y_test)
        assert result["auc_changed_by"] < 0.05

    def test_returns_calibration_curve_arrays(self, time_series_classification_data):
        X, y = time_series_classification_data
        split = 450
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(X_train, y_train)
        result = calibration_analysis(model, X_test, y_test, n_bins=3)
        assert len(result["fraction_of_positives_before"]) > 0
        assert len(result["mean_predicted_before"]) == len(result["fraction_of_positives_before"])


class TestComputeLearningCurve:
    def test_returns_expected_columns(self, time_series_classification_data):
        X, y = time_series_classification_data
        model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
        result = compute_learning_curve(model, X, y, n_splits=3)
        expected_cols = {"train_size", "train_score_mean", "train_score_std", "test_score_mean", "test_score_std"}
        assert expected_cols.issubset(set(result.columns))

    def test_train_sizes_increasing(self, time_series_classification_data):
        X, y = time_series_classification_data
        model = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
        result = compute_learning_curve(model, X, y, n_splits=3)
        sizes = result["train_size"].values
        assert (sizes[:-1] < sizes[1:]).all()


class TestEvaluateOnLabel:
    def test_returns_none_when_test_split_has_no_positives(self):
        n = 100
        X = np.random.RandomState(1).normal(0, 1, (n, 2))
        y = np.array([1] * 5 + [0] * (n - 5))  # positives only in the training portion (first 70%)
        result = evaluate_on_label(X, y)
        assert result is None

    def test_returns_valid_auc_for_separable_signal(self):
        rng = np.random.RandomState(1)
        n = 500
        X = rng.normal(0, 1, (n, 2))
        y = (X[:, 0] > 0).astype(int)
        # shuffle so positives appear throughout, not just in one region
        shuffle_idx = rng.permutation(n)
        result = evaluate_on_label(X[shuffle_idx], y[shuffle_idx])
        assert result is not None
        assert 0 <= result <= 1


class TestCompareCombinedVsPerPatternAuc:
    def test_reveals_a_gap_between_combined_and_per_pattern_auc(self):
        """The direct reproduction of file 25's closing argument: a
        feature that's a strong, clean proxy for ONE specific pattern
        should score notably differently when evaluated against its own
        true label versus a blended combined label.

        Uses DISTINCT seeds for generating the synthetic labels versus
        the shuffle inside compare_combined_vs_per_pattern_auc. Reusing
        the same seed for both is a real numpy gotcha: rng.choice(n, k,
        replace=False) and a later rng.permutation(n) seeded identically
        are both Fisher-Yates-based on the same underlying state and can
        come out correlated -- confirmed here by testing, not assumed --
        which produced a systematically empty test split regardless of
        which single shared seed value was tried.
        """
        data_rng = np.random.RandomState(50)
        n = 600
        pattern_a = np.zeros(n, dtype=int)
        pattern_b = np.zeros(n, dtype=int)
        a_idx = data_rng.choice(n, 60, replace=False)
        remaining = np.setdiff1d(np.arange(n), a_idx)
        b_idx = data_rng.choice(remaining, 60, replace=False)
        pattern_a[a_idx] = 1
        pattern_b[b_idx] = 1

        # feature 0 is a strong, clean proxy for pattern A specifically;
        # feature 1 is pure noise w.r.t. both patterns
        X = np.column_stack([
            np.where(pattern_a == 1, data_rng.normal(5, 0.3, n), data_rng.normal(0, 1, n)),
            data_rng.normal(0, 1, n),
        ])
        combined = ((pattern_a == 1) | (pattern_b == 1)).astype(int)

        result = compare_combined_vs_per_pattern_auc(
            X, {"pattern_a": pattern_a, "pattern_b": pattern_b}, combined, random_state=7
        )
        pattern_a_auc = result.loc[result["label"] == "pattern_a (per-pattern)", "auc"].iloc[0]
        combined_auc = result.loc[result["label"] == "combined (blended)", "auc"].iloc[0]
        assert not np.isnan(pattern_a_auc) and not np.isnan(combined_auc)
        # pattern A's own AUC should be notably HIGHER than the combined
        # label's AUC, since the combined label dilutes A's clean signal
        # with pattern B days the feature says nothing about
        assert pattern_a_auc > combined_auc

    def test_returns_one_row_per_pattern_plus_combined(self):
        data_rng = np.random.RandomState(51)
        n = 400
        pattern_a = np.zeros(n, dtype=int)
        pattern_a[data_rng.choice(n, 40, replace=False)] = 1
        combined = pattern_a.copy()
        X = data_rng.normal(0, 1, (n, 2))

        result = compare_combined_vs_per_pattern_auc(X, {"pattern_a": pattern_a}, combined, random_state=13)
        assert len(result) == 2  # combined + 1 pattern
