import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ml.models.logistic_regression import (
    LogisticRegressionScratch,
    threshold_sweep,
    best_threshold_by_f1,
    get_coefficient_importance,
)


@pytest.fixture
def separable_data():
    rng = np.random.RandomState(42)
    n = 400
    X = rng.normal(0, 1, (n, 2))
    y = (X[:, 0] * 2 + X[:, 1] - 1 > 0).astype(int)
    return X, y


class TestLogisticRegressionScratch:
    def test_cost_decreases_monotonically(self, separable_data):
        """File 21 Section 4's convexity claim: BCE loss for logistic
        regression is convex, so gradient descent should not increase
        cost between iterations (allowing for tiny float noise)."""
        X, y = separable_data
        model = LogisticRegressionScratch(learning_rate=0.1, iterations=500)
        model.fit(X, y)
        diffs = np.diff(model.cost_history)
        assert (diffs <= 1e-9).all(), "cost increased at some iteration -- not convex behavior"

    def test_predict_proba_between_zero_and_one(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_achieves_reasonable_auc_on_separable_data(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(learning_rate=0.1, iterations=1000).fit(X, y)
        proba = model.predict_proba(X)
        auc = roc_auc_score(y, proba)
        assert auc > 0.85

    def test_roughly_matches_sklearn_auc_on_same_data(self, separable_data):
        """Not exact numerical equality (different optimizers -- gradient
        descent vs sklearn's default lbfgs solver -- will converge to
        slightly different points), but should land in the same
        performance ballpark on genuinely separable data."""
        X, y = separable_data
        scratch = LogisticRegressionScratch(learning_rate=0.1, iterations=1000).fit(X, y)
        sklearn_model = LogisticRegression().fit(X, y)

        scratch_auc = roc_auc_score(y, scratch.predict_proba(X))
        sklearn_auc = roc_auc_score(y, sklearn_model.predict_proba(X)[:, 1])
        assert abs(scratch_auc - sklearn_auc) < 0.05

    def test_predict_respects_custom_threshold(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        low_thresh_preds = model.predict(X, threshold=0.01)
        high_thresh_preds = model.predict(X, threshold=0.99)
        # near-0 threshold should flag far more positives than near-1 threshold
        assert low_thresh_preds.sum() >= high_thresh_preds.sum()

    def test_predict_proba_before_fit_raises(self):
        model = LogisticRegressionScratch()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.array([[1, 2]]))

    def test_handles_extreme_z_without_overflow_warning(self):
        """The z-clipping in sigmoid() exists specifically so this
        doesn't raise a RuntimeWarning or produce NaN."""
        model = LogisticRegressionScratch()
        model.weights = np.array([1000.0, 1000.0])
        model.bias = 1000.0
        result = model.predict_proba(np.array([[10.0, 10.0]]))
        assert np.isfinite(result).all()


class TestThresholdSweep:
    def test_returns_expected_columns(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        sweep = threshold_sweep(y, model.predict_proba(X))
        assert list(sweep.columns) == ["threshold", "precision", "recall", "f1"]

    def test_recall_decreases_as_threshold_increases(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        sweep = threshold_sweep(y, model.predict_proba(X), thresholds=[0.1, 0.5, 0.9])
        recalls = sweep.sort_values("threshold")["recall"].values
        assert recalls[0] >= recalls[1] >= recalls[2]

    def test_precision_generally_increases_as_threshold_increases(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        sweep = threshold_sweep(y, model.predict_proba(X), thresholds=[0.1, 0.9])
        sweep_sorted = sweep.sort_values("threshold")
        assert sweep_sorted.iloc[1]["precision"] >= sweep_sorted.iloc[0]["precision"] - 0.05


class TestBestThresholdByF1:
    def test_returns_a_value_from_the_swept_thresholds(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        thresholds = [0.2, 0.4, 0.6, 0.8]
        best = best_threshold_by_f1(y, model.predict_proba(X), thresholds)
        assert best in thresholds


class TestGetCoefficientImportance:
    def test_works_with_scratch_model(self, separable_data):
        X, y = separable_data
        model = LogisticRegressionScratch(iterations=500).fit(X, y)
        result = get_coefficient_importance(model, ["feat_a", "feat_b"])
        assert set(result["feature"]) == {"feat_a", "feat_b"}
        assert "coefficient" in result.columns

    def test_works_with_sklearn_model(self, separable_data):
        X, y = separable_data
        model = LogisticRegression().fit(X, y)
        result = get_coefficient_importance(model, ["feat_a", "feat_b"])
        assert set(result["feature"]) == {"feat_a", "feat_b"}

    def test_sorted_by_absolute_coefficient_descending(self, separable_data):
        X, y = separable_data
        model = LogisticRegression().fit(X, y)
        result = get_coefficient_importance(model, ["feat_a", "feat_b"])
        abs_coefs = result["coefficient"].abs().values
        assert (abs_coefs[:-1] >= abs_coefs[1:]).all()

    def test_raises_on_model_without_coefficients(self):
        class NotAModel:
            pass
        with pytest.raises(ValueError, match="neither"):
            get_coefficient_importance(NotAModel(), ["a"])

    def test_stronger_feature_ranked_first(self):
        """feat_a has coefficient weight ~2x feat_b by construction --
        importance ranking should reflect that."""
        rng = np.random.RandomState(9)
        n = 500
        X = rng.normal(0, 1, (n, 2))
        y = (X[:, 0] * 4 + X[:, 1] * 0.5 + rng.normal(0, 0.3, n) > 0).astype(int)
        model = LogisticRegression().fit(X, y)
        result = get_coefficient_importance(model, ["strong", "weak"])
        assert result.iloc[0]["feature"] == "strong"
