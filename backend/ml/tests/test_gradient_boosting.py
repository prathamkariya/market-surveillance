import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, accuracy_score

from ml.models.gradient_boosting import (
    GradientBoostingScratch,
    train_xgboost_model,
    xgboost_feature_importance,
    DEFAULT_XGB_PARAMS,
)


@pytest.fixture
def classification_data():
    rng = np.random.RandomState(20)
    n = 600
    X = rng.uniform(-2, 2, (n, 3))
    y = ((X[:, 0] + X[:, 1] * 0.5 + rng.normal(0, 0.4, n)) > 0).astype(int)
    X_train, X_test = X[:450], X[450:]
    y_train, y_test = y[:450], y[450:]
    return X_train, X_test, y_train, y_test


class TestGradientBoostingScratch:
    def test_initial_log_odds_matches_base_rate(self):
        y = np.array([0] * 30 + [1] * 70)  # 70% base rate
        model = GradientBoostingScratch(n_estimators=1).fit(np.zeros((100, 1)), y)
        expected = np.log(0.7 / 0.3)
        assert model.initial_log_odds == pytest.approx(expected, abs=1e-6)

    def test_raises_on_all_same_class(self):
        """Base rate of 0 or 1 makes log-odds undefined (division by zero
        or log(0)) -- should fail loudly, not silently produce inf/nan."""
        y = np.zeros(50)
        model = GradientBoostingScratch()
        with pytest.raises(ValueError, match="Base rate"):
            model.fit(np.zeros((50, 1)), y)

    def test_achieves_good_auc_on_separable_data(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = GradientBoostingScratch(n_estimators=80, learning_rate=0.1, max_depth=3)
        model.fit(X_train, y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test))
        assert auc > 0.85

    def test_more_estimators_generally_reduces_training_error(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        few_trees = GradientBoostingScratch(n_estimators=5, learning_rate=0.1, max_depth=3)
        few_trees.fit(X_train, y_train)
        many_trees = GradientBoostingScratch(n_estimators=100, learning_rate=0.1, max_depth=3)
        many_trees.fit(X_train, y_train)

        few_train_auc = roc_auc_score(y_train, few_trees.predict_proba(X_train))
        many_train_auc = roc_auc_score(y_train, many_trees.predict_proba(X_train))
        assert many_train_auc >= few_train_auc

    def test_staged_predict_proba_final_stage_matches_predict_proba(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = GradientBoostingScratch(n_estimators=20, learning_rate=0.1, max_depth=3)
        model.fit(X_train, y_train)
        staged = model.staged_predict_proba(X_test)
        final_predict_proba = model.predict_proba(X_test)
        assert np.allclose(staged[-1], final_predict_proba)

    def test_staged_predict_proba_has_one_entry_per_tree(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = GradientBoostingScratch(n_estimators=15, learning_rate=0.1, max_depth=3)
        model.fit(X_train, y_train)
        staged = model.staged_predict_proba(X_test)
        assert len(staged) == 15

    def test_predict_proba_before_fit_raises(self):
        model = GradientBoostingScratch()
        with pytest.raises(RuntimeError):
            model.predict_proba(np.zeros((5, 2)))

    def test_predict_proba_bounded_zero_one(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        model = GradientBoostingScratch(n_estimators=50).fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestTrainXgboostModel:
    def _make_df_data(self, classification_data):
        X_train, X_test, y_train, y_test = classification_data
        cols = ["f0", "f1", "f2"]
        return (
            pd.DataFrame(X_train, columns=cols),
            pd.Series(y_train),
            pd.DataFrame(X_test, columns=cols),
            pd.Series(y_test),
        )

    def test_trains_and_predicts_successfully(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            num_boost_round=100, early_stopping_rounds=15,
        )
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test_df, feature_names=list(X_test_df.columns))
        proba = model.predict(dtest)
        auc = roc_auc_score(y_test_s, proba)
        assert auc > 0.8

    def test_early_stopping_uses_fewer_rounds_than_ceiling(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            num_boost_round=2000, early_stopping_rounds=20,
        )
        assert model.best_iteration < 2000

    def test_custom_params_override_defaults(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            params={"max_depth": 2}, num_boost_round=50, early_stopping_rounds=10,
        )
        # sanity: model trained without error and produces valid predictions
        import xgboost as xgb
        dtest = xgb.DMatrix(X_test_df, feature_names=list(X_test_df.columns))
        proba = model.predict(dtest)
        assert np.isfinite(proba).all()

    def test_heavier_regularization_reduces_or_maintains_train_test_gap(self, classification_data):
        """File 24 Section 9's claim: reg_alpha/reg_lambda penalize
        extreme leaf values, which should shrink (or at least not
        obviously worsen) the train/test AUC gap versus no regularization."""
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data(classification_data)
        import xgboost as xgb

        no_reg_model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            params={"reg_alpha": 0, "reg_lambda": 0},
            num_boost_round=200, early_stopping_rounds=200,  # effectively disable early stop
        )
        heavy_reg_model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            params={"reg_alpha": 1.0, "reg_lambda": 5.0},
            num_boost_round=200, early_stopping_rounds=200,
        )

        dtrain = xgb.DMatrix(X_train_df, feature_names=list(X_train_df.columns))
        dtest = xgb.DMatrix(X_test_df, feature_names=list(X_test_df.columns))

        no_reg_gap = (
            roc_auc_score(y_train_s, no_reg_model.predict(dtrain))
            - roc_auc_score(y_test_s, no_reg_model.predict(dtest))
        )
        heavy_reg_gap = (
            roc_auc_score(y_train_s, heavy_reg_model.predict(dtrain))
            - roc_auc_score(y_test_s, heavy_reg_model.predict(dtest))
        )
        assert heavy_reg_gap <= no_reg_gap + 0.05


class TestXgboostFeatureImportance:
    def test_returns_feature_and_importance_columns(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data_static(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            num_boost_round=50, early_stopping_rounds=10,
        )
        result = xgboost_feature_importance(model, importance_type="gain")
        assert list(result.columns) == ["feature", "importance"]

    def test_sorted_descending_by_importance(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data_static(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            num_boost_round=50, early_stopping_rounds=10,
        )
        result = xgboost_feature_importance(model, importance_type="gain")
        values = result["importance"].values
        assert (values[:-1] >= values[1:]).all()

    def test_strongest_feature_ranked_first(self, classification_data):
        X_train_df, y_train_s, X_test_df, y_test_s = self._make_df_data_static(classification_data)
        model = train_xgboost_model(
            X_train_df, y_train_s, X_test_df, y_test_s,
            num_boost_round=100, early_stopping_rounds=20,
        )
        result = xgboost_feature_importance(model, importance_type="gain")
        # f0 has the largest coefficient (1.0) in the data-generating process
        assert result.iloc[0]["feature"] == "f0"

    @staticmethod
    def _make_df_data_static(classification_data):
        X_train, X_test, y_train, y_test = classification_data
        cols = ["f0", "f1", "f2"]
        return (
            pd.DataFrame(X_train, columns=cols),
            pd.Series(y_train),
            pd.DataFrame(X_test, columns=cols),
            pd.Series(y_test),
        )
