"""
Gradient Boosting, matching file 24 Sections 5-9.

GradientBoostingScratch boosts in LOG-ODDS space, matching file 21
Section 4's logit framing for logistic regression -- the (y - p) residual
form is a direct consequence of pairing sigmoid with cross-entropy, not
an independent design choice (same point file 21 makes about the
logistic regression gradient).

train_xgboost_model wraps the xgb.DMatrix/xgb.train pattern from Section
8 as a reusable function with early stopping and sane defaults, instead
of a one-off script with hardcoded feature names and train/test objects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingScratch:
    """Binary classification gradient boosting. Boosts F(x) (the running
    log-odds prediction) by repeatedly fitting a regression tree to the
    CURRENT residual (y - sigmoid(F)), then adding a shrunk (learning_rate-
    scaled) copy of that tree's predictions back into F. File 24 Section 5.
    """

    def __init__(self, n_estimators: int = 50, learning_rate: float = 0.1, max_depth: int = 3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.trees: list[DecisionTreeRegressor] = []
        self.initial_log_odds: float | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingScratch":
        X = np.asarray(X)
        y = np.asarray(y)

        base_rate = y.mean()
        if base_rate <= 0 or base_rate >= 1:
            raise ValueError(
                f"Base rate is {base_rate}; need at least one example of "
                f"each class to compute a finite initial log-odds."
            )
        self.initial_log_odds = float(np.log(base_rate / (1 - base_rate)))

        F = np.full(len(y), self.initial_log_odds)

        self.trees = []
        for _ in range(self.n_estimators):
            current_probabilities = self._sigmoid(F)
            residuals = y - current_probabilities

            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=len(self.trees))
            tree.fit(X, residuals)
            self.trees.append(tree)

            F = F + self.lr * tree.predict(X)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.initial_log_odds is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.asarray(X)
        F = np.full(len(X), self.initial_log_odds)
        for tree in self.trees:
            F = F + self.lr * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def staged_predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        """Probability predictions after each successive tree is added --
        useful for plotting how quickly the model converges, and for
        picking a smaller n_estimators than what it was trained with
        without retraining."""
        if self.initial_log_odds is None:
            raise RuntimeError("Call fit() before staged_predict_proba().")
        X = np.asarray(X)
        F = np.full(len(X), self.initial_log_odds)
        stages = []
        for tree in self.trees:
            F = F + self.lr * tree.predict(X)
            stages.append(self._sigmoid(F))
        return stages


DEFAULT_XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,  # XGBoost's analogue to Random Forest's feature subsampling
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "seed": 42,
}


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict | None = None,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50,
    verbose_eval: bool | int = False,
) -> xgb.Booster:
    """File 24 Section 8's xgb.train pattern, as a function instead of a
    one-off script. Early stopping means num_boost_round is a ceiling,
    not a target -- the actual number of trees used is model.best_iteration,
    picked by test-set AUC, not guessed in advance.
    """
    merged_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_train.columns))

    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(
        merged_params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return model


def xgboost_feature_importance(model: xgb.Booster, importance_type: str = "gain") -> pd.DataFrame:
    """File 24's importance-by-gain vs importance-by-weight distinction.

    importance_type='weight': how many times a feature was used to split
        (a simple split-count, the same naive notion file 22's embedded
        importance uses).
    importance_type='gain': average improvement in the loss function
        each time this feature is used to split. Usually the more
        meaningful of the two -- a feature used rarely but decisively
        can outrank one used often but weakly.
    """
    scores = model.get_score(importance_type=importance_type)
    if not scores:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame(
        {"feature": list(scores.keys()), "importance": list(scores.values())}
    ).sort_values("importance", ascending=False).reset_index(drop=True)
