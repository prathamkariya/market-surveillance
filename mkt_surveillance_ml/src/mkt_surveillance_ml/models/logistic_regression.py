"""
Logistic Regression, matching file 21 Sections 7-11 and 15-16.

LogisticRegressionScratch is preserved with identical numerics to the
notes -- same zero-init, same z-clipping for numerical stability, same
1e-15 epsilon in the log-loss. This is the piece worth defending line by
line in an interview; changing any of these details for "cleanliness"
would make it a different (and less honest) implementation than what
was actually derived from the math.

threshold_sweep and get_coefficient_importance turn Sections 15 and 16's
one-off print statements into reusable functions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


class LogisticRegressionScratch:
    """Logistic Regression via batch gradient descent on binary
    cross-entropy loss. Built to expose every mechanical step the
    sklearn version hides -- see file 21 Section 8 for the derivation
    of why BCE (not MSE) is the theoretically correct loss here, and
    why the gradient has exactly this (y_hat - y) form.
    """

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.cost_history: list[float] = []

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        # Clipping avoids overflow in np.exp for very confident/wrong
        # predictions -- a numerical-stability detail, not a change to
        # the underlying math.
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            cost = -(1 / n_samples) * np.sum(
                y * np.log(y_predicted + 1e-15)
                + (1 - y) * np.log(1 - y_predicted + 1e-15)
            )
            self.cost_history.append(cost)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Call fit() before predict_proba().")
        linear_model = np.dot(np.asarray(X), self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def threshold_sweep(
    y_true: np.ndarray, y_pred_proba: np.ndarray, thresholds: list[float] | None = None
) -> pd.DataFrame:
    """File 21 Section 15: precision/recall/F1 at a range of decision
    thresholds. The default 0.5 threshold is a convention, not a law --
    for surveillance specifically, where missing a manipulation event
    (false negative) is usually costlier than a false alarm, a lower
    threshold trading precision for recall is often the right call. This
    function makes that tradeoff visible instead of leaving 0.5 unexamined.
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    for t in thresholds:
        y_pred_at_t = (y_pred_proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred_at_t, zero_division=0),
            "recall": recall_score(y_true, y_pred_at_t, zero_division=0),
            "f1": f1_score(y_true, y_pred_at_t, zero_division=0),
        })
    return pd.DataFrame(rows)


def best_threshold_by_f1(
    y_true: np.ndarray, y_pred_proba: np.ndarray, thresholds: list[float] | None = None
) -> float:
    """Pick the threshold maximizing F1 from the sweep above."""
    sweep = threshold_sweep(y_true, y_pred_proba, thresholds)
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def get_coefficient_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """File 21 Section 16. Works for either LogisticRegressionScratch
    (has .weights) or sklearn's LogisticRegression (has .coef_) --
    accepts either without the caller needing to know which one they have.

    Sign matters here in a way tree-based importance (files 22-24) can't
    offer: a positive coefficient means the feature increasing raises the
    log-odds of the positive class, holding other features constant. Tree
    importance tells you a feature mattered; this tells you which direction.
    """
    if hasattr(model, "coef_"):
        coefficients = model.coef_[0]
    elif hasattr(model, "weights"):
        coefficients = model.weights
    else:
        raise ValueError(
            "Model has neither .coef_ (sklearn) nor .weights "
            "(LogisticRegressionScratch) -- can't extract coefficients."
        )
    return pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    }).sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)
