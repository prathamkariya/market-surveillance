"""
Random Forest, matching file 23 Sections 2-4.

File 23's own version of DecisionTreeClassifierScratch is an intentional,
explicitly-commented duplicate ("redefined here for a self-contained
file") of file 22's -- fine for a standalone note, not fine here.
RandomForestScratch below imports the ONE canonical version from
models/decision_tree.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mkt_surveillance_ml.models.decision_tree import DecisionTreeClassifierScratch


def bootstrap_sample(
    X: pd.DataFrame, y: pd.Series, random_state: int | None = None
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Sampling WITH replacement, same size as the original dataset.
    File 23 Section 2. Some rows appear multiple times; ~37% (1/e, per
    Section 2's bootstrap-math derivation) don't appear at all -- those
    excluded rows are exactly what out-of-bag scoring uses for a "free"
    validation set later.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    return X.iloc[indices], y.iloc[indices], indices


class RandomForestScratch:
    """Bootstrap aggregating (bagging) of DecisionTreeClassifierScratch,
    plus per-tree random feature subsetting. File 23 Sections 2-4.

    Documented simplification, not a silent one: this restricts the
    feature subset ONCE per tree, at construction time. sklearn's
    RandomForestClassifier re-randomizes the feature subset at EVERY
    individual split within a tree, which decorrelates trees more
    aggressively than this version does. Both are legitimate designs;
    they are not the same algorithm, and claiming numerical parity with
    sklearn here would be false -- see test_random_forest.py's comparison
    test for what that difference in decorrelation strength looks like in
    practice.
    """

    def __init__(
        self,
        n_estimators: int = 20,
        max_depth: int = 5,
        max_features: str = "sqrt",
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[DecisionTreeClassifierScratch] = []
        self.tree_feature_subsets: list[np.ndarray] = []

    def _get_max_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestScratch":
        X = np.asarray(X)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_features_per_split = self._get_max_features(n_features)

        self.trees = []
        self.tree_feature_subsets = []
        for _ in range(self.n_estimators):
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[boot_idx], y[boot_idx]

            feature_subset = rng.choice(n_features, n_features_per_split, replace=False)
            self.tree_feature_subsets.append(feature_subset)

            tree = DecisionTreeClassifierScratch(max_depth=self.max_depth, min_samples_split=10)
            tree.fit(X_boot[:, feature_subset], y_boot)
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("Call fit() before predict_proba().")
        X = np.asarray(X)
        all_predictions = np.zeros((self.n_estimators, len(X)))
        for i, (tree, feature_subset) in enumerate(zip(self.trees, self.tree_feature_subsets)):
            all_predictions[i] = tree.predict(X[:, feature_subset])
        return all_predictions.mean(axis=0)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
