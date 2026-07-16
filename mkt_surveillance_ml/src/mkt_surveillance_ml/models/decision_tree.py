"""
Decision Tree Classifier, matching file 22 Sections 1-5.

DecisionTreeClassifierScratch is the CANONICAL version. File 23 (Random
Forest) redefines an identical copy of this class in the notes -- that's
fine in a standalone walkthrough, not fine in a package, where two
"identical" copies drifting apart over time is a real maintenance risk.
random_forest.py imports THIS module rather than redefining the class.

entropy() and information_gain() are kept as free functions (not methods)
because file 22 demonstrates them standalone first, independent of any
tree -- keeping that separation makes the "IG is fundamentally a
measure over label distributions, computed before any tree exists"
point (Section 3) visible in the code structure, not just in the prose
mentioning it.
"""
from __future__ import annotations

import numpy as np


def entropy(y: np.ndarray) -> float:
    """H(S) = -sum(p_i * log2(p_i)). File 22 Section 1.

    H=0: every example in the set belongs to the same class (pure).
    H=1: exact 50/50 split for binary labels (maximally impure).
    Log base 2 specifically gives this "bits needed to specify the
    class" interpretation -- other log bases still work for COMPARING
    splits but lose that interpretation (Section 2).

    Floating-point note: the +1e-10 inside log2 exists so a class
    probability of exactly 0 never hits log2(0) (domain error / -inf).
    Side effect: for a genuinely pure set, p=1.0 and log2(1.0 + 1e-10)
    is a tiny POSITIVE number, so entropy comes out as a tiny NEGATIVE
    float (~-1e-10) instead of exactly 0.0. That's this exact formula's
    known behavior, not a bug -- worth being able to say precisely if
    asked, rather than being surprised by it.
    """
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))


def information_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """IG = H(parent) - weighted_average(H(left), H(right)). File 22
    Section 3. This is the exact quantity _best_split maximizes,
    exhaustively, at every single node.
    """
    n = len(y)
    n_left, n_right = len(y_left), len(y_right)
    h_parent = entropy(y)
    h_children = (n_left / n) * entropy(y_left) + (n_right / n) * entropy(y_right)
    return h_parent - h_children


class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # only set for leaf nodes


class DecisionTreeClassifierScratch:
    """Greedy, recursive binary-split tree via exhaustive information-gain
    search. File 22 Sections 4-5.

    Known cost, preserved deliberately rather than "optimized away": the
    exhaustive search in _best_split re-evaluates every feature and every
    unique threshold at every node, and information_gain recomputes
    entropy(y) for the parent on every single candidate split rather than
    once per node. This is genuinely why trees get slow on wide,
    high-cardinality data -- reproducing that cost here is more honest
    than silently caching it away, since the cost IS the thing worth
    understanding.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: DecisionTreeNode | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifierScratch":
        X = np.asarray(X)
        y = np.asarray(y)
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria, each for a distinct reason: max_depth bounds
        # complexity (prevents memorizing noise); min_samples_split refuses
        # to trust a "pattern" found in too few examples to be more than
        # chance; n_classes == 1 means the node is already pure and further
        # splitting cannot improve it.
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_classes == 1
        ):
            leaf_value = np.argmax(np.bincount(y.astype(int)))
            return DecisionTreeNode(value=leaf_value)

        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            leaf_value = np.argmax(np.bincount(y.astype(int)))
            return DecisionTreeNode(value=leaf_value)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionTreeNode(best_feature, best_threshold, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray, n_features: int):
        """Exhaustive search over every feature and every candidate
        threshold, evaluating information gain for each. No shortcut --
        this IS the computational core of the whole algorithm.
        """
        best_gain = -1.0
        best_feature = None
        best_threshold = None

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            for threshold in thresholds:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold

                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                gain = information_gain(y, y[left_indices], y[right_indices])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: DecisionTreeNode):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
