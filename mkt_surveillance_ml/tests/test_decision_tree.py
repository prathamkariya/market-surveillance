import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from mkt_surveillance_ml.models.decision_tree import (
    entropy,
    information_gain,
    DecisionTreeNode,
    DecisionTreeClassifierScratch,
)


class TestEntropy:
    def test_pure_set_has_zero_entropy(self):
        y = np.array([0, 0, 0, 0, 0])
        assert entropy(y) == pytest.approx(0.0, abs=1e-6)

    def test_balanced_binary_set_has_entropy_one(self):
        y = np.array([0, 1])
        assert entropy(y) == pytest.approx(1.0, abs=1e-4)

    def test_entropy_increases_toward_balance(self):
        pure = entropy(np.array([0] * 10))
        skewed = entropy(np.array([0] * 9 + [1]))
        balanced = entropy(np.array([0] * 5 + [1] * 5))
        assert pure < skewed < balanced

    def test_entropy_is_never_negative(self):
        # tolerance accounts for the documented +1e-10-inside-log2 floating
        # point quirk: a genuinely pure set can compute to a hair below zero
        for y in [np.array([0, 0, 1]), np.array([0, 1, 1, 1, 1]), np.array([0])]:
            assert entropy(y) >= -1e-9


class TestInformationGain:
    def test_perfect_split_has_higher_gain_than_useless_split(self):
        y_parent = np.array([0] * 20 + [1] * 20)
        y_left_good = np.array([0] * 20)
        y_right_good = np.array([1] * 20)
        ig_good = information_gain(y_parent, y_left_good, y_right_good)

        y_left_useless = np.array([0] * 10 + [1] * 10)
        y_right_useless = np.array([0] * 10 + [1] * 10)
        ig_useless = information_gain(y_parent, y_left_useless, y_right_useless)

        assert ig_good > ig_useless

    def test_perfect_split_gain_equals_parent_entropy(self):
        """A perfect split reduces child entropy to exactly 0, so gain
        should equal the parent's own entropy."""
        y_parent = np.array([0] * 20 + [1] * 20)
        y_left = np.array([0] * 20)
        y_right = np.array([1] * 20)
        ig = information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(entropy(y_parent), abs=1e-6)

    def test_useless_split_has_near_zero_gain(self):
        y_parent = np.array([0] * 20 + [1] * 20)
        y_left = np.array([0] * 10 + [1] * 10)
        y_right = np.array([0] * 10 + [1] * 10)
        ig = information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(0.0, abs=1e-6)

    def test_information_gain_never_negative_for_any_valid_split(self):
        """A property of entropy (concavity): splitting can never increase
        weighted average impurity beyond the parent's."""
        rng = np.random.RandomState(0)
        y = rng.randint(0, 2, 50)
        for _ in range(20):
            split_point = rng.randint(1, 49)
            perm = rng.permutation(50)
            y_shuffled = y[perm]
            ig = information_gain(y_shuffled, y_shuffled[:split_point], y_shuffled[split_point:])
            assert ig >= -1e-9


class TestDecisionTreeClassifierScratch:
    def test_perfectly_separable_data_achieves_perfect_accuracy(self):
        rng = np.random.RandomState(1)
        n = 200
        X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)])
        y = (X[:, 0] > 0).astype(int)  # trivially splittable on feature 0
        tree = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=2).fit(X, y)
        preds = tree.predict(X)
        assert accuracy_score(y, preds) == 1.0

    def test_max_depth_limits_tree_growth(self):
        """A depth-1 tree should NOT perfectly fit data that genuinely
        needs more than one split to separate."""
        rng = np.random.RandomState(2)
        n = 400
        X = rng.uniform(-2, 2, (n, 2))
        # XOR-like pattern: needs at least 2 splits to separate correctly
        y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
        shallow_tree = DecisionTreeClassifierScratch(max_depth=1, min_samples_split=2).fit(X, y)
        deep_tree = DecisionTreeClassifierScratch(max_depth=5, min_samples_split=2).fit(X, y)

        shallow_acc = accuracy_score(y, shallow_tree.predict(X))
        deep_acc = accuracy_score(y, deep_tree.predict(X))
        assert deep_acc > shallow_acc

    def test_min_samples_split_prevents_splitting_tiny_nodes(self):
        rng = np.random.RandomState(3)
        X = rng.uniform(-1, 1, (20, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifierScratch(max_depth=10, min_samples_split=100)
        tree.fit(X, y)
        # min_samples_split of 100 on 20 samples should force an immediate leaf
        assert tree.root.value is not None

    def test_pure_node_stops_splitting_even_with_depth_remaining(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1, 1, 1, 1])  # already pure
        tree = DecisionTreeClassifierScratch(max_depth=10, min_samples_split=1).fit(X, y)
        assert tree.root.value == 1
        assert tree.root.left is None and tree.root.right is None

    def test_predict_before_fit_raises(self):
        tree = DecisionTreeClassifierScratch()
        with pytest.raises(RuntimeError):
            tree.predict(np.array([[1.0, 2.0]]))

    def test_generalizes_reasonably_to_held_out_data(self):
        rng = np.random.RandomState(4)
        n = 600
        X = rng.uniform(-2, 2, (n, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_train, X_test = X[:450], X[450:]
        y_train, y_test = y[:450], y[450:]

        tree = DecisionTreeClassifierScratch(max_depth=5, min_samples_split=10).fit(X_train, y_train)
        test_acc = accuracy_score(y_test, tree.predict(X_test))
        assert test_acc > 0.8

    def test_deeper_tree_without_min_samples_guard_overfits_more_than_shallow(self):
        """A concrete demonstration of overfitting: an unconstrained deep
        tree should fit noisy training data better than it generalizes,
        more so than a shallow, properly regularized tree."""
        rng = np.random.RandomState(5)
        n = 300
        X = rng.uniform(-2, 2, (n, 2))
        y = (X[:, 0] + X[:, 1] + rng.normal(0, 1.5, n) > 0).astype(int)  # noisy
        X_train, X_test = X[:200], X[200:]
        y_train, y_test = y[:200], y[200:]

        overfit_tree = DecisionTreeClassifierScratch(max_depth=20, min_samples_split=2).fit(X_train, y_train)
        regularized_tree = DecisionTreeClassifierScratch(max_depth=3, min_samples_split=20).fit(X_train, y_train)

        overfit_train_acc = accuracy_score(y_train, overfit_tree.predict(X_train))
        overfit_test_acc = accuracy_score(y_test, overfit_tree.predict(X_test))
        regularized_train_acc = accuracy_score(y_train, regularized_tree.predict(X_train))
        regularized_test_acc = accuracy_score(y_test, regularized_tree.predict(X_test))

        overfit_gap = overfit_train_acc - overfit_test_acc
        regularized_gap = regularized_train_acc - regularized_test_acc
        assert overfit_gap > regularized_gap
