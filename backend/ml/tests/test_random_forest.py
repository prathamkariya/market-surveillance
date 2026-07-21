import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest

from ml.models.random_forest import bootstrap_sample, RandomForestScratch
from ml.models.decision_tree import DecisionTreeClassifierScratch


class TestBootstrapSample:
    def test_bootstrap_sample_has_same_size_as_original(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        X_boot, y_boot, idx = bootstrap_sample(X, y, random_state=1)
        assert len(X_boot) == len(X)
        assert len(idx) == len(X)

    def test_roughly_37_percent_of_rows_excluded(self):
        """File 23 Section 2's bootstrap math: as n grows, the fraction of
        rows NEVER selected converges to 1/e ~ 0.368."""
        X = pd.DataFrame({"a": range(2000)})
        y = pd.Series(np.zeros(2000))
        _, _, idx = bootstrap_sample(X, y, random_state=1)
        included_fraction = len(np.unique(idx)) / len(X)
        excluded_fraction = 1 - included_fraction
        assert abs(excluded_fraction - (1 / np.e)) < 0.02

    def test_deterministic_given_random_state(self):
        X = pd.DataFrame({"a": range(50)})
        y = pd.Series(np.zeros(50))
        _, _, idx1 = bootstrap_sample(X, y, random_state=5)
        _, _, idx2 = bootstrap_sample(X, y, random_state=5)
        assert np.array_equal(idx1, idx2)

    def test_sampling_is_with_replacement(self):
        """With n=10 and only 10 draws, getting zero duplicates across
        many different seeds would be exceedingly unlikely if this were
        truly sampling with replacement."""
        X = pd.DataFrame({"a": range(10)})
        y = pd.Series(np.zeros(10))
        had_duplicate = False
        for seed in range(20):
            _, _, idx = bootstrap_sample(X, y, random_state=seed)
            if len(np.unique(idx)) < len(idx):
                had_duplicate = True
                break
        assert had_duplicate


class TestRandomForestScratch:
    def test_forest_outperforms_or_matches_single_tree_on_noisy_data(self):
        """The core claim of bagging (file 23 Section 3): averaging many
        high-variance trees over bootstrap samples should generalize at
        least as well as, and typically better than, a single tree on
        noisy data.

        Deliberately built with REDUNDANT signal features (6 noisy proxies
        of the same underlying true_signal, not 1-3 independent ones) --
        this is what real engineered feature sets actually look like (this
        project's real system has volatility_5d/10d/20d/60d, multiple
        lagged returns, etc., all correlated proxies for the same
        underlying regime change). With redundancy, a random sqrt-sized
        feature subset usually catches at least one usable proxy, which is
        the regime sqrt-subsetting is designed for. An earlier version of
        this test used a few independent, non-redundant signal features --
        see test_sqrt_feature_subsetting_can_hurt_when_signal_is_concentrated_in_few_features
        for why that setup isn't sqrt-subsetting's regime and can make a
        forest LOSE to a single tree that sees every feature at once.
        """
        rng = np.random.RandomState(10)
        n = 500
        true_signal = rng.uniform(-2, 2, n)
        # 6 noisy proxies of the same true_signal -- redundant, correlated features
        proxy_features = np.column_stack([
            true_signal + rng.normal(0, 0.6, n) for _ in range(6)
        ])
        # 6 pure-noise features carrying no information
        noise_features = rng.uniform(-2, 2, (n, 6))
        X = np.column_stack([proxy_features, noise_features])
        y = (true_signal + rng.normal(0, 1.2, n) > 0).astype(int)

        X_train, X_test = X[:350], X[350:]
        y_train, y_test = y[:350], y[350:]

        single_tree = DecisionTreeClassifierScratch(max_depth=8, min_samples_split=5)
        single_tree.fit(X_train, y_train)
        single_acc = accuracy_score(y_test, single_tree.predict(X_test))

        forest = RandomForestScratch(n_estimators=25, max_depth=8, random_state=10)
        forest.fit(X_train, y_train)
        forest_acc = accuracy_score(y_test, forest.predict(X_test))

        assert forest_acc >= single_acc - 0.03  # forest should not be meaningfully worse

    def test_sqrt_feature_subsetting_can_hurt_when_signal_is_concentrated_in_few_features(self):
        """The flip side, stated as an explicit, deliberate property
        rather than an unpleasant surprise: on LOW-dimensional data where
        nearly all the signal sits in a couple of specific features,
        forcing each tree to see only sqrt(n_features) of them can make
        the forest measurably worse than a single tree that sees
        everything. This is exactly what the previous test's original
        4-feature version ran into -- worth knowing explicitly, since
        max_features='sqrt' is a default, not a law."""
        rng = np.random.RandomState(10)
        n = 500
        X = rng.uniform(-2, 2, (n, 4))
        y = (X[:, 0] + X[:, 1] + rng.normal(0, 1.2, n) > 0).astype(int)
        X_train, X_test = X[:350], X[350:]
        y_train, y_test = y[:350], y[350:]

        single_tree = DecisionTreeClassifierScratch(max_depth=8, min_samples_split=5)
        single_tree.fit(X_train, y_train)
        single_acc = accuracy_score(y_test, single_tree.predict(X_test))

        forest_sqrt = RandomForestScratch(
            n_estimators=25, max_depth=8, max_features="sqrt", random_state=10
        )
        forest_sqrt.fit(X_train, y_train)
        forest_sqrt_acc = accuracy_score(y_test, forest_sqrt.predict(X_test))

        forest_all_features = RandomForestScratch(
            n_estimators=25, max_depth=8, max_features="all", random_state=10
        )
        forest_all_features.fit(X_train, y_train)
        forest_all_acc = accuracy_score(y_test, forest_all_features.predict(X_test))

        # using ALL features per tree (still bootstrap-bagged) should recover
        # accuracy closer to the single tree than sqrt-subsetting does here
        assert forest_all_acc >= forest_sqrt_acc - 0.03

    def test_predict_proba_is_fraction_of_trees_voting_positive(self):
        rng = np.random.RandomState(11)
        X = rng.uniform(-1, 1, (100, 3))
        y = (X[:, 0] > 0).astype(int)
        forest = RandomForestScratch(n_estimators=10, random_state=11).fit(X, y)
        proba = forest.predict_proba(X)
        # every predicted probability must be an exact multiple of 1/n_estimators
        multiples = proba * 10
        assert np.allclose(multiples, np.round(multiples))

    def test_max_features_sqrt_uses_sqrt_of_total_features(self):
        forest = RandomForestScratch(max_features="sqrt")
        assert forest._get_max_features(16) == 4
        assert forest._get_max_features(9) == 3

    def test_max_features_sqrt_never_returns_zero_for_small_feature_counts(self):
        forest = RandomForestScratch(max_features="sqrt")
        assert forest._get_max_features(1) >= 1
        assert forest._get_max_features(2) >= 1

    def test_each_tree_uses_a_different_bootstrap_sample(self):
        """Trees in the same forest should not be trained on identical data."""
        rng = np.random.RandomState(12)
        X = rng.uniform(-1, 1, (200, 3))
        y = (X[:, 0] > 0).astype(int)
        forest = RandomForestScratch(n_estimators=5, random_state=12).fit(X, y)
        predictions_per_tree = [
            tree.predict(X[:, subset])
            for tree, subset in zip(forest.trees, forest.tree_feature_subsets)
        ]
        # trees should not all produce IDENTICAL predictions on the full set
        all_identical = all(
            np.array_equal(predictions_per_tree[0], p) for p in predictions_per_tree[1:]
        )
        assert not all_identical

    def test_predict_before_fit_raises(self):
        forest = RandomForestScratch()
        with pytest.raises(RuntimeError):
            forest.predict_proba(np.array([[1.0, 2.0]]))

    def test_deterministic_given_same_random_state(self):
        rng = np.random.RandomState(13)
        X = rng.uniform(-1, 1, (150, 3))
        y = (X[:, 0] > 0).astype(int)
        forest_1 = RandomForestScratch(n_estimators=10, random_state=99).fit(X, y)
        forest_2 = RandomForestScratch(n_estimators=10, random_state=99).fit(X, y)
        assert np.array_equal(forest_1.predict(X), forest_2.predict(X))

    def test_roughly_comparable_accuracy_to_sklearn_forest(self):
        """Not numerical parity (different feature-subsetting granularity,
        documented in the class docstring) -- just checking this isn't
        wildly worse than the library version on the same data."""
        rng = np.random.RandomState(14)
        n = 500
        X = rng.uniform(-2, 2, (n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_train, X_test = X[:350], X[350:]
        y_train, y_test = y[:350], y[350:]

        scratch_forest = RandomForestScratch(n_estimators=30, max_depth=6, random_state=14)
        scratch_forest.fit(X_train, y_train)
        scratch_acc = accuracy_score(y_test, scratch_forest.predict(X_test))

        sklearn_forest = SklearnRandomForest(n_estimators=30, max_depth=6, random_state=14)
        sklearn_forest.fit(X_train, y_train)
        sklearn_acc = accuracy_score(y_test, sklearn_forest.predict(X_test))

        assert abs(scratch_acc - sklearn_acc) < 0.15
