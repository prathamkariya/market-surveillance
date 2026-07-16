import numpy as np
import pytest

from mkt_surveillance_ml.models.clustering import KMeansScratch, select_optimal_k


def make_three_blobs(random_state=0, n_per_blob=60):
    rng = np.random.RandomState(random_state)
    blob_1 = rng.normal([0, 0], 0.3, (n_per_blob, 2))
    blob_2 = rng.normal([8, 8], 0.3, (n_per_blob, 2))
    blob_3 = rng.normal([0, 8], 0.3, (n_per_blob, 2))
    X = np.vstack([blob_1, blob_2, blob_3])
    true_labels = np.array([0] * n_per_blob + [1] * n_per_blob + [2] * n_per_blob)
    return X, true_labels


class TestKMeansScratch:
    def test_inertia_decreases_monotonically(self):
        """File 26 Section 3.1's claim: inertia should never increase
        between iterations of Lloyd's algorithm."""
        X, _ = make_three_blobs()
        km = KMeansScratch(n_clusters=3, max_iters=50, random_state=1)
        km.fit(X)
        diffs = np.diff(km.inertia_history)
        assert (diffs <= 1e-6).all()

    def test_bad_initialization_can_merge_and_split_blobs_with_n_init_one(self):
        """Documents a real, discovered-not-assumed failure: with
        n_init=1 and random_state=1, the initial draw of 3 "random actual
        data points" happens to grab two points from the same blob and
        none from another. Result: two of three final centroids both land
        inside one true blob (splitting it), while the third sits almost
        exactly between the other two true blobs (merging them). This is
        exactly the limitation the class docstring names -- confirmed
        here by inspecting the actual centroids, not asserted from theory.
        """
        X, true_labels = make_three_blobs()
        km = KMeansScratch(n_clusters=3, random_state=1, n_init=1)
        pred_labels = km.fit(X)

        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, pred_labels)
        assert ari < 0.7  # confirms the bad-initialization failure actually occurs

    def test_n_init_greater_than_one_recovers_well_separated_clusters(self):
        """The fix: repeating from several random starts and keeping the
        lowest-inertia run should reliably recover the true partition,
        even using the SAME random_state that fails at n_init=1 above."""
        X, true_labels = make_three_blobs()
        km = KMeansScratch(n_clusters=3, random_state=1, n_init=10)
        pred_labels = km.fit(X)

        # cluster ID assignment is arbitrary (label 0 from KMeans might
        # correspond to true label 2) -- check the PARTITION matches, not
        # the exact label values, via adjusted rand index
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(true_labels, pred_labels)
        assert ari > 0.95

    def test_more_inits_never_produces_worse_final_inertia(self):
        """n_init selects by lowest inertia across runs, so more runs
        should never make the chosen result worse than fewer runs would
        have found (same candidate pool, superset of it)."""
        X, _ = make_three_blobs()
        km_few = KMeansScratch(n_clusters=3, random_state=3, n_init=1)
        km_few.fit(X)
        km_many = KMeansScratch(n_clusters=3, random_state=3, n_init=15)
        km_many.fit(X)
        assert km_many.inertia_history[-1] <= km_few.inertia_history[-1] + 1e-9

    def test_converged_flag_set_when_centroids_stabilize(self):
        X, _ = make_three_blobs()
        km = KMeansScratch(n_clusters=3, max_iters=100, tol=1e-3, random_state=1)
        km.fit(X)
        assert km.converged_ is True
        assert km.n_iterations_ < 100

    def test_no_stdout_side_effect_on_convergence(self, capsys):
        """The hardening this module makes over the original notes:
        fit() should not print anything -- convergence info is exposed
        as attributes instead."""
        X, _ = make_three_blobs()
        km = KMeansScratch(n_clusters=3, random_state=1)
        km.fit(X)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_raises_when_n_clusters_exceeds_n_samples(self):
        X = np.array([[1.0, 1.0], [2.0, 2.0]])
        km = KMeansScratch(n_clusters=5)
        with pytest.raises(ValueError, match="cannot exceed"):
            km.fit(X)

    def test_predict_before_fit_raises(self):
        km = KMeansScratch(n_clusters=3)
        with pytest.raises(RuntimeError):
            km.predict(np.array([[1.0, 1.0]]))

    def test_predict_assigns_new_points_to_nearest_centroid(self):
        X, _ = make_three_blobs()
        km = KMeansScratch(n_clusters=3, random_state=1)
        km.fit(X)
        new_point_near_blob_1 = np.array([[0.1, 0.1]])
        predicted_cluster = km.predict(new_point_near_blob_1)[0]
        # whichever cluster ID got assigned to points near [0,0] during fit
        # should also be assigned to this new nearby point
        labels_at_fit = km.predict(X)
        blob_1_cluster_id = labels_at_fit[0]  # first 60 points are blob 1
        assert predicted_cluster == blob_1_cluster_id

    def test_deterministic_given_same_random_state(self):
        """fit() returns the label array directly (not self) -- compare
        the two returned arrays."""
        X, _ = make_three_blobs()
        labels_1 = KMeansScratch(n_clusters=3, random_state=7).fit(X)
        labels_2 = KMeansScratch(n_clusters=3, random_state=7).fit(X)
        assert np.array_equal(labels_1, labels_2)


class TestSelectOptimalK:
    def test_recommends_true_k_for_well_separated_blobs(self):
        X, _ = make_three_blobs()
        result = select_optimal_k(X, k_range=range(2, 7))
        assert result.iloc[0]["k"] == 3

    def test_returns_sorted_by_silhouette_descending(self):
        X, _ = make_three_blobs()
        result = select_optimal_k(X, k_range=range(2, 6))
        scores = result["silhouette_score"].values
        assert (scores[:-1] >= scores[1:]).all()

    def test_includes_inertia_column_for_context(self):
        X, _ = make_three_blobs()
        result = select_optimal_k(X, k_range=range(2, 5))
        assert "inertia" in result.columns

    def test_inertia_is_monotonically_non_increasing_across_k(self):
        """The exact reason inertia alone can't select K (Section 5):
        it never goes up as K increases."""
        X, _ = make_three_blobs()
        result = select_optimal_k(X, k_range=range(2, 8))
        by_k = result.sort_values("k")
        inertias = by_k["inertia"].values
        assert (np.diff(inertias) <= 1e-6).all()
