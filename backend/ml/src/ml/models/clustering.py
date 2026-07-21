"""
K-Means Clustering, matching file 26 Sections 2-5.

KMeansScratch's algorithm (centroid init, assignment, update, convergence
check) is unchanged from the notes. One hardening: the notes' fit() calls
print() on convergence -- that's fine in a walkthrough, not fine in a
library function a caller might use inside a larger pipeline or a test
suite (uncontrollable stdout noise). Converted to stored attributes
(converged_, n_iterations_) instead -- same information, no side effect
on stdout, and the algorithm's actual behavior (centroids, labels,
inertia) is byte-for-byte identical either way.

select_optimal_k encodes Section 5's explicit argument: inertia CANNOT be
minimized directly to pick K (it's monotonically non-increasing and hits
a trivial, useless zero at K=n_points) -- silhouette score is the more
principled criterion, so that's what this function optimizes, using
inertia only as a secondary, informational column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansScratch:
    """n_init=1 reproduces the notes' exact single-run behavior --
    including its exact known failure mode (see
    test_bad_initialization_can_merge_and_split_blobs_with_n_init_one in
    test_clustering.py): picking K random ACTUAL data points as initial
    centroids can, by bad luck, draw more than one point from the same
    true cluster and none from another, converging to a locally-optimal
    but globally-wrong partition. n_init>1 repeats the SAME single-run
    algorithm from different random starting points and keeps whichever
    run reaches the lowest final inertia -- this is exactly what
    sklearn's own KMeans does by default (n_init=10) to work around this
    same documented limitation. The underlying Lloyd's-algorithm mechanics
    (assignment, centroid update, convergence check) are unchanged either way.
    """

    def __init__(
        self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4,
        random_state: int | None = None, n_init: int = 1,
    ):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.centroids: np.ndarray | None = None
        self.inertia_history: list[float] = []
        self.converged_: bool = False
        self.n_iterations_: int = 0

    def fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        n_samples = X.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError(
                f"n_clusters ({self.n_clusters}) cannot exceed n_samples ({n_samples})."
            )

        # n_init=1 uses self.random_state DIRECTLY as the single run's seed --
        # byte-for-byte identical to the notes' original single-run fit().
        # Only derive multiple distinct seeds when actually running more than
        # one init, since there's no canonical "the" seed to preserve there.
        if self.n_init == 1:
            run_seeds = [self.random_state]
        else:
            base_seed = np.random.RandomState(self.random_state)
            run_seeds = [int(base_seed.randint(0, 2**31 - 1)) for _ in range(self.n_init)]

        best_result = None
        best_inertia = np.inf

        for run_seed in run_seeds:
            result = self._fit_once(X, run_seed)
            final_inertia = result["inertia_history"][-1]
            if final_inertia < best_inertia:
                best_inertia = final_inertia
                best_result = result

        self.centroids = best_result["centroids"]
        self.inertia_history = best_result["inertia_history"]
        self.converged_ = best_result["converged"]
        self.n_iterations_ = best_result["n_iterations"]
        return best_result["labels"]

    def _fit_once(self, X: np.ndarray, seed: int) -> dict:
        """One complete run of Lloyd's algorithm from one random
        initialization -- unchanged from the notes' original fit()."""
        rng = np.random.RandomState(seed)
        n_samples = X.shape[0]

        # Initialize centroids by picking K random ACTUAL data points --
        # simple and common, with a real limitation (sensitivity to a bad
        # draw, see class docstring) that's why sklearn's KMeans defaults
        # to k-means++ init AND n_init=10 rather than a single naive draw.
        random_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_indices].copy()
        inertia_history = []
        converged = False
        n_iterations = 0

        for iteration in range(self.max_iters):
            labels = self._assign_labels_for(X, centroids)

            # Inertia (within-cluster sum of squares) should decrease
            # monotonically -- tested explicitly in test_clustering.py.
            inertia = self._compute_inertia_for(X, labels, centroids)
            inertia_history.append(inertia)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
                for k in range(self.n_clusters)
            ])

            centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)))
            centroids = new_centroids
            n_iterations = iteration + 1

            if centroid_shift < self.tol:
                converged = True
                break

        return {
            "centroids": centroids,
            "labels": labels,
            "inertia_history": inertia_history,
            "converged": converged,
            "n_iterations": n_iterations,
        }

    def _assign_labels_for(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _compute_inertia_for(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign_labels_for(np.asarray(X), self.centroids)


def select_optimal_k(
    X: np.ndarray, k_range: range = range(2, 9), random_state: int = 42, n_init: int = 10
) -> pd.DataFrame:
    """File 26 Section 5's argument as a reusable function: inertia is
    monotonically non-increasing in K and trivially reaches zero at
    K=n_points (every point its own cluster) -- USELESS as a selection
    signal on its own. Silhouette score is the principled criterion;
    inertia is reported alongside for context, not as the selector.

    Returns a DataFrame sorted by silhouette_score, descending -- the
    top row is the recommended K.
    """
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        rows.append({"k": k, "silhouette_score": score, "inertia": km.inertia_})
    return pd.DataFrame(rows).sort_values("silhouette_score", ascending=False).reset_index(drop=True)
