"""
Anomaly Detection, matching file 27 Sections 1-7 and 11.

IsolationForestScratch wraps the notes' raw functions (IsolationTreeNode,
build_isolation_tree, path_length, anomaly_score_formula) into a proper
fit/score_samples/predict class -- the notes demonstrate the mechanism
via a script and a loop, not as something with an interface a caller
could plug into a pipeline. Wrapping changes nothing about how a tree is
built, how a path length is measured, or how the final score is computed.

contamination_sensitivity_sweep and per_pattern_detection_degradation turn
file 27's most important arguments (Section 3.1: contamination is a
THRESHOLD choice, not something the algorithm discovers; Section 11: one
global contamination value under-detects rarer patterns even when set to
the exact true combined rate) into reusable, testable functions -- this
second point is the direct empirical basis for detection/multi_pattern.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class IsolationTreeNode:
    def __init__(
        self, split_feature=None, split_value=None, left=None, right=None,
        size=None, depth=None,
    ):
        self.split_feature = split_feature
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size  # number of points that reached this node (only set at leaves/early stops)
        self.depth = depth


def build_isolation_tree(
    X: np.ndarray, current_depth: int, max_depth: int, rng: np.random.RandomState
) -> IsolationTreeNode:
    """Random split on a random feature at a random value within that
    feature's observed range. File 27 Section 1 -- the entire mechanism
    anomaly detection here rests on: genuine outliers get isolated
    (reach a leaf) in FEWER random splits on average, since they don't
    need a lucky, narrowly-targeted split to end up alone.
    """
    n_samples = X.shape[0]

    if current_depth >= max_depth or n_samples <= 1:
        return IsolationTreeNode(size=n_samples, depth=current_depth)

    n_features = X.shape[1]
    split_feature = rng.randint(0, n_features)
    feature_values = X[:, split_feature]
    min_val, max_val = feature_values.min(), feature_values.max()

    if min_val == max_val:
        return IsolationTreeNode(size=n_samples, depth=current_depth)

    # THIS is the entire mechanism: a RANDOM split value, not an
    # information-gain-optimized one like file 22's decision trees.
    split_value = rng.uniform(min_val, max_val)

    left_mask = feature_values < split_value
    right_mask = ~left_mask

    left = build_isolation_tree(X[left_mask], current_depth + 1, max_depth, rng)
    right = build_isolation_tree(X[right_mask], current_depth + 1, max_depth, rng)

    return IsolationTreeNode(split_feature, split_value, left, right, depth=current_depth)


def _average_path_length_of_unsuccessful_search(n: int) -> float:
    """c(n): the expected path length of an UNSUCCESSFUL search in a
    Binary Search Tree of n points -- a normalization constant so scores
    are comparable across different dataset sizes. Applied EXACTLY ONCE,
    inside anomaly_score_formula, over the size of the set being scored --
    NOT per-leaf inside path_length (an early draft of this module
    mistakenly applied it a second time there; path_length below is the
    simple, depth-only version that matches file 27 exactly).
    """
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)


def path_length(point: np.ndarray, node: IsolationTreeNode) -> float:
    """How many splits until this point ends up isolated (alone, or at a
    depth/size-limited leaf). File 27 Section 1 -- deliberately just
    node.depth at a leaf, no per-leaf normalization. Genuine outliers:
    short paths, on average, across many randomly-built trees. Normal,
    densely-clustered points: long paths, since many splits are needed
    before a lucky one separates them from their neighbors.
    """
    if node.split_feature is None:  # reached a leaf
        return node.depth
    if point[node.split_feature] < node.split_value:
        return path_length(point, node.left)
    return path_length(point, node.right)


def anomaly_score_formula(avg_path_length: float, n_samples: int) -> float:
    """s(x,n) = 2^(-avg_path_length / c(n)). File 27 Section 2. The ONLY
    place c(n) normalization is applied in this module.

    s -> 1: avg_path_length << c(n), meaning this point isolates far
        faster than typical -- strong anomaly signal.
    s -> 0.5: avg_path_length ~ c(n), meaning this point isolates about
        as fast as a random point would -- ambiguous / normal.
    s -> 0 (well below 0.5): avg_path_length >> c(n) -- point needs MORE
        splits than typical to isolate, i.e. it's especially well
        clustered, not anomalous at all.
    """
    if n_samples <= 1:
        return 0.0
    c_n = _average_path_length_of_unsuccessful_search(n_samples)
    return 2 ** (-avg_path_length / c_n)


class IsolationForestScratch:
    """Ensemble of random isolation trees, scored via anomaly_score_formula.
    File 27 Sections 1-2 assembled into a reusable fit/score_samples/predict
    interface (the notes demonstrate this as a script over one global
    X_anomaly array, not as something with a class boundary).

    Unlike sklearn's convention (LOWER score = more anomalous), this
    returns scores where HIGHER = more anomalous, matching the formula
    exactly as derived in Section 2 (s -> 1 means anomalous). Documented
    explicitly here since mixing the two conventions silently is a very
    easy way to invert every downstream threshold decision without
    noticing -- see test_isolation_forest.py's explicit convention test.
    """

    def __init__(
        self, n_estimators: int = 100, max_samples: int = 256,
        contamination: float = 0.05, random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees: list[IsolationTreeNode] = []
        self.max_depth_: int | None = None
        self.threshold_: float | None = None
        self.n_samples_fit_: int | None = None

    def fit(self, X: np.ndarray) -> "IsolationForestScratch":
        X = np.asarray(X)
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        subsample_size = min(self.max_samples, n_samples)
        self.n_samples_fit_ = subsample_size
        # ceil(log2(subsample_size)): the expected depth for isolating a
        # POINT SAMPLED AT RANDOM in a tree over subsample_size points --
        # trees deeper than this mostly measure noise, not signal.
        self.max_depth_ = int(np.ceil(np.log2(max(subsample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, subsample_size, replace=False)
            tree = build_isolation_tree(X[sample_idx], 0, self.max_depth_, rng)
            self.trees.append(tree)

        scores = self.score_samples(X)
        self.threshold_ = float(np.quantile(scores, 1 - self.contamination))
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous. See class docstring for why this is
        the OPPOSITE convention from sklearn.IsolationForest.score_samples.

        c(n) normalization uses the number of samples the trees were
        actually built from (`self.n_samples_fit_`, capped at `max_samples`),
        not the size of the array being scored. This is intentional so that
        scoring one point at a time (the live-inference case) and scoring a
        full batch (the offline-evaluation case) produce comparable,
        consistent scores for the same underlying data."""
        if not self.trees or getattr(self, 'n_samples_fit_', None) is None:
            raise RuntimeError("Call fit() before score_samples().")
        X = np.asarray(X)
        n_samples_for_normalization = self.n_samples_fit_
        scores = np.zeros(len(X))
        for i, point in enumerate(X):
            avg_path = np.mean([path_length(point, tree) for tree in self.trees])
            scores[i] = anomaly_score_formula(avg_path, n_samples_for_normalization)
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal -- using the threshold fixed at fit()
        time from the contamination parameter. contamination sets WHERE
        this threshold sits; it does not change any score above."""
        if self.threshold_ is None:
            raise RuntimeError("Call fit() before predict().")
        scores = self.score_samples(X)
        return (scores >= self.threshold_).astype(int)


def contamination_sensitivity_sweep(
    X: np.ndarray, contamination_values: list[float] | None = None, random_state: int = 42
) -> pd.DataFrame:
    """File 27 Section 3.1 / Block 5's argument as a function: n_flagged
    scales almost exactly with contamination, which is direct evidence
    that contamination is a THRESHOLD CHOICE on an already-computed score
    distribution, not something the algorithm verifies against the data.
    Setting contamination=0.1 does not mean the model found evidence that
    10% of points are anomalous -- it means it was told to flag whichever
    10% score worst, whether or not a natural break in the score
    distribution falls anywhere near that cutoff.
    """
    if contamination_values is None:
        contamination_values = [0.02, 0.05, 0.1, 0.15, 0.25]
    rows = []
    for c in contamination_values:
        model = IsolationForest(contamination=c, random_state=random_state, n_estimators=100)
        preds = model.fit_predict(X)
        n_flagged = int((preds == -1).sum())
        rows.append({
            "contamination": c,
            "n_flagged": n_flagged,
            "pct_flagged": n_flagged / len(X) * 100,
        })
    return pd.DataFrame(rows)


def compare_isolation_forest_and_lof(
    X: np.ndarray, contamination: float = 0.06, n_neighbors: int = 20, random_state: int = 42,
) -> dict:
    """File 27 Section 9-10's ensemble-agreement comparison, as a
    reusable function.

    Isolation Forest measures GLOBAL isolation (far from the entire data
    cloud). LOF measures LOCAL density (sparser than one's own immediate
    neighborhood, even if not far from the cloud overall). They can and
    do disagree -- a point flagged by IF alone is typically far from
    everything globally but not meaningfully sparser than its own
    neighborhood; a point flagged by LOF alone typically sits in a
    locally sparse pocket without being dramatically far from the
    dataset as a whole. Both disagreement categories are informative,
    not noise to be argued away.
    """
    iso = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=100)
    iso_preds = iso.fit_predict(X)
    iso_is_anomaly = (iso_preds == -1).astype(int)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    lof_preds = lof.fit_predict(X)
    lof_is_anomaly = (lof_preds == -1).astype(int)

    both = int(((iso_is_anomaly == 1) & (lof_is_anomaly == 1)).sum())
    iso_only = int(((iso_is_anomaly == 1) & (lof_is_anomaly == 0)).sum())
    lof_only = int(((iso_is_anomaly == 0) & (lof_is_anomaly == 1)).sum())

    return {
        "iso_is_anomaly": iso_is_anomaly,
        "lof_is_anomaly": lof_is_anomaly,
        "both_agree": both,
        "isolation_forest_only": iso_only,
        "lof_only": lof_only,
        "agreement_table": pd.crosstab(
            iso_is_anomaly, lof_is_anomaly,
            rownames=["Isolation Forest"], colnames=["LOF"],
        ),
    }


def per_pattern_detection_rate_under_single_contamination(
    X: np.ndarray, pattern_labels: np.ndarray, random_state: int = 42,
) -> pd.DataFrame:
    """File 27 Section 11's specific experiment as a reusable function:
    fit ONE Isolation Forest with contamination set to the TRUE combined
    anomaly rate (as favorable a setting as a single global threshold can
    get), then measure detection rate separately for each labeled
    pattern. Rarer patterns get under-detected relative to more common
    ones even under this best-case calibration -- the direct empirical
    argument for per-pattern models in detection/multi_pattern.py, not
    an assumption made without evidence.

    pattern_labels: array of strings, e.g. 'normal', 'pump', 'wash' --
    'normal' is treated as the non-anomalous baseline; every other label
    is a distinct pattern to report a separate detection rate for.
    """
    is_anomalous = pattern_labels != "normal"
    true_combined_rate = is_anomalous.mean()

    model = IsolationForest(contamination=true_combined_rate, random_state=random_state, n_estimators=100)
    preds = model.fit_predict(X)
    flagged = preds == -1

    rows = []
    for pattern in sorted(set(pattern_labels)):
        if pattern == "normal":
            continue
        pattern_mask = pattern_labels == pattern
        n_total = int(pattern_mask.sum())
        n_flagged = int((pattern_mask & flagged).sum())
        rows.append({
            "pattern": pattern,
            "n_total": n_total,
            "n_flagged": n_flagged,
            "detection_rate": n_flagged / n_total if n_total > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("n_total").reset_index(drop=True)
