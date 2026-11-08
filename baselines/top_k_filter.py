"""
Top-k Filter-Ranking Baseline
=============================

Deterministic baseline asked for in editor comment E.4: score every gene once with
the same per-feature score Omega_i that the metaheuristics average in Eq. (11), and
keep the k best.

For objectives whose per-feature score does not depend on the other selected genes
(mi, sim_uncertainty, gain_ratio, information_gain, chi2), Eq. (11) is the mean of
fixed scores, so under |S| >= k its maximum is attained by the top-k genes. ReliefF
weights depend on the subset through the nearest-neighbour distances, so for it the
top-k is only a heuristic.

Score: ofunctions.allObjectiveFunctionWeka on all columns, one call per objective.
Ties: broken by column index, lower index first.

Typical Usage:
-------------
    import weka.core.jvm as jvm
    from baselines.top_k_filter import top_k, TopKFilter

    jvm.start(packages=True)
    subset = top_k(X, y, 'sim_uncertainty', k=10)

    # Scores learned on the training part of each fold only
    pipeline = Pipeline([('top', TopKFilter('sim_uncertainty', k=10)), ('clf', clf)])
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

import ofunctions
from metaheuristics.base import MIN_LENGTH_SUBSET

OBJECTIVE_FUNCTIONS = ['relieff', 'mi', 'sim_uncertainty', 'gain_ratio', 'chi2']
"""Objectives of the manuscript, in the order of pipeline.py."""

SEPARABLE_OBJECTIVES = {'mi', 'sim_uncertainty', 'gain_ratio', 'information_gain', 'chi2'}
"""Objectives whose per-feature score depends only on that feature and the labels."""

KNOWN_OBJECTIVES = SEPARABLE_OBJECTIVES | {'relieff'}


def rank_features(X, y, of):
    """
    Score every column of X once and rank the columns, best first.

    Parameters:
    ----------
    X : numpy.ndarray
        Data matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Class labels.
    of : str
        Objective name accepted by ofunctions.allObjectiveFunctionWeka.

    Returns:
    -------
    ranking : numpy.ndarray of int
        Column indices by score descending; ties by column index ascending.
    scores : numpy.ndarray of float
        Score of each column, in column order.

    Notes:
    -----
    Weka objectives need a running JVM (weka.core.jvm.start).
    """
    if of not in KNOWN_OBJECTIVES:
        raise ValueError(f"Unknown objective '{of}'; expected one of {sorted(KNOWN_OBJECTIVES)}")
    n_features = X.shape[1]
    result = ofunctions.allObjectiveFunctionWeka(list(range(n_features)), X, y, type_of=of, get_mean=False)
    scores = _scores_in_column_order(result, n_features, of)
    # lexsort sorts by the last key first: score descending, then index ascending
    ranking = np.lexsort((np.arange(n_features), -scores))
    return ranking, scores


def _scores_in_column_order(result, n_features, of):
    # allObjectiveFunctionWeka returns 0 instead of raising when the evaluation fails
    if np.isscalar(result):
        raise RuntimeError(f"Scoring with '{of}' failed; is the Weka JVM running?")
    result = np.asarray(result, dtype=np.float64)
    if of == 'mi':
        # One score per column, in column order
        scores = result
    else:
        # Weka Ranker: rows of [column index, score], sorted by score
        scores = np.full(n_features, np.nan)
        scores[result[:, 0].astype(int)] = result[:, 1]
    if scores.shape != (n_features,) or np.isnan(scores).any():
        raise RuntimeError(f"Scoring with '{of}' did not return one score per column")
    return scores


def top_k(X, y, of, k=MIN_LENGTH_SUBSET):
    """
    Indices of the k top-ranked columns, in ascending order (as the metaheuristics report subsets).
    """
    if k > X.shape[1]:
        raise ValueError(f'k = {k} is larger than the number of features ({X.shape[1]})')
    ranking, _ = rank_features(X, y, of)
    return sorted(int(i) for i in ranking[:k])


class TopKFilter(SelectorMixin, BaseEstimator):
    """
    Keep the k top-ranked columns, with scores learned in fit only.

    Placed after scaling and SMOTE in a CV pipeline, the ranking is learned from
    the training part of each fold (E.4, variant 4.2b).

    Parameters:
    ----------
    of : str
        Objective name accepted by ofunctions.allObjectiveFunctionWeka.
    k : int
        Number of columns kept.
    """

    def __init__(self, of, k=MIN_LENGTH_SUBSET):
        self.of = of
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        if self.k > X.shape[1]:
            raise ValueError(f'k = {self.k} is larger than the number of features ({X.shape[1]})')
        self.n_features_in_ = X.shape[1]
        self.ranking_, self.scores_ = rank_features(X, np.asarray(y), self.of)
        return self

    def _get_support_mask(self):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.ranking_[:self.k]] = True
        return mask
