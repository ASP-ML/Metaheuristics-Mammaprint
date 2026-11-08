"""
Tests for the top-k filter-ranking baseline (editor comment E.4).

The unit tests replace the objective function, so they run without Weka.
The integration tests need a JVM (set JAVA_HOME) and use data/train_normal_rtf.csv.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from baselines import top_k_filter
from baselines.top_k_filter import TopKFilter, rank_features, top_k


def fake_objective(result):
    """Stand-in for ofunctions.allObjectiveFunctionWeka that always returns `result`."""
    def objective(positions, data, labels, type_of, get_mean=True):
        return result
    return objective


# ── Unit tests (no JVM) ───────────────────────────────────────────────────────

def test_weka_rows_are_ranked_by_score_then_index(monkeypatch):
    # Weka's Ranker returns [column index, score] rows; columns 0 and 2 tie
    weka_rows = np.array([[1, 0.9], [2, 0.5], [0, 0.5], [3, 0.1]])
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective(weka_rows))
    ranking, scores = rank_features(np.zeros((5, 4)), np.zeros(5), 'sim_uncertainty')
    assert ranking.tolist() == [1, 0, 2, 3]
    assert scores.tolist() == [0.5, 0.9, 0.5, 0.1]


def test_mi_scores_in_column_order_are_ranked(monkeypatch):
    # 'mi' returns one score per column, in column order; columns 1 and 2 tie
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective([0.1, 0.3, 0.3, 0.2]))
    ranking, scores = rank_features(np.zeros((5, 4)), np.zeros(5), 'mi')
    assert ranking.tolist() == [1, 2, 3, 0]
    assert scores.tolist() == [0.1, 0.3, 0.3, 0.2]


def test_top_k_returns_the_k_best_columns_in_ascending_order(monkeypatch):
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective([0.1, 0.4, 0.3, 0.2]))
    assert top_k(np.zeros((5, 4)), np.zeros(5), 'mi', k=2) == [1, 2]


def test_failed_scoring_raises(monkeypatch):
    # allObjectiveFunctionWeka returns 0 when the evaluation fails (e.g. the JVM is not running)
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective(0))
    with pytest.raises(RuntimeError):
        rank_features(np.zeros((5, 4)), np.zeros(5), 'chi2')


def test_missing_columns_raise(monkeypatch):
    weka_rows = np.array([[1, 0.9], [0, 0.5]])
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective(weka_rows))
    with pytest.raises(RuntimeError):
        rank_features(np.zeros((5, 4)), np.zeros(5), 'chi2')


def test_unknown_objective_raises():
    with pytest.raises(ValueError):
        rank_features(np.zeros((5, 4)), np.zeros(5), 'variance')


def test_k_larger_than_the_number_of_features_raises(monkeypatch):
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', fake_objective([0.1, 0.4]))
    with pytest.raises(ValueError):
        top_k(np.zeros((5, 2)), np.zeros(5), 'mi', k=3)


def test_selector_scores_only_the_rows_it_is_fitted_on(monkeypatch):
    seen_rows = []
    def column_means(positions, data, labels, type_of, get_mean=True):
        seen_rows.append(data.shape[0])
        return list(data.mean(axis=0))
    monkeypatch.setattr(top_k_filter.ofunctions, 'allObjectiveFunctionWeka', column_means)
    rng = np.random.default_rng(45)
    X = rng.random((40, 6)) + np.array([0, 3, 0, 2, 0, 1])
    y = np.tile([0, 1], 20)

    pipeline = Pipeline([('top', TopKFilter('mi', k=3)), ('clf', LogisticRegression())])
    pipeline.fit(X[:30], y[:30])

    assert seen_rows == [30]
    assert np.flatnonzero(pipeline.named_steps['top'].get_support()).tolist() == [1, 3, 5]
    assert pipeline.named_steps['top'].transform(X[30:]).shape == (10, 3)


# ── Integration tests (Weka JVM) ──────────────────────────────────────────────

SEPARABLE = ['mi', 'sim_uncertainty', 'gain_ratio', 'chi2']


@pytest.fixture(scope='module')
def weka_data():
    jvm = pytest.importorskip('weka.core.jvm')
    try:
        jvm.start(packages=True)
    except Exception as error:
        pytest.skip(f'Weka JVM not available: {error}')
    data = pd.read_csv('data/train_normal_rtf.csv')
    labels = data['label'].to_numpy(dtype=np.intp)
    yield data.drop(columns=['label']).to_numpy(), labels
    jvm.stop()


def merit(subset, X, y, of):
    return top_k_filter.ofunctions.allObjectiveFunctionWeka(list(subset), X, y, type_of=of)


@pytest.mark.parametrize('of', SEPARABLE)
def test_scores_of_separable_objectives_do_not_depend_on_the_other_genes(weka_data, of):
    X, y = weka_data
    _, scores = rank_features(X, y, of)
    subset = [3, 10, 50, 63]
    _, sub_scores = rank_features(X[:, subset], y, of)
    np.testing.assert_allclose(sub_scores, scores[subset], rtol=0, atol=1e-12)


def test_relieff_scores_depend_on_the_other_genes(weka_data):
    X, y = weka_data
    _, scores = rank_features(X, y, 'relieff')
    subset = [3, 10, 50, 63]
    _, sub_scores = rank_features(X[:, subset], y, 'relieff')
    assert not np.allclose(sub_scores, scores[subset], rtol=0, atol=1e-12)


@pytest.mark.parametrize('of', SEPARABLE)
def test_top_10_maximizes_the_mean_score_over_subsets_of_at_least_10_genes(weka_data, of):
    X, y = weka_data
    best = merit(top_k(X, y, of, k=10), X, y, of)
    rng = np.random.default_rng(45)
    for _ in range(30):
        size = rng.integers(10, 21)
        subset = sorted(rng.choice(X.shape[1], size, replace=False))
        assert merit(subset, X, y, of) <= best + 1e-12
