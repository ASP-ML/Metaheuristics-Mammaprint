"""
Tests for top10_baseline.py (editor comment E.4).

The objective function is replaced by the mean of fixed per-column scores, so no JVM is needed.
"""

import numpy as np
import pandas as pd
import pytest

import top10_baseline
from top10_baseline import compare, top_k_subsets

SCORES = np.arange(20, dtype=float)
"""Column j scores j: the top-10 subset is columns 10-19, with merit 14.5."""


@pytest.fixture
def separable_objective(monkeypatch):
    def mean_score(positions, data, labels, type_of, get_mean=True):
        return float(np.mean(SCORES[list(positions)]))
    monkeypatch.setattr(top10_baseline.ofunctions, 'allObjectiveFunctionWeka', mean_score)


def resume_row(meta, of, subset):
    return {'Meta': meta, 'OF': of, 'IT': 50, 'Subset': str(subset), 'Length': len(subset), 'Settings': ''}


def top_row(of):
    return {**resume_row('TOP10', of, list(range(10, 20))), 'Merit': 14.5}


def test_top_k_subsets_follow_the_resume_layout(monkeypatch):
    monkeypatch.setattr(top10_baseline, 'top_k', lambda data, labels, of, k: list(range(k)))
    monkeypatch.setattr(top10_baseline.ofunctions, 'allObjectiveFunctionWeka', lambda *args, **kwargs: 0.5)
    rows = top_k_subsets(np.zeros((5, 20)), np.zeros(5), ['mi'], 10)
    # pipeline_classifiers.py reads Meta, OF, IT, Subset and Settings by position
    assert list(rows.columns[:6]) == ['Meta', 'OF', 'IT', 'Subset', 'Length', 'Settings']
    assert rows.iloc[0].tolist()[:5] == ['TOP10', 'mi', 0, str(list(range(10))), 10]


def test_compare_flags_metaheuristics_below_the_top_10_optimum(separable_objective):
    top = pd.DataFrame([top_row('mi')])
    mh = pd.DataFrame([resume_row('DE', 'mi', list(range(10, 20))), resume_row('GA', 'mi', list(range(5, 15)))])
    rows = compare(np.zeros((5, 20)), np.zeros(5), top, mh, 10).set_index('Meta')

    assert rows.loc['DE', 'MH_optimal'] and rows.loc['DE', 'Top_ge_MH']
    assert rows.loc['DE', 'Jaccard'] == 1.0
    assert not rows.loc['GA', 'MH_optimal'] and rows.loc['GA', 'Top_ge_MH']
    assert rows.loc['GA', 'Intersection'] == 5
    assert rows.loc['GA', 'Jaccard'] == pytest.approx(5 / 15)
    assert rows.loc['GA', 'Merit_gap'] == pytest.approx(5.0)


def test_compare_does_not_judge_subsets_below_the_minimum_length(separable_objective):
    # A single top gene beats the top-10 mean, but it is outside the |S| >= 10 constraint
    top = pd.DataFrame([top_row('mi')])
    mh = pd.DataFrame([resume_row('DE', 'mi', [19])])
    row = compare(np.zeros((5, 20)), np.zeros(5), top, mh, 10).iloc[0]
    assert row['Merit_gap'] < 0
    assert row['Top_ge_MH'] is None and row['MH_optimal'] is None


def test_compare_does_not_judge_relieff(separable_objective):
    top = pd.DataFrame([top_row('relieff')])
    mh = pd.DataFrame([resume_row('DE', 'relieff', list(range(10, 20)))])
    row = compare(np.zeros((5, 20)), np.zeros(5), top, mh, 10).iloc[0]
    assert not row['Separable']
    assert row['Top_ge_MH'] is None and row['MH_optimal'] is None


def test_compare_skips_objectives_without_a_top_10(separable_objective):
    top = pd.DataFrame([top_row('mi')])
    mh = pd.DataFrame([resume_row('DE', 'information_gain', list(range(10, 20))), resume_row('DE', 'mi', list(range(10, 20)))])
    assert compare(np.zeros((5, 20)), np.zeros(5), top, mh, 10)['OF'].tolist() == ['mi']
