"""
Tests for latex_tables/performance_test_table.py: LaTeX table of the hold-out test results.

The last test uses the R3_v2 files and is skipped when they are missing.
"""

import os

import pytest

from latex_tables.performance_test_table import (build_table, classifier_label, format_genes,
                                                 format_value, gene_lines, load_results, load_subsets)

METRICS = [('f1', 'F1-score'), ('pre', 'PRE'), ('rec', 'Sn'), ('spe', 'Sp'), ('acc', 'ACC')]
R3_V2_RESULTS = 'experiment_results/R3_v2/test_results_15_schemes.csv'
R3_V2_SUBSETS = 'experiment_results/R3_v2/best_results_step_R3_v2.csv'


def result(classifier, f1=0.5, pre=0.5, rec=0.5, spe=0.5, acc=0.5):
    return {'classifier': classifier, 'f1': f1, 'pre': pre, 'rec': rec, 'spe': spe, 'acc': acc}


@pytest.mark.parametrize('raw, label', [
    ('RF estimators: 60', 'RF (t=60)'),
    ('Knn k = 3', 'kNN (k=3)'),
    ('Naive bayes', 'NB'),
    ('Naive', 'NB'),
    ('SVM Kernel = poly', 'SVM (Poly)'),
    ('SVM (Poly)', 'SVM (Poly)'),
    (' Lasso', 'Lasso'),
    ('LDA', 'LDA'),
])
def test_classifier_label(raw, label):
    assert classifier_label(raw) == label


def test_unknown_classifier_raises():
    with pytest.raises(ValueError):
        classifier_label('XGBoost')


@pytest.mark.parametrize('value, decimals, text', [
    (0.625, 2, '$0.63$'),       # 5/8: exact half, rounded up as in the submitted tables
    (23 / 40, 2, '$0.58$'),     # 0.575 is stored as 0.57499...; the tables print 0.58
    (0.675, 2, '$0.68$'),
    (2 / 3, 2, '$0.67$'),
    (1, 2, '$1.00$'),
    (0.0, 2, '$0.00$'),
    (10, 0, '$10$'),
])
def test_values_are_rounded_half_up(value, decimals, text):
    assert format_value(value, decimals) == text


def test_genes_are_sorted_and_one_based():
    assert format_genes([9, 2], 1) == ['$g^{3}$', '$g^{10}$']


def test_long_subset_continues_on_next_line():
    lines = gene_lines(format_genes(list(range(12)), 1), 10)
    assert len(lines) == 2
    assert lines[0].endswith('$g^{10}$,')
    assert lines[1] == '$g^{11}$, $g^{12}$'


def test_table_layout():
    results = {'DE_mi': result('Naive bayes', f1=0.644), 'DE_chi2': result('RF estimators: 100')}
    subsets = {'DE_mi': {'subset': list(range(12)), 'classifier': 'Naive'},
               'DE_chi2': {'subset': [3, 1], 'classifier': 'RF estimators: 100'}}
    latex = build_table(results, subsets, ['DE'], ['mi', 'chi2'], METRICS, genes_per_line=10,
                        highlight='DE_chi2', worst_case=result('RF estimators: 60'), n_genes=70)
    lines = latex.splitlines()

    # DE_mi takes two rows, DE_chi2 one
    assert r'\multirow{3}{*}{DE}' in lines
    mi_row = next(line for line in lines if 'aMI' in line)
    assert '$0.64$' in mi_row and 'NB' in mi_row
    continuation = lines[lines.index(mi_row) + 1]
    assert '$g^{11}$' in continuation
    # every body row has 10 columns
    for line in lines:
        if line.endswith(r'\\') and 'multicolumn' not in line:
            assert line.count('&') == 9
    chi2_row = next(line for line in lines if 'aCHI2' in line)
    assert r'\textbf{aCHI2}' in chi2_row and r'$\mathbf{g}^{\mathbf{2}}$' in chi2_row
    worst = next(line for line in lines if 'Worst case' in line)
    assert 'Whole set' in worst and '$70$' in worst and 'RF (t=60)' in worst


def test_classifier_mismatch_raises():
    results = {'DE_mi': result('Naive bayes')}
    subsets = {'DE_mi': {'subset': [1], 'classifier': 'Knn k = 3'}}
    with pytest.raises(ValueError):
        build_table(results, subsets, ['DE'], ['mi'], METRICS)


def test_missing_scheme_raises():
    results = {'DE_mi': result('Naive bayes')}
    subsets = {'DE_mi': {'subset': [1], 'classifier': 'Naive'}}
    with pytest.raises(KeyError):
        build_table(results, subsets, ['DE'], ['mi', 'chi2'], METRICS)


@pytest.mark.skipif(not (os.path.exists(R3_V2_RESULTS) and os.path.exists(R3_V2_SUBSETS)),
                    reason='R3_v2 result files not present')
def test_r3_v2_table():
    latex = build_table(load_results(R3_V2_RESULTS), load_subsets(R3_V2_SUBSETS),
                        ['DE', 'GA', 'SA'], ['mi', 'sim_uncertainty', 'gain_ratio', 'relieff', 'chi2'],
                        METRICS, highlight='DE_sim_uncertainty')
    # DE relieff has 18 genes, so the DE block has 6 rows
    assert r'\multirow{6}{*}{DE}' in latex
    assert r'\multirow{5}{*}{GA}' in latex
    # DE aSU subset as in the manuscript: g10 g12 g15 g22 g43 g45 g51 g59 g64 g66
    su_row = next(line for line in latex.splitlines() if 'aSU' in line and 'mathbf' in line)
    for gene in (10, 12, 15, 22, 43, 45, 51, 59, 64, 66):
        assert rf'\mathbf{{{gene}}}' in su_row
