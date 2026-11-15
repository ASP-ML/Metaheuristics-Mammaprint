"""
Tests for latex_tables/clinical_comparison_table.py: comparison with MammaPrint, NIH and St. Gallen.

The expected values of the clinical criteria are the ones computed by hand for the hold-out test set.
"""

import pytest

from latex_tables.clinical_comparison_table import build_table, metrics_from_counts


def test_mammaprint_metrics():
    m = metrics_from_counts(tp=9, tn=7, fp=13, fn=1)
    assert m['acc'] == pytest.approx(16 / 30)
    assert m['sn'] == pytest.approx(0.90)
    assert m['sp'] == pytest.approx(0.35)
    assert m['pre'] == pytest.approx(9 / 22)
    assert m['npv'] == pytest.approx(0.875)
    assert m['f1'] == pytest.approx(0.5625)
    assert m['bacc'] == pytest.approx(0.625)
    assert m['mcc'] == pytest.approx(0.2665, abs=1e-4)


def test_nih_metrics():
    m = metrics_from_counts(tp=10, tn=1, fp=19, fn=0)
    assert m['npv'] == pytest.approx(1.0)
    assert m['f1'] == pytest.approx(20 / 39)
    assert m['mcc'] == pytest.approx(0.1313, abs=1e-4)


def test_undefined_metrics_of_st_gallen():
    m = metrics_from_counts(tp=10, tn=0, fp=20, fn=0)
    assert m['npv'] is None and m['mcc'] is None
    assert m['sp'] == 0 and m['f1'] == pytest.approx(0.5)


def test_table_marks_undefined_values_and_best_values():
    columns = [(['A'], {'tp': 10, 'tn': 14, 'fp': 6, 'fn': 0}),
               (['St. Gallen'], {'tp': 10, 'tn': 0, 'fp': 20, 'fn': 0})]
    latex = build_table(columns)
    npv_row = next(line for line in latex.splitlines() if line.startswith('NPV &'))
    assert r'$\mathbf{1.00}$' in npv_row and r'n.d.$^{a}$' in npv_row
    mcc_row = next(line for line in latex.splitlines() if line.startswith('MCC &'))
    assert r'n.d.$^{b}$' in mcc_row
    assert 'Not defined: no patient is predicted negative' in latex
    assert 'Not defined: all patients are predicted in the same class' in latex
    # Sn ties at 1.00: both bold
    sn_row = next(line for line in latex.splitlines() if line.startswith('Sn &'))
    assert sn_row.count(r'\mathbf{1.00}') == 2
    # scaled to the text width, header on one line per entry
    assert r'\resizebox{\textwidth}{!}{%' in latex and latex.count(r'\end{tabular}') == 1


def test_columns_from_different_test_sets_are_refused():
    columns = [(['Holdout'], {'tp': 10, 'tn': 14, 'fp': 6, 'fn': 0}),
               (['NKI33'], {'tp': 13, 'tn': 11, 'fp': 8, 'fn': 1})]
    with pytest.raises(ValueError):
        build_table(columns)
