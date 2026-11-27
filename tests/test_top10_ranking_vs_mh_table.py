"""
Tests for latex_tables/top10_ranking_vs_mh_table.py, the LaTeX table of the top-10 ranking versus the
metaheuristic signatures (test_metrics.csv written by top10_ranking_vs_mh.ipynb).
"""

import csv

import pytest

from latex_tables.top10_ranking_vs_mh_table import build_table, classifier_label, format_ci, run

METRICS = ['F1', 'Sn', 'Sp', 'ACC']


def row(pair, meta, of, classifier, signature, model, length=10, same_as_mh=False, value=0.5):
    record = {'Pair': pair, 'Meta': meta, 'OF': of, 'Classifier': classifier, 'Signature': signature,
              'Model': model, 'Length': length, 'Same_as_MH': same_as_mh}
    for metric in METRICS:
        record.update({metric: value, f'{metric}_low': value - 0.1, f'{metric}_high': value + 0.1})
    return record


def pair(number, meta, of, classifier, top10_model=None, same_as_mh=False, mh_length=10):
    return [row(number, meta, of, classifier, 'MH', f'{meta}-{of}-{classifier}', length=mh_length, value=0.8),
            row(number, meta, of, classifier, 'TOP10', top10_model or f'TOP10-{of}-{classifier}', same_as_mh=same_as_mh)]


def test_classifier_labels_follow_the_manuscript():
    assert classifier_label('RF estimators: 60') == 'RF (t=60)'
    assert classifier_label('Knn k = 3') == 'kNN (k=3)'
    assert classifier_label('Naive') == 'NB'
    assert classifier_label('SVM (Poly)') == 'SVM (Poly)'


def test_values_are_printed_with_their_interval_rounded_half_up():
    # 0.625 is printed as 0.63, as in the other tables of the manuscript
    assert format_ci(0.625, 0.3049, 0.875) == '0.63 [0.30, 0.88]'


def test_each_pair_spans_two_rows_that_cannot_be_split_across_pages():
    table = build_table(pair(1, 'DE', 'mi', 'Naive'), caption='C', label='L')
    lines = [line for line in table.splitlines() if 'Naive' in line or ' & & & ' in line or 'NB' in line]
    mh, top = lines[0], lines[1]
    assert '\\multirow{2}{*}{DE}' in mh and '\\multirow{2}{*}{aMI}' in mh and '\\multirow{2}{*}{NB}' in mh
    assert mh.rstrip().endswith('\\\\*')          # no page break between the two rows of a pair
    assert top.startswith(' & & & Top-10')


def test_the_header_is_repeated_on_every_page():
    table = build_table(pair(1, 'DE', 'mi', 'Naive'), caption='C', label='L')
    assert '\\begin{longtable}' in table and '\\endfirsthead' in table and '\\endhead' in table
    assert table.count('Signature & L & F1') == 2


def test_rows_follow_the_manuscript_order_of_metaheuristics_and_objectives():
    rows = pair(2, 'GA', 'mi', 'Naive') + pair(1, 'DE', 'chi2', 'Naive') + pair(3, 'DE', 'mi', 'Naive')
    table = build_table(rows, caption='C', label='L')
    order = [table.index(text) for text in ('{DE} & \\multirow{2}{*}{aMI}', '{DE} & \\multirow{2}{*}{aCHI2}',
                                            '{GA} & \\multirow{2}{*}{aMI}')]
    assert order == sorted(order)


def test_a_top10_model_shared_by_several_pairs_is_marked_and_explained():
    rows = (pair(1, 'DE', 'chi2', 'RF estimators: 100', top10_model='TOP10-chi2-RF')
            + pair(2, 'SA', 'chi2', 'RF estimators: 100', top10_model='TOP10-chi2-RF'))
    table = build_table(rows, caption='C', label='L')
    assert table.count('Top-10$^{a}$') == 2
    assert '$^{a}$ Same model in DE + aCHI2 and SA + aCHI2' in table


def test_a_top10_equal_to_the_signature_is_marked_and_explained():
    table = build_table(pair(1, 'DE', 'gain_ratio', 'RF estimators: 60', same_as_mh=True), caption='C', label='L')
    assert 'Top-10$^{\\dagger}$' in table
    assert '$^{\\dagger}$ The top-10 genes are the metaheuristic signature' in table


def test_genes_and_metrics_are_printed_for_both_rows():
    table = build_table(pair(1, 'DE', 'relieff', 'Knn k = 3', mh_length=18), caption='C', label='L')
    assert '& MH & 18 & 0.80 [0.70, 0.90]' in table
    assert '& Top-10 & 10 & 0.50 [0.40, 0.60]' in table


def test_a_pair_without_both_signatures_stops_the_script():
    with pytest.raises(ValueError, match='DE \\+ aMI'):
        build_table(pair(1, 'DE', 'mi', 'Naive')[:1], caption='C', label='L')


def test_run_reads_the_metrics_file_and_writes_the_table(tmp_path):
    rows = pair(1, 'DE', 'mi', 'Naive')
    metrics_file, output_file = tmp_path / 'test_metrics.csv', tmp_path / 'out' / 'table.tex'
    with open(metrics_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run(str(metrics_file), str(output_file), caption='Caption text', label='tableTop10')
    text = output_file.read_text()
    assert '\\caption{Caption text}\\label{tableTop10}' in text
    assert '0.80 [0.70, 0.90]' in text
