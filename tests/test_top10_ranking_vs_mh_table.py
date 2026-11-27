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


def table_of(rows, color=None):
    return build_table(rows, caption='C', label='L', color=color)


def body_lines(table):
    """The rows of the tabular between the header and the end, without \\hline."""
    lines = table.splitlines()
    start = lines.index('\\hline', lines.index('\\hline') + 1) + 1
    end = lines.index('\\end{tabular}}')
    return [line for line in lines[start:end] if line != '\\hline']


def test_classifier_labels_follow_the_manuscript():
    assert classifier_label('RF estimators: 60') == 'RF ($t=60$)'
    assert classifier_label('Knn k = 3') == 'kNN ($k=3$)'
    assert classifier_label('Naive') == 'NB'
    assert classifier_label('SVM (Poly)') == 'SVM (Poly)'


def test_values_are_printed_with_their_interval_rounded_half_up():
    # 0.625 is printed as 0.63, as in the other tables of the manuscript
    assert format_ci(0.625, 0.3049, 0.875) == '0.63 [0.30, 0.88]'


def test_the_table_is_a_float_scaled_to_the_text_width_like_the_other_tables():
    table = table_of(pair(1, 'DE', 'mi', 'Naive'))
    assert table.startswith('\\begin{table}[!ht]')
    assert '\\resizebox{\\textwidth}{!}{%' in table and '\\begin{tabular}{llllcccc}' in table
    assert '\\caption{C}' in table and '\\label{L}' in table
    assert table.rstrip().endswith('\\end{table}')
    # Only standard LaTeX: no longtable or multirow packages needed
    assert 'longtable' not in table and '\\multirow' not in table


def test_each_pair_is_two_consecutive_rows_named_once():
    mh, top = body_lines(table_of(pair(1, 'DE', 'mi', 'Naive')))
    assert mh.startswith('DE--aMI & NB & MH & 10 &')
    assert top.startswith(' &  & Top-10 & 10 &')


def test_genes_and_metrics_are_printed_for_both_rows():
    mh, top = body_lines(table_of(pair(1, 'DE', 'relieff', 'Knn k = 3', mh_length=18)))
    assert mh == 'DE--aReliefF & kNN ($k=3$) & MH & 18 & 0.80 [0.70, 0.90] & 0.80 [0.70, 0.90] & 0.80 [0.70, 0.90] & 0.80 [0.70, 0.90] \\\\'
    assert top.endswith('& Top-10 & 10 & 0.50 [0.40, 0.60] & 0.50 [0.40, 0.60] & 0.50 [0.40, 0.60] & 0.50 [0.40, 0.60] \\\\')


def test_rows_follow_the_manuscript_order_and_metaheuristics_are_separated_by_a_line():
    rows = pair(2, 'GA', 'mi', 'Naive') + pair(1, 'DE', 'chi2', 'Naive') + pair(3, 'DE', 'mi', 'Naive')
    table = table_of(rows)
    order = [table.index(name) for name in ('DE--aMI &', 'DE--aCHI2 &', 'GA--aMI &')]
    assert order == sorted(order)
    between_de_and_ga = table[table.index('DE--aCHI2 &'):table.index('GA--aMI &')]
    between_de_pairs = table[table.index('DE--aMI &'):table.index('DE--aCHI2 &')]
    assert '\\hline' in between_de_and_ga and '\\hline' not in between_de_pairs


def test_a_top10_model_shared_by_several_pairs_is_marked_and_explained():
    rows = (pair(1, 'DE', 'chi2', 'RF estimators: 100', top10_model='TOP10-chi2-RF')
            + pair(2, 'SA', 'chi2', 'RF estimators: 100', top10_model='TOP10-chi2-RF'))
    table = table_of(rows)
    assert table.count('Top-10$^{a}$') == 2
    assert '$^{a}$ Same model in DE--aCHI2 and SA--aCHI2' in table


def test_a_top10_equal_to_the_signature_is_marked_and_explained():
    table = table_of(pair(1, 'DE', 'gain_ratio', 'RF estimators: 60', same_as_mh=True))
    assert 'Top-10$^{\\dagger}$' in table
    assert '$^{\\dagger}$ The top-10 genes are the metaheuristic signature' in table


def test_notes_go_in_a_box_under_the_table():
    table = table_of(pair(1, 'DE', 'mi', 'Naive'))
    assert table.index('\\end{tabular}}') < table.index('\\parbox{\\textwidth}{\\footnotesize')


def test_text_can_be_coloured():
    table = table_of(pair(1, 'DE', 'mi', 'Naive'), color='blue')
    assert '\\caption{\\textcolor{blue}{C}}' in table
    mh, top = body_lines(table)
    assert mh.startswith('\\textcolor{blue}{DE--aMI} & \\textcolor{blue}{NB} & \\textcolor{blue}{MH}')
    assert top.startswith(' &  & \\textcolor{blue}{Top-10}')
    assert '\\footnotesize\\textcolor{blue}{' in table


def test_a_pair_without_both_signatures_stops_the_script():
    with pytest.raises(ValueError, match='DE--aMI'):
        table_of(pair(1, 'DE', 'mi', 'Naive')[:1])


def test_run_reads_the_metrics_file_and_writes_the_table(tmp_path):
    rows = pair(1, 'DE', 'mi', 'Naive')
    metrics_file, output_file = tmp_path / 'test_metrics.csv', tmp_path / 'out' / 'table.tex'
    with open(metrics_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    run(str(metrics_file), str(output_file), caption='Caption text', label='tableTop10', color=None)
    text = output_file.read_text()
    assert '\\caption{Caption text}' in text and '\\label{tableTop10}' in text
    assert '0.80 [0.70, 0.90]' in text
