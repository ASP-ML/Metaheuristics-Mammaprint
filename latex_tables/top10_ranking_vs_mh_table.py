"""
LaTeX table of the top-10 ranking versus the metaheuristic signatures (label tableTop10RankingVsMH).

Input (METRICS_FILE): test_metrics.csv, written by top10_ranking_vs_mh.ipynb (stage 3). One row per pair and
signature (15 pairs, 30 rows). Columns used: Pair, Meta, OF, Classifier, Signature (MH or TOP10), Model, Length,
Same_as_MH, and F1, Sn, Sp, ACC with their _low and _high 95% limits.

The table has the layout of the other tables of the manuscript: a table float scaled to the text width
(\\resizebox), notes in a box under it, and, with COLOR = 'blue', every text in blue. It needs no package beyond
graphicx (\\resizebox) and xcolor (\\textcolor).

Each pair takes two consecutive rows, MH (the metaheuristic signature) and Top-10 (the top-10 ranking of the same
objective), with the same classifier; the scheme and classifier are written on the first row only. Rows follow
the order of the manuscript (DE, GA, SA; aMI, aSU, aGR, aReliefF, aCHI2), with a line between metaheuristics.

Marks, explained in the notes:
    a, b, ...  a Top-10 model shared by several pairs (same top-10 genes, classifier and parameters): it was
               trained and tested once, so those rows are not independent results
    dagger     the top-10 genes are the metaheuristic signature, so both rows are the same model

Values are printed as "value [low, high]" with 2 decimals, rounded half up as in the other tables (0.625 -> 0.63).

Run from any folder with:  python latex_tables/top10_ranking_vs_mh_table.py
Settings are the constants in the __main__ block. Paths are relative to the repository root.
"""

import csv
import os
import re
from decimal import ROUND_HALF_UP, Decimal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

META_ORDER = ['DE', 'GA', 'SA']
OBJECTIVE_ORDER = ['mi', 'sim_uncertainty', 'gain_ratio', 'relieff', 'chi2']
OBJECTIVE_LABELS = {'mi': 'aMI', 'sim_uncertainty': 'aSU', 'gain_ratio': 'aGR', 'relieff': 'aReliefF', 'chi2': 'aCHI2'}
METRICS = ['F1', 'Sn', 'Sp', 'ACC']
METRIC_HEADERS = ['F1-score', 'Sn', 'Sp', 'ACC']
SIGNATURE_LABELS = {'MH': 'MH', 'TOP10': 'Top-10'}
SAME_MARK = '\\dagger'


def classifier_label(raw):
    """'RF estimators: 60' -> 'RF ($t=60$)', 'Knn k = 3' -> 'kNN ($k=3$)', 'Naive' -> 'NB', 'SVM (Poly)' -> 'SVM (Poly)'."""
    text = raw.strip()
    lower = text.lower()
    number = re.search(r'\d+', text)
    if lower.startswith('rf') and number:
        return f'RF ($t={number.group()}$)'
    if lower.startswith('knn') and number:
        return f'kNN ($k={number.group()}$)'
    if lower.startswith('naive') or lower == 'nb':
        return 'NB'
    if lower.startswith('svm'):
        return 'SVM (Poly)' if 'poly' in lower else 'SVM'
    if lower == 'lda':
        return 'LDA'
    raise ValueError(f'Unknown classifier name: {raw!r}')


def round_half_up(value, decimals=2):
    return str(Decimal(str(float(value))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def format_ci(value, low, high, decimals=2):
    """'0.63 [0.30, 0.88]'."""
    return f'{round_half_up(value, decimals)} [{round_half_up(low, decimals)}, {round_half_up(high, decimals)}]'


def scheme_name(record):
    """'DE--aSU', as in the other tables."""
    return f"{record['Meta']}--{OBJECTIVE_LABELS[record['OF']]}"


def is_true(value):
    return str(value).strip().lower() in ('true', '1')


def group_pairs(rows):
    """{(meta, of): {'MH': row, 'TOP10': row}} in the manuscript order. Stops if a pair is incomplete."""
    pairs = {}
    for record in rows:
        key = (record['Meta'], record['OF'])
        if record['Signature'] in pairs.setdefault(key, {}):
            raise ValueError(f'{scheme_name(record)} has two {record["Signature"]} rows')
        pairs[key][record['Signature']] = record
    for key, signatures in pairs.items():
        if set(signatures) != set(SIGNATURE_LABELS):
            raise ValueError(f'{scheme_name(next(iter(signatures.values())))} needs one MH and one TOP10 row')
    return dict(sorted(pairs.items(), key=lambda item: (META_ORDER.index(item[0][0]), OBJECTIVE_ORDER.index(item[0][1]))))


def shared_models(pairs):
    """Letter for each Top-10 model used by more than one pair, and the schemes that use it, in order of appearance."""
    users = {}
    for signatures in pairs.values():
        top = signatures['TOP10']
        users.setdefault(top['Model'], []).append(scheme_name(top))
    shared = [model for model, names in users.items() if len(names) > 1]
    return {model: (chr(ord('a') + i), users[model]) for i, model in enumerate(shared)}


def build_table(rows, caption, label, decimals=2, color='blue'):
    """The LaTeX table, as a string. With color=None the text keeps the document's colour."""
    def paint(text):
        return f'\\textcolor{{{color}}}{{{text}}}' if color and text else text

    def line(cells):
        return ' & '.join(paint(cell) for cell in cells) + ' \\\\'

    pairs = group_pairs(rows)
    shared = shared_models(pairs)
    any_same = any(is_true(signatures['TOP10']['Same_as_MH']) for signatures in pairs.values())

    notes = ['MH: metaheuristic signature. Top-10: the 10 best-ranked genes of the same objective on the training '
             'partition, with the same classifier and parameters. L: number of genes. Values with 95\\% confidence '
             'intervals.']
    notes += [f'$^{{{mark}}}$ Same model in {" and ".join(names)}: same top-10 genes, classifier and parameters, '
              f'trained and tested once.' for mark, names in shared.values()]
    if any_same:
        notes.append(f'$^{{{SAME_MARK}}}$ The top-10 genes are the metaheuristic signature: both rows are the same model.')

    lines = [
        '\\begin{table}[!ht]',
        '\\linespread{1}\\selectfont',
        '\\centering',
        '\\small',
        f'\\caption{{{paint(caption)}}}',
        f'\\label{{{label}}}',
        '\\resizebox{\\textwidth}{!}{%',
        '\\begin{tabular}{llllcccc}',
        '\\hline',
        line(['Scheme', 'Classifier', 'Signature', 'L'] + METRIC_HEADERS),
        line(['', '', '', ''] + ['[95\\% CI]'] * len(METRICS)),
        '\\hline',
    ]
    previous_meta = None
    for (meta, of), signatures in pairs.items():
        if previous_meta is not None and meta != previous_meta:
            lines.append('\\hline')
        previous_meta = meta
        mh, top = signatures['MH'], signatures['TOP10']
        mark = ''
        if top['Model'] in shared:
            mark = f'$^{{{shared[top["Model"]][0]}}}$'
        elif is_true(top['Same_as_MH']):
            mark = f'$^{{{SAME_MARK}}}$'
        metrics = lambda record: [format_ci(record[m], record[f'{m}_low'], record[f'{m}_high'], decimals) for m in METRICS]
        lines.append(line([scheme_name(mh), classifier_label(mh['Classifier']), SIGNATURE_LABELS['MH'], str(mh['Length'])]
                          + metrics(mh)))
        lines.append(line(['', '', SIGNATURE_LABELS['TOP10'] + mark, str(top['Length'])] + metrics(top)))
    lines += [
        '\\hline',
        '\\end{tabular}}',
        '\\par\\smallskip',
        f'\\parbox{{\\textwidth}}{{\\footnotesize{paint(" ".join(notes))}}}',
        '\\end{table}',
    ]
    return '\n'.join(lines) + '\n'


def run(metrics_file, output_file, caption, label, decimals=2, color='blue'):
    """Read METRICS_FILE and write the table."""
    with open(metrics_file, newline='') as file:
        rows = list(csv.DictReader(file))
    table = build_table(rows, caption, label, decimals, color)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(table)
    return table


if __name__ == "__main__":
    METRICS_FILE = os.path.join(REPO_ROOT, 'experiment_results/top10_ranking_vs_mh/test_metrics.csv')
    OUTPUT_FILE = os.path.join(REPO_ROOT, 'latex_tables/output/top10_ranking_vs_mh.tex')
    LABEL = 'tableTop10RankingVsMH'
    CAPTION = ('Exploratory comparison on the same 30 real test patients (10 positive and 20 negative) of each '
               'finalist with its metaheuristic signature (MH) and with the top-10 ranking of the same objective '
               '(Top-10), keeping the classifier and its parameters. 95\\% confidence intervals: percentile bootstrap '
               'stratified by class for the F1-score; Wilson score interval for Sn, Sp and ACC.')
    DECIMALS = 2
    COLOR = 'blue'      # text colour of the table; None for the document's colour

    run(METRICS_FILE, OUTPUT_FILE, CAPTION, LABEL, DECIMALS, COLOR)
    print('Written to', OUTPUT_FILE)
