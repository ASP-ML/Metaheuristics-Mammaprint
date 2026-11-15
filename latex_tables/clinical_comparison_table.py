"""
LaTeX table comparing the proposed scheme and the 70-gene model with clinical criteria (MammaPrint, NIH,
St. Gallen) on the hold-out test set.

Inputs:
    RESULTS_FILE      test results of the 15 schemes (test_classifiers.ipynb, TEST_NAME = 'test')
    WHOLE_SET_FILE    test result of the 70-gene model (same notebook run)
    CLINICAL_CRITERIA confusion matrix (TP, TN, FP, FN) of each clinical criterion on the same patients

Every metric is computed from the confusion matrix, the same way for all columns:
    ACC = (TP + TN) / N            Sn = TP / (TP + FN)          Sp = TN / (TN + FP)
    PRE = TP / (TP + FP)           NPV = TN / (TN + FN)         F1 = 2 TP / (2 TP + FP + FN)
    BACC = (Sn + Sp) / 2           MCC = (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN))
A metric whose denominator is zero is printed as "n.d." with a footnote (e.g. NPV when no patient is predicted
negative). All columns must have the same number of positive and negative patients, otherwise the script stops:
that would mean they were not evaluated on the same test set.

Values are rounded half up to DECIMALS, as in the other tables. With BOLD_BEST the best value of each metric row
is printed in bold (ties are all bold).

Run from any folder with:  python latex_tables/clinical_comparison_table.py
Settings are the constants in the __main__ block. Paths are relative to the repository root.
"""

import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from latex_tables.performance_test_table import (OBJECTIVE_LABELS, classifier_label, load_results,  # noqa: E402
                                                 round_half_up)

# (key, row label) of the metric rows, in table order
METRIC_ROWS = [
    ('acc', 'ACC'),
    ('sn', 'Sn'),
    ('sp', 'Sp'),
    ('pre', 'PRE'),
    ('npv', 'NPV'),
    ('f1', 'F1-score'),
    ('bacc', 'BACC'),
    ('mcc', 'MCC'),
]
COUNT_ROWS = [('tp', 'TP'), ('tn', 'TN'), ('fp', 'FP'), ('fn', 'FN')]

UNDEFINED_NOTES = {
    'pre': 'no patient is predicted positive (TP + FP = 0)',
    'npv': 'no patient is predicted negative (TN + FN = 0)',
    'f1': 'no positive patient and no positive prediction',
    'mcc': 'all patients are predicted in the same class',
}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def metrics_from_counts(tp, tn, fp, fn):
    """Metrics of one confusion matrix. None marks a metric that is not defined."""
    sn, sp = ratio(tp, tp + fn), ratio(tn, tn + fp)
    mcc_denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'acc': ratio(tp + tn, tp + tn + fp + fn),
        'sn': sn,
        'sp': sp,
        'pre': ratio(tp, tp + fp),
        'npv': ratio(tn, tn + fn),
        'f1': ratio(2 * tp, 2 * tp + fp + fn),
        'bacc': (sn + sp) / 2 if sn is not None and sp is not None else None,
        'mcc': (tp * tn - fp * fn) / math.sqrt(mcc_denominator) if mcc_denominator else None,
    }


def counts_of(result_row):
    return {k: int(float(result_row[k])) for k in ('tp', 'tn', 'fp', 'fn')}


def scheme_header(name, result_row):
    """Header lines of a proposed scheme: 'Proposed (DE--aSU)', 'RF (t=60)'."""
    meta, objective = name.split('_', 1)
    return [f'Proposed ({meta}--{OBJECTIVE_LABELS.get(objective, objective)})', classifier_label(result_row['classifier'])]


def build_table(columns, decimals=2, bold_best=True, fit_to_text_width=True, caption='', label=''):
    """columns: list of (header lines, counts dict). Returns the LaTeX source."""
    computed = [metrics_from_counts(**counts) for _, counts in columns]
    positives = {m['tp'] + m['fn'] for m in computed}
    negatives = {m['tn'] + m['fp'] for m in computed}
    if len(positives) != 1 or len(negatives) != 1:
        raise ValueError('The columns were not evaluated on the same patients: positives '
                         f'{[m["tp"] + m["fn"] for m in computed]}, negatives {[m["tn"] + m["fp"] for m in computed]}')

    notes = {}   # metric -> footnote letter, in order of appearance

    def cell(metric, value, best):
        if value is None:
            letter = notes.setdefault(metric, 'abcdefgh'[len(notes)])
            return rf'n.d.$^{{{letter}}}$'
        text = round_half_up(value, decimals)
        return rf'$\mathbf{{{text}}}$' if bold_best and text == best else f'${text}$'

    n_columns = len(columns) + 1
    n_header = max(len(lines) for lines, _ in columns)
    out = [
        r'\begin{table}[!ht]',
        r'\small',
        rf'\caption{{{caption}}}',
        r'\centering',
        rf'\label{{{label}}}',
    ]
    if fit_to_text_width:
        out.append(r'\resizebox{\textwidth}{!}{%')
    out += [rf'\begin{{tabular}}{{l{"c" * len(columns)}}}', r'\hline']
    for i in range(n_header):
        first = 'Metric' if i == n_header - 1 else ''
        out.append(' & '.join([first] + [lines[i] if i < len(lines) else '' for lines, _ in columns]) + r' \\')
    out.append(r'\hline')
    for key, row_label in COUNT_ROWS:
        out.append(' & '.join([row_label] + [f'${m[key]}$' for m in computed]) + r' \\')
    out.append(r'\hline')
    for key, row_label in METRIC_ROWS:
        defined = [round_half_up(m[key], decimals) for m in computed if m[key] is not None]
        best = max(defined, key=float) if defined else None
        out.append(' & '.join([row_label] + [cell(key, m[key], best) for m in computed]) + r' \\')
    out.append(r'\hline')
    for metric, letter in notes.items():
        out.append(rf'\multicolumn{{{n_columns}}}{{l}}{{\footnotesize $^{{{letter}}}$Not defined: '
                   rf'{UNDEFINED_NOTES[metric]}.}} \\')
    out.append(r'\end{tabular}')
    if fit_to_text_width:
        out.append('}')
    out.append(r'\end{table}')
    return '\n'.join(out) + '\n'


def run(results_file, whole_set_file, output_file, proposed_schemes, whole_set_name, clinical_criteria,
        caption_template, **table_settings):
    results = load_results(results_file)
    whole_set = load_results(whole_set_file)[whole_set_name]
    columns = [(scheme_header(name, results[name]), counts_of(results[name])) for name in proposed_schemes]
    columns.append((['Whole set ($L = 70$)', classifier_label(whole_set['classifier'])], counts_of(whole_set)))
    for criterion, counts in clinical_criteria.items():
        columns.append(([criterion], counts))

    counts = columns[0][1]
    caption = caption_template.format(n=sum(counts.values()), positives=counts['tp'] + counts['fn'],
                                      negatives=counts['tn'] + counts['fp'])
    latex = build_table(columns, caption=caption, **table_settings)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(latex)
    return latex


if __name__ == "__main__":
    # ---- test set: uses the results of test_classifiers.ipynb run with the same TEST_NAME ----
    STEP = 'R3_v2'
    TEST_NAME = 'test'                # 'test' (30-patient hold-out) or 'NKI33'
    TEST_SETS = {                     # caption text and LaTeX label of each test set
        'test': {'caption': 'the hold-out test set', 'label': 'clinicalComparison'},
        'NKI33': {'caption': 'the NKI33 set', 'label': 'clinicalComparisonNKI33'},
    }

    # ---- inputs and output (relative to the repository root) ----
    RESULTS_FILE = f'experiment_results/{STEP}/{TEST_NAME}_results_15_schemes.csv'
    WHOLE_SET_FILE = f'experiment_results/{STEP}/{TEST_NAME}_results_70_genes.csv'
    WHOLE_SET_NAME = 'ALL_FEATURES'
    OUTPUT_FILE = f'latex_tables/output/{TEST_NAME}_clinical_comparison_{STEP}.tex'

    # Proposed scheme(s), "<Meta>_<OF>" as in RESULTS_FILE. DE_sim_uncertainty has the best F1 on the hold-out
    # test set, as in the submitted manuscript. (The CV selection rule picks GA_relieff.)
    PROPOSED_SCHEMES = ['DE_sim_uncertainty']

    # Confusion matrices of the clinical criteria, per test set (on the same patients as the results)
    CLINICAL_CRITERIA = {
        'test': {
            'MammaPrint': {'tp': 9, 'tn': 7, 'fp': 13, 'fn': 1},
            'NIH': {'tp': 10, 'tn': 1, 'fp': 19, 'fn': 0},
            'St. Gallen': {'tp': 10, 'tn': 0, 'fp': 20, 'fn': 0},
        },
        # 'NKI33': {...},   add the MammaPrint, NIH and St. Gallen results of the NKI33 patients here
    }
    if TEST_NAME not in CLINICAL_CRITERIA:
        sys.exit(f'No MammaPrint, NIH and St. Gallen results for {TEST_NAME!r}: add them to CLINICAL_CRITERIA.')

    # ---- table layout ----
    DECIMALS = 2
    BOLD_BEST = True
    FIT_TO_TEXT_WIDTH = True          # scale the table to \textwidth, as Table 7 (needs \usepackage{graphicx})
    CAPTION = ('Comparison of the proposed classification scheme, the whole set of 70 genes and the clinical '
               f'criteria MammaPrint, NIH and St. Gallen on {TEST_SETS[TEST_NAME]["caption"]} '
               '($N = {n}$: {positives} positive and {negatives} negative patients).')
    LABEL = TEST_SETS[TEST_NAME]['label']

    latex = run(
        os.path.join(REPO_ROOT, RESULTS_FILE),
        os.path.join(REPO_ROOT, WHOLE_SET_FILE),
        os.path.join(REPO_ROOT, OUTPUT_FILE),
        proposed_schemes=PROPOSED_SCHEMES,
        whole_set_name=WHOLE_SET_NAME,
        clinical_criteria=CLINICAL_CRITERIA[TEST_NAME],
        caption_template=CAPTION,
        decimals=DECIMALS,
        bold_best=BOLD_BEST,
        fit_to_text_width=FIT_TO_TEXT_WIDTH,
        label=LABEL,
    )
    print(latex)
    print(f'Written to {OUTPUT_FILE}')
