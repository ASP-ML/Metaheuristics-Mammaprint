"""
LaTeX table of the hold-out test performance of the selected classification schemes
(table performanceTestSet in the manuscript).

Inputs:
    RESULTS_FILE  test metrics, one row per scheme, as written by test_classifiers.ipynb.
                  Columns: name, classifier, auc, acc, pre, mse, rec, f1, mcc, spe, tn, fp, fn, tp.
                  name is "<Meta>_<OF>", e.g. DE_sim_uncertainty.
    SUBSETS_FILE  gene subset of each scheme, e.g. best_results_step_R3_v2.csv.
                  Columns used: Meta, OF, Subset (0-based gene indices), Length, Classifier.

The two files are joined on "<Meta>_<OF>". The classifier of each scheme must be the same in both
files, otherwise the script stops: that means the results were not produced from those subsets.

Genes are printed sorted and 1-based (GENE_INDEX_OFFSET = 1): index 9 is printed as g^{10}.
A subset longer than GENES_PER_LINE continues on extra rows of the table.

Metrics are rounded half up, as in the submitted tables: 0.625 is printed as 0.63 and 23/40 = 0.575
as 0.58. (Python's own formatting would print 0.62 and 0.57.)

The worst-case row (all genes) is read from WORST_CASE_FILE, a file in the RESULTS_FILE layout,
using the row named WORST_CASE_NAME. With WORST_CASE_FILE = None the row is left out.

The table needs \\usepackage{multirow} and \\usepackage{graphicx} in the LaTeX preamble.

Run from any folder with:  python latex_tables/performance_test_table.py
Settings are the constants in the __main__ block. Paths are relative to the repository root.
"""

import ast
import csv
import os
import re
from decimal import ROUND_HALF_UP, Decimal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# OF column value -> name printed in the Function column
OBJECTIVE_LABELS = {
    'mi': 'aMI',
    'sim_uncertainty': 'aSU',
    'gain_ratio': 'aGR',
    'relieff': 'aReliefF',
    'chi2': 'aCHI2',
    'information_gain': 'aIG',
}

SVM_KERNEL_LABELS = {'poly': 'Poly', 'rbf': 'RBF', 'linear': 'Linear', 'sigmoid': 'Sigmoid'}


def classifier_label(raw):
    """Classifier name as printed in the table: 'RF estimators: 60' -> 'RF (t=60)',
    'Knn k = 3' -> 'kNN (k=3)', 'Naive bayes' -> 'NB', 'SVM Kernel = poly' -> 'SVM (Poly)'."""
    text = raw.strip()
    lower = text.lower()
    number = re.search(r'\d+', text)
    if lower.startswith('rf') and number:
        return f'RF (t={number.group()})'
    if lower.startswith('knn') and number:
        return f'kNN (k={number.group()})'
    if lower.startswith('naive') or lower == 'nb':
        return 'NB'
    if lower.startswith('svm'):
        kernel = re.search(r'(?:=|\()\s*([a-z]+)', lower)
        kernel = kernel.group(1) if kernel else 'poly'
        return f'SVM ({SVM_KERNEL_LABELS.get(kernel, kernel.capitalize())})'
    if lower in ('lda', 'qda'):
        return text.upper()
    if lower == 'lasso':
        return 'Lasso'
    raise ValueError(f'Unknown classifier name: {raw!r}')


def load_results(path):
    """Test metrics by scheme name."""
    results = {}
    with open(path, newline='') as file:
        for row in csv.DictReader(file):
            name = row['name'].strip()
            if name in results:
                raise ValueError(f'{path}: scheme {name} appears more than once')
            results[name] = row
    return results


def load_subsets(path):
    """Gene subset (0-based) and classifier by scheme name '<Meta>_<OF>'."""
    subsets = {}
    with open(path, newline='') as file:
        for row in csv.DictReader(file):
            name = f"{row['Meta'].strip()}_{row['OF'].strip()}"
            subset = list(ast.literal_eval(row['Subset']))
            if row.get('Length') and int(row['Length']) != len(subset):
                raise ValueError(f'{path}: {name} has Length {row["Length"]} but {len(subset)} genes')
            if name in subsets:
                raise ValueError(f'{path}: scheme {name} appears more than once')
            subsets[name] = {'subset': subset, 'classifier': row['Classifier']}
    return subsets


def format_genes(subset, offset, bold=False):
    """One LaTeX gene symbol per gene, sorted by index."""
    if bold:
        return [rf'$\mathbf{{g}}^{{\mathbf{{{g + offset}}}}}$' for g in sorted(subset)]
    return [rf'$g^{{{g + offset}}}$' for g in sorted(subset)]


def gene_lines(genes, genes_per_line):
    """Split the gene symbols into the text of each table row."""
    chunks = [genes[i:i + genes_per_line] for i in range(0, len(genes), genes_per_line)]
    lines = [', '.join(chunk) for chunk in chunks]
    return [line + ',' for line in lines[:-1]] + lines[-1:]


def round_half_up(value, decimals):
    """Rounded value as text. repr() gives the shortest decimal form of the float, so 23/40 is read
    as 0.575 and not as its binary approximation 0.57499..."""
    quantum = Decimal(1).scaleb(-decimals)
    return str(Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def format_value(value, decimals, bold=False):
    text = round_half_up(value, decimals)
    return rf'$\mathbf{{{text}}}$' if bold else f'${text}$'


def bold_text(text, bold):
    return rf'\textbf{{{text}}}' if bold else text


def table_row(cells):
    return ' & '.join(cells) + r' \\'


def scheme_rows(name, result, genes, metrics, decimals, genes_per_line, gene_offset, bold):
    """Cells of the rows of one scheme, without the Meta column."""
    objective = name.split('_', 1)[1]
    lines = gene_lines(format_genes(genes, gene_offset, bold), genes_per_line)
    first = [
        bold_text(OBJECTIVE_LABELS.get(objective, objective), bold),
        format_value(len(genes), 0, bold),
        lines[0],
        bold_text(classifier_label(result['classifier']), bold),
    ] + [format_value(result[column], decimals, bold) for column, _ in metrics]
    continuation = [['', '', line, ''] + [''] * len(metrics) for line in lines[1:]]
    return [first] + continuation


def build_table(results, subsets, metas, objectives, metrics, decimals=2, genes_per_line=11,
                gene_offset=1, highlight=None, worst_case=None, n_genes=70,
                caption='', label=''):
    """LaTeX source of the table."""
    names = [f'{meta}_{objective}' for meta in metas for objective in objectives]
    missing = [n for n in names if n not in results or n not in subsets]
    if missing:
        raise KeyError(f'Schemes missing from the results or subsets file: {missing}')
    if highlight is not None and highlight not in names:
        raise KeyError(f'HIGHLIGHT_SCHEME {highlight} is not one of the table schemes')
    for name in names:
        tested = classifier_label(results[name]['classifier'])
        selected = classifier_label(subsets[name]['classifier'])
        if tested != selected:
            raise ValueError(f'{name}: results file has {tested}, subsets file has {selected}. '
                             'The two files do not belong together.')

    n_columns = 5 + len(metrics)
    out = [
        r'\begin{table}[!ht]',
        r'\small',
        rf'\caption{{{caption}}}',
        r'\centering',
        rf'\label{{{label}}}',
        r'\resizebox{\textwidth}{!}{%',
        rf'\begin{{tabular}}{{{"l" * n_columns}}}',
        r'\hline',
        table_row(['Meta', 'Function', 'L', 'Subset', 'Classifier'] + [h for _, h in metrics]),
        r'\hline',
    ]
    for meta in metas:
        block = []
        for objective in objectives:
            name = f'{meta}_{objective}'
            block += scheme_rows(name, results[name], subsets[name]['subset'], metrics, decimals,
                                 genes_per_line, gene_offset, bold=(name == highlight))
        out.append(rf'\multirow{{{len(block)}}}{{*}}{{{meta}}}')
        out += [' ' + table_row([''] + cells) for cells in block]
        out.append(r'\hline')

    if worst_case is not None:
        out.append(table_row(
            [r'\multicolumn{2}{c}{$^*$Worst case}', format_value(n_genes, 0), 'Whole set',
             classifier_label(worst_case['classifier'])]
            + [format_value(worst_case[column], decimals) for column, _ in metrics]))
        out.append(r'\hline')

    out += [r'\end{tabular}', '}', r'\end{table}']
    return '\n'.join(out) + '\n'


def run(results_file, subsets_file, output_file, worst_case_file=None, worst_case_name=None,
        **table_settings):
    results = load_results(results_file)
    subsets = load_subsets(subsets_file)
    worst_case = None
    if worst_case_file is not None:
        worst_rows = load_results(worst_case_file)
        if worst_case_name not in worst_rows:
            raise KeyError(f'{worst_case_file} has no row named {worst_case_name}')
        worst_case = worst_rows[worst_case_name]

    latex = build_table(results, subsets, worst_case=worst_case, **table_settings)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(latex)
    return latex


if __name__ == "__main__":
    # ---- test set: uses the results of test_classifiers.ipynb run with the same TEST_NAME ----
    STEP = 'R3_v2'
    TEST_NAME = 'test'                # 'test' (30-patient hold-out) or 'NKI33'
    TEST_SETS = {                     # caption text and LaTeX label of each test set
        'test': {'caption': 'the hold-out test set', 'label': 'performanceTestSet'},
        'NKI33': {'caption': 'the NKI33 set', 'label': 'performanceNKI33'},
    }

    # ---- inputs and output (relative to the repository root) ----
    RESULTS_FILE = f'experiment_results/{STEP}/{TEST_NAME}_results_15_schemes.csv'
    SUBSETS_FILE = f'experiment_results/{STEP}/best_results_step_{STEP}.csv'
    OUTPUT_FILE = f'latex_tables/output/{TEST_NAME}_performance_{STEP}.tex'

    # Worst-case row (all genes). A file in the RESULTS_FILE layout; None leaves the row out.
    WORST_CASE_FILE = f'experiment_results/{STEP}/{TEST_NAME}_results_70_genes.csv'
    WORST_CASE_NAME = 'ALL_FEATURES'
    N_GENES = 70

    # ---- table layout ----
    METAS = ['DE', 'GA', 'SA']
    OBJECTIVES = ['mi', 'sim_uncertainty', 'gain_ratio', 'relieff', 'chi2']
    METRICS = [('f1', 'F1-score'), ('pre', 'PRE'), ('rec', 'Sn'), ('spe', 'Sp'), ('acc', 'ACC')]
    DECIMALS = 2
    GENES_PER_LINE = 11
    GENE_INDEX_OFFSET = 1             # Subset holds 0-based indices; the paper uses g^1 ... g^70
    HIGHLIGHT_SCHEME = 'DE_sim_uncertainty'   # row printed in bold (the final model); None for none
    CAPTION = ('Classification performance of the best-selected classification scheme on '
               f'{TEST_SETS[TEST_NAME]["caption"]}.')
    LABEL = TEST_SETS[TEST_NAME]['label']

    latex = run(
        os.path.join(REPO_ROOT, RESULTS_FILE),
        os.path.join(REPO_ROOT, SUBSETS_FILE),
        os.path.join(REPO_ROOT, OUTPUT_FILE),
        worst_case_file=os.path.join(REPO_ROOT, WORST_CASE_FILE) if WORST_CASE_FILE else None,
        worst_case_name=WORST_CASE_NAME,
        metas=METAS,
        objectives=OBJECTIVES,
        metrics=METRICS,
        decimals=DECIMALS,
        genes_per_line=GENES_PER_LINE,
        gene_offset=GENE_INDEX_OFFSET,
        highlight=HIGHLIGHT_SCHEME,
        n_genes=N_GENES,
        caption=CAPTION,
        label=LABEL,
    )
    print(latex)
    print(f'Written to {OUTPUT_FILE}')
    if WORST_CASE_FILE is None:
        print('Worst-case row left out: set WORST_CASE_FILE to add it.')
