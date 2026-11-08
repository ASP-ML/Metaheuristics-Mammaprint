"""
Final model selection (editor comment E.2)

Chooses the final model from the fold-level cross-validation results of the training partition.
This script does not read the test set.

The written rule, with what was known when it was fixed, is in protocol/selection_rule.md.

Rule, applied to a group of candidate schemes (manuscript Section 2.6.5):
    1. The scheme with the highest mean F1 is the reference.
    2. A scheme is tied with the reference when the Wilcoxon rank-sum test on the fold F1 values,
       adjusted by Benjamini-Hochberg over the comparisons with the reference, gives p >= 0.05.
    3. Among the tied schemes, the highest mean Sp wins, then Sn, then PRE. Means are compared at
       two decimals, as printed in the tables.
    4. If the tie persists, the simplest classifier wins:
       NB = LDA < kNN (smaller k first) < SVM < RF (fewer trees first).

Step 1 applies the rule inside each metaheuristic x objective combination.
Step 2 applies it to the 15 winners of step 1, and adds "fewer genes" to item 4, before the
simplest classifier. Its winner is the final model.

Any other Meta in the results (top-10 baseline, 70-gene model) is a comparator: it goes through
step 1 to choose its classifier and cannot become the final model.

Input (HISTORY_FILE): one row per scheme x fold, in the layout of classifiers.writeHistory.
Input (SUBSET_FILES): CSV files with the columns Meta, OF and Subset (0-based gene indices).

Outputs (OUTPUT_DIR):
    final_model.json     final model, the other winners and the comparators with their genes and
                         CV metrics, the rule settings, and the SHA-256 of the input files
    selection_table.csv  one row per scheme per step: metrics, p_BH, RBC, Tied, Selected

Freezing: run this on the corrected classifier results with OUTPUT_DIR = 'protocol', and commit
final_model.json before the test evaluation runs. The test evaluation must read the model from
that file.
"""

import ast
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, mannwhitneyu

ALPHA = 0.05
DECIMALS = 2
N_FOLDS = 30

PROPOSED_METAS = ('DE', 'GA', 'SA')
OBJECTIVE_FUNCTIONS = ('relieff', 'mi', 'sim_uncertainty', 'gain_ratio', 'chi2')

CLASSIFIER_COMPLEXITY = {
    'Naive bayes': 0,
    'LDA': 0,
    'Knn k = 3': 1,
    'Knn k = 5': 2,
    'Knn k = 7': 3,
    'SVC poly': 4,
    'RF estimators: 60': 5,
    'RF estimators: 80': 6,
    'RF estimators: 100': 7,
}
"""The classifier grid of the manuscript (Tables 4-6), and the order of step 4 of the rule."""

COMBINATION = ['Meta', 'OF']
SCHEME = ['Meta', 'OF', 'Classifier']
REQUIRED_COLUMNS = ['Meta', 'OF', 'Length', 'Settings', 'Classifier', 'Fold',
                    'F1', 'Recall', 'Precision', 'Specificity']
CV_METRICS = ['F1', 'F1_SD', 'Sp', 'Sp_SD', 'Sn', 'Sn_SD', 'PRE', 'PRE_SD']


def load_history(path):
    # pipeline_classifiers.py writes the column as Spec; the submitted file has Specificity
    history = pd.read_csv(path).rename(columns={'Spec': 'Specificity'})
    missing = [column for column in REQUIRED_COLUMNS if column not in history.columns]
    if missing:
        raise ValueError(f'{path} has no column {missing}')
    return history


def keep_study_schemes(history):
    """Drop classifiers outside the manuscript's grid and objectives the manuscript does not use."""
    in_grid = history['Classifier'].isin(list(CLASSIFIER_COMPLEXITY))
    is_comparator = ~history['Meta'].isin(PROPOSED_METAS)
    in_study = history['OF'].isin(OBJECTIVE_FUNCTIONS) | is_comparator
    return history[in_grid & in_study].reset_index(drop=True)


def check_history(history):
    """Raise ValueError if a partial or mixed run would decide the final model."""
    proposed = history[history['Meta'].isin(PROPOSED_METAS)]
    found = set(zip(proposed['Meta'], proposed['OF']))
    expected = {(meta, of) for meta in PROPOSED_METAS for of in OBJECTIVE_FUNCTIONS}
    if found != expected:
        raise ValueError(f'Wrong combinations: missing {sorted(expected - found)}, '
                         f'unexpected {sorted(found - expected)}')

    for (meta, of), group in history.groupby(COMBINATION):
        missing = set(CLASSIFIER_COMPLEXITY) - set(group['Classifier'])
        if missing:
            raise ValueError(f'{meta}-{of} is missing classifiers {sorted(missing)}')
        lengths = group['Length'].unique()
        if len(lengths) != 1:
            raise ValueError(f'{meta}-{of} has several subset lengths: {sorted(lengths)}')
        for classifier, scheme in group.groupby('Classifier'):
            if len(scheme) != N_FOLDS or scheme['Fold'].nunique() != N_FOLDS:
                raise ValueError(f'{meta}-{of}-{classifier} does not have {N_FOLDS} different folds')


def compare_with_best(folds):
    """One row per scheme: CV metrics and the comparison with the highest mean F1."""
    grouped = folds.groupby(SCHEME)
    table = grouped.agg(Length=('Length', 'first'), Settings=('Settings', 'first'),
                        F1=('F1', 'mean'), F1_SD=('F1', 'std'),
                        Sp=('Specificity', 'mean'), Sp_SD=('Specificity', 'std'),
                        Sn=('Recall', 'mean'), Sn_SD=('Recall', 'std'),
                        PRE=('Precision', 'mean'), PRE_SD=('Precision', 'std')).reset_index()
    f1_values = [scheme['F1'].to_numpy() for _, scheme in grouped]
    best = table['F1'].idxmax()
    others = [i for i in table.index if i != best]

    p_values = []
    rbc = np.zeros(len(table))
    for i in others:
        result = mannwhitneyu(f1_values[best], f1_values[i], alternative='two-sided', method='asymptotic')
        # SciPy returns NaN when all the values are equal
        p_values.append(1.0 if np.isnan(result.pvalue) else result.pvalue)
        rbc[i] = 1 - 2 * result.statistic / (len(f1_values[best]) * len(f1_values[i]))

    p_bh = np.ones(len(table))
    p_bh[others] = false_discovery_control(p_values, method='bh')

    table['p_BH'] = p_bh
    table['RBC'] = rbc
    table['Best_F1'] = table.index == best
    table['Tied'] = p_bh >= ALPHA
    return table


def tie_break_key(row, fewer_genes):
    """Sort key of steps 3-4 of the rule; the names at the end only make the order deterministic."""
    metrics = tuple(-round(row[metric], DECIMALS) for metric in ('Sp', 'Sn', 'PRE'))
    genes = (row['Length'],) if fewer_genes else ()
    return metrics + genes + (CLASSIFIER_COMPLEXITY[row['Classifier']], row['Meta'], row['OF'], row['Classifier'])


def apply_rule(folds, fewer_genes=False):
    """Apply the rule to the schemes in `folds`. Returns the table of compare_with_best, plus Selected."""
    table = compare_with_best(folds)
    tied = table[table['Tied']]
    selected = min(tied.index, key=lambda i: tie_break_key(tied.loc[i], fewer_genes))
    table['Selected'] = table.index == selected
    return table


def select_per_combination(history):
    """Step 1: one winner per metaheuristic x objective combination."""
    tables = [apply_rule(group) for _, group in history.groupby(COMBINATION)]
    return pd.concat(tables, ignore_index=True)


def select_final_model(history):
    """Steps 1 and 2. In the table of step 2, the Selected row is the final model."""
    step1 = select_per_combination(history)
    winners = step1[step1['Selected'] & step1['Meta'].isin(PROPOSED_METAS)]
    step2 = apply_rule(history.merge(winners[SCHEME], on=SCHEME), fewer_genes=True)
    return step1, step2


def load_subsets(paths):
    """Sorted gene indices per (Meta, OF). Raises ValueError if a combination has two different subsets."""
    subsets = {}
    for path in paths:
        table = pd.read_csv(path)
        for meta, of, subset in zip(table['Meta'], table['OF'], table['Subset']):
            genes = sorted(ast.literal_eval(subset))
            if subsets.setdefault((meta, of), genes) != genes:
                raise ValueError(f'{meta}-{of} has two different subsets ({path})')
    return subsets


def scheme_record(row, subsets):
    """A scheme with its genes. A wrong length means the subset file is not from the run that was evaluated."""
    meta, of = row['Meta'], row['OF']
    if (meta, of) not in subsets:
        raise ValueError(f'No subset for {meta}-{of}')
    subset = subsets[(meta, of)]
    if len(subset) != row['Length']:
        raise ValueError(f'{meta}-{of}: the CV results used {row["Length"]} genes, '
                         f'the subset file has {len(subset)}')
    return {
        'meta': meta,
        'objective': of,
        'classifier': row['Classifier'],
        'settings': row['Settings'],
        'length': int(row['Length']),
        'subset': subset,
        'cv': {metric: float(row[metric]) for metric in CV_METRICS},
    }


def build_final_model(step1, step2, subsets, inputs):
    """Content of final_model.json."""
    comparators = step1[step1['Selected'] & ~step1['Meta'].isin(PROPOSED_METAS)]
    return {
        'final_model': scheme_record(step2[step2['Selected']].iloc[0], subsets),
        'other_winners': [scheme_record(row, subsets) for _, row in step2[~step2['Selected']].iterrows()],
        'comparators': [scheme_record(row, subsets) for _, row in comparators.iterrows()],
        'gene_numbering': '0-based column index; manuscript gene number = index + 1',
        'rule': {'alpha': ALPHA, 'decimals': DECIMALS, 'n_folds': N_FOLDS,
                 'classifier_complexity': CLASSIFIER_COMPLEXITY},
        'inputs': inputs,
    }


def file_sha256(path):
    with open(path, 'rb') as file:
        return hashlib.sha256(file.read()).hexdigest()


def run(history_file, subset_files, output_dir):
    """Select the final model from the CSV files and write the two outputs."""
    loaded = load_history(history_file)
    history = keep_study_schemes(loaded)
    ignored = (set(loaded['Classifier']) - set(history['Classifier'])) | (set(loaded['OF']) - set(history['OF']))
    if ignored:
        print(f'Ignored, not in the study: {sorted(ignored)}')
    check_history(history)
    step1, step2 = select_final_model(history)
    inputs = {
        'history_file': str(history_file),
        'history_sha256': file_sha256(history_file),
        'subset_files': [{'path': str(path), 'sha256': file_sha256(path)} for path in subset_files],
    }
    frozen = build_final_model(step1, step2, load_subsets(subset_files), inputs)

    os.makedirs(output_dir, exist_ok=True)
    table = pd.concat([step1.assign(Step=1), step2.assign(Step=2)], ignore_index=True)
    table.to_csv(os.path.join(output_dir, 'selection_table.csv'), index=False)
    with open(os.path.join(output_dir, 'final_model.json'), 'w') as file:
        json.dump(frozen, file, indent=2)
    return step2, frozen


if __name__ == "__main__":
    # Preview on the submitted results. They are affected by E.1, so this is not the frozen choice.
    # To freeze: point HISTORY_FILE to the corrected classifier run, make sure SUBSET_FILES covers
    # every comparator in it, and set OUTPUT_DIR = 'protocol'.
    HISTORY_FILE = 'statistic_test/results/lasso_full_history.csv'
    SUBSET_FILES = [
        '/mnt/Cris-SW/Research/MetaheuristicExperiment/experiment_results/R3/class_step_3_report_train_normal_rtf.csv',
        'results/top10_baseline/top10_subsets.csv',
    ]
    OUTPUT_DIR = 'results/model_selection_preview'

    step2, frozen = run(HISTORY_FILE, SUBSET_FILES, OUTPUT_DIR)

    columns = ['Meta', 'OF', 'Classifier', 'Length', 'F1', 'Sp', 'Sn', 'PRE', 'p_BH', 'Tied', 'Selected']
    print(step2.sort_values('F1', ascending=False)[columns].round(3).to_string(index=False))
    final = frozen['final_model']
    print(f"\nFinal model: {final['meta']}-{final['objective']} + {final['classifier']}, "
          f"{final['length']} genes (0-based): {final['subset']}")
    print(f'Written to {OUTPUT_DIR}')
