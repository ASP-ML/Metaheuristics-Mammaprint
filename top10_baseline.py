"""
Top-10 filter-ranking baseline (editor comment E.4)

1. Ranks the 70 genes once per objective on the file the reducer searched and keeps the top 10.
2. Compares each metaheuristic subset with the top 10 of its objective: |S|, intersection,
   Jaccard and Eq. (11) value.

Outputs (OUTPUT_DIR):
    top10_subsets.csv  top-10 subsets in the resume layout read by pipeline_classifiers.py
                       (Meta = TOP10), ready for the classifier run
    top10_vs_mh.csv    one row per metaheuristic x objective

Sanity check: for separable objectives, among subsets with at least K genes, the top-10 merit
must be >= the metaheuristic merit.
"""

import ast
import os

import numpy as np
import pandas as pd
import weka.core.jvm as jvm

import ofunctions
from baselines.top_k_filter import OBJECTIVE_FUNCTIONS, SEPARABLE_OBJECTIVES, top_k
from jaccard import jaccard

MERIT_TOLERANCE = 1e-12


def load_data(path):
    # Same as utils.processDataSet, which cannot be imported without TensorFlow
    data = pd.read_csv(path)
    labels = data['label'].to_numpy(dtype=np.intp)
    return data.drop(columns=['label']).to_numpy(), labels


def top_k_subsets(data, labels, objective_functions, k):
    """One row per objective; the first six columns follow the resume layout of pipeline_classifiers.py."""
    rows = []
    for of in objective_functions:
        subset = top_k(data, labels, of, k)
        rows.append({
            'Meta': f'TOP{k}',
            'OF': of,
            'IT': 0,
            'Subset': str(subset),
            'Length': len(subset),
            'Settings': f'Ranking: {of} Ties: lower index Evals: 1',
            'Merit': ofunctions.allObjectiveFunctionWeka(subset, data, labels, type_of=of),
        })
    return pd.DataFrame(rows)


def compare(data, labels, top_df, mh_df, k):
    """One row per metaheuristic subset against the top-k subset of the same objective."""
    top_by_of = top_df.set_index('OF')
    mh_df = mh_df[mh_df['OF'].isin(top_by_of.index)].drop_duplicates(['Meta', 'OF', 'Subset'])
    rows = []
    for _, mh in mh_df.iterrows():
        of = mh['OF']
        top = top_by_of.loc[of]
        mh_subset = ast.literal_eval(mh['Subset'])
        top_subset = ast.literal_eval(top['Subset'])
        merit_mh = ofunctions.allObjectiveFunctionWeka(mh_subset, data, labels, type_of=of)
        merit_top = float(top['Merit'])
        # The top-k is the optimum only for separable objectives and subsets with at least k genes
        judged = of in SEPARABLE_OBJECTIVES and len(mh_subset) >= k
        rows.append({
            'OF': of,
            'Separable': of in SEPARABLE_OBJECTIVES,
            'Meta': mh['Meta'],
            'Length_MH': len(mh_subset),
            'Length_Top': len(top_subset),
            'Intersection': len(set(mh_subset) & set(top_subset)),
            'Jaccard': jaccard(mh_subset, top_subset),
            'Merit_MH': merit_mh,
            'Merit_Top': merit_top,
            # Positive: the top-k subset has the higher Eq. (11) value
            'Merit_gap': merit_top - merit_mh,
            'Top_ge_MH': (merit_top >= merit_mh - MERIT_TOLERANCE) if judged else None,
            'MH_optimal': (abs(merit_top - merit_mh) <= MERIT_TOLERANCE) if judged else None,
        })
    return pd.DataFrame(rows)


def print_sanity_check(comparison, k):
    judged = comparison[comparison['Top_ge_MH'].notna()]
    failures = judged[~judged['Top_ge_MH'].astype(bool)]
    optimal = judged[judged['MH_optimal'].astype(bool)]
    skipped = comparison[comparison['Separable'] & (comparison['Length_MH'] < k)]
    print(f'\nSeparable objectives, MH subsets with >= {k} genes: top-{k} merit >= MH merit in '
          f'{len(judged) - len(failures)}/{len(judged)}; MH reached the top-{k} optimum in {len(optimal)}/{len(judged)} '
          f'({", ".join(optimal["Meta"] + "-" + optimal["OF"]) or "none"}).')
    if len(skipped) > 0:
        print(f'Not judged (fewer than {k} genes): ' + ', '.join(skipped['Meta'] + '-' + skipped['OF']))
    if len(failures) > 0:
        print('SANITY CHECK FAILED for: ' + ', '.join(failures['Meta'] + '-' + failures['OF']))


if __name__ == "__main__":
    DATA_FILE = 'train_normal_rtf.csv'
    MH_SUBSETS_FILE = 'resumes/resume_class_step_3_report_train_normal_rtf.csv'
    K = 10
    OUTPUT_DIR = 'results/top10_baseline'

    data, labels = load_data(f'data/{DATA_FILE}')
    mh_df = pd.read_csv(MH_SUBSETS_FILE)
    jvm.start(packages=True)
    try:
        top_df = top_k_subsets(data, labels, OBJECTIVE_FUNCTIONS, K)
        comparison = compare(data, labels, top_df, mh_df, K)
    finally:
        # Write only after the JVM stops: while it runs, JPype's import hook can break the lazy
        # imports pandas does inside to_csv (seen with jlink-trimmed Java runtimes)
        jvm.stop()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    top_df.to_csv(f'{OUTPUT_DIR}/top{K}_subsets.csv', index=False)
    comparison.to_csv(f'{OUTPUT_DIR}/top{K}_vs_mh.csv', index=False)
    print(top_df[['OF', 'Subset', 'Merit']].to_string(index=False))
    print()
    print(comparison.to_string(index=False))
    print_sanity_check(comparison, K)
