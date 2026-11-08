import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = 'charts/jaccard'

OF_LABELS = {
    'chi2':             'χ²',
    'gain_ratio':       'Gain Ratio',
    'information_gain': 'Info. Gain',
    'mi':               'Mutual Info.',
    'relieff':          'ReliefF',
    'sim_uncertainty':  'Sym. Uncert.',
}

def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)

def get_subset(df, model, name):
    row = df[(df['Model'] == model) & (df['Name'] == name)]
    return ast.literal_eval(row.iloc[0]['Subset'])

if __name__ == '__main__':
    # --- Load data ---
    r1 = pd.read_csv('experiment_results/R1/subset_r1.csv')
    r2 = pd.read_csv('experiment_results/R2/subset_r2.csv')
    r3 = pd.read_csv('experiment_results/R3/subset_r3.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Stability across rounds ────────────────────────────────────────────────
    combinations = r1[['Model', 'Name']].drop_duplicates().values.tolist()

    rows = []
    for model, name in combinations:
        s1 = get_subset(r1, model, name)
        s2 = get_subset(r2, model, name)
        s3 = get_subset(r3, model, name)
        rows.append({
            'Model': model,
            'OF': OF_LABELS.get(name, name),
            'J(R1,R1)': round(jaccard(s1, s1), 4),
            'J(R1,R2)': round(jaccard(s1, s2), 4),
            'J(R2,R3)': round(jaccard(s2, s3), 4),
        })

    stability = pd.DataFrame(rows)
    print(stability.to_string(index=False))
    stability.to_csv(f'{OUTPUT_DIR}/jaccard_stability.csv', index=False)

    # # Plot grouped bar chart
    # fig, ax = plt.subplots(figsize=(14, 6))
    # x = np.arange(len(stability))
    # width = 0.28
    # labels = [f"{r['Model']}\n{r['OF']}" for _, r in stability.iterrows()]
    # ax.bar(x - width, stability['J(R1,R2)'], width, label='J(R1, R2)', color='#4472C4')
    # ax.bar(x,         stability['J(R2,R3)'], width, label='J(R2, R3)', color='#70AD47')
    # ax.bar(x + width, stability['J(R1,R3)'], width, label='J(R1, R3)', color='#FF0000', alpha=0.7)
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels, fontsize=9)
    # ax.set_ylabel('Jaccard Similarity', fontsize=12)
    # ax.set_title('Feature Subset Stability Across Rounds', fontsize=14)
    # ax.set_ylim(0, 1.05)
    # ax.legend()
    # ax.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{OUTPUT_DIR}/jaccard_stability.png', dpi=300)
    # plt.close()

    # # ── 2. Pairwise heatmap within each round ─────────────────────────────────────
    # def pairwise_heatmap(df, round_label):
    #     combos = df[['Model', 'Name']].drop_duplicates().values.tolist()
    #     keys   = [f"{m}\n{OF_LABELS.get(n, n)}" for m, n in combos]
    #     n = len(combos)
    #     matrix = np.zeros((n, n))
    #     for i, (m1, n1) in enumerate(combos):
    #         s1 = ast.literal_eval(df[(df['Model']==m1) & (df['Name']==n1)].iloc[0]['Subset'])
    #         for j, (m2, n2) in enumerate(combos):
    #             s2 = ast.literal_eval(df[(df['Model']==m2) & (df['Name']==n2)].iloc[0]['Subset'])
    #             matrix[i, j] = jaccard(s1, s2)

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     sns.heatmap(matrix, annot=True, fmt='.2f', xticklabels=keys, yticklabels=keys,
    #                 cmap='YlOrRd', vmin=0, vmax=1, ax=ax, linewidths=0.5)
    #     ax.set_title(f'Pairwise Jaccard Similarity — {round_label}', fontsize=14)
    #     plt.tight_layout()
    #     plt.savefig(f'{OUTPUT_DIR}/jaccard_heatmap_{round_label}.png', dpi=300)
    #     plt.close()
    #     print(f'Saved heatmap for {round_label}')

    # pairwise_heatmap(r1, 'R1')
    # pairwise_heatmap(r2, 'R2')
    # pairwise_heatmap(r3, 'R3')

    print(f'\nDone. All outputs saved to {OUTPUT_DIR}/')