"""
Tests for select_final_model.py: choose the final model from training CV results only
(editor comment E.2).

The unit tests build fold-level results by hand. The tests at the end use the submitted CV
results and are skipped when those files are missing.
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from select_final_model import (apply_rule, build_final_model, check_history, keep_study_schemes,
                                load_history, load_subsets, run, select_final_model, select_per_combination)

SUBMITTED_HISTORY = 'statistic_test/results/lasso_full_history.csv'
SUBMITTED_SUBSETS = ('/mnt/Cris-SW/Research/MetaheuristicExperiment/experiment_results/R3/'
                     'class_step_3_report_train_normal_rtf.csv')

GRID = ['Naive bayes', 'LDA', 'Knn k = 3', 'Knn k = 5', 'Knn k = 7', 'SVC poly',
        'RF estimators: 60', 'RF estimators: 80', 'RF estimators: 100']
METAS = ['DE', 'GA', 'SA']
OBJECTIVES = ['relieff', 'mi', 'sim_uncertainty', 'gain_ratio', 'chi2']

# Spread of one scheme's F1 across its 30 folds. It has mean 0, so a scheme's mean F1 is exactly `f1`.
PATTERN = np.linspace(-0.02, 0.02, 30)


def scheme(classifier='LDA', f1=0.70, sp=0.60, sn=0.70, pre=0.70, meta='DE', of='mi', length=10):
    """30 fold rows of one scheme, in the layout of classifiers.writeHistory."""
    return pd.DataFrame({
        'Meta': meta, 'OF': of, 'Generation': 100, 'Length': length, 'Settings': 'Pop: 35',
        'Classifier': classifier, 'Fold': range(1, 31), 'AUC': 0.7, 'Accuracy': 0.7, 'Precision': pre,
        'MSE': 0.3, 'Recall': sn, 'F1': f1 + PATTERN, 'MCC': 0.4, 'Specificity': sp,
    })


def folds(*schemes):
    return pd.concat(schemes, ignore_index=True)


def winner(table):
    selected = table[table['Selected']]
    assert len(selected) == 1
    return selected.iloc[0]


def row_of(table, classifier):
    return table[table['Classifier'] == classifier].iloc[0]


def make_history(best=('GA', 'relieff', 'Knn k = 3'), comparators=()):
    """Complete study: 3 metaheuristics x 5 objectives x 9 classifiers. `best` clearly has the top F1."""
    rows = []
    for meta in METAS:
        for of in OBJECTIVES:
            for classifier in GRID:
                f1 = 0.90 if (meta, of, classifier) == best else 0.70
                rows.append(scheme(classifier, f1=f1, meta=meta, of=of))
    for meta, of in comparators:
        rows.extend(scheme(classifier, meta=meta, of=of) for classifier in GRID)
    return folds(*rows)


def make_subsets(history):
    """One 10-gene subset per combination; every subset is different."""
    subsets = {}
    for i, key in enumerate(sorted(set(zip(history['Meta'], history['OF'])))):
        subsets[key] = list(range(i, i + 10))
    return subsets


# --- The rule: highest mean F1, tie test, then Sp, Sn, PRE, complexity ------------------------


def test_highest_f1_wins_when_no_scheme_is_tied():
    table = apply_rule(folds(scheme('RF estimators: 60', f1=0.90, sp=0.50),
                             scheme('LDA', f1=0.70, sp=0.90)))
    assert winner(table)['Classifier'] == 'RF estimators: 60'


def test_tied_scheme_with_higher_sp_beats_the_highest_f1():
    table = apply_rule(folds(scheme('RF estimators: 60', f1=0.701, sp=0.60, sn=0.90),
                             scheme('RF estimators: 80', f1=0.700, sp=0.70, sn=0.70)))
    assert winner(table)['Classifier'] == 'RF estimators: 80'


def test_sp_equal_at_two_decimals_passes_to_sn():
    table = apply_rule(folds(scheme('RF estimators: 60', sp=0.604, sn=0.70, pre=0.90),
                             scheme('RF estimators: 80', sp=0.601, sn=0.80, pre=0.60)))
    assert winner(table)['Classifier'] == 'RF estimators: 80'


def test_sn_equal_at_two_decimals_passes_to_pre():
    table = apply_rule(folds(scheme('RF estimators: 60', sn=0.704, pre=0.60),
                             scheme('RF estimators: 80', sn=0.701, pre=0.80)))
    assert winner(table)['Classifier'] == 'RF estimators: 80'


@pytest.mark.parametrize('simpler, more_complex', [
    ('LDA', 'Knn k = 3'),
    ('Naive bayes', 'Knn k = 3'),
    ('Knn k = 3', 'Knn k = 5'),
    ('Knn k = 5', 'Knn k = 7'),
    ('Knn k = 7', 'SVC poly'),
    ('SVC poly', 'RF estimators: 60'),
    ('RF estimators: 60', 'RF estimators: 80'),
    ('RF estimators: 80', 'RF estimators: 100'),
])
def test_simpler_classifier_wins_when_all_metrics_are_equal(simpler, more_complex):
    table = apply_rule(folds(scheme(more_complex), scheme(simpler)))
    assert winner(table)['Classifier'] == simpler


def test_equal_complexity_gives_the_same_winner_in_any_input_order():
    first = apply_rule(folds(scheme('LDA'), scheme('Naive bayes')))
    second = apply_rule(folds(scheme('Naive bayes'), scheme('LDA')))
    assert winner(first)['Classifier'] == winner(second)['Classifier'] == 'LDA'


def test_tie_uses_the_bh_adjusted_p_value():
    # Raw p of 'RF estimators: 60' against the best is 0.0271: significant alone, not after BH
    table = apply_rule(folds(scheme('LDA', f1=0.700),
                             scheme('RF estimators: 60', f1=0.693),
                             scheme('RF estimators: 80', f1=0.698)))
    compared = row_of(table, 'RF estimators: 60')
    # 0.0271 x 2 comparisons: the best scheme is not compared with itself
    assert compared['p_BH'] == pytest.approx(0.0542, abs=1e-3)
    assert bool(compared['Tied'])


def test_scheme_significantly_below_the_best_is_not_tied():
    table = apply_rule(folds(scheme('LDA', f1=0.90), scheme('RF estimators: 60', f1=0.70)))
    assert not row_of(table, 'RF estimators: 60')['Tied']


def test_best_scheme_is_reported_as_in_tables_4_to_6():
    table = apply_rule(folds(scheme('LDA', f1=0.90), scheme('RF estimators: 60', f1=0.70)))
    best = row_of(table, 'LDA')
    assert bool(best['Best_F1']) and bool(best['Tied'])
    assert best['p_BH'] == 1.0
    assert best['RBC'] == 0.0


def test_rbc_is_minus_one_when_every_fold_is_below_the_best():
    table = apply_rule(folds(scheme('LDA', f1=0.90), scheme('RF estimators: 60', f1=0.70)))
    assert row_of(table, 'RF estimators: 60')['RBC'] == -1.0


def test_identical_constant_f1_counts_as_tied():
    # SciPy returns NaN when all the values are equal
    first, second = scheme('LDA'), scheme('RF estimators: 60', sp=0.80)
    first['F1'] = second['F1'] = 0.7
    table = apply_rule(folds(first, second))
    assert winner(table)['Classifier'] == 'RF estimators: 60'


def test_a_single_candidate_is_selected():
    assert winner(apply_rule(scheme('LDA')))['Classifier'] == 'LDA'


def test_means_and_sd_are_reported():
    table = apply_rule(scheme('LDA', f1=0.70, sp=0.60, sn=0.75, pre=0.80))
    row = table.iloc[0]
    assert (row['F1'], row['Sp'], row['Sn'], row['PRE']) == pytest.approx((0.70, 0.60, 0.75, 0.80))
    # Sample SD of 30 evenly spaced values in [-0.02, 0.02]; the population SD would be 0.01194
    assert row['F1_SD'] == pytest.approx(0.01214, abs=1e-4)


# --- Step 1 and step 2 ------------------------------------------------------------------------


def test_step_1_selects_one_winner_per_combination():
    table = select_per_combination(folds(
        scheme('LDA', f1=0.90, meta='DE', of='mi'), scheme('RF estimators: 60', f1=0.70, meta='DE', of='mi'),
        scheme('LDA', f1=0.70, meta='GA', of='mi'), scheme('RF estimators: 60', f1=0.90, meta='GA', of='mi')))
    winners = table[table['Selected']]
    assert set(zip(winners['Meta'], winners['Classifier'])) == {('DE', 'LDA'), ('GA', 'RF estimators: 60')}


def test_step_2_only_compares_the_step_1_winners():
    # GA's LDA has the best Sp overall, but it loses inside its own combination
    _, step2 = select_final_model(folds(
        scheme('LDA', f1=0.90, sp=0.60, meta='DE', of='mi'),
        scheme('RF estimators: 60', f1=0.89, sp=0.60, meta='GA', of='mi'),
        scheme('LDA', f1=0.70, sp=0.95, meta='GA', of='mi')))
    assert set(zip(step2['Meta'], step2['Classifier'])) == {('DE', 'LDA'), ('GA', 'RF estimators: 60')}


def test_step_2_final_model_can_differ_from_the_highest_f1():
    _, step2 = select_final_model(folds(
        scheme('LDA', f1=0.701, sp=0.60, meta='DE', of='mi'),
        scheme('LDA', f1=0.700, sp=0.70, meta='GA', of='mi')))
    assert winner(step2)['Meta'] == 'GA'
    assert row_of(step2[step2['Meta'] == 'DE'], 'LDA')['Best_F1']


def test_step_2_prefers_fewer_genes_before_the_simpler_classifier():
    _, step2 = select_final_model(folds(
        scheme('LDA', length=18, meta='DE', of='mi'),
        scheme('RF estimators: 100', length=10, meta='GA', of='mi')))
    assert winner(step2)['Meta'] == 'GA'


def test_step_1_does_not_use_the_number_of_genes():
    table = select_per_combination(folds(scheme('RF estimators: 100', length=10), scheme('LDA', length=18)))
    assert winner(table)['Classifier'] == 'LDA'


def test_comparators_get_a_step_1_winner_but_cannot_be_the_final_model():
    step1, step2 = select_final_model(folds(
        scheme('LDA', f1=0.70, meta='DE', of='mi'),
        scheme('LDA', f1=0.95, meta='TOP10', of='mi')))
    assert winner(step1[step1['Meta'] == 'TOP10'])['Classifier'] == 'LDA'
    assert set(step2['Meta']) == {'DE'}


# --- Loading and checking the fold results ----------------------------------------------------


@pytest.mark.parametrize('column', ['Specificity', 'Spec'])
def test_load_history_reads_both_specificity_headers(tmp_path, column):
    # The submitted file says Specificity; pipeline_classifiers.py writes Spec
    path = tmp_path / 'history.csv'
    scheme('LDA', sp=0.61).rename(columns={'Specificity': column}).to_csv(path, index=False)
    assert load_history(path)['Specificity'].iloc[0] == 0.61


def test_load_history_rejects_a_missing_column(tmp_path):
    path = tmp_path / 'history.csv'
    scheme('LDA').drop(columns=['Recall']).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Recall'):
        load_history(path)


def test_keep_study_schemes_drops_classifiers_outside_the_paper_grid():
    kept = keep_study_schemes(folds(scheme('LDA'), scheme('Lasso'), scheme('Knn k = 9'),
                                    scheme('RF estimators: 7')))
    assert set(kept['Classifier']) == {'LDA'}


def test_keep_study_schemes_drops_the_sixth_objective_only_for_metaheuristics():
    kept = keep_study_schemes(folds(scheme('LDA', meta='DE', of='mi'),
                                    scheme('LDA', meta='DE', of='information_gain'),
                                    scheme('LDA', meta='ALL70', of='none')))
    assert set(zip(kept['Meta'], kept['OF'])) == {('DE', 'mi'), ('ALL70', 'none')}


def test_check_history_accepts_a_complete_study():
    check_history(make_history(comparators=[('TOP10', 'mi'), ('ALL70', 'none')]))


def test_check_history_rejects_a_missing_combination():
    history = make_history()
    history = history[~((history['Meta'] == 'SA') & (history['OF'] == 'chi2'))]
    with pytest.raises(ValueError, match='combinations'):
        check_history(history)


def test_check_history_rejects_a_missing_classifier():
    # For example, RF is commented out in classifiers.pipeline_updated
    history = make_history()
    history = history[~((history['Meta'] == 'DE') & (history['OF'] == 'mi')
                        & (history['Classifier'] == 'RF estimators: 60'))]
    with pytest.raises(ValueError, match='classifiers'):
        check_history(history)


def test_check_history_rejects_a_comparator_without_the_whole_grid():
    history = make_history(comparators=[('TOP10', 'mi')])
    history = history[~((history['Meta'] == 'TOP10') & (history['Classifier'] == 'LDA'))]
    with pytest.raises(ValueError, match='classifiers'):
        check_history(history)


def test_check_history_rejects_a_missing_fold():
    history = make_history()
    with pytest.raises(ValueError, match='folds'):
        check_history(history.iloc[1:])


def test_check_history_rejects_a_repeated_fold():
    history = make_history()
    history.loc[history.index[0], 'Fold'] = 2
    with pytest.raises(ValueError, match='folds'):
        check_history(history)


def test_check_history_rejects_two_subset_lengths_in_one_combination():
    history = make_history()
    history.loc[(history['Meta'] == 'DE') & (history['OF'] == 'mi') & (history['Classifier'] == 'LDA'),
                'Length'] = 1
    with pytest.raises(ValueError, match='lengths'):
        check_history(history)


# --- Subsets and the frozen file --------------------------------------------------------------


def test_load_subsets_reads_one_subset_per_combination_from_several_files(tmp_path):
    first, second = tmp_path / 'first.csv', tmp_path / 'second.csv'
    pd.DataFrame({'Meta': ['DE', 'DE'], 'OF': ['mi', 'mi'], 'Subset': ['[3, 1, 2]', '[3, 1, 2]']}).to_csv(first)
    pd.DataFrame({'Meta': ['TOP10'], 'OF': ['mi'], 'Subset': ['[5, 4]']}).to_csv(second)
    assert load_subsets([first, second]) == {('DE', 'mi'): [1, 2, 3], ('TOP10', 'mi'): [4, 5]}


def test_load_subsets_rejects_two_subsets_for_one_combination(tmp_path):
    path = tmp_path / 'subsets.csv'
    pd.DataFrame({'Meta': ['DE', 'DE'], 'OF': ['mi', 'mi'], 'Subset': ['[1, 2]', '[1, 3]']}).to_csv(path)
    with pytest.raises(ValueError, match='DE'):
        load_subsets([path])


def test_build_final_model_holds_the_frozen_scheme_and_its_genes():
    history = make_history(comparators=[('TOP10', 'mi')])
    subsets = make_subsets(history)
    subsets[('GA', 'relieff')] = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    step1, step2 = select_final_model(history)
    frozen = build_final_model(step1, step2, subsets, inputs={'history_sha256': 'abc'})

    final = frozen['final_model']
    assert (final['meta'], final['objective'], final['classifier']) == ('GA', 'relieff', 'Knn k = 3')
    assert final['subset'] == [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    assert final['length'] == 10
    assert final['cv']['F1'] == pytest.approx(0.90)
    assert len(frozen['other_winners']) == 14
    assert [c['meta'] for c in frozen['comparators']] == ['TOP10']
    assert frozen['inputs'] == {'history_sha256': 'abc'}


def test_build_final_model_rejects_a_subset_with_the_wrong_length():
    # The stale resumes/ file has 1-3-gene subsets where the CV results used 10 genes
    history = make_history()
    subsets = make_subsets(history)
    subsets[('DE', 'mi')] = [33]
    step1, step2 = select_final_model(history)
    with pytest.raises(ValueError, match="DE.*mi"):
        build_final_model(step1, step2, subsets, inputs={})


def test_build_final_model_rejects_a_missing_subset():
    history = make_history()
    subsets = make_subsets(history)
    del subsets[('SA', 'chi2')]
    step1, step2 = select_final_model(history)
    with pytest.raises(ValueError, match="SA.*chi2"):
        build_final_model(step1, step2, subsets, inputs={})


def write_inputs(tmp_path, history):
    history_file = tmp_path / 'history.csv'
    history.to_csv(history_file, index=False)
    subsets_file = tmp_path / 'subsets.csv'
    pd.DataFrame([{'Meta': m, 'OF': o, 'Subset': str(s)} for (m, o), s in make_subsets(history).items()]
                 ).to_csv(subsets_file, index=False)
    return history_file, subsets_file


def test_run_says_which_schemes_it_ignored(tmp_path, capsys):
    history = folds(make_history(), scheme('Lasso', meta='DE', of='mi'),
                    scheme('LDA', meta='DE', of='information_gain'))
    history_file, subsets_file = write_inputs(tmp_path, history)

    run(history_file, [subsets_file], tmp_path / 'out')

    printed = capsys.readouterr().out
    assert 'Lasso' in printed
    assert 'information_gain' in printed


def test_run_says_nothing_when_no_scheme_is_ignored(tmp_path, capsys):
    history_file, subsets_file = write_inputs(tmp_path, make_history())

    run(history_file, [subsets_file], tmp_path / 'out')

    assert 'Ignored' not in capsys.readouterr().out


def test_run_writes_the_frozen_file_and_the_table_from_the_csv_files(tmp_path):
    # A Lasso scheme with a perfect score must be ignored: it is not in the paper's grid
    history = folds(make_history(), scheme('Lasso', f1=0.97, meta='GA', of='relieff'))
    history_file, subsets_file = write_inputs(tmp_path, history)
    output_dir = tmp_path / 'protocol'

    run(history_file, [subsets_file], output_dir)

    frozen = json.loads((output_dir / 'final_model.json').read_text())
    assert (frozen['final_model']['meta'], frozen['final_model']['classifier']) == ('GA', 'Knn k = 3')
    assert frozen['inputs']['history_sha256'] == hashlib.sha256(history_file.read_bytes()).hexdigest()

    table = pd.read_csv(output_dir / 'selection_table.csv')
    assert (table['Step'] == 1).sum() == 135  # 15 combinations x 9 classifiers
    assert (table['Step'] == 2).sum() == 15
    assert table[(table['Step'] == 2) & table['Selected']]['Meta'].tolist() == ['GA']


# --- The submitted CV results -----------------------------------------------------------------

needs_submitted_history = pytest.mark.skipif(not os.path.exists(SUBMITTED_HISTORY),
                                             reason='submitted CV results not on disk')

TABLE_8_WINNERS = {
    ('DE', 'mi'): 'Naive bayes', ('DE', 'sim_uncertainty'): 'RF estimators: 60',
    ('DE', 'gain_ratio'): 'RF estimators: 60', ('DE', 'relieff'): 'Knn k = 3',
    ('DE', 'chi2'): 'RF estimators: 100',
    ('GA', 'mi'): 'RF estimators: 60', ('GA', 'sim_uncertainty'): 'RF estimators: 80',
    ('GA', 'gain_ratio'): 'RF estimators: 80', ('GA', 'relieff'): 'Knn k = 3', ('GA', 'chi2'): 'SVC poly',
    ('SA', 'mi'): 'RF estimators: 100', ('SA', 'sim_uncertainty'): 'RF estimators: 60',
    ('SA', 'gain_ratio'): 'RF estimators: 80', ('SA', 'relieff'): 'Knn k = 5',
    ('SA', 'chi2'): 'RF estimators: 100',
}


@pytest.fixture
def submitted():
    history = keep_study_schemes(load_history(SUBMITTED_HISTORY))
    check_history(history)
    return select_final_model(history)


@needs_submitted_history
def test_submitted_results_match_table_8_except_two_hand_selection_slips(submitted):
    step1, _ = submitted
    winners = step1[step1['Selected']]
    selected = {(m, o): c for m, o, c in zip(winners['Meta'], winners['OF'], winners['Classifier'])}
    differences = {key: selected[key] for key in TABLE_8_WINNERS if selected[key] != TABLE_8_WINNERS[key]}
    # GA-aSU: RF(100) has the higher PRE. SA-aCHI2: SVM(Poly) is tied and has the higher Sp (Table 6).
    assert differences == {('GA', 'sim_uncertainty'): 'RF estimators: 100', ('SA', 'chi2'): 'SVC poly'}


@needs_submitted_history
def test_submitted_results_select_ga_relieff_knn3(submitted):
    _, step2 = submitted
    final = winner(step2)
    assert (final['Meta'], final['OF'], final['Classifier'], final['Length']) == ('GA', 'relieff', 'Knn k = 3', 11)
    tied = step2[step2['Tied']]
    assert set(zip(tied['Meta'], tied['OF'])) == {('DE', 'relieff'), ('GA', 'relieff'), ('SA', 'relieff')}


@needs_submitted_history
def test_submitted_headline_model_is_not_tied_with_the_best(submitted):
    _, step2 = submitted
    headline = step2[(step2['Meta'] == 'DE') & (step2['OF'] == 'sim_uncertainty')].iloc[0]
    assert headline['Classifier'] == 'RF estimators: 60'
    assert not headline['Tied']


@needs_submitted_history
@pytest.mark.skipif(not os.path.exists(SUBMITTED_SUBSETS), reason='submitted subsets not on disk')
def test_submitted_final_model_holds_the_table_3_genes(submitted):
    step1, step2 = submitted
    frozen = build_final_model(step1, step2, load_subsets([SUBMITTED_SUBSETS]), inputs={})
    # Table 3, GA-aReliefF: g3, g7, g21, g23, g29, g38, g39, g50, g51, g58, g68 (minus 1: 0-based)
    assert frozen['final_model']['subset'] == [2, 6, 20, 22, 28, 37, 38, 49, 50, 57, 67]
