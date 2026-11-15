"""
Tests for confidence_intervals.py: 95% confidence intervals and paired comparisons on the test
set (editor comment E.3).

All the tests use made-up patients. The expected values were worked out separately from the
script: the Wilson formula written out by hand, binomial probabilities, and exact McNemar p-values.
"""

import json

import numpy as np
import pandas as pd
import pytest

from confidence_intervals import (ci_table, compare_models, draw_resamples, format_ci, load_predictions,
                                  mcnemar_exact, model_name, models_to_compare, predictions_by_model, run,
                                  wilson_ci)

# 10 positive and 20 negative patients, as in the corrected test set
N_POS, N_NEG = 10, 20
Y_TRUE = np.array([1] * N_POS + [0] * N_NEG)


def predicted(tp, tn):
    """Predictions with `tp` positives and `tn` negatives right; the first patients of each class are the right ones."""
    positives = [1] * tp + [0] * (N_POS - tp)
    negatives = [0] * tn + [1] * (N_NEG - tn)
    return np.array(positives + negatives)


def model_rows(model, y_pred, y_true=Y_TRUE):
    return pd.DataFrame({'Model': model, 'Patient': [f'P{i:02d}' for i in range(len(y_true))],
                         'y_true': y_true, 'y_pred': y_pred})


def row(table, metric, column='Model', name=None):
    selected = table[table['Metric'] == metric]
    if name is not None:
        selected = selected[selected[column] == name]
    assert len(selected) == 1
    return selected.iloc[0]


# --- Wilson intervals -------------------------------------------------------------------------


@pytest.mark.parametrize('successes, total, low, high', [
    (18, 20, 0.6990, 0.9721),
    (9, 10, 0.5958, 0.9821),
    (17, 20, 0.6396, 0.9476),
    (0, 10, 0.0, 0.2775),
    (10, 10, 0.7225, 1.0),
])
def test_wilson_ci_matches_the_formula(successes, total, low, high):
    assert wilson_ci(successes, total) == pytest.approx((low, high), abs=1e-4)


def test_wilson_ci_is_undefined_without_cases():
    # Precision of a model that predicts no positives: 0 of 0
    assert np.isnan(wilson_ci(0, 0)).all()


# --- Table 7: one interval per metric and model -------------------------------------------------


@pytest.fixture
def one_model_table():
    # TP 9, FN 1, TN 17, FP 3
    y_pred = predicted(tp=9, tn=17)
    return ci_table(Y_TRUE, {'A': y_pred}, draw_resamples(Y_TRUE))


@pytest.mark.parametrize('metric, value, low, high', [
    ('Sn', 0.9000, 0.5958, 0.9821),    # 9 of 10 positives
    ('Sp', 0.8500, 0.6396, 0.9476),    # 17 of 20 negatives
    ('PRE', 0.7500, 0.4677, 0.9111),   # 9 of 12 predicted positive
    ('ACC', 0.8667, 0.7032, 0.9469),   # 26 of 30 patients
])
def test_count_metrics_use_the_right_numerator_and_denominator(one_model_table, metric, value, low, high):
    result = row(one_model_table, metric)
    assert (result['Value'], result['Low'], result['High']) == pytest.approx((value, low, high), abs=1e-4)
    assert 'Wilson' in result['Method']


def test_f1_uses_the_bootstrap(one_model_table):
    # Value: 2 x 9 / (2 x 9 + 3 + 1). Bounds: worked out from the two binomial counts, without this script.
    result = row(one_model_table, 'F1')
    assert result['Value'] == pytest.approx(18 / 22)
    assert (result['Low'], result['High']) == pytest.approx((0.667, 0.952), abs=1e-3)
    assert 'bootstrap' in result['Method']


def test_table_has_the_counts_and_the_text_for_the_paper(one_model_table):
    result = row(one_model_table, 'Sn')
    assert (result['TP'], result['FN'], result['TN'], result['FP']) == (9, 1, 17, 3)
    assert result['Text'] == '0.90 [0.60, 0.98]'


def test_perfect_model_has_f1_of_one_in_every_resample():
    result = row(ci_table(Y_TRUE, {'A': Y_TRUE}, draw_resamples(Y_TRUE)), 'F1')
    assert (result['Value'], result['Low'], result['High']) == (1.0, 1.0, 1.0)


def test_precision_is_not_available_when_nothing_is_predicted_positive():
    result = row(ci_table(Y_TRUE, {'A': np.zeros(30, dtype=int)}, draw_resamples(Y_TRUE)), 'PRE')
    assert np.isnan(result['Value'])
    assert result['Text'] == 'n/a'


def test_every_model_gets_all_five_metrics():
    table = ci_table(Y_TRUE, {'A': predicted(9, 17), 'B': predicted(8, 15)}, draw_resamples(Y_TRUE))
    assert set(zip(table['Model'], table['Metric'])) == {
        (model, metric) for model in 'AB' for metric in ['F1', 'PRE', 'Sn', 'Sp', 'ACC']}


# --- Resampling -------------------------------------------------------------------------------


def test_each_resample_keeps_the_number_of_positives_and_negatives():
    positives, negatives = draw_resamples(Y_TRUE, resamples=50)
    assert positives.shape == (50, N_POS) and negatives.shape == (50, N_NEG)
    assert (Y_TRUE[positives] == 1).all() and (Y_TRUE[negatives] == 0).all()


def test_each_resample_can_repeat_a_patient():
    positives, _ = draw_resamples(Y_TRUE, resamples=50)
    assert any(len(set(sample)) < N_POS for sample in positives)


def test_the_same_seed_draws_the_same_patients():
    # Checked on the draws: at 100,000 resamples the interval barely changes even with an unfixed seed
    first = draw_resamples(Y_TRUE, resamples=50)
    second = draw_resamples(Y_TRUE, resamples=50)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_a_different_seed_draws_other_patients():
    first = draw_resamples(Y_TRUE, resamples=50, seed=1)
    second = draw_resamples(Y_TRUE, resamples=50, seed=2)
    assert not np.array_equal(first[0], second[0])


def test_a_model_gets_the_same_interval_whatever_the_other_models_are():
    draws = draw_resamples(Y_TRUE)
    alone = ci_table(Y_TRUE, {'A': predicted(9, 17)}, draws)
    together = ci_table(Y_TRUE, {'B': predicted(5, 11), 'A': predicted(9, 17)}, draws)
    assert row(alone, 'F1')['Low'] == row(together, 'F1', name='A')['Low']
    assert row(alone, 'F1')['High'] == row(together, 'F1', name='A')['High']


# --- Paired comparison ------------------------------------------------------------------------


@pytest.mark.parametrize('only_a, only_b, p_value', [
    (6, 0, 0.03125),   # 2 x 0.5^6
    (0, 6, 0.03125),
    (3, 0, 0.25),      # 2 x 0.5^3
    (0, 0, 1.0),       # the models never disagree
    (4, 4, 1.0),
])
def test_mcnemar_exact_p_values(only_a, only_b, p_value):
    assert mcnemar_exact(only_a, only_b) == pytest.approx(p_value)


def test_comparison_counts_only_the_patients_where_the_models_disagree():
    table = compare_models(Y_TRUE, {'A': Y_TRUE, 'B': predicted(7, 17)}, 'A', ['B'], draw_resamples(Y_TRUE))
    result = row(table, 'F1')
    assert (result['Only_reference_correct'], result['Only_comparator_correct']) == (6, 0)
    assert result['McNemar_p'] == pytest.approx(0.03125)


@pytest.mark.parametrize('metric, delta', [
    ('F1', 0.30),    # 1.00 - 14 / 20
    ('Sn', 0.30),    # 10/10 - 7/10
    ('Sp', 0.15),    # 20/20 - 17/20
    ('ACC', 0.20),   # 30/30 - 24/30
])
def test_difference_is_reference_minus_comparator(metric, delta):
    table = compare_models(Y_TRUE, {'A': Y_TRUE, 'B': predicted(7, 17)}, 'A', ['B'], draw_resamples(Y_TRUE))
    assert row(table, metric)['Delta'] == pytest.approx(delta)


def test_difference_interval_comes_from_the_same_resampled_patients():
    # B only fails on one positive patient. A difference in Sn then equals how many times that patient
    # was drawn, over 10, so it can never be negative. The count follows a Binomial(10, 0.1):
    # P(0) = 0.35 > 0.025 and P(<=2) = 0.93 < 0.975 <= P(<=3) = 0.987, so the interval is [0, 0.3].
    # Drawing different patients for each model would give a negative lower bound.
    table = compare_models(Y_TRUE, {'A': Y_TRUE, 'B': predicted(9, 20)}, 'A', ['B'], draw_resamples(Y_TRUE))
    result = row(table, 'Sn')
    assert (result['Low'], result['High']) == pytest.approx((0.0, 0.3))
    assert result['Text'] == '0.10 [0.00, 0.30]'


def test_identical_models_have_no_difference():
    table = compare_models(Y_TRUE, {'A': predicted(9, 17), 'B': predicted(9, 17)}, 'A', ['B'],
                           draw_resamples(Y_TRUE))
    result = row(table, 'F1')
    assert (result['Delta'], result['Low'], result['High']) == (0.0, 0.0, 0.0)
    assert result['McNemar_p'] == 1.0


def test_p_values_are_corrected_over_the_comparators():
    # Raw p 0.03125 (6 errors more) and 0.25 (3 errors more). BH: 0.03125 x 2 / 1 and 0.25 x 2 / 2.
    table = compare_models(Y_TRUE, {'A': Y_TRUE, 'B': predicted(7, 17), 'C': predicted(9, 18)}, 'A', ['B', 'C'],
                           draw_resamples(Y_TRUE))
    assert row(table, 'F1', 'Comparator', 'B')['McNemar_p_BH'] == pytest.approx(0.0625)
    assert row(table, 'F1', 'Comparator', 'C')['McNemar_p_BH'] == pytest.approx(0.25)


def test_no_comparators_gives_an_empty_table():
    table = compare_models(Y_TRUE, {'A': Y_TRUE}, 'A', [], draw_resamples(Y_TRUE))
    assert table.empty


def test_each_comparator_gets_the_four_differences():
    table = compare_models(Y_TRUE, {'A': Y_TRUE, 'B': predicted(7, 17), 'C': predicted(9, 18)}, 'A', ['B', 'C'],
                           draw_resamples(Y_TRUE))
    assert set(zip(table['Comparator'], table['Metric'])) == {
        (comparator, metric) for comparator in 'BC' for metric in ['F1', 'Sn', 'Sp', 'ACC']}


# --- Text -------------------------------------------------------------------------------------


@pytest.mark.parametrize('value, low, high, text', [
    (0.9, 0.5958, 0.9821, '0.90 [0.60, 0.98]'),
    (-0.1, -0.25, 0.05, '-0.10 [-0.25, 0.05]'),
    (-0.001, -0.2, 0.2, '0.00 [-0.20, 0.20]'),    # never "-0.00"
    (float('nan'), float('nan'), float('nan'), 'n/a'),
])
def test_format_ci(value, low, high, text):
    assert format_ci(value, low, high) == text


# --- Reading and checking the predictions -------------------------------------------------------


def write_csv(tmp_path, table):
    path = tmp_path / 'predictions.csv'
    table.to_csv(path, index=False)
    return path


def test_load_predictions_rejects_a_missing_column(tmp_path):
    path = write_csv(tmp_path, model_rows('A', Y_TRUE).drop(columns=['y_pred']))
    with pytest.raises(ValueError, match='y_pred'):
        load_predictions(path)


def test_predictions_by_model_lines_up_the_patients_of_every_model():
    # The same patients in a different row order must not be compared with the wrong patient
    first = model_rows('A', predicted(9, 17))
    second = model_rows('B', predicted(8, 15)).iloc[::-1]
    y_true, by_model = predictions_by_model(pd.concat([first, second], ignore_index=True))
    assert np.array_equal(y_true, Y_TRUE)
    assert np.array_equal(by_model['B'], predicted(8, 15))


def test_predictions_are_rejected_when_a_model_has_other_patients():
    first = model_rows('A', Y_TRUE)
    second = model_rows('B', Y_TRUE).iloc[1:]
    with pytest.raises(ValueError, match='patients'):
        predictions_by_model(pd.concat([first, second], ignore_index=True))


def test_predictions_are_rejected_when_a_patient_has_two_true_labels():
    second = model_rows('B', Y_TRUE)
    second.loc[0, 'y_true'] = 0
    with pytest.raises(ValueError, match='labels'):
        predictions_by_model(pd.concat([model_rows('A', Y_TRUE), second], ignore_index=True))


def test_predictions_are_rejected_when_a_patient_is_repeated():
    table = model_rows('A', Y_TRUE)
    table.loc[1, 'Patient'] = 'P00'
    with pytest.raises(ValueError, match='repeated'):
        predictions_by_model(table)


@pytest.mark.parametrize('column', ['y_true', 'y_pred'])
def test_predictions_are_rejected_when_a_label_is_not_0_or_1(column):
    table = model_rows('A', Y_TRUE)
    table[column] = table[column].astype(float)
    table.loc[0, column] = 0.7
    with pytest.raises(ValueError, match='0 or 1'):
        predictions_by_model(table)


def test_predictions_are_rejected_without_both_classes():
    table = model_rows('A', np.ones(30, dtype=int), y_true=np.ones(30, dtype=int))
    with pytest.raises(ValueError, match='positive and negative'):
        predictions_by_model(table)


# --- Which models to compare, from the frozen file ----------------------------------------------


FROZEN = {
    'final_model': {'meta': 'GA', 'objective': 'relieff', 'classifier': 'Knn k = 3'},
    'other_winners': [{'meta': 'DE', 'objective': 'mi', 'classifier': 'LDA'}],
    'comparators': [{'meta': 'TOP10', 'objective': 'mi', 'classifier': 'SVC poly'},
                    {'meta': 'ALL70', 'objective': 'none', 'classifier': 'RF estimators: 60'}],
}


def test_model_name_joins_the_parts_of_a_scheme():
    assert model_name(FROZEN['final_model']) == 'GA-relieff-Knn k = 3'


def test_models_to_compare_reads_the_final_model_and_comparators_from_the_frozen_file(tmp_path):
    path = tmp_path / 'final_model.json'
    path.write_text(json.dumps(FROZEN))
    reference, comparators = models_to_compare(path, extra_comparators=['LASSO', 'SVM-RFE'])
    assert reference == 'GA-relieff-Knn k = 3'
    # The other winners are not comparators
    assert comparators == ['TOP10-mi-SVC poly', 'ALL70-none-RF estimators: 60', 'LASSO', 'SVM-RFE']


# --- The whole run ----------------------------------------------------------------------------


def prepare_run(tmp_path, models):
    predictions_file = write_csv(tmp_path, pd.concat(
        [model_rows(name, y_pred) for name, y_pred in models.items()], ignore_index=True))
    frozen_file = tmp_path / 'final_model.json'
    frozen_file.write_text(json.dumps(FROZEN))
    return predictions_file, frozen_file


def test_run_writes_both_tables(tmp_path):
    predictions_file, frozen_file = prepare_run(tmp_path, {
        'GA-relieff-Knn k = 3': Y_TRUE,
        'DE-mi-LDA': predicted(8, 16),
        'TOP10-mi-SVC poly': predicted(7, 17),
        'ALL70-none-RF estimators: 60': predicted(9, 18),
        'LASSO': predicted(8, 18),
    })
    output_dir = tmp_path / 'out'

    run(predictions_file, frozen_file, ['LASSO'], output_dir)

    intervals = pd.read_csv(output_dir / 'test_ci.csv')
    assert len(intervals) == 25    # 5 models x 5 metrics
    comparisons = pd.read_csv(output_dir / 'test_comparisons.csv')
    assert set(comparisons['Reference']) == {'GA-relieff-Knn k = 3'}
    # The other winner gets intervals but is not compared
    assert set(comparisons['Comparator']) == {'TOP10-mi-SVC poly', 'ALL70-none-RF estimators: 60', 'LASSO'}


def test_run_stops_if_the_frozen_file_does_not_exist(tmp_path):
    # Intervals cannot be produced before the model is frozen
    predictions_file, _ = prepare_run(tmp_path, {'GA-relieff-Knn k = 3': Y_TRUE})
    with pytest.raises(FileNotFoundError):
        run(predictions_file, tmp_path / 'missing.json', [], tmp_path / 'out')


def test_run_stops_if_the_final_model_has_no_predictions(tmp_path):
    predictions_file, frozen_file = prepare_run(tmp_path, {'SOMETHING-else': Y_TRUE})
    with pytest.raises(ValueError, match='GA-relieff-Knn k = 3'):
        run(predictions_file, frozen_file, [], tmp_path / 'out')


def test_run_stops_if_a_comparator_has_no_predictions(tmp_path):
    predictions_file, frozen_file = prepare_run(tmp_path, {
        'GA-relieff-Knn k = 3': Y_TRUE,
        'TOP10-mi-SVC poly': predicted(7, 17),
        'ALL70-none-RF estimators: 60': predicted(9, 18),
    })
    with pytest.raises(ValueError, match='SVM-RFE'):
        run(predictions_file, frozen_file, ['SVM-RFE'], tmp_path / 'out')
