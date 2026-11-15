"""
95% confidence intervals and paired comparisons on the test set (editor comment E.3)

Input (PREDICTIONS_FILE): one row per model and test patient, with the columns
    Model    name of the model. For a scheme it is Meta-OF-Classifier, as in
             'GA-relieff-Knn k = 3' (see model_name); the baselines are 'LASSO' and 'SVM-RFE'
    Patient  patient id
    y_true   true label, 0 or 1
    y_pred   predicted label, 0 or 1

Input (FINAL_MODEL_FILE): the file written by select_final_model.py. It says which model is the
final model and which ones are comparators. The script stops if the file does not exist, so
intervals cannot be produced before the model is frozen.

Intervals:
    Sn, Sp, PRE, ACC  Wilson score interval
    F1                percentile bootstrap. Patients are drawn with replacement, separately among
                      positives and negatives, so every resample keeps the class counts of the test set

Comparison of the final model with each comparator, on the same patients:
    difference        final model minus comparator, in F1, Sn, Sp and ACC. Its interval is a percentile
                      bootstrap in which both models are scored on the same resampled patients
    McNemar           exact test on the patients where exactly one of the two models is right,
                      corrected by Benjamini-Hochberg over the comparators

Outputs (OUTPUT_DIR):
    test_ci.csv           one row per model and metric: Value, Low, High, Text, Method, confusion counts
    test_comparisons.csv  one row per comparator and metric: Delta, Low, High, Text, McNemar p-values
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, false_discovery_control

CONFIDENCE = 0.95
RESAMPLES = 100_000
SEED = 45

METRICS = ['F1', 'PRE', 'Sn', 'Sp', 'ACC']
DIFFERENCE_METRICS = ['F1', 'Sn', 'Sp', 'ACC']

WILSON_METHOD = 'Wilson score interval'
BOOTSTRAP_METHOD = f'Percentile bootstrap stratified by class, {RESAMPLES} resamples, seed {SEED}'

REQUIRED_COLUMNS = ['Model', 'Patient', 'y_true', 'y_pred']
COMPARISON_COLUMNS = ['Reference', 'Comparator', 'Metric', 'Delta', 'Low', 'High', 'Text',
                      'Only_reference_correct', 'Only_comparator_correct', 'McNemar_p']


def load_predictions(path):
    predictions = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in predictions.columns]
    if missing:
        raise ValueError(f'{path} has no column {missing}')
    return predictions


def predictions_by_model(predictions):
    """
    True labels and each model's predictions, as arrays in the same patient order.

    Raises ValueError if the models cannot be compared patient by patient.
    """
    for column in ('y_true', 'y_pred'):
        if not predictions[column].isin([0, 1]).all():
            raise ValueError(f'{column} must be 0 or 1')
    if predictions.duplicated(['Model', 'Patient']).any():
        raise ValueError('A patient is repeated within a model')
    n_labels = predictions.groupby('Patient')['y_true'].nunique()
    if (n_labels > 1).any():
        raise ValueError(f'Patients with two true labels: {n_labels[n_labels > 1].index.tolist()}')

    patients = sorted(predictions['Patient'].unique())
    by_model = {}
    for model, rows in predictions.groupby('Model'):
        rows = rows.sort_values('Patient')
        if rows['Patient'].tolist() != patients:
            raise ValueError(f'{model} does not have the same patients as the other models')
        by_model[model] = rows['y_pred'].to_numpy(dtype=int)

    y_true = predictions.drop_duplicates('Patient').sort_values('Patient')['y_true'].to_numpy(dtype=int)
    if y_true.min() == y_true.max():
        raise ValueError('The test set needs both positive and negative patients')
    return y_true, by_model


def wilson_ci(successes, total):
    """Wilson interval of a proportion; NaN if there are no cases (0 of 0)."""
    if total == 0:
        return np.nan, np.nan
    interval = binomtest(int(successes), int(total)).proportion_ci(confidence_level=CONFIDENCE, method='wilson')
    return interval.low, interval.high


def percentile_ci(values):
    tail = 50 * (1 - CONFIDENCE)
    low, high = np.percentile(values, [tail, 100 - tail])
    return low, high


def format_ci(value, low, high):
    """Text for the tables, such as '0.90 [0.60, 0.98]'."""
    if np.isnan(value):
        return 'n/a'
    # Adding 0.0 turns -0.0 into 0.0
    return '{:.2f} [{:.2f}, {:.2f}]'.format(*(round(number, 2) + 0.0 for number in (value, low, high)))


def metrics_from_counts(tp, tn, n_pos, n_neg):
    """The metrics of Table 7. The counts can be numbers or arrays. PRE is NaN if nothing is predicted positive."""
    fn, fp = n_pos - tp, n_neg - tn
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(tp, tp + fp)
    return {
        'F1': 2 * tp / (2 * tp + fp + fn),
        'PRE': precision,
        'Sn': tp / n_pos,
        'Sp': tn / n_neg,
        'ACC': (tp + tn) / (n_pos + n_neg),
    }


def draw_resamples(y_true, resamples=RESAMPLES, seed=SEED):
    """Indices of the resampled patients: one array for the positives and one for the negatives, one row per resample."""
    rng = np.random.default_rng(seed)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    return (positives[rng.integers(0, len(positives), (resamples, len(positives)))],
            negatives[rng.integers(0, len(negatives), (resamples, len(negatives)))])


def bootstrap_metrics(y_pred, draws):
    """The metrics of one model on every resample. Every model uses the same `draws`."""
    positives, negatives = draws
    tp = (y_pred[positives] == 1).sum(axis=1)
    tn = (y_pred[negatives] == 0).sum(axis=1)
    return metrics_from_counts(tp, tn, positives.shape[1], negatives.shape[1])


def confusion_counts(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tp, fn, tn, fp


def ci_table(y_true, by_model, draws):
    """One row per model and metric, with its value and 95% interval."""
    rows = []
    for model, y_pred in by_model.items():
        tp, fn, tn, fp = confusion_counts(y_true, y_pred)
        values = metrics_from_counts(tp, tn, tp + fn, tn + fp)
        intervals = {
            'F1': percentile_ci(bootstrap_metrics(y_pred, draws)['F1']),
            'PRE': wilson_ci(tp, tp + fp),
            'Sn': wilson_ci(tp, tp + fn),
            'Sp': wilson_ci(tn, tn + fp),
            'ACC': wilson_ci(tp + tn, tp + fn + tn + fp),
        }
        for metric in METRICS:
            low, high = intervals[metric]
            rows.append({
                'Model': model, 'Metric': metric, 'Value': values[metric], 'Low': low, 'High': high,
                'Text': format_ci(values[metric], low, high),
                'Method': BOOTSTRAP_METHOD if metric == 'F1' else WILSON_METHOD,
                'TP': tp, 'FN': fn, 'TN': tn, 'FP': fp,
            })
    return pd.DataFrame(rows)


def mcnemar_exact(only_a, only_b):
    """Exact McNemar p-value from the patients where exactly one of the two models is right."""
    if only_a + only_b == 0:
        return 1.0
    return binomtest(only_a, only_a + only_b, 0.5).pvalue


def compare_models(y_true, by_model, reference, comparators, draws):
    """One row per comparator and metric: reference minus comparator, on the same patients."""
    y_reference = by_model[reference]
    reference_correct = y_reference == y_true
    tp, fn, tn, fp = confusion_counts(y_true, y_reference)
    reference_values = metrics_from_counts(tp, tn, tp + fn, tn + fp)
    reference_resampled = bootstrap_metrics(y_reference, draws)

    rows = []
    p_values = []
    for comparator in comparators:
        y_comparator = by_model[comparator]
        comparator_correct = y_comparator == y_true
        only_reference = int((reference_correct & ~comparator_correct).sum())
        only_comparator = int((~reference_correct & comparator_correct).sum())
        p_value = mcnemar_exact(only_reference, only_comparator)
        p_values.append(p_value)

        tp, fn, tn, fp = confusion_counts(y_true, y_comparator)
        values = metrics_from_counts(tp, tn, tp + fn, tn + fp)
        resampled = bootstrap_metrics(y_comparator, draws)
        for metric in DIFFERENCE_METRICS:
            delta = reference_values[metric] - values[metric]
            low, high = percentile_ci(reference_resampled[metric] - resampled[metric])
            rows.append({
                'Reference': reference, 'Comparator': comparator, 'Metric': metric,
                'Delta': delta, 'Low': low, 'High': high, 'Text': format_ci(delta, low, high),
                'Only_reference_correct': only_reference, 'Only_comparator_correct': only_comparator,
                'McNemar_p': p_value,
            })

    table = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    corrected = dict(zip(comparators, false_discovery_control(p_values, method='bh')))
    table['McNemar_p_BH'] = table['Comparator'].map(corrected)
    return table


def model_name(record):
    """Name of a scheme of final_model.json in the Model column of the predictions."""
    return f"{record['meta']}-{record['objective']}-{record['classifier']}"


def models_to_compare(final_model_file, extra_comparators):
    """The final model and its comparators: those of the frozen file, plus `extra_comparators`."""
    with open(final_model_file) as file:
        frozen = json.load(file)
    comparators = [model_name(record) for record in frozen['comparators']] + list(extra_comparators)
    return model_name(frozen['final_model']), comparators


def run(predictions_file, final_model_file, extra_comparators, output_dir):
    """Compute both tables from the CSV files and write them."""
    reference, comparators = models_to_compare(final_model_file, extra_comparators)
    y_true, by_model = predictions_by_model(load_predictions(predictions_file))
    missing = [model for model in [reference] + comparators if model not in by_model]
    if missing:
        raise ValueError(f'No predictions for {missing}')

    draws = draw_resamples(y_true)
    intervals = ci_table(y_true, by_model, draws)
    comparisons = compare_models(y_true, by_model, reference, comparators, draws)

    os.makedirs(output_dir, exist_ok=True)
    intervals.to_csv(os.path.join(output_dir, 'test_ci.csv'), index=False)
    comparisons.to_csv(os.path.join(output_dir, 'test_comparisons.csv'), index=False)
    return reference, y_true, intervals, comparisons


if __name__ == "__main__":
    PREDICTIONS_FILE = 'results/v2/test_predictions.csv'
    FINAL_MODEL_FILE = 'protocol/final_model.json'
    EXTRA_COMPARATORS = ['LASSO', 'SVM-RFE']
    OUTPUT_DIR = 'results/v2'

    reference, y_true, intervals, comparisons = run(PREDICTIONS_FILE, FINAL_MODEL_FILE, EXTRA_COMPARATORS, OUTPUT_DIR)

    print(f'{len(y_true)} test patients: {(y_true == 1).sum()} positive, {(y_true == 0).sum()} negative')
    print(f'Final model: {reference}\n')
    print(intervals.pivot(index='Model', columns='Metric', values='Text')[METRICS].to_string())
    print(f'\nFinal model minus comparator, with the McNemar p-values:')
    table = comparisons.pivot(index='Comparator', columns='Metric', values='Text')[DIFFERENCE_METRICS]
    table['McNemar p (BH)'] = comparisons.groupby('Comparator')['McNemar_p_BH'].first().round(3)
    print(table.to_string())
    print(f'\nWritten to {OUTPUT_DIR}')
