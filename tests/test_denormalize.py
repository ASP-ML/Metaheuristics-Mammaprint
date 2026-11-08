"""
Tests for denormalize.py: recover the original-scale train/test split from the
min-max normalized files.

The unit tests build small synthetic data. The integration tests use the CSVs in
data/ and are skipped when those files are missing.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from denormalize import LABEL, denormalize_split, inverse_min_max, match_rows

RAW_FILE = 'data/rtfDataSet.csv'
TRAIN_NORMAL_FILE = 'data/train_normal_rtf.csv'
TEST_NORMAL_FILE = 'data/test_normal_rtf.csv'


def make_raw(n_rows=20, n_genes=5, seed=0):
    """Random raw dataset in the layout of rtfDataSet.csv: gene columns '0'..'n-1' plus label."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(np.round(rng.uniform(-1, 1, (n_rows, n_genes)), 3),
                      columns=[str(i) for i in range(n_genes)])
    df[LABEL] = rng.integers(0, 2, n_rows)
    return df


def normalize(raw_df):
    """Min-max scale the gene columns of raw_df, fitting on all of its rows."""
    genes = raw_df.drop(columns=LABEL)
    out = pd.DataFrame(MinMaxScaler().fit_transform(genes), columns=genes.columns)
    out[LABEL] = raw_df[LABEL].to_numpy()
    return out


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_inverse_min_max_undoes_sklearn_scaler():
    x = make_raw().drop(columns=LABEL).to_numpy()
    scaled = MinMaxScaler().fit_transform(x)
    recovered = inverse_min_max(scaled, x.min(axis=0), x.max(axis=0))
    np.testing.assert_allclose(recovered, x, atol=1e-12)


def test_match_rows_finds_shuffled_rows():
    raw = make_raw().drop(columns=LABEL).to_numpy()
    order = np.array([7, 2, 19, 0])
    np.testing.assert_array_equal(match_rows(raw[order], raw), order)


def test_match_rows_rejects_row_not_in_raw():
    raw = make_raw().drop(columns=LABEL).to_numpy()
    with pytest.raises(ValueError, match='no matching raw row'):
        match_rows(raw[[0]] + 0.5, raw)


def test_match_rows_rejects_two_rows_matching_same_raw_row():
    raw = make_raw().drop(columns=LABEL).to_numpy()
    with pytest.raises(ValueError, match='same raw row'):
        match_rows(raw[[3, 3]], raw)


def test_denormalize_split_returns_raw_rows_in_split_order():
    raw = make_raw()
    order = [5, 0, 12, 3]
    split = normalize(raw).iloc[order].reset_index(drop=True)

    out, idx = denormalize_split(split, raw)

    np.testing.assert_array_equal(idx, order)
    pd.testing.assert_frame_equal(out, raw.iloc[order].reset_index(drop=True))


def test_denormalize_split_rejects_label_mismatch():
    raw = make_raw()
    split = normalize(raw).iloc[[1, 2]].reset_index(drop=True)
    split.loc[0, LABEL] = 1 - split.loc[0, LABEL]
    with pytest.raises(ValueError, match='label'):
        denormalize_split(split, raw)


def test_denormalize_split_rejects_different_columns():
    raw = make_raw()
    split = normalize(raw).iloc[[1, 2]].drop(columns='4')
    with pytest.raises(ValueError, match='columns'):
        denormalize_split(split, raw)


# ── Integration tests (real data files) ───────────────────────────────────────

data_files_present = all(os.path.exists(f) for f in (RAW_FILE, TRAIN_NORMAL_FILE, TEST_NORMAL_FILE))
needs_data = pytest.mark.skipif(not data_files_present, reason='data CSVs not found')


@pytest.fixture(scope='module')
def real_splits():
    raw = pd.read_csv(RAW_FILE)
    train_norm = pd.read_csv(TRAIN_NORMAL_FILE)
    test_norm = pd.read_csv(TEST_NORMAL_FILE)
    train, train_idx = denormalize_split(train_norm, raw)
    test, test_idx = denormalize_split(test_norm, raw)
    return raw, train_norm, test_norm, train, train_idx, test, test_idx


@needs_data
def test_real_split_shapes_and_columns(real_splits):
    raw, _, _, train, _, test, _ = real_splits
    assert train.shape == (356, 71)
    assert test.shape == (40, 71)
    assert list(train.columns) == list(raw.columns) == list(test.columns)


@needs_data
def test_real_train_and_test_partition_the_raw_rows(real_splits):
    raw, _, _, _, train_idx, _, test_idx = real_splits
    assert not set(train_idx) & set(test_idx)
    assert sorted(np.concatenate([train_idx, test_idx])) == list(range(len(raw)))


@needs_data
def test_real_split_renormalizes_to_the_given_files(real_splits):
    raw, train_norm, test_norm, train, _, test, _ = real_splits
    genes = raw.drop(columns=LABEL)
    lo, hi = genes.min(), genes.max()
    for out, norm in ((train, train_norm), (test, test_norm)):
        renorm = (out.drop(columns=LABEL) - lo) / (hi - lo)
        np.testing.assert_allclose(renorm.to_numpy(), norm.drop(columns=LABEL).to_numpy(), atol=1e-12)
        np.testing.assert_array_equal(out[LABEL].to_numpy(), norm[LABEL].to_numpy())
