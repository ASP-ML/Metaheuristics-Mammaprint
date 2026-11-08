"""
Denormalize the Train/Test Split
================================

data/train_normal_rtf.csv and data/test_normal_rtf.csv were made by min-max scaling
data/rtfDataSet.csv to [0, 1], fitting on all 396 rows (test included), and then
splitting the rows. data/normal_rtf.csv is that scaled file before the split.

This script inverts the scaling gene by gene,

    x = x_norm * (max - min) + min,    min, max: column range of rtfDataSet.csv,

and writes the split on the original scale.

Every recovered row is checked against rtfDataSet.csv: it must equal exactly one raw
row with the same label, and train and test must not share a raw row. The output keeps
the row order of the normalized files (row i of train_rtf.csv is row i of
train_normal_rtf.csv) and takes its values from the matched raw row, so they carry no
floating-point noise from the inversion.

rtfDataSet.csv has no patient ID column, so a row's ID is its 0-based position in that
file (raw_id 0 is the first data row, line 2 of the CSV). The script also writes
data/split_raw_ids.csv, one line per row of each split:

    split      train or test
    split_row  0-based position in train_rtf.csv / test_rtf.csv (and *_normal_rtf.csv)
    raw_id     0-based position in rtfDataSet.csv
    label      class label

Run with: python denormalize.py
"""

import numpy as np
import pandas as pd

LABEL = 'label'

TOLERANCE = 1e-9
"""Largest absolute difference, per gene, for a recovered row to count as a raw row."""


def inverse_min_max(normalized, data_min, data_max):
    """Undo x_norm = (x - min) / (max - min) column by column."""
    return normalized * (data_max - data_min) + data_min


def match_rows(recovered, raw, tol=TOLERANCE):
    """
    Position in `raw` of the row equal to each row of `recovered`.

    Raises ValueError if a recovered row is farther than `tol` from every raw row, or if
    two recovered rows match the same raw row.
    """
    diff = np.abs(recovered[:, None, :] - raw[None, :, :]).max(axis=2)
    idx = diff.argmin(axis=1)
    unmatched = np.flatnonzero(diff[np.arange(len(idx)), idx] > tol)
    if len(unmatched):
        raise ValueError(f'{len(unmatched)} recovered rows have no matching raw row '
                         f'(first: row {unmatched[0]})')
    if len(np.unique(idx)) != len(idx):
        raise ValueError('two recovered rows match the same raw row')
    return idx


def denormalize_split(normalized_df, raw_df):
    """
    Bring one normalized split back to the scale of `raw_df`.

    `raw_df` is the dataset the scaler was fitted on: its column range gives the
    min/max to invert with, and its rows are the reference each recovered row must match.

    Returns the original-scale split (raw rows, in the row order of `normalized_df`) and
    the position in `raw_df` of each of those rows.
    """
    if list(normalized_df.columns) != list(raw_df.columns):
        raise ValueError('normalized and raw files have different columns')

    raw_genes = raw_df.drop(columns=LABEL).to_numpy(dtype=float)
    norm_genes = normalized_df.drop(columns=LABEL).to_numpy(dtype=float)
    recovered = inverse_min_max(norm_genes, raw_genes.min(axis=0), raw_genes.max(axis=0))
    idx = match_rows(recovered, raw_genes)

    if not np.array_equal(normalized_df[LABEL].to_numpy(), raw_df[LABEL].to_numpy()[idx]):
        raise ValueError('a recovered row matches a raw row with a different label')
    return raw_df.iloc[idx].reset_index(drop=True), idx


if __name__ == "__main__":
    RAW_FILE = 'data/rtfDataSet.csv'
    SPLITS = {
        'train': ('data/train_normal_rtf.csv', 'data/train_rtf.csv'),
        'test': ('data/test_normal_rtf.csv', 'data/test_rtf.csv'),
    }
    IDS_FILE = 'data/split_raw_ids.csv'

    raw = pd.read_csv(RAW_FILE)
    id_tables = []
    for name, (normalized_file, output_file) in SPLITS.items():
        split, idx = denormalize_split(pd.read_csv(normalized_file), raw)
        split.to_csv(output_file, index=False)
        id_tables.append(pd.DataFrame({
            'split': name,
            'split_row': np.arange(len(idx)),
            'raw_id': idx,
            'label': split[LABEL].to_numpy(),
        }))
        print(f'{normalized_file} -> {output_file}: {len(split)} rows, '
              f'labels {split[LABEL].value_counts().sort_index().to_dict()}')

    ids = pd.concat(id_tables, ignore_index=True)
    if ids['raw_id'].duplicated().any():
        raise ValueError('train and test share a raw row')
    ids.to_csv(IDS_FILE, index=False)
    print(f'{len(ids)} of {len(raw)} rows of {RAW_FILE} recovered, none in two splits; IDs in {IDS_FILE}')
