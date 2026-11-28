# Source of manuscript Tables 4–6 (cross-validation results)

The mean(SD) values in Tables 4, 5 and 6 of the manuscript (DE, GA and SA classification schemes, three-times stratified ten-fold cross-validation) come from **`results/meanByFold.xlsx`**. `results/meanByFold.csv` holds the same values.

`results/` is in `.gitignore`, so these files exist only on the local disk. A byte-identical copy is in `/mnt/Cris-SW/Research/StatisticTest/LASSO_CLEANED/`.

## How the files were produced

Run the scripts from this folder, in this order:

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `process_classifiers.py` | `results/lasso_full_history.csv`: one row per fold, 30 folds (3 × 10) per scheme | `results/meanByFold.csv`: mean and SD per scheme. `results/meanFullHistoryByFold.csv`: the per-fold rows plus the scheme's mean F1 |
| 2 | `main_statistic_test.py` | `results/meanFullHistoryByFold.csv` | `results/statistic_results/{Meta}_{OF}_{F1}.csv`: Mann–Whitney U test of each scheme against the pivot (highest mean F1), RBC and Benjamini–Hochberg p-values |
| 3 | `concat_statistic_test.py` | `results/statistic_results/` | `results/combined_statistic_results.csv` |

The *p*-value (BH) and RBC columns of the tables come from step 3.

## Standard deviation

`process_classifiers.py` uses `statistics.stdev`, the **sample** SD (divides by n − 1, with n = 30 folds).

Do not take SDs from the `*_DESV` columns of `experiment_results/R3/class_step_3_report_train_normal_rtf.csv`. Those are population SDs (divide by n) and give slightly smaller values. For example, SA–aGR RF(t=60) Sn: sample SD 0.1266 rounds to 0.13, population SD 0.1244 rounds to 0.12.

## Names in the file and in the manuscript

| Manuscript | `meanByFold` |
|---|---|
| F1-score, PRE, Sn, Sp, ACC | `F1_*`, `PRE_*`, `REC_*`, `SPEC_*`, `ACC_*` (`_MEAN` and `_STD`) |
| aMI, aSU, aGR, aReliefF, aCHI2 | `mi`, `sim_uncertainty`, `gain_ratio`, `relieff`, `chi2` |
| kNN(k=5), RF(t=60), NB, SVM (Poly), LDA | `Knn k = 5`, `RF estimators: 60`, `Naive bayes`, `SVC poly`, `LDA` |

Table 4 is DE, Table 5 is GA and Table 6 is SA.

## Check

On 28 November 2026, every mean(SD) cell of Tables 4–6 (89 rows × 5 metrics × mean and SD = 890 cells) was compared with `results/meanByFold.csv`, rounded to two decimals. All 890 match.
