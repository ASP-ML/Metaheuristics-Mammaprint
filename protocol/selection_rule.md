# Final model selection rule

Manuscript: CMPBUP-D-25-00249R1, *Filter-based Metaheuristics Optimization for Prognostic Gene Signature Selection in Breast Cancer*. This file addresses editor comment E.2.

Written on 8 November 2026. Implemented in [`select_final_model.py`](../select_final_model.py).

**The final model is chosen only from cross-validation (CV) results on the training partition. The test set plays no part in the choice.** The chosen model's test result is reported whatever it turns out to be, and it cannot change the choice.

## Candidates

| | |
|---|---|
| Metaheuristics | DE, GA, SA |
| Objective functions | aReliefF, aMI, aSU, aGR, aCHI2 |
| Classifiers | NB, LDA, kNN (k = 3, 5, 7), SVM (polynomial kernel), RF (t = 60, 80, 100) |

A *scheme* is one metaheuristic, one objective function and one classifier, which gives 135 schemes. Each scheme has 30 fold results, from three repeats of stratified ten-fold CV. FFBP is not a candidate because Tables 4–6 of the manuscript report no FFBP results.

## The rule

The rule is applied to a group of candidate schemes:

1. **Reference.** The scheme with the highest mean F1 is the reference. If two schemes have exactly the same mean F1, the first in alphabetical order is the reference.
2. **Ties.** Each other scheme is compared with the reference using a two-sided Wilcoxon rank-sum test on the 30 fold F1 values (asymptotic, with continuity correction). The p-values are adjusted by Benjamini-Hochberg over those comparisons, without comparing the reference with itself. A scheme is tied with the reference if its adjusted p ≥ 0.05. The reference counts as tied.
3. **Metrics.** Among the tied schemes, the highest mean specificity (Sp) wins, then the highest mean sensitivity (Sn), then the highest mean precision (PRE). These means are compared after rounding to two decimals, as printed in the tables.
4. **Simplicity.** If the tie persists, the simplest classifier wins, in this order: NB = LDA < kNN (k = 3 < 5 < 7) < SVM < RF (t = 60 < 80 < 100).
5. **Order of last resort.** If schemes are still tied, the first in alphabetical order (metaheuristic, objective function, classifier) wins. This only makes the result reproducible.

## Two steps

**Step 1: the best scheme per combination.** The rule is applied separately to each of the 15 metaheuristic and objective function combinations. This gives 15 winners.

**Step 2: the final model.** The rule is applied to the 15 winners, with one addition: in item 4, the scheme with fewer genes wins before the simplest classifier is considered. The winner is the final model.

## Comparators

The top-10 filter-ranking baselines and the 70-gene model are comparators. Each goes through step 1 to choose its classifier. They do not take part in step 2 and cannot become the final model.

## Checks

The script stops with an error, and selects nothing, if any of these is not met:

- all 15 combinations are present;
- every combination, including the comparators, has all nine classifiers;
- every scheme has 30 different folds;
- all the schemes of a combination used a subset of the same length;
- each subset in the subset files has the length used in the CV results.

Classifiers outside the list above, and the unused sixth objective function (information gain), are ignored, and the script prints what it ignored.

## Order of work

1. Commit this file and `select_final_model.py`.
2. Produce the corrected CV results on the training partition.
3. Run the script on those results with `OUTPUT_DIR = 'protocol'`. Commit `protocol/final_model.json` and `protocol/selection_table.csv`, and tag that commit `v2-selection-frozen`.
4. Only then run the test evaluation. It takes the final model from `protocol/final_model.json`.

Other schemes may also be evaluated on the test set for transparency. They must be reported as not used for selection.

Any change to this rule after step 3 is a protocol deviation and must be reported in the manuscript.

## What was known when this rule was written

This is not a blind pre-registration, and the following should be taken into account when reading the results:

- **The submitted results were already known.** The CV and test results of these schemes were reported in the submitted manuscript. Its final model, DE + aSU + RF (t = 60), was chosen using test performance. Those results were computed on data in which SMOTE and min-max scaling had been applied before the split.
- **The test patients have been inspected.** The corrected test set is made up of the 30 real patients (20 negative, 10 positive) of the submitted 40-sample test set, so test results may remain optimistic. It is a corrected hold-out of the same cohort, not an independent validation.
- **The rule was tried on the submitted CV results.** This was done on 8 November 2026 to check the code. It selected GA + aReliefF + kNN (k = 3), with 11 genes. DE + aSU + RF (t = 60) was not tied with the reference (adjusted p = 0.002). These numbers are not results of the corrected analysis.

### What this file adds to Section 2.6.5 of the submitted manuscript

Section 2.6.5 already gave the order F1, statistical tie, Sp, Sn, PRE, lowest complexity, for each combination. This file adds:

- step 2, because the submitted manuscript had no rule for choosing across combinations;
- fewer genes as a tie-break in step 2;
- the comparison at two decimals, which was not stated before. On the submitted results, it reproduces 13 of the 15 winners in Table 8, against 11 without rounding, and the final model is the same either way;
- the exact order of classifier complexity and the order of last resort;
- the handling of the comparators.

With this rule, two winners differ from the submitted Table 8: GA + aSU is RF (t = 100) rather than RF (t = 80), because of its higher PRE, and SA + aCHI2 is SVM rather than RF (t = 100), because it is tied and has the higher Sp.
