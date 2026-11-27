# Which data the metaheuristics searched in the submitted run

Manuscript CMPBUP-D-25-00249R1, *Filter-based Metaheuristics Optimization for Prognostic Gene Signature Selection in Breast Cancer*.
Checked on 24–27 November 2026. Every result below is reproduced by [`reducer_input_check.ipynb`](reducer_input_check.ipynb), which runs on the files in this repository.

## Summary

Section 2.6.1 of the manuscript (lines 432–436) says that *"the training set was used for feature selection"* and that the hold-out test set was *"strictly isolated and unseen during all phases of model development"*.

For the gene subsets reported in the manuscript, this is not the case. **The reducer (DE, GA and SA, three rounds) searched `data/normal_rtf.csv`, which holds all 396 rows: the 356 training rows and the 40 rows of the test set, including the 30 real test patients.** The classifiers were trained on `data/train_normal_rtf.csv` (356 rows), as described.

So the test patients influenced which genes were selected, and the same patients were then used to evaluate those genes. The test results of the submitted schemes are optimistic, by an amount we cannot measure without re-running the reducer. Re-running only the classifiers does not remove this, because the subsets themselves are affected.

The names of the result files (`…_train_normal_rtf.csv`) suggest otherwise, but the name comes from the classifier step, which did use the training file.

## Evidence

There are four independent pieces of evidence. Each is one section of the notebook.

**1. The data (section 2).** `normal_rtf.csv` is exactly the 356 rows of `train_normal_rtf.csv` plus the 40 rows of `test_normal_rtf.csv`. The 30 real test patients are among those 40 rows.

**2. The code (section 3).** In the old pipeline (`evidence/pipeline.py`, `pipeline_2.py`, `pipeline_3.py`), the two menu options that run the multi-round reducer load the full file. The classifier option loads the training file:

| Menu option | What it runs | Data it loads |
|---|---|---|
| 6, *Cargar y correr configuración* | reducer, round 1 | `processDataSet('normal_rtf.csv')`, with the comment `#Here mh_dict is generated with all data` |
| *Correr paso desde archivo de resumen* (9 in `pipeline.py`, 8 in the other two) | reducer, rounds 2–3 | `processDataSet('normal_rtf.csv')`, same comment |
| 7, *Ejecutar pipeline de clasificadores* | classifier cross-validation | `NAME_FILE = 'train_normal_rtf'` |

Options 4 and 5 also run metaheuristics, but on a data set picked by hand in option 1, and they write no round reports.

**3. The merits (sections 4 and 5.1).** The reducer's own round results (`evidence/minimum_length_resume_step_{1,2,3}.csv`, dated 21 September 2023) hold the same subsets as Tables 2 and 3, with the merit computed during the search. We recomputed these merits with this repository's objective function (`ofunctions.allObjectiveFunctionWeka`). **All 54 merits (18 per round) are reproduced to 6 decimals on `normal_rtf.csv`, and none on `train_normal_rtf.csv`.** Round 3 for DE:

| DE subset | Logged by the reducer | On `train_normal_rtf.csv` (356) | On `normal_rtf.csv` (396) |
|---|---|---|---|
| aMI | 0.082621 | 0.084706 | **0.082621** |
| aSU | 0.078548 | 0.083722 | **0.078548** |
| aGR | 0.103818 | 0.109775 | **0.103818** |
| aReliefF | 0.014495 | 0.013992 | **0.014495** |
| aCHI2 | 37.181303 | 35.447290 | **37.181303** |

The "Whole set" column of Table 3 (70 genes, section 5.2) agrees: for every objective, the value on `normal_rtf.csv` is the closer one to the printed value.

**4. The DE subsets (section 5.3).** For aMI, aSU, aGR, aCHI2 and aIG (information gain, run by the reducer but not reported in the manuscript), a gene's score does not depend on the other genes. Under the constraint |S| ≥ 10, the maximum of Eq. (11) is therefore the 10 best-ranked genes. **DE returned exactly the top 10 of `normal_rtf.csv` for all five objectives, and the top 10 of `train_normal_rtf.csv` only for aGR.** On the training data alone, three of the ten genes would have been different for aMI, aSU and aCHI2 (two for aIG):

| Objective | DE kept (because of the test rows) | Top 10 of the training rows instead |
|---|---|---|
| aMI | g4, g58, g68 | g22, g24, g36 |
| aSU | g12, g45, g59 | g33, g35, g68 |
| aCHI2 | g39, g50, g59 | g10, g13, g33 |
| aGR | none | none |

The scoring code is not the cause. The objective functions in this repository's `ofunctions.py` are identical to those in both copies of the old project. We checked this separately, on 24 November, by comparing the function bodies.

## Table 2: the aCHI2 merits are normalized

This is not an error. Table 2 prints the aCHI2 merits on a 0–1 scale (round 3: DE 1.000, GA 0.879, SA 0.778), while Table 3 prints the raw values (37.181, 34.505, 32.255). The normalization is in [`evidence/processDataset.ipynb`](evidence/processDataset.ipynb) (25 September 2023): a hand-typed list of 90 aCHI2 merits (3 metaheuristics × 10 iteration budgets × 3 rounds) scaled with `MinMaxScaler`. Every aCHI2 value in Table 2 is

  (raw − 14.998303) / (37.1813027 − 14.998303)

where 14.998303 is the lowest value in the list and 37.1813027 is the highest (DE's subset). See section 6 of the notebook.

Suggested caption note for Table 2: *"aCHI2 merits are min-max normalized over the best subsets of all metaheuristics, iteration budgets and rounds; Table 3 reports the raw values."*

## What to change in the manuscript

Status on 27 November 2026:

| # | Change | Status |
|---|---|---|
| 1 | Section 2.6.1, lines 432–436: the claim that the test set was unseen during feature selection | Open: depends on the E.1 decision |
| 2 | Tables 2 and 3: say that the merits were computed on the 396 rows, or replace them | Open: depends on the E.1 decision |
| 3 | Table 2 caption: say that the aCHI2 column is normalized | Open |
| 4 | Editor comment E.4: what the search added for the separable objectives | Open |

1. **Section 2.6.1, lines 432–436.** As written, the claim does not hold for the reported subsets. There are two ways to fix it, and choosing between them is part of editor comment E.1:
   - re-run the reducer on `train_normal_rtf.csv`, which changes the subsets and every result built on them;
   - or keep the subsets and state that they were selected on the full 396-row data set.
2. **Tables 2 and 3.** All merits were computed on the 396 rows. Say so, or replace them if the reducer is re-run.
3. **Table 2 caption.** State that the aCHI2 column is normalized (see above).
4. **Editor comment E.4.** On the data it searched, DE found exactly the top-10 ranking for aMI, aSU, aGR and aCHI2. It reached the maximum of Eq. (11), and this is why it "converged in the first round" (lines 564–566). For these objectives, the search added nothing over a ranking.

## Caveats

- `evidence/pipeline.py` and `pipeline_2.py` were last modified on 16 March 2026, and `pipeline_3.py` on 23 April 2025. Both dates are after the run (September 2023), so these files may not be exactly the code that ran. Evidence 1, 3 and 4 do not depend on the code: they use the run's output files and the data.
- The submitted test set had 40 rows: 30 real patients and 10 SMOTE rows. The revision evaluates only the 30 real patients. All 40 rows are in `normal_rtf.csv`.

## Files

| File | Content |
|---|---|
| [`reducer_input_check.ipynb`](reducer_input_check.ipynb) | All the checks, with their outputs saved. It reads only, writes nothing. |
| [`evidence/`](evidence/) | Unmodified copies of files of the old project, which is outside this repository. `evidence/.gitattributes` stops git from converting their line endings, so they stay byte-identical to the originals. |

The original files are in `/mnt/Cris-SW/Research/MetaheuristicCollection/`:

| Copy in `evidence/` | Original | Last modified | SHA-256 |
|---|---|---|---|
| `pipeline.py` | `pipeline.py` | 2026-03-16 | `97f38c534ab2ff55dd3aa52aea9d1fc7ee530fc66d14c73d7cf4ef414d186d4e` |
| `pipeline_2.py` | `pipeline_2.py` | 2026-03-16 | `e359c37eac34de7a16c30cbf28d2605daf534f14fb16829720e93f1935b53606` |
| `pipeline_3.py` | `pipeline_3.py` | 2025-04-23 | `4e555782f268ef2433d501e9448f8e99e4c52d314bc07b71d73b54ae7f346837` |
| `minimum_length_resume_step_1.csv` | `resumes/minimum_length_resume_step_1.csv` | 2023-09-21 | `5060c387e2db6b694d9764251397b72a25601f5b2a336276f02f9e39616eded1` |
| `minimum_length_resume_step_2.csv` | `resumes/minimum_length_resume_step_2.csv` | 2023-09-21 | `bf0e00efca516e102351e7c22f869947999369ca9b4b6f8d9932c8d629038f87` |
| `minimum_length_resume_step_3.csv` | `resumes/minimum_length_resume_step_3.csv` | 2023-09-21 | `fb28f7ff581a5e3528cc768263c3e4b3b8ba264a0fb3ef200f2028f86f42e1a6` |
| `processDataset.ipynb` | `notebooks/processDataset.ipynb` | 2023-09-25 | `326349373ad084aa9b2dbfd3f9a1e5c69067832c3ae841717fddbd7a23c6b4bc` |

The data files, `subset_r{1,2,3}.xlsx` (Tables 2–3), `ofunctions.py` and `baselines/top_k_filter.py` are already in the repository.

## How to reproduce

Open `reducer_input_check.ipynb` with the project's `.venv` kernel and run all cells. Section 5 uses Weka, which needs a Java runtime. Set `JAVA_HOME` in section 1; on the machine used for this check there is no system Java, so it points to the runtime bundled with JetBrains Rider. If `JAVA_HOME` is already set in the environment, that value is used instead. The Weka JVM can be started only once per kernel, so restart the kernel before running all cells again.
