# Lab 04: Data pipeline and quality

Code: `src/` (run `cd lab04/src && python run_pipeline.py`). Outputs: `results/`. Config: `src/config.yaml`. Data card: `results/data_card.md`. Environment: `results/versions.txt` (made by `src/print_versions.py`). Seed 42.

## 1. Goal

Build a reproducible, leakage-free data pipeline for the UCI Adult dataset (`fetch_openml(name='adult', version=2, as_frame=True)`) and check its quality before any modelling.

- **Task:** predict whether a person's income is above 50K from 14 census features (6 numeric, 8 categorical). The target `income` is `>50K` (about 24%) or `<=50K`.
- **Success metrics:** F1 of the positive class and accuracy on a held-out test set. The classes are imbalanced (about 76/24), so a model that always answers `<=50K` already reaches about 76% accuracy. F1 is therefore the main metric.
- **Constraints:** no leakage (the split comes first, and every statistic is learned on train only) and a reproducible run (fixed seed, YAML config, saved transformer, hash of the processed data, one command).
- **Risks:** the data is from 1994, covers the US only and contains sensitive attributes (race, sex), see the data card.

## 2. Method

1. **Ingestion.** The raw table is read with `fetch_openml` and saved unchanged. The pipeline logs 48842 rows and 15 columns and stops if the shape differs.
2. **Validation.** The schema (14 features plus `income`), the integer and category types, the ranges (age 17-90, hours 1-99, capital-gain 0-99999, capital-loss 0-4356), the allowed categories, missing values and exact duplicates are checked on the raw data. Violations are reported and no row is dropped.
3. **Split and cleaning.** A 70/30 stratified split is done before any cleaning. On train only, the pipeline fits a median imputer (numeric) and a mode imputer (categorical), computes the 1st and 99th percentiles for clipping, and drops exact duplicates. The test set is only transformed, never fitted on and never de-duplicated. The fitted transformer is saved with `joblib.dump` to `results/preprocessor.joblib`.
4. **Derived features.** `capital_net = capital-gain - capital-loss` and `hours_per_age = hours-per-week / (age - 16)`. Both are row-wise, so they use no statistics from other rows and cannot leak. I compare a `LogisticRegression` baseline (standardised numeric features, one-hot categorical features) with and without them.
5. **Label audit.** I look for feature-identical rows with conflicting labels. I also draw 50 random rows and check each against the data dictionary (ranges, `education` vs `education-num`, valid `sex` and `income`).
6. **Reproducibility.** A SHA256 hash of the processed train and test files is stored in `results/manifest.yaml` together with the config and library versions.

## 3. Results

**Ingestion and split**

| Item | Value |
|---|---|
| Raw rows x columns | 48842 x 15 |
| Train after split | 34189 rows |
| Exact duplicates removed from train | 24 (train is 34165 rows) |
| Test (untouched) | 14653 rows |

**Validation (`results/validation_report.md`)**

| Check | Result |
|---|---|
| Schema, type, range and category violations | 0 |
| Missing `workclass` | 2799 (5.7%) |
| Missing `occupation` | 2809 (5.8%) |
| Missing `native-country` | 857 (1.8%) |
| Exact duplicate rows in the raw data | 52 |

The missing values arrive as `NaN` (6465 cells), not as the `?` string the task describes. The pipeline treats both as missing.

**Logistic regression baseline (test set)**

| Features | Accuracy | F1 |
|---|---|---|
| Without derived features | 0.8540 | 0.6601 |
| With `capital_net` and `hours_per_age` | 0.8578 | 0.6696 |

**Label quality (`results/label_audit_sample50.csv`)**

| Check | Result |
|---|---|
| Groups of identical features with conflicting income | 5 groups, 10 rows |
| Random sample of 50 rows with impossible values | 0 flagged |

**Reproducibility.** SHA256 of the processed data: `e13af11f6bea0de8b9a74e07029ae25b91cb45e04becbf6cc8e8cfe66c0e77f9`.

## 4. Conclusions

1. The order of steps is what keeps the pipeline leakage-free. Splitting first and learning medians, modes, clipping bounds and the duplicate list from train alone means the test score is an honest estimate. The test set is only transformed, so it stays at 14653 rows.
2. The data is clean in format but not in content. There were no schema or range violations, yet about 6% of `workclass` and `occupation` values are missing, 52 rows are exact duplicates and 5 feature-identical groups carry different labels. These conflicts (10 rows) are a small but real limit on how well any model can do.
3. The derived features helped a little: accuracy rose by 0.004 and F1 by 0.010. This is one deterministic run, so I read it as a small improvement and not as proof. Accuracy alone would hide this, because the majority-class baseline is already around 76%. The saved hash, config and one-command run let anyone check these numbers.

## 5. Limitations

- Task 6 allows either a duplicate-conflict audit or a manual check. I did both, but the "manual" check of the 50 rows is rule-based, with code comparing each row to the data dictionary.
- The conflict audit uses the full raw data (before the split), so it describes the dataset and not only the train set.
