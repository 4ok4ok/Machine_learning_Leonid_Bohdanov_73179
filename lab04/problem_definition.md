# Task 1: Problem Definition

**Goal.** Predict whether a person's annual income exceeds $50K from 1994 US Census
demographic and employment attributes (UCI Adult, loaded with
`fetch_openml(name='adult', version=2, as_frame=True)`). This is a binary
classification problem. The point of the lab is not model quality but a trustworthy
data pipeline that others can reproduce.

**Inputs.** 14 features: 6 numerical (`age`, `fnlwgt`, `education-num`, `capital-gain`,
`capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`,
`marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).

**Output.** `income`: `>50K` (positive class, ~24%) vs `<=50K`.

**Success metrics.** Accuracy and F1 of the positive class (`>50K`) on a held-out test
set. Because classes are imbalanced (~76/24), accuracy alone is misleading (a majority
predictor already scores ~76%), so F1 is the primary metric and accuracy is reported
alongside it.

**Constraints.**
1. *No leakage:* the train/test split (70/30, stratified) happens before any cleaning.
   Imputation values (median / mode), clipping bounds (1st/99th percentile) and
   duplicate removal are computed on train only; test is only transformed.
   Derived features are row-wise, so they use no cross-row statistics.
2. *Reproducibility:* fixed seed, config in YAML, fitted preprocessor saved with
   joblib, SHA256 of the processed data stored, and the whole pipeline runs with
   `python run_pipeline.py`.
3. *Data quality is explicit:* schema, range, missing-value, duplicate and
   label-conflict checks are executed and reported; violations are reported, never
   silently dropped.

**Risks.** Data is from 1994 and US-only; it contains sensitive attributes (race, sex,
native-country), so any model built on it can encode societal bias (see `data_card.md`).
