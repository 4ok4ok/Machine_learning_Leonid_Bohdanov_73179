# Data Card: UCI Adult (Census Income), OpenML v2

## Source
UCI Machine Learning Repository "Adult" dataset (Becker & Kohavi, 1996), extracted from the
1994 US Census database; loaded via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
CC BY 4.0 as listed by the UCI repository (verify on the source page before redistribution).

## Size
- Raw: 48842 rows x 15 columns (14 features + `income`).
- Train: 34189 rows -> 34165 after removing
  24 exact duplicates. Test: 14653 rows (untouched).
- Class balance (positive `>50K`): 23.9%.
- Missing values: {'workclass': '5.7%', 'occupation': '5.8%', 'native-country': '1.8%'}.

## Processing summary
70/30 stratified split (seed 42) before cleaning; median/mode imputation and
1st/99th percentile clipping fitted on train only; processed data SHA256 =
`e13af11f6bea0de8b9a74e07029ae25b91cb45e04becbf6cc8e8cfe66c0e77f9`; fitted transformer in `results/preprocessor.joblib`.

## Quality findings
- Schema/range violations: 0 (see `results/validation_report.md`).
- Duplicate-conflict audit: 5 feature-identical groups with
  conflicting labels (10 rows).
- Random sample of 50 rows checked against the data dictionary:
  0 flagged (see `results/label_audit_sample50.csv`).
- Baseline LR: without derived features {'accuracy': 0.8539548215382515, 'f1': 0.6601016518424396}, with {'accuracy': 0.8577765645260357, 'f1': 0.6696258719086874}.

## Known biases
- 1994 US data: not representative of other countries or today; the $50K threshold is not
  inflation-adjusted.
- Heavily skewed `native-country` (mostly United-States) and `race`; men are
  over-represented, and income labels differ strongly by sex and race, so models can
  reproduce discrimination. Evaluate fairness before any real use.
- Missing values (workclass, occupation, native-country) are not random.
- `fnlwgt` is a census sampling weight, not a personal attribute, and is easy to misuse.
- Label noise: identical feature rows with different labels exist (see audit).

## Privacy concerns
Data is anonymised (no names/IDs), but quasi-identifiers (age, sex, race, native-country,
occupation) can enable re-identification when linked with other data. It includes
sensitive attributes (race, sex, marital status). Do not use for decisions about real
individuals.
