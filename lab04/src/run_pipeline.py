"""Lab 4 pipeline. Run from lab04/src:  python run_pipeline.py
Steps: ingest -> validate -> split -> clean (train-fit) -> derived features + LR
       -> label audit -> SHA256 + manifest -> data card."""
import hashlib
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocessing import (CATEGORICAL, DERIVED, NUMERIC, POSITIVE, TARGET,
                           AdultPreprocessor, add_derived)

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent                       # lab04/; paths in config.yaml are relative to it
log = logging.getLogger("lab4")

RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
EDU_MAP = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
           "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
           "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
           "Prof-school": 15, "Doctorate": 16}
LABELS = {"<=50K", ">50K"}


# ---------------------------------------------------------------- 2. ingestion
def ingest(cfg):
    d = cfg["dataset"]
    bunch = fetch_openml(name=d["name"], version=d["version"], as_frame=True)
    df = bunch.data.copy()
    df[TARGET] = bunch.target.values          # raw values, no cleaning
    log.info("INGEST rows=%d cols=%d", df.shape[0], df.shape[1])
    if df.shape != (d["expected_rows"], d["expected_cols"]):
        raise ValueError(f"Unexpected shape {df.shape}, expected "
                         f"({d['expected_rows']}, {d['expected_cols']})")
    raw_path = ROOT / cfg["paths"]["raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)          # snapshot of the raw data
    return df


# --------------------------------------------------------------- 3. validation
def is_missing(s):
    return s.isna() | (s.astype(object) == "?")


def validate(df):
    viol = []

    def add(check, detail, n):
        viol.append({"check": check, "detail": detail, "rows": int(n)})
        log.warning("VIOLATION %s | %s | rows=%d", check, detail, n)

    expected = NUMERIC + CATEGORICAL + [TARGET]
    for c in set(expected) - set(df.columns):
        add("schema", f"missing column {c}", 0)
    for c in set(df.columns) - set(expected):
        add("schema", f"unexpected column {c}", 0)

    for c in NUMERIC:                                       # int types
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            add("type", f"{c}: dtype {s.dtype} is not numeric", len(s))
        else:
            n = int(((s.dropna() % 1) != 0).sum())
            if n:
                add("type", f"{c}: non-integer values", n)
    for c in CATEGORICAL + [TARGET]:                        # category types
        if not (isinstance(df[c].dtype, pd.CategoricalDtype) or df[c].dtype == object):
            add("type", f"{c}: dtype {df[c].dtype} is not category", len(df))

    for c, (lo, hi) in RANGES.items():                      # ranges
        bad = df[c].notna() & ~df[c].between(lo, hi)
        if bad.any():
            add("range", f"{c} outside [{lo}, {hi}]", bad.sum())

    bad = ~df[TARGET].astype(str).isin(LABELS)              # allowed categories
    if bad.any():
        add("category", f"{TARGET} not in {sorted(LABELS)}", bad.sum())
    bad = ~df["sex"].astype(str).isin({"Male", "Female"})
    if bad.any():
        add("category", "sex not in {Male, Female}", bad.sum())

    missing = {c: int(is_missing(df[c]).sum()) for c in df.columns}
    missing = {c: n for c, n in missing.items() if n}
    n_nan = int(df[NUMERIC + CATEGORICAL].isna().sum().sum())
    n_q = int((df[CATEGORICAL].astype(object) == "?").sum().sum())
    n_dup = int(df.duplicated().sum())
    return {"rows": len(df), "cols": df.shape[1], "n_violations": len(viol),
            "violations": viol, "missing_per_column": missing,
            "missing_encoding": {"NaN_cells": n_nan, "question_mark_cells": n_q},
            "exact_duplicate_rows": n_dup,
            "dtypes": {c: str(t) for c, t in df.dtypes.items()}}


def write_validation(rep, out):
    (out / "validation_report.json").write_text(json.dumps(rep, indent=2))
    md = ["# Validation report", "",
          f"Rows: {rep['rows']}, columns: {rep['cols']}",
          f"Violations: **{rep['n_violations']}** (rows are reported, never dropped)", ""]
    for v in rep["violations"]:
        md.append(f"- `{v['check']}`: {v['detail']} (rows: {v['rows']})")
    md += ["", "## Missing values ('?' or NaN)", "", "| column | missing |", "|---|---|"]
    md += [f"| {c} | {n} |" for c, n in rep["missing_per_column"].items()]
    md += ["", f"Encoding found: {rep['missing_encoding']}",
           f"Exact duplicate rows (features + label): {rep['exact_duplicate_rows']}"]
    (out / "validation_report.md").write_text("\n".join(md))


# ------------------------------------------------------------ 5. LR evaluation
def evaluate(train, test, derived, seed):
    tr, te = train, test
    num = list(NUMERIC)
    if derived:
        tr, te = add_derived(tr), add_derived(te)
        num += DERIVED
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    model = Pipeline([("prep", ct), ("clf", LogisticRegression(max_iter=2000, random_state=seed))])
    y_tr = (tr[TARGET].astype(str) == POSITIVE).astype(int)
    y_te = (te[TARGET].astype(str) == POSITIVE).astype(int)
    model.fit(tr[num + CATEGORICAL], y_tr)
    pred = model.predict(te[num + CATEGORICAL])
    return {"accuracy": float(accuracy_score(y_te, pred)), "f1": float(f1_score(y_te, pred))}


# ------------------------------------------------------------- 6. label audit
def row_flags(r):
    f = []
    for c, (lo, hi) in RANGES.items():
        if pd.notna(r[c]) and not lo <= r[c] <= hi:
            f.append(f"{c} out of range")
    if pd.notna(r["education"]) and EDU_MAP.get(str(r["education"])) != r["education-num"]:
        f.append("education vs education-num mismatch")
    if str(r["sex"]) not in {"Male", "Female"}:
        f.append("invalid sex")
    if str(r[TARGET]) not in LABELS:
        f.append("invalid income label")
    return "; ".join(f)


def label_audit(df, cfg, out):
    feats = [c for c in df.columns if c != TARGET]
    g = df.groupby(feats, dropna=False, observed=True)[TARGET].agg(["nunique", "size"])
    conf = g[g["nunique"] > 1]
    sample = df.sample(cfg["audit"]["sample_size"], random_state=cfg["seed"]).copy()
    sample["flags"] = sample.apply(row_flags, axis=1)
    sample.to_csv(out / "label_audit_sample50.csv")
    res = {"conflict_groups": int(len(conf)), "conflict_rows": int(conf["size"].sum()),
           "sample_size": len(sample), "sample_rows_flagged": int((sample["flags"] != "").sum())}
    log.info("LABEL AUDIT %s", res)
    return res


# -------------------------------------------------------------------- 7. hash
def sha256_files(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def write_data_card(path, cfg, rep, stats, audit, metrics, digest):
    miss = {c: f"{n / rep['rows']:.1%}" for c, n in rep["missing_per_column"].items()}
    txt = f"""# Data Card: UCI Adult (Census Income), OpenML v2

## Source
UCI Machine Learning Repository "Adult" dataset (Becker & Kohavi, 1996), extracted from the
1994 US Census database; loaded via `fetch_openml(name='adult', version=2, as_frame=True)`.

## License
CC BY 4.0 as listed by the UCI repository (verify on the source page before redistribution).

## Size
- Raw: {rep['rows']} rows x {rep['cols']} columns (14 features + `income`).
- Train: {stats['train_raw']} rows -> {stats['train_clean']} after removing
  {stats['train_dups']} exact duplicates. Test: {stats['test']} rows (untouched).
- Class balance (positive `>50K`): {stats['pos_rate']:.1%}.
- Missing values: {miss}.

## Processing summary
70/30 stratified split (seed {cfg['seed']}) before cleaning; median/mode imputation and
1st/99th percentile clipping fitted on train only; processed data SHA256 =
`{digest}`; fitted transformer in `results/preprocessor.joblib`.

## Quality findings
- Schema/range violations: {rep['n_violations']} (see `results/validation_report.md`).
- Duplicate-conflict audit: {audit['conflict_groups']} feature-identical groups with
  conflicting labels ({audit['conflict_rows']} rows).
- Random sample of {audit['sample_size']} rows checked against the data dictionary:
  {audit['sample_rows_flagged']} flagged (see `results/label_audit_sample50.csv`).
- Baseline LR: without derived features {metrics['without_derived']}, with {metrics['with_derived']}.

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
"""
    Path(path).write_text(txt)


# ------------------------------------------------------------------------ main
def main(cfg=None):
    cfg = cfg or yaml.safe_load((SRC / "config.yaml").read_text())
    out = ROOT / cfg["paths"]["results"]
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "pipeline.log", mode="w")])
    seed = cfg["seed"]

    raw = ingest(cfg)
    rep = validate(raw)
    write_validation(rep, out)

    # split BEFORE any cleaning
    train, test = train_test_split(raw, test_size=cfg["split"]["test_size"],
                                   stratify=raw[TARGET], random_state=seed)
    n_train_raw = len(train)
    train = train.drop_duplicates()                      # train only; test untouched
    n_dups = n_train_raw - len(train)
    log.info("SPLIT train=%d test=%d; removed %d exact duplicates from train",
             n_train_raw, len(test), n_dups)

    pre = AdultPreprocessor(cfg["cleaning"]["lower_quantile"],
                            cfg["cleaning"]["upper_quantile"]).fit(train)   # train only
    joblib.dump(pre, out / "preprocessor.joblib")
    train_c, test_c = pre.transform(train), pre.transform(test)
    train_c.to_csv(out / "train_processed.csv", index=False, float_format="%.10g")
    test_c.to_csv(out / "test_processed.csv", index=False, float_format="%.10g")

    metrics = {"without_derived": evaluate(train_c, test_c, False, seed),
               "with_derived": evaluate(train_c, test_c, True, seed)}
    log.info("METRICS %s", metrics)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    audit = label_audit(raw, cfg, out)

    digest = sha256_files([out / "train_processed.csv", out / "test_processed.csv"])
    log.info("PROCESSED SHA256 %s", digest)
    manifest = {"config": cfg, "processed_sha256": digest,
                "files_hashed": ["train_processed.csv", "test_processed.csv"],
                "versions": {"python": sys.version.split()[0], "sklearn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__}}
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    stats = {"train_raw": n_train_raw, "train_clean": len(train_c), "train_dups": n_dups,
             "test": len(test_c), "pos_rate": float((raw[TARGET].astype(str) == POSITIVE).mean())}
    write_data_card(ROOT / cfg["paths"]["data_card"], cfg, rep, stats, audit, metrics, digest)
    log.info("DONE")


if __name__ == "__main__":
    main()
