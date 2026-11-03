"""Shared definitions + the fitted preprocessor (kept in its own module so that
joblib can unpickle it from anywhere: `from preprocessing import AdultPreprocessor`)."""
import numpy as np
import pandas as pd

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
TARGET = "income"
POSITIVE = ">50K"
DERIVED = ["capital_net", "hours_per_age"]


class AdultPreprocessor:
    """'?' / NaN -> missing; median (numeric) / mode (categorical) imputation;
    numeric clipping to [p1, p99]. Everything is learned in fit() from TRAIN only."""

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def _prep(self, df):
        X = df.copy()
        for c in NUMERIC:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        for c in CATEGORICAL:
            s = X[c].astype(object)
            X[c] = s.where(s.notna() & (s != "?"), np.nan)
        return X

    def fit(self, df):
        X = self._prep(df)
        self.medians_ = X[NUMERIC].median()
        self.lower_ = X[NUMERIC].quantile(self.lower_q)
        self.upper_ = X[NUMERIC].quantile(self.upper_q)
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in CATEGORICAL}
        return self

    def transform(self, df):
        X = self._prep(df)
        X[NUMERIC] = X[NUMERIC].fillna(self.medians_)
        X[NUMERIC] = X[NUMERIC].clip(lower=self.lower_, upper=self.upper_, axis=1)
        for c in CATEGORICAL:
            X[c] = X[c].fillna(self.modes_[c])
        return X


def add_derived(df):
    """Row-wise features -> no statistics from other rows -> cannot leak."""
    d = df.copy()
    d["capital_net"] = d["capital-gain"] - d["capital-loss"]
    d["hours_per_age"] = d["hours-per-week"] / (d["age"] - 16).clip(lower=1)
    return d
