"""
Data loading & preprocessing for the multi-site UCI Heart Disease dataset.

Four real, independently-collected clinical sites are used as "environments":
  Cleveland   (Cleveland Clinic Foundation, USA)
  Hungary     (Hungarian Institute of Cardiology, Budapest)
  Switzerland (University Hospital, Zurich/Basel)
  VA          (V.A. Medical Center, Long Beach, USA)

Each site differs in acquisition protocol, patient population, and label
prevalence (e.g. Switzerland is >85% disease-positive after binarization,
Hungary is majority disease-negative), which produces genuine, documented
distribution shift -- not a synthetic split.

We keep the 11 features that are populated across ALL four sites (the UCI
docs and multiple independent analyses note that `ca` and `thal` are almost
entirely missing outside Cleveland, so we drop them to avoid manufacturing
missingness that doesn't reflect the real acquisition process).

Feature groups double as "modalities" for the missing-modality robustness
test, since they correspond to genuinely different measurement procedures:
  demographic        : age, sex
  resting/clinical    : cp, trestbps, chol, fbs, restecg
  exercise-test       : thalach, exang, oldpeak, slope
"""
import numpy as np
import pandas as pd
from pathlib import Path

COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

KEEP = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope"]

MODALITY_GROUPS = {
    "demographic": ["age", "sex"],
    "clinical": ["cp", "trestbps", "chol", "fbs", "restecg"],
    "exercise": ["thalach", "exang", "oldpeak", "slope"],
}

SITES = ["cleveland", "hungarian", "switzerland", "va"]
SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}


def load_site(raw_dir: Path, site: str) -> pd.DataFrame:
    path = raw_dir / SITE_FILES[site]
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    df = df[KEEP + ["num"]].copy()
    for c in KEEP:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["num"] = pd.to_numeric(df["num"], errors="coerce")
    df = df.dropna(subset=["num"])
    df["y"] = (df["num"] > 0).astype(int)
    df["site"] = site
    return df.reset_index(drop=True)


def load_all(raw_dir: Path) -> pd.DataFrame:
    dfs = [load_site(raw_dir, s) for s in SITES]
    return pd.concat(dfs, ignore_index=True)


def site_label_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("site")["y"].agg(["count", "mean"]).rename(
        columns={"count": "n", "mean": "disease_prevalence"})


def modality_column_indices():
    """Indices of KEEP columns belonging to each modality group."""
    idx = {}
    for name, cols in MODALITY_GROUPS.items():
        idx[name] = [KEEP.index(c) for c in cols]
    return idx


def fit_preprocessor(train_df: pd.DataFrame):
    """Median imputation + standardization fit ONLY on training environments."""
    X = train_df[KEEP].values.astype(float)
    medians = np.nanmedian(X, axis=0)
    X_imp = np.where(np.isnan(X), medians, X)
    mean = X_imp.mean(axis=0)
    std = X_imp.std(axis=0)
    std[std == 0] = 1.0
    return {"medians": medians, "mean": mean, "std": std}


def apply_preprocessor(df: pd.DataFrame, prep: dict) -> np.ndarray:
    X = df[KEEP].values.astype(float)
    X = np.where(np.isnan(X), prep["medians"], X)
    X = (X - prep["mean"]) / prep["std"]
    return X
