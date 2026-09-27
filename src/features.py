"""
src/features.py
---------------
Feature engineering for the IEEE-CIS fraud dataset.

Critical rule: ALL statistics (missing-value thresholds, frequency maps,
imputation medians, scaler parameters) are fitted on the TRAIN split only,
then applied to validation and test.  Nothing from val/test leaks into train.
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
ARTIFACT    = PROC_DIR / "feature_artifacts.pkl"

# Columns dropped from encoding / frequency tables (identifiers + target)
ID_COLS     = ["TransactionID", "TransactionDT", "isFraud"]

# Frequency-encoding targets (fitted on train)
FREQ_COLS   = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "P_emaildomain", "R_emaildomain",
]

# Aggregation group for card1-based amount features
AGG_GROUP   = "card1"

# Missing threshold: drop columns with > 90 % missing (computed on train)
MISSING_THRESH = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_amt(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"].astype(np.float32))
    df["amt_decimal"]        = (df["TransactionAmt"] % 1).astype(np.float32)
    return df


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TransactionDT is seconds offset from a reference; mod to get hour/dow
    df["hour_of_day"]  = ((df["TransactionDT"] // 3600) % 24).astype(np.int8)
    df["day_of_week"]  = ((df["TransactionDT"] // 86400) % 7).astype(np.int8)
    return df


def _freq_encode(
    df: pd.DataFrame,
    freq_maps: dict[str, pd.Series],
) -> pd.DataFrame:
    df = df.copy()
    for col, freq_map in freq_maps.items():
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(freq_map).fillna(0).astype(np.float32)
    return df


def _card1_agg(
    df: pd.DataFrame,
    agg_stats: dict[str, float],
) -> pd.DataFrame:
    """Apply pre-computed card1 aggregation statistics (train-fitted)."""
    df = df.copy()
    df["card1_amt_mean"] = df[AGG_GROUP].map(agg_stats["mean"]).fillna(
        agg_stats["global_mean"]
    ).astype(np.float32)
    df["card1_amt_std"]  = df[AGG_GROUP].map(agg_stats["std"]).fillna(
        agg_stats["global_std"]
    ).astype(np.float32)
    df["amt_over_card1_mean"] = (
        df["TransactionAmt"] / (df["card1_amt_mean"] + 1e-6)
    ).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Fit (train only)
# ---------------------------------------------------------------------------

def fit_transform(
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Fit all feature-engineering artifacts on train, transform train.

    Returns
    -------
    X_train  : pd.DataFrame  (features, no isFraud)
    artifacts: dict          (all fitted objects needed for val/test)
    """
    print("Fitting feature artifacts on train ...")

    # 1. Drop high-missing columns (computed on train only)
    miss_rate   = train.isnull().mean()
    drop_cols   = miss_rate[miss_rate > MISSING_THRESH].index.tolist()
    # Always keep these even if somehow high-missing
    keep_always = ID_COLS + ["TransactionAmt"] + FREQ_COLS
    drop_cols   = [c for c in drop_cols if c not in keep_always]
    print(f"  Dropping {len(drop_cols)} columns with >{MISSING_THRESH:.0%} missing")

    train_fe = train.drop(columns=drop_cols, errors="ignore")

    # 2. Engineered features
    train_fe = _log_amt(train_fe)
    train_fe = _time_features(train_fe)

    # 3. Frequency encoding — fit on train
    freq_maps: dict[str, pd.Series] = {}
    for col in FREQ_COLS:
        if col in train_fe.columns:
            freq_maps[col] = train_fe[col].value_counts(normalize=True)
    train_fe = _freq_encode(train_fe, freq_maps)

    # 4. card1 aggregation stats — fit on train
    grp = train_fe.groupby(AGG_GROUP)["TransactionAmt"]
    agg_stats = {
        "mean":        grp.mean().astype(np.float32),
        "std":         grp.std().fillna(0).astype(np.float32),
        "global_mean": float(train_fe["TransactionAmt"].mean()),
        "global_std":  float(train_fe["TransactionAmt"].std()),
    }
    train_fe = _card1_agg(train_fe, agg_stats)

    # 5. Drop ID-like columns before returning features
    target    = train_fe["isFraud"].copy()
    drop_feat = [c for c in ID_COLS if c in train_fe.columns] + FREQ_COLS
    X_train   = train_fe.drop(columns=drop_feat, errors="ignore")

    # 6. Identify numeric columns for LR pipeline
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # 7. Impute + scale for Logistic Regression (fitted on train only)
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_num   = X_train[num_cols].values
    X_imp   = imputer.fit_transform(X_num)
    scaler.fit(X_imp)

    # 8. Collect remaining categorical columns (for LightGBM / XGBoost)
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")

    artifacts = {
        "drop_cols":   drop_cols,
        "freq_maps":   freq_maps,
        "agg_stats":   agg_stats,
        "imputer":     imputer,
        "scaler":      scaler,
        "num_cols":    num_cols,
        "cat_cols":    cat_cols,
        "feature_cols": X_train.columns.tolist(),
    }

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT, "wb") as f:
        pickle.dump(artifacts, f)
    print(f"  Artifacts saved -> {ARTIFACT}")
    print(f"  X_train shape: {X_train.shape}  |  fraud rate: {target.mean():.4%}")

    return X_train, target, artifacts


def transform(
    df: pd.DataFrame,
    artifacts: dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Apply fitted artifacts to validation or test set.
    No fitting — only transforming.
    """
    drop_cols  = artifacts["drop_cols"]
    freq_maps  = artifacts["freq_maps"]
    agg_stats  = artifacts["agg_stats"]
    cat_cols   = artifacts["cat_cols"]
    feat_cols  = artifacts["feature_cols"]

    df_fe = df.drop(columns=drop_cols, errors="ignore")
    df_fe = _log_amt(df_fe)
    df_fe = _time_features(df_fe)
    df_fe = _freq_encode(df_fe, freq_maps)
    df_fe = _card1_agg(df_fe, agg_stats)

    target   = df_fe["isFraud"].copy()
    drop_feat = [c for c in ID_COLS if c in df_fe.columns] + FREQ_COLS
    X = df_fe.drop(columns=drop_feat, errors="ignore")

    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Align columns to training set (add missing as NaN, drop extras)
    for col in feat_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feat_cols]

    return X, target


def get_lr_matrix(
    X: pd.DataFrame,
    artifacts: dict[str, Any],
    top_features: list[str] | None = None,
) -> np.ndarray:
    """
    Return imputed + scaled numeric matrix for Logistic Regression.

    If top_features is provided (e.g. top-50 by LightGBM importance),
    only those columns are used — this keeps the LR matrix tractable
    and avoids one-hot explosion on high-cardinality categoricals.
    """
    num_cols = artifacts["num_cols"]
    imputer  = artifacts["imputer"]
    scaler   = artifacts["scaler"]

    if top_features is not None:
        # Intersect with available numeric cols
        use_cols = [c for c in top_features if c in num_cols]
    else:
        use_cols = num_cols

    X_num = X[use_cols].values.astype(np.float64)

    # The imputer/scaler were fitted on ALL num_cols; refit on the subset
    # using the same strategy (this is safe — it's still train-only when
    # called after fit_transform).
    imp_sub = SimpleImputer(strategy="median")
    scl_sub = StandardScaler()
    # Caller is responsible for calling this only with train data first,
    # then reusing imp_sub/scl_sub for val/test.
    # To keep the API simple we return raw arrays and handle fit/transform
    # in train.py.
    return X_num, use_cols


def load_artifacts() -> dict[str, Any]:
    with open(ARTIFACT, "rb") as f:
        return pickle.load(f)
