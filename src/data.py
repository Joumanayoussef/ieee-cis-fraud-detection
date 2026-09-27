"""
src/data.py
-----------
Load, merge, downcasting, stratified sampling, time-based split, parquet cache.

Memory strategy
---------------
* 50 % stratified sample (~295 K rows, seed=42) — documented because free RAM
  was ~4.5 GB at build time; full 590 K rows need ~6 GB+ peak.
* Downcasting: float64 -> float32, int64 -> smallest int type.
* Parquet cache: after the first run the heavy CSVs are never re-read.
* Single copy rule: after each major transformation the source frame is deleted
  and gc.collect() is called before the next allocation.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR  = Path(__file__).parent.parent / "data" / "processed"
CACHE     = PROC_DIR / "merged_sample.parquet"

SAMPLE_FRAC = 0.50
SAMPLE_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.10
# test = remaining 20 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to the smallest safe type."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(force: bool = False) -> pd.DataFrame:
    """
    Load, merge, sample, and cache to parquet.

    Parameters
    ----------
    force : bool
        If True, re-read CSVs even when the cache exists.

    Returns
    -------
    pd.DataFrame  shape ~ (295_000, ~433)
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE.exists() and not force:
        print(f"Loading cached parquet: {CACHE}")
        df = pd.read_parquet(CACHE)
        print(f"  Loaded  {len(df):,} rows, {df.shape[1]} cols")
        return df

    # ---- transactions ----
    print("Reading train_transaction.csv ...")
    txn = pd.read_csv(
        RAW_DIR / "train_transaction.csv",
        dtype={
            "TransactionID": np.int32,
            "isFraud":       np.int8,
            "TransactionDT": np.int32,
        },
    )
    txn = _downcast(txn)
    print(f"  {len(txn):,} rows, {txn.shape[1]} cols — {txn.memory_usage(deep=True).sum()/1e6:.0f} MB")

    # ---- identity ----
    print("Reading train_identity.csv ...")
    idn = pd.read_csv(
        RAW_DIR / "train_identity.csv",
        dtype={"TransactionID": np.int32},
    )
    idn = _downcast(idn)
    print(f"  {len(idn):,} rows, {idn.shape[1]} cols — {idn.memory_usage(deep=True).sum()/1e6:.0f} MB")

    # ---- merge ----
    print("Left-merging on TransactionID ...")
    df = txn.merge(idn, on="TransactionID", how="left")
    del txn, idn
    gc.collect()
    print(f"  Merged: {len(df):,} rows, {df.shape[1]} cols")

    # ---- stratified 50 % sample ----
    print(f"Stratified {SAMPLE_FRAC:.0%} sample (seed={SAMPLE_SEED}) ...")
    fraud     = df[df["isFraud"] == 1].sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED)
    non_fraud = df[df["isFraud"] == 0].sample(frac=SAMPLE_FRAC, random_state=SAMPLE_SEED)
    del df
    gc.collect()

    df = pd.concat([fraud, non_fraud], ignore_index=True)
    del fraud, non_fraud
    gc.collect()

    # Sort by TransactionDT so the temporal split remains valid
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    fraud_rate = df["isFraud"].mean()
    print(f"  Sample: {len(df):,} rows | fraud rate: {fraud_rate:.4%}")
    print(f"  Memory: {df.memory_usage(deep=True).sum()/1e6:.0f} MB")

    print(f"Caching to {CACHE} ...")
    df.to_parquet(CACHE, index=False)
    print("  Done.")
    return df


def time_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-based split on TransactionDT.

    First 70 % -> train, next 10 % -> validation, last 20 % -> test.
    The data must already be sorted by TransactionDT (load_raw() guarantees this).

    Returns
    -------
    train, val, test  (DataFrames, each with isFraud column)
    """
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train = df.iloc[:n_train].copy()
    val   = df.iloc[n_train : n_train + n_val].copy()
    test  = df.iloc[n_train + n_val :].copy()

    for name, split in [("train", train), ("val", val), ("test", test)]:
        fr = split["isFraud"].mean()
        print(
            f"  {name:5s}: {len(split):>7,} rows | "
            f"fraud: {split['isFraud'].sum():>5,} ({fr:.3%}) | "
            f"DT [{split['TransactionDT'].min()}, {split['TransactionDT'].max()}]"
        )

    return train, val, test


if __name__ == "__main__":
    df = load_raw()
    print("\nTime-based split:")
    train, val, test = time_split(df)
