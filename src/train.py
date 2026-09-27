"""
src/train.py
------------
Train all five models, evaluate on the held-out test split, and write
reports/metrics.json.

Models
------
1. LogisticRegression (class_weight="balanced")          — interpretable baseline
2. LogisticRegression + SMOTE                            — resampling baseline
3. LightGBM (no class weighting)                        — gradient boosting
4. LightGBM (scale_pos_weight)                          — gradient boosting + imbalance
5. XGBoost  (scale_pos_weight)                          — gradient boosting + imbalance

Memory notes
------------
* SMOTE is applied inside an imblearn Pipeline so oversampling ONLY ever
  sees training data.
* LR uses the top-50 features by LightGBM importance (numeric only) to
  avoid one-hot explosion on high-cardinality categoricals.
* Models are saved to data/processed/ so explain.py can load them.
"""

from __future__ import annotations

import gc
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROC_DIR    = Path(__file__).parent.parent / "data" / "processed"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR  = PROC_DIR / "models"

TOP_LR_FEATURES = 50   # top features from LightGBM used in LR


# ---------------------------------------------------------------------------
# Threshold selection (on validation)
# ---------------------------------------------------------------------------

def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return probability threshold that maximises F1 on a validation set."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    best_idx = np.argmax(f1s[:-1])   # last element has no threshold
    return float(thresholds[best_idx])


def recall_at_90_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Recall at the lowest threshold where precision >= 90 %."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    mask = prec >= 0.90
    if not mask.any():
        return 0.0
    return float(rec[mask][0])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    cm     = confusion_matrix(y_true, y_pred).tolist()
    return {
        "model":            name,
        "roc_auc":          round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":           round(float(average_precision_score(y_true, y_prob)), 4),
        "threshold":        round(threshold, 4),
        "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall_at_90p":    round(recall_at_90_precision(y_true, y_prob), 4),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# LR helpers
# ---------------------------------------------------------------------------

def _make_lr_data(
    X: pd.DataFrame,
    top_features: list[str],
    imputer: SimpleImputer,
    scaler:  StandardScaler,
    fit: bool = False,
) -> np.ndarray:
    """Extract numeric top features, impute, scale."""
    num_top = [c for c in top_features if c in X.columns and X[c].dtype != "category"]
    X_sub = X[num_top].values.astype(np.float64)
    if fit:
        X_sub = imputer.fit_transform(X_sub)
        X_sub = scaler.fit_transform(X_sub)
    else:
        X_sub = imputer.transform(X_sub)
        X_sub = scaler.transform(X_sub)
    return X_sub, num_top


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_all(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val:   pd.DataFrame, y_val:   pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
) -> dict[str, Any]:
    """
    Train all five models; return dict of per-model metrics.
    Thresholds are chosen on the validation set.
    Test set is touched exactly once, for final reporting.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: list[dict] = []
    model_store: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 1. LightGBM — no weighting (train first to get feature importances) #
    # ------------------------------------------------------------------ #
    print("\n[1/5] LightGBM (no weighting) ...")
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    fraud_rate_train = pos / (neg + pos)

    cat_cols_lgb = [c for c in X_train.columns if X_train[c].dtype.name == "category"]

    lgb_ds_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols_lgb, free_raw_data=False)
    lgb_ds_val   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_cols_lgb, reference=lgb_ds_train, free_raw_data=False)

    params_lgb_base = {
        "objective":      "binary",
        "metric":         ["auc", "binary_logloss"],   # auc drives early stopping
        "learning_rate":  0.05,
        "num_leaves":     63,
        "min_child_samples": 50,
        "subsample":      0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":      0.1,
        "reg_lambda":     1.0,
        "verbose":        -1,
        "seed":           42,
    }

    cb_lgb_base = lgb.train(
        params_lgb_base,
        lgb_ds_train,
        num_boost_round=500,
        valid_sets=[lgb_ds_val],
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False), lgb.log_evaluation(-1)],
    )
    prob_val_lgb = cb_lgb_base.predict(X_val)
    thr_lgb      = best_f1_threshold(y_val.values, prob_val_lgb)
    prob_test_lgb = cb_lgb_base.predict(X_test)
    m = evaluate("LightGBM (no weight)", y_test.values, prob_test_lgb, thr_lgb)
    m["best_iteration"] = cb_lgb_base.best_iteration
    all_metrics.append(m)
    model_store["lgb_base"] = cb_lgb_base
    print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  F1={m['f1']}")

    # Feature importance for LR subset
    fi = pd.Series(
        cb_lgb_base.feature_importance("gain"),
        index=X_train.columns,
    ).sort_values(ascending=False)
    top_features = fi.head(TOP_LR_FEATURES).index.tolist()
    model_store["top_features"] = top_features
    model_store["feature_importance_lgb"] = fi

    del lgb_ds_train, lgb_ds_val
    gc.collect()

    # ------------------------------------------------------------------ #
    # 2. LightGBM — scale_pos_weight                                     #
    # ------------------------------------------------------------------ #
    print("\n[2/5] LightGBM (scale_pos_weight) ...")
    spw = neg / max(pos, 1)

    lgb_ds_train2 = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols_lgb, free_raw_data=False)
    lgb_ds_val2   = lgb.Dataset(X_val,   label=y_val,   categorical_feature=cat_cols_lgb, reference=lgb_ds_train2, free_raw_data=False)

    params_lgb_w = {**params_lgb_base, "scale_pos_weight": spw}
    cb_lgb_w = lgb.train(
        params_lgb_w,
        lgb_ds_train2,
        num_boost_round=500,
        valid_sets=[lgb_ds_val2],
        callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=False), lgb.log_evaluation(-1)],
    )
    prob_val_lgb_w  = cb_lgb_w.predict(X_val)
    thr_lgb_w       = best_f1_threshold(y_val.values, prob_val_lgb_w)
    prob_test_lgb_w = cb_lgb_w.predict(X_test)
    m = evaluate("LightGBM (scale_pos_weight)", y_test.values, prob_test_lgb_w, thr_lgb_w)
    m["best_iteration"] = cb_lgb_w.best_iteration
    all_metrics.append(m)
    model_store["lgb_weighted"] = cb_lgb_w
    print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  F1={m['f1']}")

    del lgb_ds_train2, lgb_ds_val2
    gc.collect()

    # ------------------------------------------------------------------ #
    # 3. XGBoost — scale_pos_weight                                      #
    # ------------------------------------------------------------------ #
    print("\n[3/5] XGBoost (scale_pos_weight) ...")

    # XGBoost needs numeric; convert categoricals to codes
    X_train_xgb = X_train.copy()
    X_val_xgb   = X_val.copy()
    X_test_xgb  = X_test.copy()
    for col in cat_cols_lgb:
        X_train_xgb[col] = X_train_xgb[col].cat.codes.astype(np.int16)
        X_val_xgb[col]   = X_val_xgb[col].cat.codes.astype(np.int16)
        X_test_xgb[col]  = X_test_xgb[col].cat.codes.astype(np.int16)

    dtrain = xgb.DMatrix(X_train_xgb, label=y_train, nthread=-1)
    dval   = xgb.DMatrix(X_val_xgb,   label=y_val,   nthread=-1)
    dtest  = xgb.DMatrix(X_test_xgb,  label=y_test,  nthread=-1)

    params_xgb = {
        "objective":        "binary:logistic",
        "eval_metric":      ["logloss", "auc"],
        "learning_rate":    0.05,
        "max_depth":        6,
        "min_child_weight": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "scale_pos_weight": spw,
        "seed":             42,
        "nthread":          -1,
        "tree_method":      "hist",
    }

    xgb_model = xgb.train(
        params_xgb,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    prob_val_xgb  = xgb_model.predict(dval)
    thr_xgb       = best_f1_threshold(y_val.values, prob_val_xgb)
    prob_test_xgb = xgb_model.predict(dtest)
    m = evaluate("XGBoost (scale_pos_weight)", y_test.values, prob_test_xgb, thr_xgb)
    m["best_iteration"] = xgb_model.best_iteration
    all_metrics.append(m)
    model_store["xgb"] = xgb_model
    model_store["xgb_cat_cols"] = cat_cols_lgb
    print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  F1={m['f1']}")

    del X_train_xgb, X_val_xgb, X_test_xgb, dtrain, dval, dtest
    gc.collect()

    # ------------------------------------------------------------------ #
    # 4. Logistic Regression — top-50 numeric features, class_weight      #
    # ------------------------------------------------------------------ #
    print("\n[4/5] Logistic Regression (class_weight=balanced) ...")
    print(f"  Using top-{TOP_LR_FEATURES} LightGBM features (numeric only)")

    lr_imputer = SimpleImputer(strategy="median")
    lr_scaler  = StandardScaler()

    X_train_lr, used_feats = _make_lr_data(X_train, top_features, lr_imputer, lr_scaler, fit=True)
    X_val_lr,   _          = _make_lr_data(X_val,   used_feats,   lr_imputer, lr_scaler, fit=False)
    X_test_lr,  _          = _make_lr_data(X_test,  used_feats,   lr_imputer, lr_scaler, fit=False)

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="saga",
        C=0.1,
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X_train_lr, y_train)
    prob_val_lr  = lr.predict_proba(X_val_lr)[:, 1]
    thr_lr       = best_f1_threshold(y_val.values, prob_val_lr)
    prob_test_lr = lr.predict_proba(X_test_lr)[:, 1]
    m = evaluate("LogisticRegression (balanced)", y_test.values, prob_test_lr, thr_lr)
    m["features_used"] = used_feats
    all_metrics.append(m)
    model_store["lr"] = lr
    model_store["lr_imputer"] = lr_imputer
    model_store["lr_scaler"]  = lr_scaler
    model_store["lr_features"] = used_feats
    print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  F1={m['f1']}")

    # ------------------------------------------------------------------ #
    # 5. LR + SMOTE                                                       #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Logistic Regression + SMOTE ...")
    print("  SMOTE applied inside imblearn Pipeline — only sees train data")

    smote_pipe = ImbPipeline([
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("lr",    LogisticRegression(
            max_iter=1000, solver="saga", C=0.1, random_state=42, n_jobs=-1,
        )),
    ])
    smote_pipe.fit(X_train_lr, y_train)
    prob_val_smote  = smote_pipe.predict_proba(X_val_lr)[:, 1]
    thr_smote       = best_f1_threshold(y_val.values, prob_val_smote)
    prob_test_smote = smote_pipe.predict_proba(X_test_lr)[:, 1]
    m = evaluate("LR + SMOTE", y_test.values, prob_test_smote, thr_smote)
    m["features_used"] = used_feats
    all_metrics.append(m)
    model_store["lr_smote"] = smote_pipe
    print(f"  ROC-AUC={m['roc_auc']}  PR-AUC={m['pr_auc']}  F1={m['f1']}")

    # ------------------------------------------------------------------ #
    # Store probabilities for curve plotting (returned to notebook)       #
    # ------------------------------------------------------------------ #
    model_store["test_probs"] = {
        "lgb_base":      prob_test_lgb,
        "lgb_weighted":  prob_test_lgb_w,
        "xgb":           prob_test_xgb,
        "lr":            prob_test_lr,
        "lr_smote":      prob_test_smote,
    }
    model_store["y_test"] = y_test.values

    # ------------------------------------------------------------------ #
    # Write metrics.json                                                  #
    # ------------------------------------------------------------------ #
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({"models": all_metrics, "sample_note": (
            "50% stratified sample (~295K rows) used due to 4.5 GB free RAM "
            "(full dataset requires ~6 GB+ peak). Seed=42. Sorted by "
            "TransactionDT before sampling, so time-based split remains valid."
        )}, f, indent=2)
    print(f"\nMetrics written -> {metrics_path}")

    # ------------------------------------------------------------------ #
    # Save model store                                                    #
    # ------------------------------------------------------------------ #
    store_path = MODELS_DIR / "model_store.pkl"
    with open(store_path, "wb") as f:
        pickle.dump(model_store, f)
    print(f"Models saved -> {store_path}")

    return all_metrics, model_store


def load_model_store() -> dict[str, Any]:
    store_path = MODELS_DIR / "model_store.pkl"
    with open(store_path, "rb") as f:
        return pickle.load(f)
