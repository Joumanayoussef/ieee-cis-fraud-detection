"""
src/explain.py
--------------
SHAP TreeExplainer, LIME, and permutation importance for the best model.
All plots saved to reports/figures/.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
SHAP_SAMPLE = 10_000
PERM_SAMPLE = 5_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    p = FIGURES_DIR / name
    fig.savefig(p, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {p.name}")
    return p


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def shap_analysis(
    model,
    X_test: pd.DataFrame,
    sample_size: int = SHAP_SAMPLE,
    seed: int = 42,
) -> dict[str, Any]:
    """
    SHAP TreeExplainer on a sample of the test set.

    Returns
    -------
    dict with shap_values, sample indices, and top-feature ranking
    """
    print(f"\nSHAP analysis on {sample_size:,}-row test sample ...")
    rng   = np.random.default_rng(seed)
    idx   = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    X_samp = X_test.iloc[idx].reset_index(drop=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_samp)

    # -- Global beeswarm --
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Beeswarm — Global Feature Impact (LightGBM)", fontsize=13)
    plt.tight_layout()
    _save(fig, "shap_beeswarm.png")

    # -- Global bar --
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Mean |value| — Top 20 Features", fontsize=13)
    plt.tight_layout()
    _save(fig, "shap_bar.png")

    # -- Top 3 dependence plots --
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    top3     = pd.Series(mean_abs, index=X_samp.columns).nlargest(3).index.tolist()
    for feat in top3:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, feat], show=False, ax=ax)
        ax.set_title(f"SHAP Dependence — {feat}", fontsize=12)
        plt.tight_layout()
        _save(fig, f"shap_dependence_{feat.replace('/', '_')[:40]}.png")

    return {
        "shap_values": shap_values,
        "X_sample":    X_samp,
        "sample_idx":  idx,
        "top_features": top3,
        "feature_ranking": pd.Series(mean_abs, index=X_samp.columns).sort_values(ascending=False),
    }


def shap_waterfall(
    shap_result: dict[str, Any],
    y_test_sample: np.ndarray,
    y_prob_sample: np.ndarray,
    threshold: float,
) -> None:
    """Waterfall plots for one true positive and one false positive."""
    shap_values = shap_result["shap_values"]
    y_pred      = (y_prob_sample >= threshold).astype(int)
    y_true      = y_test_sample

    # True positive: fraud predicted as fraud
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)

    for label, mask, fname in [
        ("True Positive",  tp_mask,  "shap_waterfall_tp.png"),
        ("False Positive", fp_mask,  "shap_waterfall_fp.png"),
    ]:
        idx_arr = np.where(mask)[0]
        if len(idx_arr) == 0:
            print(f"  No {label} found — skipping waterfall")
            continue
        i = idx_arr[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[i], max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {label}", fontsize=12)
        plt.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def lime_explanation(
    model,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_test:  np.ndarray,
    y_prob:  np.ndarray,
    threshold: float,
    seed: int = 42,
) -> None:
    """LIME explanation for one true positive."""
    print("\nLIME explanation for one true positive ...")

    y_pred  = (y_prob >= threshold).astype(int)
    tp_mask = (y_test == 1) & (y_pred == 1)
    tp_idx  = np.where(tp_mask)[0]
    if len(tp_idx) == 0:
        print("  No true positive found — skipping LIME")
        return
    i = tp_idx[0]

    # Encode categorical columns as integer codes — LIME only accepts floats
    cat_cols = X_train.select_dtypes(include="category").columns.tolist()
    cat_categories = {col: X_train[col].cat.categories for col in cat_cols}

    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()
    for col in cat_cols:
        X_train_enc[col] = X_train_enc[col].cat.codes.astype(np.float64)
        X_test_enc[col]  = X_test_enc[col].cat.codes.astype(np.float64)

    feature_names = X_train_enc.columns.tolist()
    X_train_np = X_train_enc.values.astype(np.float64)
    X_test_np  = X_test_enc.values.astype(np.float64)

    # Replace NaN with column medians (LIME doesn't handle NaN)
    col_medians = np.nanmedian(X_train_np, axis=0)
    for j in range(X_train_np.shape[1]):
        X_train_np[np.isnan(X_train_np[:, j]), j] = col_medians[j]
        X_test_np[np.isnan(X_test_np[:, j]), j]   = col_medians[j]

    def predict_fn(x):
        df_x = pd.DataFrame(x, columns=feature_names)
        for col in cat_cols:
            codes = df_x[col].round().astype(int).clip(0, len(cat_categories[col]) - 1)
            df_x[col] = pd.Categorical.from_codes(codes, categories=cat_categories[col])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df_x)
            return proba
        # Fallback for models that only expose predict (returns 0/1)
        p = model.predict(df_x).astype(np.float64)
        return np.column_stack([1 - p, p])

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        class_names=["Not Fraud", "Fraud"],
        mode="classification",
        random_state=seed,
        discretize_continuous=True,
    )

    exp = explainer.explain_instance(
        data_row=X_test_np[i],
        predict_fn=predict_fn,
        num_features=15,
        num_samples=500,
    )

    fig = exp.as_pyplot_figure(label=1)
    fig.suptitle(f"LIME Explanation — True Positive (test row {i})", fontsize=12)
    plt.tight_layout()
    _save(fig, "lime_true_positive.png")


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_plot(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    shap_ranking: pd.Series,
    sample_size: int = PERM_SAMPLE,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance on a validation sample; compare to SHAP ranking.
    """
    print(f"\nPermutation importance on {sample_size:,}-row val sample ...")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_val), size=min(sample_size, len(X_val)), replace=False)
    X_sub = X_val.iloc[idx]
    y_sub = y_val.iloc[idx]

    # Manual permutation importance — works with any model that has .predict()
    # (sklearn's version requires a sklearn estimator with .fit())
    def _predict_proba(X_df):
        p = model.predict(X_df)
        return p if p.ndim == 1 else p[:, 1]

    n_repeats = 5
    rng = np.random.default_rng(seed)
    baseline = roc_auc_score(y_sub.values, _predict_proba(X_sub))
    feats = X_sub.columns.tolist()
    means, stds = [], []
    for col in feats:
        drops = []
        is_cat = X_sub[col].dtype.name == "category"
        cat_cats = X_sub[col].cat.categories if is_cat else None
        for _ in range(n_repeats):
            X_p = X_sub.copy()
            if is_cat:
                perm_codes = rng.permutation(X_sub[col].cat.codes.values)
                X_p[col] = pd.Categorical.from_codes(perm_codes, categories=cat_cats)
            else:
                X_p[col] = rng.permutation(X_sub[col].values)
            drops.append(baseline - roc_auc_score(y_sub.values, _predict_proba(X_p)))
        means.append(float(np.mean(drops)))
        stds.append(float(np.std(drops)))

    perm_df = pd.DataFrame({
        "feature":   feats,
        "perm_mean": means,
        "perm_std":  stds,
    }).sort_values("perm_mean", ascending=False).head(30).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    # Permutation
    axes[0].barh(perm_df["feature"][::-1], perm_df["perm_mean"][::-1], xerr=perm_df["perm_std"][::-1])
    axes[0].set_title("Permutation Importance (top 30)", fontsize=12)
    axes[0].set_xlabel("Mean decrease in ROC-AUC")
    # SHAP comparison
    shap_top30 = shap_ranking.head(30)
    axes[1].barh(shap_top30.index[::-1], shap_top30.values[::-1])
    axes[1].set_title("SHAP Mean |value| (top 30)", fontsize=12)
    axes[1].set_xlabel("Mean |SHAP value|")
    plt.suptitle("Permutation vs SHAP Ranking", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "permutation_vs_shap.png")

    return perm_df


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr_curves(
    test_probs: dict[str, np.ndarray],
    y_test: np.ndarray,
    metrics: list[dict],
) -> None:
    """Plot ROC and PR curves for all models."""
    from sklearn.metrics import roc_curve, precision_recall_curve

    label_map = {
        "lgb_base":     "LightGBM (no weight)",
        "lgb_weighted": "LightGBM (scale_pos_weight)",
        "xgb":          "XGBoost (scale_pos_weight)",
        "lr":           "LR (balanced)",
        "lr_smote":     "LR + SMOTE",
    }
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for (key, prob), color in zip(test_probs.items(), colors):
        label = label_map.get(key, key)
        # ROC
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = next((m["roc_auc"] for m in metrics if m["model"] == label), None)
        axes[0].plot(fpr, tpr, color=color, lw=1.8, label=f"{label} (AUC={auc_val})")
        # PR
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap_val = next((m["pr_auc"] for m in metrics if m["model"] == label), None)
        axes[1].plot(rec, prec, color=color, lw=1.8, label=f"{label} (AP={ap_val})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — All Models"); axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PR Curves — All Models"); axes[1].legend(fontsize=8)

    plt.tight_layout()
    _save(fig, "roc_pr_curves.png")
