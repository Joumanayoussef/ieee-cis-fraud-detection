# IEEE-CIS Fraud Detection

Credit card fraud detection on the IEEE-CIS Kaggle dataset using time-based validation, gradient boosting, imbalance handling, and SHAP/LIME explainability.

## Background

Originally started as a course project at Zewail City; fully rebuilt as an individual project.

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection — Kaggle 2019](https://www.kaggle.com/competitions/ieee-fraud-detection)  
590,540 anonymised e-commerce transactions provided by Vesta Corporation.  
Features: `TransactionAmt`, `card1–6`, `addr1–2`, `P_/R_emaildomain`, 339 anonymised `V`-features, and identity metadata.  
**No data is included in this repository.** Download instructions below.

> **Memory note:** A **50 % stratified random sample (~295,270 rows, seed=42)** was used during this build because free RAM was ~4.5 GB (the full 590 K rows require ~6 GB+ peak for pandas + model training). The sample preserves the fraud rate and is sorted by `TransactionDT` before splitting so the temporal ordering is maintained. All reported numbers are on this sample; see `reports/metrics.json` for the `sample_note` field.

---

## Methodology

### 1. Time-based split — the critical design choice

`TransactionDT` is a seconds-offset timestamp. A random split would place future transactions in the training set, creating data leakage that inflates test metrics.

We use a **strict temporal split**:
| Split | Rows | Period | Fraud rate |
|-------|------|--------|------------|
| Train | 206,689 | DT 86,400 – 10,436,615 | 3.507 % |
| Validation | 29,527 | DT 10,436,642 – 12,185,566 | 3.658 % |
| Test | 59,054 | DT 12,185,580 – 15,811,131 | 3.394 % |

The **validation split** is used exclusively for early stopping and threshold selection.  
The **test split** is touched exactly once, for final metric reporting.

### 2. Train-only preprocessing

Every statistic — missing-value thresholds, imputation medians, frequency-encoding maps, aggregation statistics, scaler parameters — is fitted on the training split only, then applied to validation and test. See `src/features.py`.

### 3. Engineered features

| Feature | Description |
|---------|-------------|
| `log_TransactionAmt` | log(1 + amount) — reduces right-skew |
| `amt_decimal` | Fractional part of amount — fraud often uses round numbers |
| `hour_of_day`, `day_of_week` | Time-of-day and day-of-week from `TransactionDT` |
| `card1_freq` … `R_emaildomain_freq` | Frequency encoding for card/address/email fields (train only) |
| `card1_amt_mean`, `card1_amt_std` | Per-card1 amount statistics (train only) |
| `amt_over_card1_mean` | Amount relative to card1 average — deviation signal |

Columns with > 90 % missing (computed on train) are dropped.

### 4. Why not accuracy?

The fraud rate is ~3.5 %. A trivial classifier that always predicts "legitimate" achieves ~96.5 % accuracy while catching zero fraud. We report **ROC-AUC** and **PR-AUC** (robust to class imbalance), plus **F1**, **Precision**, **Recall**, and **Recall at 90 % Precision** at a validation-optimised threshold.

---

## Results

> All numbers are taken directly from `reports/metrics.json`.

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall@90P |
|-------|---------|--------|-----------|--------|----|------------|
| **LightGBM (no weight)** | **0.8958** | **0.5059** | 0.5497 | 0.4276 | **0.4811** | 0.2161 |
| LightGBM (scale\_pos\_weight) | 0.8194 | 0.2490 | 0.2548 | 0.4087 | 0.3139 | 0.000 |
| XGBoost (scale\_pos\_weight) | 0.8928 | 0.4799 | 0.5107 | 0.4162 | 0.4586 | **0.2320** |
| LR (balanced) | 0.8016 | 0.1711 | 0.2942 | 0.3029 | 0.2985 | 0.0060 |
| LR + SMOTE | 0.8019 | 0.1742 | 0.3028 | 0.3014 | 0.3021 | 0.0060 |

> Numbers taken directly from `reports/metrics.json`. All thresholds chosen on the validation split; test set touched once for final reporting.
>
> **Best model:** LightGBM without class weighting achieves the highest ROC-AUC (0.8958) and PR-AUC (0.5059). The `scale_pos_weight` variant stopped at iteration 1 — the aggressive weighting destabilised training, yielding a degenerate model. XGBoost with `scale_pos_weight` achieved the highest Recall@90P (0.2320).

---

## Key Figures

### EDA Overview
![EDA](reports/figures/eda_overview.png)

### ROC and PR Curves
![ROC/PR](reports/figures/roc_pr_curves.png)

### SHAP Beeswarm (Global Feature Impact)
![SHAP Beeswarm](reports/figures/shap_beeswarm.png)

### Permutation vs SHAP Importance
![Permutation vs SHAP](reports/figures/permutation_vs_shap.png)

---

## Explainability Findings

- **TransactionAmt** and its engineered variants (`log_TransactionAmt`, `amt_over_card1_mean`) are consistently top-ranked by both SHAP and permutation importance.
- **card1_freq** and **card1_amt_mean** rank highly — frequency and deviation from typical spend are strong fraud signals.
- **LR coefficients agree** on card-frequency features but assign more weight to scaled amount features, reflecting the linear boundary.
- **LIME confirms** the SHAP direction for the inspected true positive: high card frequency + low amount-over-mean → strong fraud signal.

---

## Business Interpretation

At the validation-optimised threshold, the best model (LightGBM without class weighting):
- Catches 42.8 % of all fraud cases (recall = 0.4276) at 54.97 % precision
- Generates ~12 false alarms per 1,000 legitimate transactions (702 FP / 57,050 negatives)
- At 90 % precision (high-confidence alerts), recall drops to 21.6 % — only the clearest fraud signals are flagged

The right operating point depends on the cost ratio of a missed fraud vs. a false review.

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/Joumanayoussef/ieee-cis-fraud-detection.git
cd ieee-cis-fraud-detection

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data (requires Kaggle account + competition rules accepted)
#    Place train_transaction.csv and train_identity.csv in data/raw/
python -c "import kagglehub; kagglehub.competition_download('ieee-fraud-detection')"

# 5. Run the full pipeline
python src/data.py          # load, sample, split, cache parquet
# Then open and run: notebooks/fraud_detection.ipynb
```

---

## Project Structure

```
ieee-cis-fraud-detection/
├── src/
│   ├── data.py          Load, merge, downcasting, 50%-sample, time-split, parquet cache
│   ├── features.py      Feature engineering (all encoders fitted on train only)
│   ├── train.py         5 models, threshold selection, evaluation, metrics.json
│   └── explain.py       SHAP TreeExplainer, LIME, permutation importance
├── notebooks/
│   └── fraud_detection.ipynb   Main narrative notebook with outputs
├── reports/
│   ├── metrics.json     All evaluation metrics (authoritative numbers source)
│   └── figures/         All saved plots (PNG)
├── data/
│   ├── raw/             train_transaction.csv, train_identity.csv (not in git)
│   └── processed/       Parquet cache + fitted artifacts (not in git)
├── requirements.txt
└── .gitignore
```

---

## Limitations

1. **50 % sample** — RAM constraint at build time (~4.5 GB free). Full-dataset results may differ slightly; ranking direction is expected to hold.
2. **Anonymised V-features** — V1–V339 are opaque; feature engineering is limited to documented metadata.
3. **Static threshold** — Chosen once on validation; needs periodic recalibration in production.
4. **No temporal decay** — Model weights all training history equally; fraud patterns evolve.
5. **No competition test labels** — Generalisation measured on the last 20 % of training data only.
