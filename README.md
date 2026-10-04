# Seismic Bumps Hazard Prediction

Machine-learning pipeline that predicts **hazardous high-energy seismic events in underground coal mines**, built on the UCI *Seismic-bumps* dataset. The project covers data cleaning, feature engineering, exploratory data analysis (EDA), imbalanced-class classification, and energy regression.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Modeling & Results](#modeling--results)
  - [Task 1: Classification](#task-1-classification-predict-class)
  - [Task 2: Regression](#task-2-regression-predict-energy)
- [Key Takeaways](#key-takeaways)
- [Limitations & Future Work](#limitations--future-work)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset Source & Citation](#dataset-source--citation)

---

## Overview

Each record summarizes the seismic activity observed during an **8-hour work shift** in a Polish coal mine. The goal is to predict whether a **hazardous seismic bump (energy > 10⁴ J)** will occur during the **next shift**.

| Task | Target | Type |
|---|---|---|
| **1. Hazard prediction** | `class` (0 = non-hazardous, 1 = hazardous) | Binary classification, heavily imbalanced |
| **2. Energy prediction** | `energy` (total bump energy in the shift) | Regression |

The main challenge is **class imbalance**: only **170 of 2,584 shifts (6.58 %)** are hazardous, so plain accuracy is misleading and the work focuses on recall, F1, and ROC-AUC for the minority class.

---

## Dataset

- **Source:** [UCI Machine Learning Repository, Seismic-bumps](https://archive.ics.uci.edu/dataset/266/seismic+bumps)
- **Size:** 2,584 observations × 18 input features (+ `id` and `class`)
- **Quality:** no missing values, no duplicate rows

| Feature | Description |
|---|---|
| `seismic` | Hazard from the shift-long seismic method (`a` = safe, `b` = hazardous) |
| `seismoacoustic` | Hazard from the seismoacoustic method (`a`, `b`, `c`) |
| `shift` | Shift type: `W` = coal-getting (production), `N` = preparation |
| `genergy` | Seismic energy recorded by the most active geophone in the previous shift |
| `gpuls` | Number of pulses recorded by that geophone |
| `gdenergy` | % deviation of energy vs. the last 8 shifts |
| `gdpuls` | % deviation of pulses vs. the last 8 shifts |
| `ghazard` | Hazard assessed by the seismoacoustic method in the current shift |
| `nbumps` | Total number of seismic bumps recorded |
| `nbumps2` … `nbumps89` | Bump counts per energy range (10²–10³ J up to 10⁸–10¹⁰ J) |
| `energy` | Total energy of the bumps recorded in the shift |
| `maxenergy` | Maximum energy of a single bump in the shift |
| **`class`** | **Target:** 1 if a hazardous bump occurred in the next shift, else 0 |

---

## Project Workflow

```
Load & inspect ─► EDA ─► Feature engineering ─► Outlier handling (log1p)
        ─► Encoding ─► Drop constant columns ─► Stratified train/test split ─► Scaling ─► SMOTE
        ─► Modeling ─► Hyperparameter tuning ─► Threshold tuning (CV) ─► Test evaluation ─► Regression
```

---

## Exploratory Data Analysis

*(Notebook: `eda_team3.ipynb`)*

### 1. The target is highly imbalanced

93.42 % of shifts are safe and 6.58 % are hazardous. A model that always predicts "safe" would already score above 93 % accuracy.

![Target distribution](images/target_distribution.png)

### 2. Shift type is the strongest single predictor

Production shifts (`W`) are hazardous **9.2 %** of the time versus **1.8 %** for preparation shifts (`N`), a ~5× gap. The `seismic` method also separates the classes (`b` = 9.6 % vs. `a` = 4.9 %), while `seismoacoustic` and `ghazard` stay close to the 6.6 % base rate (`ghazard = c` shows 0 %, but it has only 30 rows, so this is noise).

![Hazard rate by categorical feature](images/categorical_vs_target.png)

### 3. Sensor features point the same way, but weakly

Hazardous shifts have higher median `genergy`, `gpuls`, `nbumps`, `energy`, and `maxenergy`, but the safe/hazardous distributions overlap heavily. 56.7 % of shifts have zero bumps (and therefore zero energy).

![genergy and gpuls distributions](images/genergy_gpuls_distribution.png)
![Numeric features vs target](images/numeric_vs_target.png)

### 4. Strong multicollinearity and some useless features

- `energy`, `maxenergy`, and `avg_energy_per_bump` are correlated at ≈ **1.00**.
- `genergy` and `gpuls` are correlated at **0.78**.
- `nbumps` has the strongest numeric correlation with the target (**0.25**).
- `gdenergy` and `gdpuls` have almost no relationship with the target (0.00 / 0.02 Pearson, still near zero with Spearman).
- `nbumps6`, `nbumps7`, `nbumps89` (and the derived `has_high_energy_bump`) are **constant zeros** and carry no information.

![Correlation matrix](images/correlation_matrix.png)

### 5. Outliers matter for some features

Share of hazardous shifts among IQR outliers vs. normal readings:

| Feature | Outlier shifts | Normal shifts | Reading |
|---|---|---|---|
| `nbumps` | **22.4 %** | 4.7 % | Strongest outlier → danger link |
| `genergy` | **17.8 %** | 5.9 % | Unusually high energy is a warning sign |
| `gdenergy` | 5.0 % | 6.7 % | No signal |
| `gdpuls` | 6.3 % | 6.6 % | No signal |
| `gpuls` | 0.0 % | 6.9 % | Outliers were not dangerous |

![Outlier analysis](images/outlier_analysis.png)

---

## Data Preprocessing

*(Notebook: `Data_Cleaning_modeling.ipynb`)*

### Feature engineering (12 new features)

| Group | Features |
|---|---|
| Bump counts | `has_any_bump`, `has_high_energy_bump`, `high_energy_bumps` (nbumps4–89), `low_energy_bumps` (nbumps2–3) |
| Energy ratios | `high_energy_ratio`, `avg_energy_per_bump`, `energy_per_pulse` |
| Deviation | `both_positive_dev`, `both_negative_dev`, `dev_magnitude` (√(gdenergy² + gdpuls²)) |
| Hazard agreement | `hazard_agreement` (all 3 hazard sources agree), `any_high_hazard` (any source flags `c`) |

### Outlier handling

Seven right-skewed columns were transformed with **`log1p`**: `genergy`, `gpuls`, `energy`, `maxenergy`, `avg_energy_per_bump`, `energy_per_pulse`, `dev_magnitude`. (`gdenergy` / `gdpuls` were left as-is because they contain negative values.)

| Column | Mean / Median before | Mean / Median after |
|---|---|---|
| `genergy` | 90,243 / 25,485 | 10.22 / 10.15 |
| `gpuls` | 538.6 / 379 | 5.77 / 5.94 |
| `dev_magnitude` | 71.6 / 54.1 | 3.96 / 4.01 |

### Encoding, splitting, scaling, balancing

1. **Encoding:** `OrdinalEncoder` for `seismic` (a < b), `seismoacoustic` and `ghazard` (a < b < c); one-hot (`drop_first`) for `shift` → `shift_W`.
2. **Drop constant columns:** `nbumps6`, `nbumps7`, `nbumps89` and `has_high_energy_bump` contain only zeros and were dropped before the split.
3. **Split:** stratified 80/20, `random_state=42` → **2,067 train / 517 test** (**26 features**; `id` and `class` excluded).
4. **Scaling:** `StandardScaler` fitted on the training set only.
5. **SMOTE** on the training set only: **1,931 : 136 → 1,931 : 1,931** (test set left untouched).

---

## Modeling & Results

### Task 1: Classification (predict `class`)

Test set: 517 shifts, of which **34 are hazardous**. All models were trained on the SMOTE-balanced training data.

#### Baseline models (threshold = 0.5)

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.75 | 0.16 | **0.62** | **0.25** | **0.7651** |
| Random Forest | 0.90 | 0.21 | 0.18 | 0.19 | 0.7245 |
| XGBoost | 0.92 | 0.29 | 0.15 | 0.20 | 0.7414 |

**The accuracy paradox:** Random Forest and XGBoost reach 90–92 % accuracy but catch only 15–18 % of hazardous shifts. Logistic Regression has lower accuracy (0.75) but detects about 21 of the 34 hazardous shifts, which is what matters in a safety-critical setting.

| Logistic Regression | Random Forest | XGBoost |
|---|---|---|
| ![LR](images/cm_logistic_regression.png) | ![RF](images/cm_random_forest.png) | ![XGB](images/cm_xgboost.png) |

#### Hyperparameter tuning (Random Forest)

`RandomizedSearchCV` (15 iterations, 3-fold CV, scoring = F1) selected:

```python
{'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5,
 'min_samples_leaf': 1, 'class_weight': 'balanced_subsample'}
```

At the default 0.5 threshold the tuned model gives precision 0.21, recall **0.15** and F1 0.17, so it does not beat the baseline. The decision threshold was tuned next.

#### Decision-threshold tuning (5-fold cross-validation on the training data)

To avoid choosing the threshold on the test set, thresholds were compared on **out-of-fold probabilities of the training data** (scaling and SMOTE are applied inside each fold). The threshold with the best F1 was selected, and the test set was used **once** at the end.

| Threshold | LR precision | LR recall | LR F1 | RF precision | RF recall | RF F1 |
|---|---|---|---|---|---|---|
| 0.2 | 0.08 | 0.89 | 0.15 | 0.16 | 0.60 | **0.25** |
| 0.3 | 0.10 | 0.84 | 0.18 | 0.16 | 0.38 | 0.23 |
| 0.4 | 0.12 | 0.76 | 0.21 | 0.19 | 0.30 | 0.24 |
| 0.5 | 0.14 | 0.65 | 0.24 | 0.17 | 0.15 | 0.16 |
| 0.7 | 0.21 | 0.42 | **0.28** | 0.23 | 0.04 | 0.07 |

(LR = Logistic Regression, RF = tuned Random Forest. Best thresholds: **0.7** for LR and **0.2** for RF.)

#### Final evaluation on the test set (517 shifts, 34 hazardous)

| Model | Threshold | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7 (CV-chosen) | 0.89 | **0.26** | 0.41 | **0.32** | **0.7651** |
| Tuned Random Forest | 0.2 (CV-chosen) | 0.79 | 0.17 | **0.56** | 0.26 | 0.7290 |

| Logistic Regression (0.7) | Tuned Random Forest (0.2) |
|---|---|
| ![LR final](images/cm_final_logistic_regression.png) | ![RF final](images/cm_final_random_forest.png) |

Lowering the threshold trades precision for recall. The F1-optimal threshold gives Logistic Regression the best F1 (0.32), while the tuned Random Forest at 0.2 catches more hazardous shifts (recall 0.56) with more false alarms. If missing a hazardous shift is considered more costly than a false alarm, a lower threshold can be chosen.

### Task 2: Regression (predict `energy`)

Features: numeric columns of the original data, excluding `energy`, `maxenergy`, `id` and `class`. An initial R² of ≈ 0.99 was traced to `maxenergy` leaking the target (≈ 91 % of feature importance), so it was removed.

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Linear Regression** | **6498.33** | 1951.44 | **0.8666** |
| Random Forest Regressor | 6863.94 | **1843.72** | 0.8511 |
| XGBoost Regressor | 7702.00 | 2090.58 | 0.8125 |

Top Random Forest feature importances: `nbumps5` (0.604), `nbumps4` (0.256), `gdenergy` (0.034), `gdpuls` (0.031), `nbumps3` (0.026).

---

## Key Takeaways

1. **Accuracy is the wrong metric** for this problem (6.6 % positives); recall, F1, and ROC-AUC on the hazardous class are what matter.
2. **Logistic Regression is the strongest baseline** (best recall 0.62 and ROC-AUC 0.7651); Random Forest and XGBoost miss most hazardous shifts at the default threshold.
3. **Threshold choice matters and is a business decision.** Chosen with cross-validation, it lifts Logistic Regression to F1 0.32 (precision 0.26, recall 0.41) on the test set, while the tuned Random Forest at 0.2 reaches recall 0.56 with more false alarms.
4. **Shift type, bump counts (`nbumps`), and `genergy`** are the most informative signals; `gdenergy` / `gdpuls` are the weakest.
5. For energy regression, **Linear Regression** has the best R² (0.8666) and RMSE, while Random Forest has the best MAE. These scores are optimistic because `nbumps4` / `nbumps5` determine `energy` almost by definition.

---

## Limitations & Future Work

Issues found in earlier versions have been fixed: `.astype(int)` no longer truncates the log-transformed features, thresholds are now chosen with cross-validation instead of on the test set, constant columns are dropped, and `id` / `class` are excluded from the regression features. Remaining points worth addressing:

- **Small positive class.** The test set has only 34 hazardous shifts, so metrics are noisy. Consider stratified k-fold CV and PR-AUC.
- **SMOTE before hyperparameter search.** `RandomizedSearchCV` runs on the already SMOTE-balanced training data, so synthetic samples can leak into its validation folds. Putting SMOTE inside each fold (as done for threshold tuning) would be cleaner.
- **Regression features.** `nbumps4` and `nbumps5` count bumps in the 10⁴–10⁶ J ranges, so they largely determine total `energy` by construction (≈ 86 % of the Random Forest importance). Treat the reported R² as an upper bound.
- **Reproducibility.** The EDA notebook reads a pre-transformation snapshot (`clean_data_for_EDA _brfore transformation.csv`) that the cleaning notebook, as submitted, does not write; save a copy of the data before the `log1p` step to regenerate it.

---

## Repository Structure

```
.
├── Data_Cleaning_modeling.ipynb   # Cleaning, feature engineering, encoding, SMOTE, models, tuning, regression
├── eda_team3.ipynb                # Full exploratory data analysis
├── images/                        # Plots used in this README
└── README.md
```

Data files expected by the notebooks:

| File | Used by |
|---|---|
| `csv_result-seismic-bumps-1.csv` | Raw data, read by the cleaning notebook |
| `clean_data_for_EDA.csv` | Output of the cleaning notebook (after `log1p`), read by the EDA notebook |
| `clean_data_for_EDA _brfore transformation.csv` | Pre-`log1p` snapshot, read by the EDA notebook |

---

## Getting Started

The notebooks were developed in **Google Colab** (data paths start with `/content/`); update the paths if you run them locally.

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
jupyter notebook
```

Suggested run order:

1. `Data_Cleaning_modeling.ipynb` — preprocessing, classification, tuning, regression
2. `eda_team3.ipynb` — exploratory analysis of the cleaned data

---

## Dataset Source & Citation

Dataset: [Seismic-bumps, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/266/seismic+bumps) (licensed CC BY 4.0).

> Sikora M., Wróbel Ł. *Application of rule induction algorithms for analysis of data collected by seismic hazard monitoring systems in coal mines.* Archives of Mining Sciences, 55(1), 2010, 91–114.
