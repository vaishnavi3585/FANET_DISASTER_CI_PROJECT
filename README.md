# FANET Disaster Dataset — EDA, Preprocessing, and Comparative ML Pipeline

## Overview
This pipeline performs comprehensive exploratory data analysis, leak-free preprocessing, and multi-model benchmarking on the **FANET Disaster Dataset** (5,000 records × 22 columns).

Four distinct machine learning paradigms are implemented, trained, and benchmarked on identical holdout test data:
1. **Random Forest Classifier** (`n_estimators=100`, `class_weight='balanced'`, `n_jobs=-1`) — Non-linear Ensemble
2. **Decision Tree Classifier** (`class_weight='balanced'`) — Interpretable Non-linear Tree
3. **Support Vector Machine (SVM)** (`kernel='rbf'`, `probability=True`, `class_weight='balanced'`) — Non-linear Kernel Method
4. **Logistic Regression** (`max_iter=1000`, `class_weight='balanced'`) — Linear Discriminative Baseline

---

## What this Pipeline Does
- **Loads `FANET Disaster Dataset.csv` accurately**:
  - Handles the `Attack_Type` column where `'None'` represents normal benign traffic (No Cyber Attack, 988 records, ~19.8% alongside Blackhole, Jamming, Replay, and Sybil).
  - Preserves true categories and confirms that the dataset contains **0 missing values** across all 22 columns and 5,000 records.
- **Dataset Understanding & Summary Statistics**:
  - Reports dataset dimensions (5,000 rows × 22 columns), data types, unique counts, duplicates, and descriptive statistics.
  - Exports column metadata and summary statistics to `outputs/reports/dataset_summary.txt` and `outputs/reports/column_overview.csv`.
- **Publication-Quality Exploratory Visualizations**:
  - `01_missing_values.png`: Visual verification of missing values (0 missing).
  - `02_target_distribution.png`: Distribution of `Network_Status` (`Normal`, `Degraded`, `Critical`) with exact counts and percentages.
  - `03_numerical_distributions.png`: 4×4 grid of histograms with KDE curves for all 16 numerical features.
  - `04_correlation_heatmap.png`: Annotated correlation matrix of telemetry features.
  - `05_key_features_by_network_status.png`: Multi-panel boxplots illustrating how key discriminators (`Latency`, `Packet_Delivery_Ratio`, `Throughput`, `Link_Stability`, etc.) separate the network health classes.
  - `06_attack_type_vs_network_status.png`: Stacked distribution displaying network impact across different cyber attack vectors.
  - `07_disaster_type_vs_network_status.png`: Stacked distribution displaying network health across disaster environments (Cyclone, Earthquake, Flood, Wildfire).
  - `category_distributions_grid.png` and individual category plots.
- **Preprocessing & Data Leakage Prevention**:
  - Excludes `UAV_ID` as it is an identifier.
  - Performs an 80% train / 20% test partition stratified on `Network_Status`.
  - Fits imputers, standard scaling, and one-hot encoding strictly on training data.
  - Exports clean data, preprocessed train/test CSVs, and cross-tabulation reports to `outputs/processed/` and `outputs/reports/`.
- **4-Model Benchmark & Comparative Evaluation**:
  - Trains and evaluates Random Forest, Decision Tree, SVM, and Logistic Regression.
  - Generates side-by-side confusion matrices (2×2 grid), multiclass ROC-AUC curves (2×2 grid), dedicated Critical-class ROC comparison, grouped metrics bar chart, and feature importances / linear weights.
  - Exports tabular benchmark metrics to `outputs/reports/model_comparison_report.csv` and detailed reports to `outputs/reports/model_comparison_summary.txt`.

---

## Evaluation Metrics Summary (Holdout Test Set, N=1,000)

All models are evaluated on the exact same **80/20 stratified train/test split** (`random_state=42`, stratified on `Network_Status`) using identical preprocessing (`StandardScaler` for numerical features and `OneHotEncoder` for categorical features, fitted strictly on training data):

| Metric | Random Forest | Decision Tree | Support Vector Machine (SVM) | Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **99.90%** | **99.90%** | 87.80% | 79.50% |
| **Macro F1-Score** | **0.9977** | **0.9977** | 0.8464 | 0.7796 |
| **Weighted F1-Score** | **0.9990** | **0.9990** | 0.8806 | 0.7999 |
| **Macro ROC-AUC (OvR)** | **1.0000** | 0.9980 | 0.9727 | 0.9372 |
| **Micro ROC-AUC (OvR)** | **1.0000** | 0.9993 | 0.9793 | 0.9482 |
| **Critical Class Recall** | **100.0%** (607/607) | **100.0%** (607/607) | 89.46% (543/607) | 80.89% (491/607) |
| **Critical Class Precision** | **100.0%** | **100.0%** | 96.45% | 91.26% |
| **Critical Class F1** | **1.0000** | **1.0000** | 0.9282 | 0.8576 |
| **Critical Class ROC-AUC** | **1.0000** | 1.0000 | 0.9811 | 0.9399 |
| **Degraded Class Recall** | **100.0%** (299/299) | **100.0%** (299/299) | 86.29% (258/299) | 73.91% (221/299) |
| **Degraded Class Precision** | **99.67%** (299/300) | **99.67%** (299/300) | 76.11% (258/339) | 63.51% (221/348) |
| **Normal Class Recall** | **98.94%** (93/94) | **98.94%** (93/94) | 81.91% (77/94) | 88.30% (83/94) |
| **Normal Class Precision** | **100.0%** (93/93) | **100.0%** (93/93) | 78.57% (77/98) | 72.81% (83/114) |

---

## Key Findings & Disaster Response Implications

1. **Tree-Based Superiority (Random Forest & Decision Tree)**:
   - Both tree-based architectures achieve **99.90% accuracy** and **100.0% recall on the Critical class** with **zero false negatives** (0 out of 607 critical disaster nodes missed).
   - This aligns with the non-linear threshold dynamics of FANET telemetry: network state transitions occur abruptly when latency exceeds ~200ms or packet delivery ratio drops below critical cutoffs.

2. **Kernel Non-Linear Boundary (SVM)**:
   - The RBF-kernel SVM captures non-linear manifolds significantly better than linear regression, achieving **87.80% accuracy**, **0.9727 Macro ROC-AUC**, and **89.46% Critical recall**.
   - With 96.45% precision on the Critical class, SVM produces very few false alarms, though it misses 64 critical nodes compared to zero misses for tree models.

3. **Linear Limitation (Logistic Regression)**:
   - Logistic Regression achieves **79.50% accuracy** and **80.89% Critical recall**, misclassifying 116 critical nodes as merely "Degraded".
   - In emergency disaster response operations, failing to detect degraded/critical nodes can result in total link disconnection.

---

## Folder Setup & Generated Artifacts

```
FANET_Disaster_ML_Starter/
│── FANET Disaster Dataset.csv
│── main.py
│── requirements.txt
│── README.md
└── outputs/
    ├── plots/
    │   ├── 01_missing_values.png
    │   ├── 02_target_distribution.png
    │   ├── 03_numerical_distributions.png
    │   ├── 04_correlation_heatmap.png
    │   ├── 05_key_features_by_network_status.png
    │   ├── 06_attack_type_vs_network_status.png
    │   ├── 07_disaster_type_vs_network_status.png
    │   ├── 08_confusion_matrices.png (and 08_confusion_matrix.png)
    │   ├── 09_roc_auc_curves.png
    │   ├── 10_critical_class_roc_curve.png
    │   ├── 11_model_comparison_metrics.png
    │   ├── 12_feature_importances.png (and 09_feature_importances.png)
    │   └── category_distributions_grid.png
    ├── reports/
    │   ├── dataset_summary.txt
    │   ├── column_overview.csv
    │   ├── preprocessing_summary.txt
    │   ├── target_class_distribution.csv
    │   ├── numeric_correlation_matrix.csv
    │   ├── attack_network_status_crosstab.csv
    │   ├── disaster_network_status_crosstab.csv
    │   ├── baseline_model_evaluation.txt
    │   ├── model_comparison_report.csv
    │   └── model_comparison_summary.txt
    └── processed/
        ├── cleaned_dataset.csv
        ├── train_preprocessed.csv
        └── test_preprocessed.csv
```

---

## How to Run
```powershell
python -m pip install -r requirements.txt
python main.py
```
