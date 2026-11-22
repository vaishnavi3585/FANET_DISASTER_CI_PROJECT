# FANET Disaster Network Health Prediction — Group Project Report

## 1. Executive Summary & Objective
In disaster relief operations (e.g., floods, earthquakes, cyclones, wildfires), terrestrial communication infrastructures frequently collapse. Flying Ad-Hoc Networks (**FANETs**) composed of Unmanned Aerial Vehicles (UAVs) provide critical dynamic communication backbones for search-and-rescue teams. 

This project builds, benchmarks, and analyzes an end-to-end Machine Learning pipeline that predicts UAV **`Network_Status`** (`Normal`, `Degraded`, `Critical`) in real-time from telemetry and cyber-attack signals.

We implemented and benchmarked **four distinct machine learning paradigms**:
1. **Random Forest Classifier** — *Bagging Ensemble (100 parallel trees)*
2. **Decision Tree Classifier** — *Interpretable Non-linear Tree*
3. **Support Vector Machine (SVM)** — *Non-linear Maximum-Margin Kernel Method (RBF)*
4. **Logistic Regression** — *Linear Discriminative Baseline*

---

## 2. Dataset Overview
- **Dataset Source**: `FANET Disaster Dataset.csv`
- **Total Records**: 5,000 UAV telemetry logs
- **Total Features**: 22 columns (16 numerical telemetry, 4 categorical environment/attack attributes, 1 identifier `UAV_ID`, and 1 target `Network_Status`)
- **Missing Values**: **0 missing values** across all 22 columns (Verified: `Attack_Type` contains 988 records with the literal string `'None'`, representing normal unattacked benign traffic).
- **Target Distribution**:
  - `Critical`: 3,034 samples (**60.68%**) — High-risk failure state
  - `Degraded`: 1,497 samples (**29.94%**) — Impaired performance
  - `Normal`: 469 samples (**9.38%**) — Optimal healthy telemetry

---

## 3. Preprocessing & Leak-Free Methodology
1. **Identifier Exclusion**: `UAV_ID` excluded from model inputs to prevent spurious identification memorization.
2. **Train/Test Split**: Exact **80% Training ($N=4{,}000$)** and **20% Testing ($N=1{,}000$)** partition. Stratified on `Network_Status` to preserve exact class proportions in both sets.
3. **Data Leakage Prevention**:
   - `SimpleImputer` and `StandardScaler` fitted **strictly on training data** and used to transform holdout test data.
   - `OneHotEncoder(handle_unknown='ignore')` applied to categorical features (`Role`, `Disaster_Type`, `Weather_Condition`, `Attack_Type`).
   - Total processed input feature dimension: **32 features**.

---

## 4. Benchmark Performance Comparison (Holdout Test Set, N=1,000)

All models evaluated under identical conditions (`random_state=42`, stratified 80/20 split, balanced class weighting):

| Metric | Random Forest (Bagging) | Decision Tree | SVM (RBF Kernel) | Logistic Regression |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **99.90%** | **99.90%** | 87.80% | 79.50% |
| **Macro F1-Score** | **0.9977** | **0.9977** | 0.8464 | 0.7796 |
| **Weighted F1-Score** | **0.9990** | **0.9990** | 0.8806 | 0.7999 |
| **Macro ROC-AUC (OvR)** | **1.0000** | 0.9980 | 0.9727 | 0.9372 |
| **Critical Class Recall** | **100.0%** (607/607) | **100.0%** (607/607) | 89.46% (543/607) | 80.89% (491/607) |
| **Critical Class Precision** | **100.0%** (607/607) | **100.0%** (607/607) | 96.45% (543/563) | 91.26% (491/538) |
| **Critical Class F1** | **1.0000** | **1.0000** | 0.9282 | 0.8576 |
| **Critical Class ROC-AUC** | **1.0000** | 1.0000 | 0.9811 | 0.9399 |
| **Degraded Class Recall** | **100.0%** (299/299) | **100.0%** (299/299) | 86.29% (258/299) | 73.91% (221/299) |
| **Degraded Class Precision** | **99.67%** (299/300) | **99.67%** (299/300) | 76.11% (258/339) | 63.51% (221/348) |
| **Normal Class Recall** | **98.94%** (93/94) | **98.94%** (93/94) | 81.91% (77/94) | 88.30% (83/94) |
| **Normal Class Precision** | **100.0%** (93/93) | **100.0%** (93/93) | 78.57% (77/98) | 72.81% (83/114) |

---

## 5. Confusion Matrices Summary
- **Random Forest**:
  $$\begin{bmatrix} 607 & 0 & 0 \\ 0 & 299 & 0 \\ 0 & 1 & 93 \end{bmatrix}$$
  *Only 1 sample misclassified out of 1,000.*
- **Decision Tree**:
  $$\begin{bmatrix} 607 & 0 & 0 \\ 0 & 299 & 0 \\ 0 & 1 & 93 \end{bmatrix}$$
- **SVM**:
  $$\begin{bmatrix} 543 & 64 & 0 \\ 20 & 258 & 21 \\ 0 & 17 & 77 \end{bmatrix}$$
  *543 Critical caught, 64 misclassified as Degraded.*
- **Logistic Regression**:
  $$\begin{bmatrix} 491 & 116 & 0 \\ 47 & 221 & 31 \\ 0 & 11 & 83 \end{bmatrix}$$
  *491 Critical caught, 116 misclassified as Degraded.*

---

## 6. Key Findings & Disaster Response Insights
1. **Tree-Based Superiority (Why Random Forest & Decision Tree achieve 99.9%)**:
   - In drone networks, failure modes are defined by sharp non-linear thresholds: when `Latency` exceeds ~200ms or `Packet_Delivery_Ratio` drops below 0.65, connection collapses.
   - Decision trees and Random Forest natively segment orthogonal hyperplanes, capturing these threshold cutoffs with **zero false negatives on critical outages**.
2. **High-Stakes Critical Recall**:
   - In search-and-rescue FANET operations, a false negative on `Critical` status means an aerial node disconnected without backup routing, potentially stranding first responders.
   - **Random Forest and Decision Tree achieved 100.0% recall on the Critical class (0 nodes missed out of 607)**.
   - In comparison, Logistic Regression missed 116 critical nodes.
3. **Most Influential Features**:
   - The top 4 features driving network status across all models are:
     1. `Latency`
     2. `Packet_Delivery_Ratio`
     3. `Throughput`
     4. `Link_Stability`

---

## 7. How to Run the Code (For Group Members)

### Prerequisites:
- Python 3.9+ or Python 3.13
- Required packages: `pip install -r requirements.txt`

### Execution:
1. Open the folder in **VS Code** or terminal.
2. Run:
   ```powershell
   python main.py
   ```
3. Generated charts are located in `outputs/plots/`, and tabular metric summaries are in `outputs/reports/`.

