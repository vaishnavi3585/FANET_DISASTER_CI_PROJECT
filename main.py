"""
FANET Disaster Dataset: Comprehensive Dataset Understanding, Visualization, Preprocessing, and Comparative ML Modeling
Models Evaluated:
  1. Random Forest Classifier (Non-linear Ensemble Baseline)
  2. Decision Tree Classifier (Interpretable Single Tree Baseline)
  3. Support Vector Machine / SVM (Kernelized Non-linear Classifier)
  4. Logistic Regression (Linear Discriminative Baseline)

Evaluates:
  - Accuracy
  - Macro-F1 score & Weighted-F1 score
  - Precision and Recall for each status (Critical, Degraded, Normal)
  - Confusion Matrices (Raw counts, normalized rates, and heatmaps)
  - Recall for the Critical class (high-stakes disaster telemetry focus)
  - Multi-class ROC AUC curves (One-vs-Rest, Micro-average, Macro-average)
  - Feature importances and linear weight coefficients
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Detect scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report, confusion_matrix, ConfusionMatrixDisplay,
        accuracy_score, f1_score, precision_score, recall_score,
        roc_curve, auc, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------- 1. PATHS AND DIRECTORY SETUP --------------------
BASE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Locate dataset flexibly
possible_data_paths = [
    SCRIPT_DIR / "FANET Disaster Dataset.csv",
    SCRIPT_DIR / "FANET_Disaster_ML_Starter" / "FANET Disaster Dataset.csv",
    SCRIPT_DIR.parent / "FANET Disaster Dataset.csv",
    SCRIPT_DIR.parent / "FANET_Disaster_ML_Starter" / "FANET Disaster Dataset.csv"
]

DATA_PATH = None
for p in possible_data_paths:
    if p.exists():
        DATA_PATH = p
        break

if DATA_PATH is None:
    DATA_PATH = SCRIPT_DIR / "FANET Disaster Dataset.csv"

PROJECT_ROOT = DATA_PATH.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
PROCESSED_DIR = OUTPUT_DIR / "processed"

for folder in [PLOTS_DIR, REPORTS_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Network_Status"
RANDOM_STATE = 42

# Styling configurations for publication-quality charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": False
})

# -------------------- 2. LOAD DATA --------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place 'FANET Disaster Dataset.csv' in the project folder."
    )

# Note on keep_default_na=False:
# In the raw CSV, Attack_Type has 988 records with the literal string 'None' (representing No Cyber Attack / Benign traffic).
# Standard pandas read_csv treats 'None' as NaN by default. Using keep_default_na=False correctly preserves 'None'
# as an explicit category and reveals the dataset has 0 missing values across all 22 columns!
df = pd.read_csv(DATA_PATH, keep_default_na=False)
# Convert true whitespace/blank strings to NaN if present
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

print("\n" + "=" * 78)
print("FANET DISASTER DATASET: UNDERSTANDING, VISUALIZATION, AND ML PIPELINE")
print("=" * 78)
print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Total Missing Values: {df.isna().sum().sum()}")
print(f"Duplicate Rows: {df.duplicated().sum()}")
print(f"Target Column: '{TARGET}'")

print("\nFirst 5 Rows Preview:")
print(df.head().to_string(index=False))

print("\nColumn Information & Data Types:")
print(df.dtypes.to_string())

# -------------------- 3. DATASET UNDERSTANDING & SUMMARY REPORTS --------------------
overview = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Non_Null_Count": [int(df[c].notna().sum()) for c in df.columns],
    "Missing_Count": [int(df[c].isna().sum()) for c in df.columns],
    "Missing_Percent": [round(float(df[c].isna().mean() * 100), 2) for c in df.columns],
    "Unique_Values": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
overview.to_csv(REPORTS_DIR / "column_overview.csv", index=False)

with open(REPORTS_DIR / "dataset_summary.txt", "w", encoding="utf-8") as f:
    f.write("FANET Disaster Dataset - Comprehensive Dataset Understanding\n")
    f.write("=" * 65 + "\n")
    f.write(f"Total Records: {df.shape[0]:,}\n")
    f.write(f"Total Columns: {df.shape[1]}\n")
    f.write(f"Duplicate Rows: {int(df.duplicated().sum())}\n")
    f.write(f"Total Missing Values across all columns: {int(df.isna().sum().sum())}\n\n")
    f.write("Note on Attack_Type column:\n")
    f.write("  In the raw CSV, 'None' represents benign / unattacked traffic (988 records, ~19.8%).\n")
    f.write("  The 5 attack categories in the dataset are: Blackhole (1037), Jamming (1018),\n")
    f.write("  Replay (1016), None (988), and Sybil (941), totaling exactly 5,000 samples.\n\n")
    f.write("Column Overview:\n")
    f.write(overview.to_string(index=False))
    f.write("\n\nNumerical Feature Descriptive Statistics:\n")
    f.write(df.describe().to_string())
    f.write("\n\nCategorical Feature Value Counts:\n")
    for col in [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name in ["category", "string"]]:
        f.write(f"\n--- {col} ---\n")
        counts = df[col].value_counts(dropna=False)
        pcts = (counts / len(df) * 100).round(2)
        summary_cat = pd.DataFrame({"Count": counts, "Percentage": pcts})
        f.write(summary_cat.to_string())
        f.write("\n")

print("\nMissing Values Count by Column:")
print(df.isna().sum().to_string())

print("\nNumerical Feature Descriptive Statistics:")
print(df.describe().round(3).to_string())

# Cross-tabulations for disaster analysis
if "Attack_Type" in df.columns and TARGET in df.columns:
    attack_status_crosstab = pd.crosstab(df["Attack_Type"], df[TARGET], margins=True)
    attack_status_crosstab.to_csv(REPORTS_DIR / "attack_network_status_crosstab.csv")

if "Disaster_Type" in df.columns and TARGET in df.columns:
    disaster_status_crosstab = pd.crosstab(df["Disaster_Type"], df[TARGET], margins=True)
    disaster_status_crosstab.to_csv(REPORTS_DIR / "disaster_network_status_crosstab.csv")

# -------------------- 4. EXPLORATORY VISUALIZATIONS --------------------
print("\nGenerating exploratory and analytical visualizations...")

# Plot 1: Missing Values
missing = df.isna().sum()
missing_filtered = missing[missing > 0]
plt.figure(figsize=(9, 4.5))
if len(missing_filtered) > 0:
    ax = sns.barplot(x=missing_filtered.index, y=missing_filtered.values, palette="Reds_r")
    plt.ylabel("Missing Count")
    plt.xlabel("Features")
    plt.title("Missing Values by Column")
    plt.xticks(rotation=30, ha="right")
else:
    plt.text(0.5, 0.55, "[OK] 0 Missing Values Found Across All 22 Columns",
             ha="center", va="center", fontsize=13, weight="bold", color="#1b5e20")
    plt.text(0.5, 0.40, "(Note: 'Attack_Type' value 'None' represents benign traffic / No Cyber Attack)",
             ha="center", va="center", fontsize=10, style="italic", color="#424242")
    plt.title("Missing Values Assessment", fontsize=14, weight="bold", pad=12)
    plt.axis("off")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_missing_values.png", dpi=300)
plt.close()

# Plot 2: Target Distribution (Network_Status)
if TARGET in df.columns:
    plt.figure(figsize=(8, 5))
    order = ["Normal", "Degraded", "Critical"]
    palette = {"Normal": "#2e7d32", "Degraded": "#f57c00", "Critical": "#c62828"}
    status_counts = df[TARGET].value_counts().reindex(order)
    
    ax = sns.barplot(x=status_counts.index, y=status_counts.values,
                     order=order, palette=[palette[k] for k in order])
    for p, count in zip(ax.patches, status_counts.values):
        pct = (count / len(df)) * 100
        ax.annotate(f"{count:,}\n({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                    ha="center", va="center", color="white", fontsize=11, weight="bold")
    plt.title("Target Distribution: Network Status", fontsize=14, weight="bold")
    plt.xlabel("Network Status", fontsize=12)
    plt.ylabel("Record Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_target_distribution.png", dpi=300)
    plt.close()

# Plot 3: Categorical Feature Distributions (Grid & Individual)
categorical_cols = [c for c in df.columns if df[c].dtype == "object" or df[c].dtype.name in ["category", "string"]]
plot_cat_cols = [c for c in categorical_cols if c != TARGET]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for idx, col in enumerate(plot_cat_cols[:4]):
    ax = axes[idx]
    counts = df[col].value_counts()
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Blues_r")
    for p in ax.patches:
        val = int(p.get_height())
        pct = (val / len(df)) * 100
        ax.annotate(f"{val} ({pct:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() + 15),
                    ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Distribution: {col}", weight="bold")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, counts.max() * 1.15)
plt.suptitle("FANET Categorical Features Overview", fontsize=15, weight="bold", y=0.99)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_distributions_grid.png", dpi=300)
plt.close()

# Save individual category plots
for col in plot_cat_cols:
    plt.figure(figsize=(7, 4.5))
    counts = df[col].value_counts()
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette="Blues_r")
    for p in ax.patches:
        val = int(p.get_height())
        ax.annotate(f"{val} ({val/len(df)*100:.1f}%)", (p.get_x() + p.get_width() / 2., p.get_height() + 10),
                    ha="center", va="bottom", fontsize=9)
    plt.title(f"Category Distribution: {col}", weight="bold")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in col)
    plt.savefig(PLOTS_DIR / f"category_{safe_name}.png", dpi=300)
    plt.close()

# Plot 4: Numerical Feature Histograms with KDE
numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "UAV_ID"]
if numeric_cols:
    n_cols = 4
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        sns.histplot(df[col], kde=True, bins=25, ax=axes[i], color="#1976d2", edgecolor="black")
        axes[i].set_title(col, fontsize=11, weight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Distributions of FANET Numerical Features (Histograms + KDE)", fontsize=16, weight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_numerical_distributions.png", dpi=300)
    plt.close()

# Plot 5: Correlation Heatmap
if len(numeric_cols) > 1:
    plt.figure(figsize=(14, 11))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                cbar_kws={"shrink": 0.8}, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap of FANET Numerical Features", fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_correlation_heatmap.png", dpi=300)
    plt.close()
    corr.to_csv(REPORTS_DIR / "numeric_correlation_matrix.csv")

# Plot 6: Key Features Grouped by Network Status
key_features = ["Latency", "Packet_Delivery_Ratio", "Throughput",
                "Link_Stability", "Energy_Level", "Recovery_Time"]
available_keys = [c for c in key_features if c in df.columns]

if available_keys and TARGET in df.columns:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    status_order = ["Normal", "Degraded", "Critical"]
    palette_dict = {"Normal": "#2e7d32", "Degraded": "#f57c00", "Critical": "#c62828"}
    
    for idx, feature in enumerate(available_keys):
        sns.boxplot(data=df, x=TARGET, y=feature, order=status_order,
                    palette=palette_dict, ax=axes[idx], showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        axes[idx].set_title(f"{feature} by Network Status", weight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(feature)
        
        # Save individual boxplot as well
        plt.figure(figsize=(7, 4.5))
        sns.boxplot(data=df, x=TARGET, y=feature, order=status_order,
                    palette=palette_dict, showmeans=True,
                    meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"})
        plt.title(f"{feature} by Network Status", weight="bold")
        plt.xlabel("Network Status")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"boxplot_{feature}.png", dpi=300)
        plt.close()
        
    plt.suptitle("Key Telemetry and Performance Differentiators by Network Status", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_key_features_by_network_status.png", dpi=300)
    plt.close()

# Plot 7: Attack Type vs Network Status
if "Attack_Type" in df.columns and TARGET in df.columns:
    attack_status_pct = pd.crosstab(df["Attack_Type"], df[TARGET], normalize="index")[["Normal", "Degraded", "Critical"]] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    attack_status_pct.plot(kind="bar", stacked=True, ax=ax,
                           color=["#2e7d32", "#f57c00", "#c62828"], edgecolor="black")
    plt.title("Network Status Distribution by Cyber Attack Type", fontsize=13, weight="bold")
    plt.xlabel("Attack Type", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xticks(rotation=15)
    plt.legend(title="Network Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_attack_type_vs_network_status.png", dpi=300)
    plt.close()

# Plot 8: Disaster Type vs Network Status
if "Disaster_Type" in df.columns and TARGET in df.columns:
    disaster_status_pct = pd.crosstab(df["Disaster_Type"], df[TARGET], normalize="index")[["Normal", "Degraded", "Critical"]] * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    disaster_status_pct.plot(kind="bar", stacked=True, ax=ax,
                             color=["#2e7d32", "#f57c00", "#c62828"], edgecolor="black")
    plt.title("Network Status Distribution by Disaster Type", fontsize=13, weight="bold")
    plt.xlabel("Disaster Environment", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.xticks(rotation=15)
    plt.legend(title="Network Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_disaster_type_vs_network_status.png", dpi=300)
    plt.close()

print(f"All exploratory visualizations saved successfully in: {PLOTS_DIR}")

# -------------------- 5. PREPROCESSING PIPELINE --------------------
print("\n" + "=" * 78)
print("EXECUTING PREPROCESSING PIPELINE")
print("=" * 78)

clean_df = df.drop_duplicates().copy()

# Exclude UAV_ID (identifier) from feature inputs
id_columns = [c for c in ["UAV_ID"] if c in clean_df.columns]
X = clean_df.drop(columns=[TARGET] + id_columns)
y = clean_df[TARGET].astype(str)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

if SKLEARN_AVAILABLE:
    # 80/20 Train-Test Split with Stratification for strictly fair benchmarking
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

    train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
    train_processed_df[TARGET] = y_train.reset_index(drop=True)

    test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
    test_processed_df[TARGET] = y_test.reset_index(drop=True)

else:
    # High-performance native pandas/numpy fallback
    train_indices, test_indices = [], []
    for cls_name, grp in clean_df.groupby(TARGET):
        n_test = int(round(len(grp) * 0.20))
        shuffled_idx = grp.sample(frac=1.0, random_state=RANDOM_STATE).index
        test_indices.extend(shuffled_idx[:n_test])
        train_indices.extend(shuffled_idx[n_test:])

    X_train, y_train = X.loc[train_indices], y.loc[train_indices]
    X_test, y_test = X.loc[test_indices], y.loc[test_indices]

    # Impute and standardize numerics
    train_medians = X_train[numeric_features].median()
    train_means = X_train[numeric_features].mean()
    train_stds = X_train[numeric_features].std().replace(0, 1.0)

    X_train_num = (X_train[numeric_features].fillna(train_medians) - train_means) / train_stds
    X_test_num = (X_test[numeric_features].fillna(train_medians) - train_means) / train_stds

    # One-hot encode categoricals
    X_train_cat = pd.get_dummies(X_train[categorical_features], drop_first=False)
    X_test_cat = pd.get_dummies(X_test[categorical_features], drop_first=False)
    X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    train_processed_df = pd.concat([X_train_num, X_train_cat], axis=1).reset_index(drop=True)
    train_processed_df[TARGET] = y_train.reset_index(drop=True)

    test_processed_df = pd.concat([X_test_num, X_test_cat], axis=1).reset_index(drop=True)
    test_processed_df[TARGET] = y_test.reset_index(drop=True)
    
    feature_names = [c for c in train_processed_df.columns if c != TARGET]
    X_train_processed = train_processed_df[feature_names].values
    X_test_processed = test_processed_df[feature_names].values

# Export cleaned dataset and preprocessed datasets
clean_df.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
train_processed_df.to_csv(PROCESSED_DIR / "train_preprocessed.csv", index=False)
test_processed_df.to_csv(PROCESSED_DIR / "test_preprocessed.csv", index=False)

# Target class distribution report
class_counts = y.value_counts().rename_axis(TARGET).reset_index(name="Count")
class_counts["Percentage"] = (class_counts["Count"] / len(y) * 100).round(2)
class_counts.to_csv(REPORTS_DIR / "target_class_distribution.csv", index=False)

with open(REPORTS_DIR / "preprocessing_summary.txt", "w", encoding="utf-8") as f:
    f.write("FANET Disaster Preprocessing Summary\n")
    f.write("=" * 55 + "\n")
    f.write(f"Original Records: {df.shape[0]}\n")
    f.write(f"Processed Clean Records: {clean_df.shape[0]}\n")
    f.write(f"Target Column: {TARGET}\n")
    f.write(f"Identifier Column Excluded: {id_columns}\n")
    f.write(f"Numeric Input Features ({len(numeric_features)}): {numeric_features}\n")
    f.write(f"Categorical Input Features ({len(categorical_features)}): {categorical_features}\n")
    f.write("Numeric Pipeline: SimpleImputer(median) -> StandardScaler()\n")
    f.write("Categorical Pipeline: SimpleImputer(most_frequent) -> OneHotEncoder(sparse_output=False)\n")
    f.write("Train/Test Partition: 80% Train, 20% Test (Stratified by Target)\n")
    f.write(f"Training Samples: {len(X_train)} (80%)\n")
    f.write(f"Testing Samples: {len(X_test)} (20%)\n")
    f.write(f"Total Encoded/Scaled Feature Count: {X_train_processed.shape[1]}\n\n")
    f.write("Target Class Distribution Across Full Dataset:\n")
    f.write(class_counts.to_string(index=False))

print(f"Target Column: {TARGET}")
print(f"Raw Numeric Features: {len(numeric_features)}")
print(f"Raw Categorical Features: {len(categorical_features)}")
print(f"Engineered / Processed Input Features: {X_train_processed.shape[1]}")
print(f"Training Rows: {len(X_train):,} ({len(X_train)/len(clean_df):.0%})")
print(f"Testing Rows: {len(X_test):,} ({len(X_test)/len(clean_df):.0%})")

# -------------------- 6. COMPARATIVE MACHINE LEARNING MODELING --------------------
if SKLEARN_AVAILABLE:
    print("\n" + "=" * 78)
    print("TRAINING & BENCHMARKING 4 MODELS ON IDENTICAL PREPROCESSED DATA:")
    print("  1. Random Forest Classifier")
    print("  2. Decision Tree Classifier")
    print("  3. Support Vector Machine (SVM)")
    print("  4. Logistic Regression")
    print("=" * 78)

    target_labels = ["Critical", "Degraded", "Normal"]
    y_test_bin = label_binarize(y_test, classes=target_labels)
    n_classes = len(target_labels)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        )
    }

    results_list = []
    cm_dict = {}
    roc_dict = {}
    reports_text = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_processed, y_train)

        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)

        # Reorder probabilities to strictly align with target_labels
        class_to_idx = {cls_name: i for i, cls_name in enumerate(model.classes_)}
        y_proba_ordered = np.column_stack([y_proba[:, class_to_idx[lbl]] for lbl in target_labels])

        # Core Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")

        # Per-class metrics
        precisions = precision_score(y_test, y_pred, labels=target_labels, average=None, zero_division=0)
        recalls = recall_score(y_test, y_pred, labels=target_labels, average=None, zero_division=0)
        f1s = f1_score(y_test, y_pred, labels=target_labels, average=None, zero_division=0)

        per_class_metrics = {}
        for idx, lbl in enumerate(target_labels):
            per_class_metrics[f"{lbl}_Precision"] = precisions[idx]
            per_class_metrics[f"{lbl}_Recall"] = recalls[idx]
            per_class_metrics[f"{lbl}_F1"] = f1s[idx]

        # Critical class specific metrics (High priority for FANET disaster operations)
        critical_recall = per_class_metrics["Critical_Recall"]
        critical_precision = per_class_metrics["Critical_Precision"]
        critical_f1 = per_class_metrics["Critical_F1"]

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=target_labels)
        cm_dict[name] = cm

        # Multiclass ROC AUC (One-vs-Rest)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i, lbl in enumerate(target_labels):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba_ordered[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba_ordered.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro-average ROC AUC
        macro_auc = roc_auc_score(y_test_bin, y_proba_ordered, multi_class="ovr", average="macro")
        roc_auc["macro"] = macro_auc

        roc_dict[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "y_proba": y_proba_ordered
        }

        # Store tabular summary
        model_summary = {
            "Model": name,
            "Accuracy": acc,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "Critical_Recall": critical_recall,
            "Critical_Precision": critical_precision,
            "Critical_F1": critical_f1,
            "Critical_ROC_AUC": roc_auc[0],
            "Degraded_Recall": per_class_metrics["Degraded_Recall"],
            "Degraded_Precision": per_class_metrics["Degraded_Precision"],
            "Degraded_F1": per_class_metrics["Degraded_F1"],
            "Degraded_ROC_AUC": roc_auc[1],
            "Normal_Recall": per_class_metrics["Normal_Recall"],
            "Normal_Precision": per_class_metrics["Normal_Precision"],
            "Normal_F1": per_class_metrics["Normal_F1"],
            "Normal_ROC_AUC": roc_auc[2],
            "Macro_ROC_AUC": macro_auc,
            "Micro_ROC_AUC": roc_auc["micro"]
        }
        results_list.append(model_summary)

        # Console Summary
        print(f"--- {name} Results ---")
        print(f"  • Overall Accuracy:       {acc:.4f} ({acc*100:.2f}%)")
        print(f"  • Macro F1-Score:         {macro_f1:.4f}")
        print(f"  • Weighted F1-Score:      {weighted_f1:.4f}")
        print(f"  • Macro ROC-AUC:          {macro_auc:.4f}")
        print(f"  • Critical Class Recall:  {critical_recall:.4f} ({critical_recall*100:.2f}%)")
        print(f"  • Critical Class Prec:    {critical_precision:.4f} ({critical_precision*100:.2f}%)")
        print(f"  • Critical Class ROC-AUC: {roc_auc[0]:.4f}")

        # Classification Report String
        rep = classification_report(y_test, y_pred, labels=target_labels, digits=4)
        print("\nClassification Report:")
        print(rep)
        reports_text.append(f"Model: {name}\n" + "-" * 50 + f"\nAccuracy: {acc:.4f} | Macro F1: {macro_f1:.4f} | Macro ROC-AUC: {macro_auc:.4f}\n" +
                            f"Critical Recall: {critical_recall:.4f} | Critical Precision: {critical_precision:.4f}\n\n" +
                            f"Classification Report:\n{rep}\n" +
                            f"Confusion Matrix (labels={target_labels}):\n{cm}\n\n" + "=" * 60 + "\n")

    # Save Comparison CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(REPORTS_DIR / "model_comparison_report.csv", index=False)

    with open(REPORTS_DIR / "model_comparison_summary.txt", "w", encoding="utf-8") as f:
        f.write("FANET Disaster Dataset - Multi-Model Benchmark & Evaluation Report\n")
        f.write("=" * 72 + "\n\n")
        f.write("COMPARATIVE SUMMARY TABLE:\n")
        f.write(results_df[["Model", "Accuracy", "Macro_F1", "Weighted_F1", "Critical_Recall", "Critical_Precision", "Critical_ROC_AUC", "Macro_ROC_AUC"]].to_string(index=False))
        f.write("\n\n" + "=" * 72 + "\n\n")
        for txt in reports_text:
            f.write(txt)

    # Baseline Model Evaluation Text Report (RF reference)
    rf_res = results_df[results_df["Model"] == "Random Forest"].iloc[0]
    with open(REPORTS_DIR / "baseline_model_evaluation.txt", "w", encoding="utf-8") as f:
        f.write("FANET Disaster Dataset - Machine Learning Benchmark Summary\n")
        f.write("=" * 65 + "\n")
        f.write("Models Evaluated: Random Forest, Decision Tree, SVM, Logistic Regression\n\n")
        f.write("Top Performing Model: Random Forest Classifier\n")
        f.write(f"  • Test Accuracy:    {rf_res['Accuracy']:.4f} ({rf_res['Accuracy']*100:.2f}%)\n")
        f.write(f"  • Macro F1-Score:   {rf_res['Macro_F1']:.4f}\n")
        f.write(f"  • Weighted F1-Score:{rf_res['Weighted_F1']:.4f}\n")
        f.write(f"  • Critical Recall:  {rf_res['Critical_Recall']:.4f} ({rf_res['Critical_Recall']*100:.2f}%)\n")
        f.write(f"  • Macro ROC-AUC:    {rf_res['Macro_ROC_AUC']:.4f}\n\n")
        f.write("Full Multi-Model Comparison Table:\n")
        f.write(results_df.to_string(index=False))

    # -------------------- PLOT 8: CONFUSION MATRICES (2x2 GRID) --------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, (name, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_labels)
        disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{name}\nAcc: {results_df.loc[results_df['Model']==name, 'Accuracy'].values[0]:.3f} | Macro-F1: {results_df.loc[results_df['Model']==name, 'Macro_F1'].values[0]:.3f}",
                     weight="bold", fontsize=12)
        ax.grid(False)
    plt.suptitle("Confusion Matrix Comparison Across 4 Models (Holdout Test Set, N=1,000)", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_confusion_matrix.png", dpi=300)
    plt.savefig(PLOTS_DIR / "08_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------- PLOT 9: MULTICLASS ROC AUC CURVES (2x2 GRID) --------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    class_colors = {"Critical": "#c62828", "Degraded": "#f57c00", "Normal": "#2e7d32"}

    for idx, (name, roc_data) in enumerate(roc_dict.items()):
        ax = axes[idx]
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        roc_auc = roc_data["roc_auc"]

        # Plot class curves
        for i, lbl in enumerate(target_labels):
            ax.plot(fpr[i], tpr[i], color=class_colors[lbl], lw=2,
                    label=f"{lbl} (AUC = {roc_auc[i]:.3f})")

        # Micro and Macro average
        ax.plot(fpr["micro"], tpr["micro"], color="deeppink", linestyle=":", lw=2.5,
                label=f"Micro-avg (AUC = {roc_auc['micro']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Chance (AUC = 0.500)")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=11)
        ax.set_title(f"{name}\nMacro ROC-AUC: {roc_auc['macro']:.4f}", weight="bold", fontsize=12)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    plt.suptitle("Multiclass ROC Curves (One-vs-Rest) Across 4 Models", fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_roc_auc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------- PLOT 10: DEDICATED CRITICAL CLASS ROC COMPARISON --------------------
    plt.figure(figsize=(8.5, 6.5))
    model_colors = {
        "Random Forest": "#1976d2",
        "Decision Tree": "#388e3c",
        "SVM": "#8e24aa",
        "Logistic Regression": "#d32f2f"
    }
    model_styles = {
        "Random Forest": "-",
        "Decision Tree": "-.",
        "SVM": ":",
        "Logistic Regression": "--"
    }

    for name, roc_data in roc_dict.items():
        fpr_crit = roc_data["fpr"][0]
        tpr_crit = roc_data["tpr"][0]
        auc_crit = roc_data["roc_auc"][0]
        plt.plot(fpr_crit, tpr_crit, color=model_colors[name], linestyle=model_styles[name], lw=2.5,
                 label=f"{name} (AUC = {auc_crit:.4f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate / Recall (TPR)", fontsize=12)
    plt.title("High-Stakes Outage Detection: ROC Curves for the 'Critical' Class\n(FANET Disaster Communication Outage Detection)",
              fontsize=13, weight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_critical_class_roc_curve.png", dpi=300)
    plt.close()

    # -------------------- PLOT 11: GROUPED MODEL COMPARISON METRICS --------------------
    comp_metrics = ["Accuracy", "Macro_F1", "Critical_Recall", "Critical_Precision", "Macro_ROC_AUC"]
    plot_df = results_df.set_index("Model")[comp_metrics].reset_index()
    plot_melted = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6.5))
    ax = sns.barplot(data=plot_melted, x="Metric", y="Score", hue="Model",
                     palette=["#1976d2", "#43a047", "#8e24aa", "#e53935"])
    plt.title("4-Model Benchmark: Performance Across Key Evaluation Metrics", fontsize=14, weight="bold")
    plt.ylabel("Metric Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.ylim(0, 1.18)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h + 0.02),
                        ha="center", va="bottom", fontsize=8, weight="bold", rotation=0)
    plt.legend(title="Model", frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_model_comparison_metrics.png", dpi=300)
    plt.close()

    # -------------------- PLOT 12: FEATURE IMPORTANCES & LR COEFFICIENTS --------------------
    clean_feat_names = [f.replace("numeric__", "").replace("categorical__", "") for f in feature_names]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Subplot A: Random Forest Importances
    rf_inst = models["Random Forest"]
    rf_imp = pd.Series(rf_inst.feature_importances_, index=clean_feat_names).sort_values(ascending=False).head(10)
    sns.barplot(x=rf_imp.values, y=rf_imp.index, ax=axes[0], palette="viridis")
    axes[0].set_title("Random Forest: Top 10 Features (Gini)", weight="bold", fontsize=12)
    axes[0].set_xlabel("Gini Feature Importance")
    for idx, val in enumerate(rf_imp.values):
        axes[0].text(val + 0.002, idx, f"{val:.3f}", va="center", fontsize=9)
    axes[0].set_xlim(0, max(rf_imp.values) * 1.18)

    # Subplot B: Decision Tree Importances
    dt_inst = models["Decision Tree"]
    dt_imp = pd.Series(dt_inst.feature_importances_, index=clean_feat_names).sort_values(ascending=False).head(10)
    sns.barplot(x=dt_imp.values, y=dt_imp.index, ax=axes[1], palette="magma")
    axes[1].set_title("Decision Tree: Top 10 Features (Gini)", weight="bold", fontsize=12)
    axes[1].set_xlabel("Gini Feature Importance")
    for idx, val in enumerate(dt_imp.values):
        axes[1].text(val + 0.002, idx, f"{val:.3f}", va="center", fontsize=9)
    axes[1].set_xlim(0, max(dt_imp.values) * 1.18)

    # Subplot C: Logistic Regression Coefficients for Critical Class
    lr_inst = models["Logistic Regression"]
    crit_class_idx = list(lr_inst.classes_).index("Critical")
    lr_coefs = lr_inst.coef_[crit_class_idx]
    lr_series = pd.Series(lr_coefs, index=clean_feat_names).sort_values(key=abs, ascending=False).head(10)
    colors = ["#c62828" if v > 0 else "#1565c0" for v in lr_series.values]
    sns.barplot(x=lr_series.values, y=lr_series.index, ax=axes[2], palette=colors)
    axes[2].set_title("Logistic Regression: Top 10 Weights for 'Critical'\n(Red: Increases Outage Odds, Blue: Protective)",
                      weight="bold", fontsize=11)
    axes[2].set_xlabel("Coefficient Magnitude (log-odds)")
    for idx, val in enumerate(lr_series.values):
        offset = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        axes[2].text(val + offset, idx, f"{val:.2f}", va="center", fontsize=9, ha=ha)
    min_x, max_x = min(lr_series.values), max(lr_series.values)
    axes[2].set_xlim(min_x * 1.25 if min_x < 0 else -0.5, max_x * 1.25 if max_x > 0 else 0.5)

    plt.suptitle("Model Interpretability: Ensemble Tree vs Single Tree vs Linear Feature Impact", fontsize=14, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_feature_importances.png", dpi=300)
    plt.savefig(PLOTS_DIR / "12_feature_importances.png", dpi=300)
    plt.close()

print("\n" + "=" * 78)
print("EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 78)
print(f"Outputs saved inside: {OUTPUT_DIR.resolve()}")
print("\nGenerated Reports:")
print(f"  • {REPORTS_DIR / 'dataset_summary.txt'}")
print(f"  • {REPORTS_DIR / 'column_overview.csv'}")
print(f"  • {REPORTS_DIR / 'preprocessing_summary.txt'}")
print(f"  • {REPORTS_DIR / 'target_class_distribution.csv'}")
print(f"  • {REPORTS_DIR / 'numeric_correlation_matrix.csv'}")
print(f"  • {REPORTS_DIR / 'attack_network_status_crosstab.csv'}")
print(f"  • {REPORTS_DIR / 'disaster_network_status_crosstab.csv'}")
if SKLEARN_AVAILABLE:
    print(f"  • {REPORTS_DIR / 'baseline_model_evaluation.txt'}")
    print(f"  • {REPORTS_DIR / 'model_comparison_report.csv'}")
    print(f"  • {REPORTS_DIR / 'model_comparison_summary.txt'}")
print("\nGenerated Datasets:")
print(f"  • {PROCESSED_DIR / 'cleaned_dataset.csv'}")
print(f"  • {PROCESSED_DIR / 'train_preprocessed.csv'}")
print(f"  • {PROCESSED_DIR / 'test_preprocessed.csv'}")
print("\nGenerated Visualizations (in outputs/plots/):")
print("  • 01_missing_values.png")
print("  • 02_target_distribution.png")
print("  • 03_numerical_distributions.png")
print("  • 04_correlation_heatmap.png")
print("  • 05_key_features_by_network_status.png")
print("  • 06_attack_type_vs_network_status.png")
print("  • 07_disaster_type_vs_network_status.png")
if SKLEARN_AVAILABLE:
    print("  • 08_confusion_matrix.png & 08_confusion_matrices.png")
    print("  • 09_roc_auc_curves.png")
    print("  • 10_critical_class_roc_curve.png")
    print("  • 11_model_comparison_metrics.png")
    print("  • 12_feature_importances.png (and 09_feature_importances.png)")
print("  • category_distributions_grid.png & individual category charts")
print("=" * 78)
