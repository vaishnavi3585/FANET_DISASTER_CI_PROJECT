"""Train the FANET Random Forest pipeline and save it as fanet_rf_pipeline.joblib.
Mirrors main.py of the project: 80/20 stratified split, random_state=42,
RandomForest(n_estimators=100, class_weight='balanced')."""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "FANET Disaster Dataset.csv"
MODEL = HERE / "fanet_rf_pipeline.joblib"
TARGET, DROP = "Network_Status", ["UAV_ID"]
NUMERIC = ["Latency", "Throughput", "Packet_Delivery_Ratio", "Signal_Strength", "Link_Stability",
           "Topology_Change_Rate", "Recovery_Time", "Route_Change_Count", "Neighbor_Count",
           "X", "Y", "Z", "Velocity", "Direction", "Energy_Level", "Obstacle_Density"]
CATEGORICAL = ["Role", "Disaster_Type", "Weather_Condition", "Attack_Type"]


def build_pipeline():
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL),
    ])
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    return Pipeline([("preprocess", pre), ("model", rf)])


def train(save=True):
    # keep_default_na=False so the string 'None' in Attack_Type stays a valid category
    df = pd.read_csv(DATA, keep_default_na=False)
    df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    X, y = df.drop(columns=DROP + [TARGET]), df[TARGET].astype(str)
    Xtr, Xte, ytr, yte = train_test_split(X[NUMERIC + CATEGORICAL], y, test_size=0.2,
                                          random_state=42, stratify=y)
    pipe = build_pipeline().fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(f"Accuracy {accuracy_score(yte, pred):.4f} | Macro-F1 {f1_score(yte, pred, average='macro'):.4f}")
    if save:
        joblib.dump(pipe, MODEL)
        print("Saved", MODEL)
    return pipe


if __name__ == "__main__":
    train()
