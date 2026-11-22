"""
AI/ML-Based FANET Network Status Prediction for Disaster Management
Mission-control style Streamlit dashboard.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# 0. CONFIG  (column names are EXACTLY those used when the model was trained)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "fanet_rf_pipeline.joblib"
SAMPLE_CSV = HERE / "FANET Disaster Dataset.csv"

NUMERIC = ["Latency", "Throughput", "Packet_Delivery_Ratio", "Signal_Strength", "Link_Stability",
           "Topology_Change_Rate", "Recovery_Time", "Route_Change_Count", "Neighbor_Count",
           "X", "Y", "Z", "Velocity", "Direction", "Energy_Level", "Obstacle_Density"]
CATEGORICAL = ["Role", "Disaster_Type", "Weather_Condition", "Attack_Type"]
FEATURES = NUMERIC + CATEGORICAL            # exact training column order
ID_COL, TARGET = "UAV_ID", "Network_Status"

ROLES = ["Leader", "Relay", "Sensor"]
DISASTERS = ["Earthquake", "Cyclone", "Flood", "Wildfire"]
WEATHER = ["Clear", "Rain", "Smoke", "Windy"]
ATTACKS = ["None", "Blackhole", "Jamming", "Replay", "Sybil"]

COLORS = {"Normal": "#22c55e", "Degraded": "#f59e0b", "Critical": "#ef4444"}
ORDER = ["Normal", "Degraded", "Critical"]

# Alert thresholds taken from the class medians in the dataset
LAT_ALERT, PDR_ALERT = 170.0, 0.72

# Accept friendly header variants in uploaded CSVs ("Packet Delivery Ratio", "Weather", ...)
ALIASES = {"weather": "Weather_Condition", "disaster type": "Disaster_Type"}

st.set_page_config(page_title="FANET Network Status Prediction", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; max-width: 1250px;}
.hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 100%); color:#fff;}
.hero h1 {font-size: 1.6rem; margin:0 0 .2rem 0; color:#fff;}
.hero p  {margin:0; opacity:.8; font-size:.92rem;}
.kpi {border-radius:12px; padding:.9rem 1.1rem; border:1px solid rgba(128,128,128,.25);
      background: rgba(128,128,128,.07); height:100%;}
.kpi .lbl {font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; opacity:.7;}
.kpi .val {font-size:1.6rem; font-weight:700; line-height:1.3;}
.kpi .sub {font-size:.8rem; opacity:.75;}
.prio {border-left:4px solid #3b82f6; padding:.2rem 0 .2rem .8rem; margin-bottom:.4rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 1. MODEL LOADING
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Random Forest pipeline…")
def load_model():
    """Load the saved sklearn Pipeline (preprocessing + Random Forest).
    If the file is missing or was saved with an incompatible scikit-learn version,
    retrain from the bundled CSV using train_model.py."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        from train_model import train
        return train(save=True)


model = load_model()


def predict(df: pd.DataFrame):
    """df must contain FEATURES. Returns (labels, probability DataFrame)."""
    X = df[FEATURES]                                  # enforce exact column order
    proba = model.predict_proba(X)
    probs = pd.DataFrame(proba, columns=model.classes_, index=df.index)[ORDER]
    return probs.idxmax(axis=1), probs


def clean_csv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise headers, keep the string 'None' valid, blank -> NaN (imputed by pipeline)."""
    raw = raw.copy()
    raw.columns = [ALIASES.get(c.strip().lower(), c.strip().replace(" ", "_")) for c in raw.columns]
    raw = raw.replace(r"^\s*$", np.nan, regex=True)
    return raw


# --------------------------------------------------------------------------
# 2. HEADER + KPI PLACEHOLDER (filled after the Tab-1 prediction runs)
# --------------------------------------------------------------------------
st.markdown("""
<div class="hero">
  <h1>🛰️ AI/ML-Based FANET Network Status Prediction for Disaster Management</h1>
  <p>Random Forest classifier · proactive UAV network health monitoring · Normal / Degraded / Critical</p>
</div>
""", unsafe_allow_html=True)
kpi_area = st.container()

tab1, tab2, tab3 = st.tabs(["🎛️ Single UAV Simulator", "📡 Batch Swarm Prediction", "📊 Model Insights"])

# --------------------------------------------------------------------------
# 3. TAB 1 - SINGLE UAV SIMULATOR
# --------------------------------------------------------------------------
with tab1:
    st.caption("Adjust the telemetry below – the prediction updates instantly.")
    inp = {}

    st.markdown('<div class="prio"><b>⭐ Priority drivers</b> – together ≈ 75 % of model importance</div>',
               unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    inp["Packet_Delivery_Ratio"] = c1.slider("Packet Delivery Ratio (0–1)", 0.50, 1.00, 0.85, 0.01,
                                             help="Importance ≈ 39.0 %")
    inp["Latency"] = c2.slider("Latency (ms)", 10.0, 300.0, 100.0, 1.0, help="Importance ≈ 36.5 %")

    with st.expander("📶 Network Performance", expanded=False):
        a, b, c = st.columns(3)
        inp["Throughput"] = a.slider("Throughput (Mbps)", 1.0, 100.0, 51.0, 0.5)
        inp["Signal_Strength"] = b.slider("Signal Strength (dBm)", -95.0, -40.0, -67.0, 0.5)
        inp["Link_Stability"] = c.slider("Link Stability", 0.30, 1.00, 0.64, 0.01)
        a, b, c = st.columns(3)
        inp["Topology_Change_Rate"] = a.slider("Topology Change Rate", 0.10, 1.00, 0.55, 0.01)
        inp["Recovery_Time"] = b.slider("Recovery Time (s)", 0.0, 50.0, 25.0, 0.5)
        inp["Route_Change_Count"] = c.slider("Route Change Count", 0, 19, 10)
        inp["Neighbor_Count"] = a.slider("Neighbor Count", 1, 14, 8)

    with st.expander("🌍 Physical & Environmental", expanded=False):
        a, b, c = st.columns(3)
        inp["X"] = a.slider("X (m)", 0.0, 1000.0, 500.0, 1.0)
        inp["Y"] = b.slider("Y (m)", 0.0, 1000.0, 500.0, 1.0)
        inp["Z"] = c.slider("Z – altitude (m)", 50.0, 300.0, 175.0, 1.0)
        inp["Velocity"] = a.slider("Velocity (m/s)", 5.0, 30.0, 17.5, 0.5)
        inp["Direction"] = b.slider("Direction (°)", 0.0, 360.0, 180.0, 1.0)
        inp["Energy_Level"] = c.slider("Energy Level (%)", 10.0, 100.0, 55.0, 1.0)
        inp["Obstacle_Density"] = a.slider("Obstacle Density", 0.10, 1.00, 0.55, 0.01)
        inp["Role"] = b.selectbox("Role", ROLES, index=1)
        inp["Disaster_Type"] = c.selectbox("Disaster Type", DISASTERS)
        inp["Weather_Condition"] = a.selectbox("Weather", WEATHER)

    with st.expander("🛡️ Cybersecurity Threat", expanded=False):
        inp["Attack_Type"] = st.selectbox("Attack Type", ATTACKS, index=0)

    row = pd.DataFrame([inp])[FEATURES]
    label, probs = predict(row)
    label, p = label.iloc[0], probs.iloc[0]
    conf = float(p[label])

    st.divider()
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("Prediction")
        if label == "Normal":
            st.success("**NORMAL** – Optimal link health. No action required; continue routine monitoring.")
        elif label == "Degraded":
            st.warning("**DEGRADED** – Link quality is falling. Recommend proactive routing adjustment "
                       "and relay repositioning before the link worsens.")
        else:
            st.error("**CRITICAL** – High risk of link failure. Trigger immediate emergency failover "
                     "and recovery procedures.")
        if inp["Attack_Type"] != "None":
            st.info(f"🛡️ Active threat flagged: **{inp['Attack_Type']}** attack.")
    with right:
        fig = go.Figure(go.Bar(
            x=ORDER, y=[p[c] * 100 for c in ORDER],
            marker_color=[COLORS[c] for c in ORDER],
            text=[f"{p[c]*100:.1f}%" for c in ORDER], textposition="outside"))
        fig.update_layout(title="Class probability", yaxis=dict(range=[0, 115], title="%"),
                          height=330, margin=dict(l=10, r=10, t=45, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# 4. KPI CARDS (top of page, reflect the Tab-1 simulator)
# --------------------------------------------------------------------------
def kpi(col, lbl, val, sub, color=None):
    style = f' style="color:{color}"' if color else ""
    col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div>'
                 f'<div class="val"{style}>{val}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)

with kpi_area:
    k1, k2, k3 = st.columns(3)
    kpi(k1, "Live prediction status", label.upper(), "Simulated UAV (Tab 1)", COLORS[label])
    kpi(k2, "Model confidence", f"{conf*100:.1f}%", "Probability of predicted class")
    alerts = []
    if inp["Packet_Delivery_Ratio"] <= PDR_ALERT:
        alerts.append(f"PDR low ({inp['Packet_Delivery_Ratio']:.2f})")
    if inp["Latency"] >= LAT_ALERT:
        alerts.append(f"Latency high ({inp['Latency']:.0f} ms)")
    kpi(k3, "Key driver alerts (PDR & Latency)", "⚠️ " + " · ".join(alerts) if alerts else "✅ Within range",
        "PDR ≤ %.2f or Latency ≥ %.0f ms" % (PDR_ALERT, LAT_ALERT),
        COLORS["Critical"] if alerts else COLORS["Normal"])

# --------------------------------------------------------------------------
# 5. TAB 2 - BATCH PREDICTION
# --------------------------------------------------------------------------
with tab2:
    st.caption("Upload multi-UAV telemetry logs. `UAV_ID` is ignored for inference but kept in the output. "
               "A `Network_Status` column, if present, is ignored.")
    up = st.file_uploader("Telemetry CSV", type="csv")
    use_sample = False
    if SAMPLE_CSV.exists():
        use_sample = st.checkbox("…or try it on the bundled FANET dataset (5,000 records)")

    source = None
    if up is not None:
        source = pd.read_csv(up, keep_default_na=False)
    elif use_sample:
        source = pd.read_csv(SAMPLE_CSV, keep_default_na=False)

    if source is not None:
        data = clean_csv(source)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            for c in NUMERIC:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            pred, pr = predict(data)
            out = data.drop(columns=[TARGET], errors="ignore").copy()
            out["Predicted_Network_Status"] = pred.values
            for c in ORDER:
                out[f"P({c})"] = pr[c].round(4).values
            if TARGET in data.columns:
                acc = (data[TARGET].astype(str).values == pred.values).mean()
                st.caption(f"Uploaded file has labels – agreement with predictions: **{acc*100:.2f}%**")

            counts = pred.value_counts()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total UAVs monitored", f"{len(out):,}")
            m2.metric("🔴 Critical", f"{counts.get('Critical', 0):,}")
            m3.metric("🟠 Degraded", f"{counts.get('Degraded', 0):,}")
            m4.metric("🟢 Normal", f"{counts.get('Normal', 0):,}")

            fig = px.scatter(out, x="Latency", y="Packet_Delivery_Ratio",
                             color="Predicted_Network_Status", color_discrete_map=COLORS,
                             category_orders={"Predicted_Network_Status": ORDER},
                             hover_data=[c for c in [ID_COL, "Role", "Attack_Type"] if c in out.columns],
                             opacity=0.7, title="Latency vs. Packet Delivery Ratio")
            fig.update_layout(height=470, xaxis_title="Latency (ms)", yaxis_title="Packet Delivery Ratio")
            st.plotly_chart(fig, width="stretch")

            front = [c for c in [ID_COL, "Predicted_Network_Status"] if c in out.columns]
            show = out[front + [c for c in out.columns if c not in front]]
            st.dataframe(show, width="stretch", height=320)
            st.download_button("⬇️ Download Predictions CSV", show.to_csv(index=False).encode("utf-8"),
                               "fanet_predictions.csv", "text/csv", type="primary")

# --------------------------------------------------------------------------
# 6. TAB 3 - MODEL INSIGHTS (static benchmark summary)
# --------------------------------------------------------------------------
with tab3:
    st.subheader("Random Forest benchmark")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", "99.9%")
    b.metric("Macro-F1", "0.998")
    c.metric("ROC-AUC", "1.000")
    d.metric("Critical recall", "100%")

    imp = pd.DataFrame({"Feature": ["Packet Delivery Ratio", "Latency", "Y"],
                        "Importance (%)": [39.0, 36.5, 1.8]}).iloc[::-1]
    fig = px.bar(imp, x="Importance (%)", y="Feature", orientation="h", text="Importance (%)",
                 title="Top feature importances", color_discrete_sequence=["#3b82f6"])
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=45, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("""
**Architecture**
- **Data:** FANET Disaster Dataset – 5,000 simulated UAV telemetry records, 80/20 stratified split.
- **Inputs:** 16 numeric (median imputation → `StandardScaler`) + 4 categorical (most-frequent imputation →
  `OneHotEncoder`) = 32 encoded features. `UAV_ID` is dropped.
- **Model:** `RandomForestClassifier(n_estimators=100, class_weight="balanced")`.
- **Output:** `Normal`, `Degraded`, `Critical` with class probabilities.
- **Insight:** PDR and Latency alone carry ≈ 75 % of the importance, so they separate the three states cleanly.
""")
