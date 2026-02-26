"""
app.py — Entry point for the Energy Consumption Forecasting Dashboard.
         Handles page configuration, sidebar, data loading, and page routing.

Run with:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st

from utils import inject_css, load_data, engineer_features

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — data loading, navigation, settings
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Energy Forecasting")
    st.markdown("---")

    _candidates = [
        "Predicting Energy Consumption.csv",
        "energy_consumption.csv",
        "energy_data.csv",
    ]
    DATA_PATH = next(
        (Path(__file__).parent / f for f in _candidates
         if (Path(__file__).parent / f).exists()),
        Path(__file__).parent / "Predicting Energy Consumption.csv",
    )
    if not DATA_PATH.exists():
        uploaded = st.file_uploader(
            "Upload CSV — supported formats:\n"
            "• 3-column: `Start time UTC`, `End time UTC`, `Electricity consumption (MWh)`\n"
            "• 3-column (normalised): `start_timestamp`, `end_timestamp`, `consumption_mwh`\n"
            "• 2-column: `timestamp`, `consumption_mwh`",
            type="csv",
        )
        if uploaded:
            DATA_PATH = uploaded
        else:
            st.warning(
                "Place `Predicting Energy Consumption.csv` (or `energy_consumption.csv`) "
                "next to `app.py`, or upload your file above."
            )
            st.stop()

    df_raw = load_data(DATA_PATH)

    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        [
            "📊 Data Explorer",
            "🔮 Forecasting",
            "🧠 LSTM Deep Learning",
            "🔬 Explainability",
            "🔍 Anomaly Detection",
            "📈 Model Comparison",
            "🧬 Synthetic Data",
            "💬 AI Chatbot",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Forecast Settings")
    HORIZON = st.slider("Forecast horizon (hours)", 24, 336, 168, 24)

    st.markdown("### Models to Run")
    run_xgb     = st.checkbox("XGBoost", value=True)
    run_rf      = st.checkbox("Random Forest", value=True)
    run_arima   = st.checkbox("ARIMA", value=False)
    run_prophet = st.checkbox("Prophet", value=False)
    run_lstm    = st.checkbox("LSTM", value=False)

    st.markdown("---")
    st.markdown(
        f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.75rem;color:#8b949e;">'
        f'DATA: {len(df_raw):,} rows<br>'
        f'{df_raw.timestamp.min().strftime("%Y-%m-%d")} → {df_raw.timestamp.max().strftime("%Y-%m-%d")}'
        f'</span>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING & TRAIN/VAL/TEST SPLITS
# ──────────────────────────────────────────────────────────────────────────────
featured = engineer_features(df_raw)
FEATURE_COLS = [c for c in featured.columns if c not in ["timestamp", "consumption_mwh"]]
n         = len(featured)
train_end = int(n * 0.75)
val_end   = int(n * 0.85)
train = featured.iloc[:train_end].reset_index(drop=True)
val   = featured.iloc[train_end:val_end].reset_index(drop=True)
test  = featured.iloc[val_end:].reset_index(drop=True)

X_train, y_train = train[FEATURE_COLS].values, train["consumption_mwh"].values
X_val,   y_val   = val[FEATURE_COLS].values,   val["consumption_mwh"].values
X_test,  y_test  = test[FEATURE_COLS].values,  test["consumption_mwh"].values

# ──────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ──────────────────────────────────────────────────────────────────────────────
ctx = dict(
    df_raw=df_raw, featured=featured, FEATURE_COLS=FEATURE_COLS,
    train=train, val=val, test=test,
    X_train=X_train, y_train=y_train,
    X_val=X_val,   y_val=y_val,
    X_test=X_test, y_test=y_test,
    HORIZON=HORIZON,
    run_xgb=run_xgb, run_rf=run_rf,
    run_arima=run_arima, run_prophet=run_prophet, run_lstm=run_lstm,
)

if page == "📊 Data Explorer":
    from model_training import page_data_explorer
    page_data_explorer(ctx)

elif page == "🔮 Forecasting":
    from model_training import page_forecasting
    page_forecasting(ctx)

elif page == "🧠 LSTM Deep Learning":
    from model_training import page_lstm
    page_lstm(ctx)

elif page == "🔬 Explainability":
    from model_comparision import page_explainability
    page_explainability(ctx)

elif page == "🔍 Anomaly Detection":
    from model_comparision import page_anomaly_detection
    page_anomaly_detection(ctx)

elif page == "📈 Model Comparison":
    from model_comparision import page_model_comparison
    page_model_comparison(ctx)

elif page == "🧬 Synthetic Data":
    from model_comparision import page_synthetic_data
    page_synthetic_data(ctx)

elif page == "💬 AI Chatbot":
    from model_comparision import page_ai_chatbot
    page_ai_chatbot(ctx)
