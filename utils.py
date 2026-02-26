"""
utils.py — Shared constants, CSS injection, helper functions,
           data loading, and feature engineering for the Energy Forecasting Dashboard.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# COLOR CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
AMBER   = "#e8a04d"
TEAL    = "#3dd6c6"
RED     = "#f05060"
MUTED   = "#8b949e"
BG_CARD = "#161b22"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Barlow, sans-serif", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#30363d"),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ──────────────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;500;700&display=swap');

:root {
    --bg-dark:   #0d1117;
    --bg-card:   #161b22;
    --bg-panel:  #1c2128;
    --amber:     #e8a04d;
    --amber-dim: #a06828;
    --teal:      #3dd6c6;
    --red:       #f05060;
    --text:      #c9d1d9;
    --text-muted:#8b949e;
    --border:    #30363d;
}

.stApp { background: var(--bg-dark); font-family: 'Barlow', sans-serif; }
[data-testid="stSidebar"] { background: var(--bg-card) !important; border-right: 1px solid var(--border); }

h1, h2, h3 { font-family: 'Barlow', sans-serif; font-weight: 700; color: #e0e6ef; }
p, li, span, div { color: var(--text); }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--amber);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    font-family: 'Share Tech Mono', monospace;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--amber);
    font-family: 'Share Tech Mono', monospace;
}
.metric-card .delta { font-size: 0.8rem; color: var(--text-muted); }

.section-header {
    border-left: 4px solid var(--amber);
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.75rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e6ef;
    letter-spacing: 0.03em;
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: var(--text) !important; }

.stButton > button {
    background: linear-gradient(135deg, var(--amber-dim), var(--amber));
    color: #0d1117;
    border: none;
    font-weight: 700;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.05em;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stTabs"] button {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--amber) !important;
    border-bottom-color: var(--amber) !important;
}

[data-testid="stExpander"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: 6px; }
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }

.badge-success { background:#1a3a2a; color:#3dd6c6; border:1px solid #3dd6c6; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Share Tech Mono',monospace; }
.badge-warn    { background:#3a2a10; color:#e8a04d; border:1px solid #e8a04d; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-family:'Share Tech Mono',monospace; }

hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.15):
    """Convert #rrggbb to rgba(r,g,b,alpha) — Plotly rejects 8-digit hex."""
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return hex_color


def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig


def metric_card(label, value, delta=""):
    return f"""
<div class="metric-card">
  <div class="label">{label}</div>
  <div class="value">{value}</div>
  <div class="delta">{delta}</div>
</div>"""


def section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def evaluate(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    rmse  = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae   = float(np.mean(np.abs(y_true - y_pred)))
    mask  = y_true != 0
    mape  = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    r2    = float(1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-8))
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data…")
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "start" in cl and col_map.get("start_timestamp") is None:
            col_map[c] = "start_timestamp"
        elif "end" in cl and col_map.get("end_timestamp") is None:
            col_map[c] = "end_timestamp"
        elif ("consumption" in cl or "mwh" in cl) and col_map.get("consumption_mwh") is None:
            col_map[c] = "consumption_mwh"
        elif cl == "timestamp" and col_map.get("timestamp") is None:
            col_map[c] = "timestamp"
    df = df.rename(columns=col_map)
    cols = list(df.columns)

    if "start_timestamp" in cols and "end_timestamp" in cols:
        df["start_timestamp"] = pd.to_datetime(df["start_timestamp"])
        df["end_timestamp"]   = pd.to_datetime(df["end_timestamp"])
        df["duration_hours"] = (
            (df["end_timestamp"] - df["start_timestamp"]).dt.total_seconds() / 3600
        )
        df["timestamp"] = df["start_timestamp"] + (df["end_timestamp"] - df["start_timestamp"]) / 2
        df = df[["timestamp", "start_timestamp", "end_timestamp", "duration_hours", "consumption_mwh"]]
    elif "timestamp" in cols:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(
            "CSV must contain either ['timestamp', 'consumption_mwh'] "
            "or ['start_timestamp'/'Start time UTC', 'end_timestamp'/'End time UTC', "
            "'consumption_mwh'/'Electricity consumption (MWh)'] columns."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp").resample("h").mean()

    Q1 = df["consumption_mwh"].quantile(0.01)
    Q3 = df["consumption_mwh"].quantile(0.99)
    IQR = Q3 - Q1
    outlier_mask = (df["consumption_mwh"] < Q1 - 3*IQR) | (df["consumption_mwh"] > Q3 + 3*IQR)
    df.loc[outlier_mask, "consumption_mwh"] = np.nan
    df["consumption_mwh"] = df["consumption_mwh"].interpolate(method="time", limit=24).bfill().ffill()
    return df.reset_index()


@st.cache_data(show_spinner="Engineering features…")
def engineer_features(df):
    df = df.copy()
    df["hour"]        = df.timestamp.dt.hour
    df["day_of_week"] = df.timestamp.dt.dayofweek
    df["month"]       = df.timestamp.dt.month
    df["quarter"]     = df.timestamp.dt.quarter
    df["day_of_year"] = df.timestamp.dt.dayofyear
    df["is_weekend"]  = (df.timestamp.dt.dayofweek >= 5).astype(int)
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]     = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]     = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    if "duration_hours" in df.columns:
        df["duration_hours"] = df["duration_hours"].fillna(1.0)
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}"] = df["consumption_mwh"].shift(lag)
    for w in [6, 12, 24, 48, 168]:
        df[f"roll_mean_{w}"] = df["consumption_mwh"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"]  = df["consumption_mwh"].shift(1).rolling(w).std()
    drop_cols = [c for c in ["start_timestamp", "end_timestamp"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df.dropna().reset_index(drop=True)
