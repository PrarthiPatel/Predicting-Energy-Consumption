"""
model_training.py — Pages 1–3 of the Energy Forecasting Dashboard:
    • page_data_explorer  — raw data exploration and seasonal analysis
    • page_forecasting    — XGBoost / RF / ARIMA / Prophet forecasts
    • page_lstm           — PyTorch LSTM training and evaluation
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    AMBER, TEAL, RED, MUTED,
    apply_layout, metric_card, section_header, evaluate,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 1: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
def page_data_explorer(ctx):
    df_raw = ctx["df_raw"]
    st.markdown("# 📊 Data Explorer")

    _has_interval = any(
        c in df_raw.columns for c in ("start_timestamp", "end_timestamp", "duration_hours")
    )
    if _has_interval:
        avg_dur_h = df_raw.get("duration_hours", pd.Series([1.0])).mean()
        st.info(
            f"🕒 **Interval-based dataset detected** (3-column format). "
            f"Representative timestamp = midpoint of each interval. "
            f"Average interval duration: **{avg_dur_h:.2f} h**"
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(metric_card("Total Records", f"{len(df_raw):,}", f"{df_raw.timestamp.min().year}–{df_raw.timestamp.max().year}"), unsafe_allow_html=True)
    with col2: st.markdown(metric_card("Mean Consumption", f"{df_raw.consumption_mwh.mean():.0f}", "MWh / hour"), unsafe_allow_html=True)
    with col3: st.markdown(metric_card("Peak Consumption", f"{df_raw.consumption_mwh.max():.0f}", "MWh"), unsafe_allow_html=True)
    with col4: st.markdown(metric_card("Min Consumption",  f"{df_raw.consumption_mwh.min():.0f}", "MWh"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full time series ──────────────────────────────────────────────────────
    section_header("Full Time Series")
    sample_size = st.slider("Downsample (every N rows)", 1, 24, 4, key="ts_sample")
    df_plot = df_raw.iloc[::sample_size]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot.timestamp, y=df_plot.consumption_mwh,
        mode="lines", line=dict(color=AMBER, width=0.8),
        fill="tozeroy", fillcolor="rgba(232,160,77,0.08)",
        name="Consumption",
    ))
    apply_layout(fig, title="Hourly Electricity Consumption", xaxis_title="Date", yaxis_title="MWh")
    st.plotly_chart(fig, use_container_width=True)

    # ── Seasonal patterns ─────────────────────────────────────────────────────
    section_header("Seasonal Patterns")
    df2 = df_raw.copy()
    df2["hour"]      = df2.timestamp.dt.hour
    df2["month"]     = df2.timestamp.dt.month
    df2["year"]      = df2.timestamp.dt.year
    df2["is_weekend"] = df2.timestamp.dt.dayofweek >= 5

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        hourly = df2.groupby(["hour", "is_weekend"])["consumption_mwh"].mean().reset_index()
        hourly["day_type"] = hourly.is_weekend.map({False: "Weekday", True: "Weekend"})
        fig2 = px.line(hourly, x="hour", y="consumption_mwh", color="day_type",
                       color_discrete_map={"Weekday": AMBER, "Weekend": TEAL},
                       markers=True, title="Average Hourly Profile")
        fig2.update_traces(marker_size=4)
        apply_layout(fig2, xaxis_title="Hour of Day", yaxis_title="Avg MWh")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        monthly = df2.groupby("month")["consumption_mwh"].mean().reset_index()
        month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["month_name"] = monthly.month.apply(lambda x: month_names[x-1])
        fig3 = px.bar(monthly, x="month_name", y="consumption_mwh",
                      color="consumption_mwh", color_continuous_scale="RdYlBu_r",
                      title="Average Monthly Consumption")
        apply_layout(fig3, xaxis_title="", yaxis_title="Avg MWh", coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col_c:
        yearly = df2.groupby("year")["consumption_mwh"].mean().reset_index()
        fig4 = px.bar(yearly, x="year", y="consumption_mwh",
                      color_discrete_sequence=[AMBER], title="Average Yearly Consumption")
        apply_layout(fig4, xaxis_title="Year", yaxis_title="Avg MWh")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Heatmap ───────────────────────────────────────────────────────────────
    section_header("Consumption Heatmap: Hour × Day of Week")
    df2["day_name"] = df2.timestamp.dt.day_name()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = df2.pivot_table(values="consumption_mwh", index="hour", columns="day_name", aggfunc="mean")
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])
    fig5 = px.imshow(pivot, color_continuous_scale="RdYlBu_r",
                     labels=dict(x="Day of Week", y="Hour of Day", color="Avg MWh"), aspect="auto")
    apply_layout(fig5, title="Average Consumption Heatmap")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Distribution ──────────────────────────────────────────────────────────
    section_header("Distribution Analysis")
    col_d, col_e = st.columns(2)
    with col_d:
        fig6 = px.histogram(df_raw, x="consumption_mwh", nbins=80,
                             color_discrete_sequence=[AMBER], title="Consumption Distribution")
        fig6.update_traces(marker_line_width=0)
        apply_layout(fig6, xaxis_title="MWh", yaxis_title="Count")
        st.plotly_chart(fig6, use_container_width=True)
    with col_e:
        fig7 = go.Figure()
        fig7.add_trace(go.Box(y=df_raw.consumption_mwh, name="Consumption",
                               marker_color=AMBER, boxmean="sd"))
        apply_layout(fig7, title="Box Plot", yaxis_title="MWh")
        st.plotly_chart(fig7, use_container_width=True)

    # ── ACF ───────────────────────────────────────────────────────────────────
    section_header("Autocorrelation (ACF)")
    from statsmodels.tsa.stattools import acf
    series = df_raw.consumption_mwh.dropna().values
    NLAGS = 72
    acf_vals = acf(series, nlags=NLAGS, fft=True)
    ci = 1.96 / np.sqrt(len(series))
    lags = np.arange(NLAGS + 1)
    fig8 = go.Figure()
    fig8.add_bar(x=lags, y=acf_vals, marker_color=AMBER, name="ACF")
    fig8.add_hline(y=ci,  line_dash="dash", line_color=RED, annotation_text=f"+{ci:.3f}")
    fig8.add_hline(y=-ci, line_dash="dash", line_color=RED, annotation_text=f"{-ci:.3f}")
    apply_layout(fig8, title="Autocorrelation Function", xaxis_title="Lag (hours)", yaxis_title="Correlation")
    st.plotly_chart(fig8, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 2: FORECASTING
# ──────────────────────────────────────────────────────────────────────────────
def page_forecasting(ctx):
    df_raw       = ctx["df_raw"]
    train        = ctx["train"]
    val          = ctx["val"]
    test         = ctx["test"]
    X_train      = ctx["X_train"]; y_train = ctx["y_train"]
    X_val        = ctx["X_val"];   y_val   = ctx["y_val"]
    X_test       = ctx["X_test"];  y_test  = ctx["y_test"]
    FEATURE_COLS = ctx["FEATURE_COLS"]
    HORIZON      = ctx["HORIZON"]
    run_xgb      = ctx["run_xgb"]
    run_rf       = ctx["run_rf"]
    run_arima    = ctx["run_arima"]
    run_prophet  = ctx["run_prophet"]

    st.markdown("# 🔮 Forecasting")

    @st.cache_resource(show_spinner="Training XGBoost…")
    def train_xgb(_X_train, _y_train, _X_val, _y_val):
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=30, eval_metric="rmse",
            random_state=42, verbosity=0, n_jobs=-1,
        )
        model.fit(_X_train, _y_train, eval_set=[(_X_val, _y_val)], verbose=False)
        return model

    @st.cache_resource(show_spinner="Training Random Forest…")
    def train_rf(_X_train, _y_train):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1,
        )
        model.fit(_X_train, _y_train)
        return model

    @st.cache_resource(show_spinner="Fitting ARIMA…")
    def train_arima(_train_series, _horizon):
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(_train_series, order=(2, 1, 2)).fit()
        return model.forecast(steps=_horizon).values

    @st.cache_resource(show_spinner="Fitting Prophet…")
    def train_prophet(_train_df, _horizon):
        from prophet import Prophet
        prophet_train = _train_df[["timestamp","consumption_mwh"]].rename(
            columns={"timestamp":"ds","consumption_mwh":"y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=True, seasonality_mode="multiplicative")
        m.fit(prophet_train)
        future = m.make_future_dataframe(periods=_horizon, freq="h", include_history=False)
        fc = m.predict(future)
        return fc["yhat"].values[:_horizon]

    pred_registry   = {}
    metric_registry = {}

    if run_xgb:
        try:
            xgb_model = train_xgb(X_train, y_train, X_val, y_val)
            xgb_preds = xgb_model.predict(X_test[:HORIZON])
            pred_registry["XGBoost"]   = xgb_preds
            metric_registry["XGBoost"] = evaluate(y_test[:HORIZON], xgb_preds)
        except Exception as e:
            st.warning(f"XGBoost failed: {e}")

    if run_rf:
        try:
            rf_model = train_rf(X_train, y_train)
            rf_preds = rf_model.predict(X_test[:HORIZON])
            pred_registry["Random Forest"]   = rf_preds
            metric_registry["Random Forest"] = evaluate(y_test[:HORIZON], rf_preds)
        except Exception as e:
            st.warning(f"Random Forest failed: {e}")

    if run_arima:
        try:
            arima_preds = train_arima(
                train.set_index("timestamp")["consumption_mwh"], HORIZON)
            pred_registry["ARIMA"]   = arima_preds
            metric_registry["ARIMA"] = evaluate(y_test[:HORIZON], arima_preds)
        except Exception as e:
            st.warning(f"ARIMA failed: {e}")

    if run_prophet:
        try:
            prophet_preds = train_prophet(train, HORIZON)
            pred_registry["Prophet"]   = prophet_preds
            metric_registry["Prophet"] = evaluate(y_test[:HORIZON], prophet_preds)
        except Exception as e:
            st.warning(f"Prophet failed: {e}")

    if not pred_registry:
        st.info("Select at least one model in the sidebar and click Train.")
        st.stop()

    # ── Metrics row ───────────────────────────────────────────────────────────
    section_header("Model Performance Metrics")
    cols = st.columns(len(pred_registry))
    colors_list = [AMBER, TEAL, RED, "#a78bfa", "#34d399"]
    for i, (name, m) in enumerate(metric_registry.items()):
        with cols[i]:
            st.markdown(
                f'<div class="metric-card" style="border-top-color:{colors_list[i%len(colors_list)]};">'
                f'<div class="label">{name}</div>'
                f'<div class="value" style="font-size:1.1rem;color:{colors_list[i%len(colors_list)]};">'
                f'RMSE {m["RMSE"]:.1f}</div>'
                f'<div class="delta">MAE {m["MAE"]:.1f} | MAPE {m["MAPE"]:.2f}% | R² {m["R²"]:.4f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Forecast overlay ──────────────────────────────────────────────────────
    section_header("Forecast vs Actual")
    fig = go.Figure()
    hist = df_raw.tail(24 * 14)
    fig.add_trace(go.Scatter(
        x=hist.timestamp, y=hist.consumption_mwh,
        mode="lines", line=dict(color="#4a7fba", width=1.2),
        fill="tozeroy", fillcolor="rgba(74,127,186,0.06)", name="Historical",
    ))
    fig.add_trace(go.Scatter(
        x=test.timestamp.values[:HORIZON], y=y_test[:HORIZON],
        mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="Actual",
    ))
    test_ts = test["timestamp"].values[:HORIZON]
    for (name, preds), color in zip(pred_registry.items(), colors_list):
        fig.add_trace(go.Scatter(
            x=test_ts, y=preds[:HORIZON],
            mode="lines", line=dict(color=color, width=1.8, dash="dash"), name=name,
        ))
    _split_ts = pd.Timestamp(test["timestamp"].iloc[0])
    fig.add_vline(x=_split_ts.timestamp() * 1000, line_dash="dot", line_color=MUTED)
    fig.add_annotation(x=_split_ts, y=1, yref="paper", text="Train/Test split",
                       showarrow=False, font=dict(color=MUTED, size=11),
                       xanchor="left", yanchor="bottom")
    apply_layout(fig, title=f"Model Forecast Comparison (next {HORIZON}h)",
                 xaxis_title="Date", yaxis_title="Consumption (MWh)")
    st.plotly_chart(fig, use_container_width=True)

    # ── Confidence intervals (XGBoost only) ──────────────────────────────────
    if "XGBoost" in pred_registry:
        section_header("XGBoost Forecast with Confidence Intervals")
        val_preds_ci = train_xgb(X_train, y_train, X_val, y_val).predict(X_val)
        std_resid = (y_val - val_preds_ci).std()
        preds_ci  = pred_registry["XGBoost"]
        lower_95 = preds_ci - 1.96 * std_resid
        upper_95 = preds_ci + 1.96 * std_resid
        lower_80 = preds_ci - 1.28 * std_resid
        upper_80 = preds_ci + 1.28 * std_resid

        fig_ci = go.Figure()
        hist2 = df_raw.tail(24 * 7)
        fig_ci.add_trace(go.Scatter(
            x=hist2.timestamp, y=hist2.consumption_mwh,
            mode="lines", line=dict(color="#4a7fba", width=1.2), name="Historical"))
        fig_ci.add_trace(go.Scatter(
            x=np.concatenate([test_ts, test_ts[::-1]]),
            y=np.concatenate([upper_95, lower_95[::-1]]),
            fill="toself", fillcolor="rgba(232,160,77,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="95% CI"))
        fig_ci.add_trace(go.Scatter(
            x=np.concatenate([test_ts, test_ts[::-1]]),
            y=np.concatenate([upper_80, lower_80[::-1]]),
            fill="toself", fillcolor="rgba(232,160,77,0.22)",
            line=dict(color="rgba(0,0,0,0)"), name="80% CI"))
        fig_ci.add_trace(go.Scatter(
            x=test_ts, y=preds_ci, mode="lines",
            line=dict(color=AMBER, width=2, dash="dash"), name="XGBoost Forecast"))
        fig_ci.add_trace(go.Scatter(
            x=test_ts, y=y_test[:HORIZON], mode="lines",
            line=dict(color=MUTED, width=1, dash="dot"), name="Actual"))
        apply_layout(fig_ci, title="XGBoost Forecast + Confidence Intervals",
                     xaxis_title="Date", yaxis_title="MWh")
        st.plotly_chart(fig_ci, use_container_width=True)
        st.caption(f"Residual std (from validation set): ±{std_resid:.1f} MWh")

    # ── Residual diagnostics ──────────────────────────────────────────────────
    best_name  = next(iter(pred_registry))
    best_preds = pred_registry[best_name]
    residuals  = y_test[:HORIZON] - best_preds[:HORIZON]

    section_header(f"Residual Diagnostics — {best_name}")
    fig_r = make_subplots(rows=2, cols=2,
        subplot_titles=["Residuals Over Time","Residual Distribution",
                        "Actual vs Predicted","Residuals vs Predicted"])
    fig_r.add_trace(go.Scatter(y=residuals, mode="lines",
                               line=dict(color=AMBER, width=0.8), name="Residual"), row=1, col=1)
    fig_r.add_hline(y=0, line_dash="dash", line_color=RED, row=1, col=1)
    fig_r.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color=AMBER,
                                  marker_line_width=0, name="Dist"), row=1, col=2)
    fig_r.add_trace(go.Scatter(x=y_test[:HORIZON], y=best_preds[:HORIZON],
                               mode="markers", marker=dict(color=TEAL, size=4, opacity=0.5),
                               name="Actual vs Pred"), row=2, col=1)
    mn, mx = y_test[:HORIZON].min(), y_test[:HORIZON].max()
    fig_r.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx],
                               line=dict(color=RED, dash="dash"), name="Perfect"), row=2, col=1)
    fig_r.add_trace(go.Scatter(x=best_preds[:HORIZON], y=residuals,
                               mode="markers", marker=dict(color="#a78bfa", size=4, opacity=0.5),
                               name="Resid vs Pred"), row=2, col=2)
    fig_r.add_hline(y=0, line_dash="dash", line_color=RED, row=2, col=2)
    fig_r.update_layout(**{}, height=600, title_text=f"Residual Diagnostics — {best_name}")
    from utils import PLOTLY_LAYOUT
    fig_r.update_layout(**PLOTLY_LAYOUT, height=600)
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    section_header("Feature Importance")
    fi_cols = st.columns(min(len(pred_registry), 2))
    fi_idx  = 0
    for name in pred_registry:
        try:
            if name == "XGBoost":
                m = train_xgb(X_train, y_train, X_val, y_val)
            elif name == "Random Forest":
                m = train_rf(X_train, y_train)
            else:
                continue
            importance = pd.Series(m.feature_importances_, index=FEATURE_COLS)
            top20 = importance.sort_values(ascending=True).tail(20)
            fig_fi = go.Figure(go.Bar(
                x=top20.values, y=top20.index, orientation="h",
                marker=dict(color=top20.values, colorscale="Viridis"),
            ))
            apply_layout(fig_fi, title=f"{name} — Top 20 Features",
                         xaxis_title="Importance", height=500)
            with fi_cols[fi_idx % 2]:
                st.plotly_chart(fig_fi, use_container_width=True)
            fi_idx += 1
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 3: LSTM DEEP LEARNING
# ──────────────────────────────────────────────────────────────────────────────
def page_lstm(ctx):
    df_raw  = ctx["df_raw"]
    train   = ctx["train"]
    val     = ctx["val"]
    test    = ctx["test"]
    y_val   = ctx["y_val"]
    y_test  = ctx["y_test"]
    HORIZON = ctx["HORIZON"]

    st.markdown("# 🧠 LSTM Deep Learning")
    st.markdown("Train a 2-layer LSTM with PyTorch on the energy time series.")

    col_l1, col_l2, col_l3, col_l4 = st.columns(4)
    with col_l1: lstm_epochs  = st.slider("Epochs", 5, 50, 20, 5)
    with col_l2: lstm_hidden  = st.slider("Hidden units", 32, 256, 128, 32)
    with col_l3: lstm_seq_len = st.slider("Seq length (h)", 24, 336, 168, 24)
    with col_l4: lstm_horizon = st.slider("LSTM horizon (h)", 6, 72, 24, 6)

    if st.button("🚀 Train LSTM", use_container_width=True):
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.caption(f"Device: **{DEVICE}**")

            lstm_mean = float(train["consumption_mwh"].mean())
            lstm_std  = float(train["consumption_mwh"].std())

            def normalize(x): return (x - lstm_mean) / lstm_std
            def denorm(x):    return x * lstm_std + lstm_mean

            class SeqDataset(Dataset):
                def __init__(self, series, seq_len, horizon):
                    self.data    = normalize(series.astype(np.float32))
                    self.seq_len = seq_len
                    self.horizon = horizon
                def __len__(self):
                    return len(self.data) - self.seq_len - self.horizon + 1
                def __getitem__(self, i):
                    x = torch.tensor(self.data[i:i+self.seq_len]).unsqueeze(-1)
                    y = torch.tensor(self.data[i+self.seq_len:i+self.seq_len+self.horizon])
                    return x, y

            class LSTMNet(nn.Module):
                def __init__(self, hidden=128, layers=2, dropout=0.2, horizon=24):
                    super().__init__()
                    self.lstm = nn.LSTM(1, hidden, layers, batch_first=True,
                                        dropout=dropout if layers > 1 else 0.0)
                    self.head = nn.Sequential(
                        nn.Linear(hidden, hidden // 2), nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden // 2, horizon),
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.head(out[:, -1, :])

            train_ds = SeqDataset(train["consumption_mwh"].values, lstm_seq_len, lstm_horizon)
            val_ds   = SeqDataset(val["consumption_mwh"].values,   lstm_seq_len, lstm_horizon)
            train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_dl   = DataLoader(val_ds,   batch_size=64)

            lstm_net  = LSTMNet(hidden=lstm_hidden, horizon=lstm_horizon).to(DEVICE)
            optimizer = torch.optim.Adam(lstm_net.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
            criterion = nn.HuberLoss()

            n_params = sum(p.numel() for p in lstm_net.parameters())
            st.markdown(f"**Params:** `{n_params:,}`")

            progress_bar = st.progress(0.0)
            status_txt   = st.empty()
            train_losses, val_losses = [], []
            best_val, best_state = float("inf"), None

            for epoch in range(lstm_epochs):
                lstm_net.train(); t_loss = 0.0
                for xb, yb in train_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(lstm_net(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(lstm_net.parameters(), 1.0)
                    optimizer.step()
                    t_loss += loss.item()
                t_loss /= len(train_dl)

                lstm_net.eval(); v_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_dl:
                        v_loss += criterion(lstm_net(xb.to(DEVICE)), yb.to(DEVICE)).item()
                v_loss /= len(val_dl)
                scheduler.step(v_loss)
                train_losses.append(t_loss); val_losses.append(v_loss)
                if v_loss < best_val:
                    best_val   = v_loss
                    best_state = {k: v.clone() for k, v in lstm_net.state_dict().items()}

                pct = (epoch + 1) / lstm_epochs
                progress_bar.progress(pct)
                status_txt.text(f"Epoch {epoch+1}/{lstm_epochs}  train={t_loss:.4f}  val={v_loss:.4f}")

            lstm_net.load_state_dict(best_state)

            # ── Training curves ────────────────────────────────────────────────
            section_header("Training Curves")
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(y=train_losses, mode="lines+markers",
                                         marker_size=4, line_color="#4a7fba", name="Train Loss"))
            fig_lc.add_trace(go.Scatter(y=val_losses, mode="lines+markers",
                                         marker_size=4, line_color=AMBER, name="Val Loss"))
            apply_layout(fig_lc, title="LSTM Huber Loss per Epoch",
                         xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig_lc, use_container_width=True)

            # ── Rolling inference ──────────────────────────────────────────────
            combined = np.concatenate([train["consumption_mwh"].values,
                                        val["consumption_mwh"].values])
            ctx_arr = normalize(combined[-lstm_seq_len:].astype(np.float32))
            all_preds    = []
            steps_needed = int(np.ceil(HORIZON / lstm_horizon))
            lstm_net.eval()
            for _ in range(steps_needed):
                xt = torch.tensor(ctx_arr).unsqueeze(0).unsqueeze(-1).to(DEVICE)
                with torch.no_grad():
                    out = lstm_net(xt).cpu().numpy().flatten()
                all_preds.extend(out)
                ctx_arr = np.concatenate([ctx_arr[lstm_horizon:], out])
            lstm_preds  = denorm(np.array(all_preds[:HORIZON]))
            metrics_lstm = evaluate(y_test[:HORIZON], lstm_preds)

            # ── KPI row ───────────────────────────────────────────────────────
            section_header("Evaluation Metrics")
            km1, km2, km3, km4 = st.columns(4)
            with km1: st.markdown(metric_card("RMSE", f"{metrics_lstm['RMSE']:.1f}", "MWh"), unsafe_allow_html=True)
            with km2: st.markdown(metric_card("MAE",  f"{metrics_lstm['MAE']:.1f}",  "MWh"), unsafe_allow_html=True)
            with km3: st.markdown(metric_card("MAPE", f"{metrics_lstm['MAPE']:.2f}", "%"),   unsafe_allow_html=True)
            with km4: st.markdown(metric_card("R²",   f"{metrics_lstm['R²']:.4f}",   ""),    unsafe_allow_html=True)

            # ── Forecast chart ────────────────────────────────────────────────
            section_header("LSTM Forecast vs Actual")
            fig_lf = go.Figure()
            hist_l = df_raw.tail(24 * 14)
            fig_lf.add_trace(go.Scatter(
                x=hist_l.timestamp, y=hist_l.consumption_mwh,
                mode="lines", line=dict(color="#4a7fba", width=1),
                fill="tozeroy", fillcolor="rgba(74,127,186,0.07)", name="Historical"))
            fig_lf.add_trace(go.Scatter(
                x=test.timestamp.values[:HORIZON], y=y_test[:HORIZON],
                mode="lines", line=dict(color=MUTED, width=1, dash="dot"), name="Actual"))
            fig_lf.add_trace(go.Scatter(
                x=test.timestamp.values[:HORIZON], y=lstm_preds,
                mode="lines", line=dict(color=TEAL, width=2, dash="dash"), name="LSTM Forecast"))
            fig_lf.add_traces([
                go.Scatter(
                    x=np.concatenate([test.timestamp.values[:HORIZON],
                                       test.timestamp.values[:HORIZON][::-1]]),
                    y=np.concatenate([lstm_preds + 1.96*metrics_lstm["RMSE"],
                                       (lstm_preds - 1.96*metrics_lstm["RMSE"])[::-1]]),
                    fill="toself", fillcolor="rgba(61,214,198,0.08)",
                    line=dict(color="rgba(0,0,0,0)"), name="95% CI"),
            ])
            apply_layout(fig_lf, title=f"LSTM Forecast — next {HORIZON}h",
                         xaxis_title="Date", yaxis_title="Consumption (MWh)")
            st.plotly_chart(fig_lf, use_container_width=True)

            # ── Residuals ─────────────────────────────────────────────────────
            section_header("Residual Analysis")
            residuals_lstm = y_test[:HORIZON] - lstm_preds
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                fig_rh = go.Figure(go.Histogram(
                    x=residuals_lstm, nbinsx=40, marker_color=TEAL, opacity=0.8))
                apply_layout(fig_rh, title="Residual Distribution",
                             xaxis_title="Error (MWh)", yaxis_title="Count")
                st.plotly_chart(fig_rh, use_container_width=True)
            with col_r2:
                fig_ap = go.Figure(go.Scatter(
                    x=y_test[:HORIZON], y=lstm_preds,
                    mode="markers", marker=dict(color=TEAL, size=4, opacity=0.5)))
                mx_v = max(y_test[:HORIZON].max(), lstm_preds.max())
                mn_v = min(y_test[:HORIZON].min(), lstm_preds.min())
                fig_ap.add_trace(go.Scatter(x=[mn_v, mx_v], y=[mn_v, mx_v],
                                              line=dict(color=RED, dash="dash"), name="Perfect"))
                apply_layout(fig_ap, title="Actual vs Predicted",
                             xaxis_title="Actual (MWh)", yaxis_title="Predicted (MWh)")
                st.plotly_chart(fig_ap, use_container_width=True)

        except ImportError:
            st.error("PyTorch not installed. Run: `pip install torch`")
        except Exception as e:
            st.error(f"LSTM training failed: {e}")
    else:
        st.info("Configure the hyperparameters above and click **🚀 Train LSTM** to begin.")
        st.markdown("""
**Architecture:**
- 2-layer LSTM with configurable hidden size
- Huber loss (robust to outliers)
- Adam optimizer + ReduceLROnPlateau scheduler
- Rolling 24-h inference to produce multi-step forecasts

**Tips:**
- Start with 20 epochs and 128 hidden units
- Increase sequence length to capture weekly patterns (168h)
- LSTM is best for capturing non-linear temporal dependencies
""")
