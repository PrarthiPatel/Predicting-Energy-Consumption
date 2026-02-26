"""
model_comparision.py — Pages 4–8 of the Energy Forecasting Dashboard:
    • page_explainability      — SHAP values, feature importance, seasonality decomposition
    • page_anomaly_detection   — rolling Z-score anomaly detection
    • page_model_comparison    — side-by-side model metrics, radar chart
    • page_synthetic_data      — Fourier-based synthetic data generator
    • page_ai_chatbot          — AI chatbot backed by Groq API (with local fallback)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from utils import (
    AMBER, TEAL, RED, MUTED,
    PLOTLY_LAYOUT, hex_to_rgba,
    apply_layout, metric_card, section_header, evaluate,
)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 4: EXPLAINABILITY
# ──────────────────────────────────────────────────────────────────────────────
def page_explainability(ctx):
    df_raw       = ctx["df_raw"]
    FEATURE_COLS = ctx["FEATURE_COLS"]
    X_train      = ctx["X_train"]; y_train = ctx["y_train"]
    X_val        = ctx["X_val"];   y_val   = ctx["y_val"]
    X_test       = ctx["X_test"]

    st.markdown("# 🔬 Explainability")
    st.markdown("SHAP values, feature importance, and seasonality decomposition.")

    @st.cache_resource(show_spinner="Training XGBoost for SHAP…")
    def train_xgb_shap(_X_train, _y_train, _X_val, _y_val):
        from xgboost import XGBRegressor
        m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         early_stopping_rounds=20, eval_metric="rmse",
                         random_state=42, verbosity=0, n_jobs=-1)
        m.fit(_X_train, _y_train, eval_set=[(_X_val, _y_val)], verbose=False)
        return m

    @st.cache_resource(show_spinner="Training Random Forest for SHAP…")
    def train_rf_shap(_X_train, _y_train):
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        m.fit(_X_train, _y_train)
        return m

    tab_fi, tab_shap, tab_decomp = st.tabs(
        ["📊 Feature Importance", "🧮 SHAP Analysis", "📉 Seasonality Decomposition"])

    # ── TAB 1: Feature Importance ─────────────────────────────────────────────
    with tab_fi:
        section_header("Tree Model Feature Importance")
        xgb_fi = train_xgb_shap(X_train, y_train, X_val, y_val)
        rf_fi  = train_rf_shap(X_train, y_train)

        col_f1, col_f2 = st.columns(2)
        for col, model, name, color in [
            (col_f1, xgb_fi, "XGBoost", AMBER),
            (col_f2, rf_fi,  "Random Forest", TEAL),
        ]:
            importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
            top20 = importance.sort_values(ascending=True).tail(20)
            fig_fi = go.Figure(go.Bar(
                x=top20.values, y=top20.index, orientation="h",
                marker=dict(color=top20.values,
                            colorscale=[[0, "#1c2128"], [1, color]], showscale=False),
                text=[f"{v:.4f}" for v in top20.values],
                textposition="outside", textfont=dict(size=9),
            ))
            apply_layout(fig_fi, title=f"{name} — Top 20 Features",
                         xaxis_title="Importance", height=550)
            with col:
                st.plotly_chart(fig_fi, use_container_width=True)

        section_header("Feature Importance Comparison (Top 15)")
        top15_xgb = pd.Series(xgb_fi.feature_importances_, index=FEATURE_COLS).nlargest(15)
        top15_rf  = pd.Series(rf_fi.feature_importances_,  index=FEATURE_COLS).nlargest(15)
        all_feats = list(set(top15_xgb.index) | set(top15_rf.index))
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="XGBoost", x=all_feats,
                                   y=[float(top15_xgb.get(f, 0)) for f in all_feats],
                                   marker_color=AMBER, opacity=0.85))
        fig_cmp.add_trace(go.Bar(name="Random Forest", x=all_feats,
                                   y=[float(top15_rf.get(f, 0)) for f in all_feats],
                                   marker_color=TEAL, opacity=0.85))
        fig_cmp.update_layout(barmode="group")
        apply_layout(fig_cmp, title="XGBoost vs Random Forest — Feature Importance",
                     xaxis_title="Feature", yaxis_title="Importance", height=400)
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── TAB 2: SHAP ────────────────────────────────────────────────────────────
    with tab_shap:
        shap_n = st.slider("SHAP sample size (n)", 100, 1000, 300, 100)
        if st.button("🧮 Compute SHAP Values", use_container_width=True):
            try:
                import shap
                xgb_shap  = train_xgb_shap(X_train, y_train, X_val, y_val)
                explainer  = shap.TreeExplainer(xgb_shap)
                shap_sample = X_test[:shap_n]
                shap_vals   = explainer.shap_values(shap_sample)

                section_header("Mean |SHAP| Value — XGBoost (Top 20)")
                mean_shap   = np.abs(shap_vals).mean(axis=0)
                shap_series = pd.Series(mean_shap, index=FEATURE_COLS).sort_values(ascending=True).tail(20)
                fig_ms = go.Figure(go.Bar(
                    x=shap_series.values, y=shap_series.index, orientation="h",
                    marker=dict(color=shap_series.values, colorscale='Blues', showscale=False),
                    text=[f"{v:.4f}" for v in shap_series.values],
                    textposition="outside", textfont=dict(size=9),
                ))
                apply_layout(fig_ms, title="SHAP Mean |Value| — Feature Importance",
                             xaxis_title="Mean |SHAP|", height=550)
                st.plotly_chart(fig_ms, use_container_width=True)

                section_header("SHAP Beeswarm — Top 10 Features")
                top10_idx   = np.argsort(mean_shap)[-10:]
                top10_names = [FEATURE_COLS[i] for i in top10_idx]
                fig_bee = go.Figure()
                for i, (idx, fname) in enumerate(zip(top10_idx, top10_names)):
                    feat_vals = shap_sample[:, idx]
                    sv        = shap_vals[:, idx]
                    norm_f    = (feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-8)
                    colors_bee = [f"hsl({int(240 - c*240)},80%,60%)" for c in norm_f]
                    fig_bee.add_trace(go.Scatter(
                        x=sv,
                        y=[i + float(np.random.uniform(-0.25, 0.25)) for _ in sv],
                        mode="markers",
                        marker=dict(color=colors_bee, size=4, opacity=0.6),
                        name=fname, showlegend=True,
                    ))
                fig_bee.add_vline(x=0, line_dash="dash", line_color=MUTED)
                fig_bee.update_layout(
                    yaxis=dict(tickmode="array", tickvals=list(range(10)),
                               ticktext=top10_names, gridcolor="#21262d"),
                )
                apply_layout(fig_bee, title="SHAP Beeswarm (Blue=low value, Red=high value)",
                             xaxis_title="SHAP Value", height=500)
                st.plotly_chart(fig_bee, use_container_width=True)

                section_header("SHAP Waterfall — Single Prediction")
                sample_idx = st.slider("Sample index", 0, shap_n - 1, 0)
                sv_single  = shap_vals[sample_idx]
                fv_single  = shap_sample[sample_idx]
                shap_row = pd.DataFrame({
                    "Feature": [f"{n_}={v:.2f}" for n_, v in zip(FEATURE_COLS, fv_single)],
                    "SHAP":    sv_single,
                }).sort_values("SHAP", key=abs, ascending=False).head(15)

                colors_wf = [AMBER if x > 0 else RED for x in shap_row["SHAP"]]
                fig_wf = go.Figure(go.Bar(
                    x=shap_row["SHAP"], y=shap_row["Feature"],
                    orientation="h", marker_color=colors_wf,
                    text=[f"{v:+.2f}" for v in shap_row["SHAP"]],
                    textposition="outside", textfont=dict(size=9),
                ))
                fig_wf.add_vline(x=0, line_dash="dash", line_color=MUTED)
                base = float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0
                apply_layout(fig_wf,
                             title=f"SHAP Waterfall — sample {sample_idx} (base ≈ {base:.0f} MWh)",
                             xaxis_title="SHAP Value", height=500)
                fig_wf.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_wf, use_container_width=True)

                section_header("SHAP Correlation Heatmap")
                top8_idx   = np.argsort(mean_shap)[-8:]
                top8_names = [FEATURE_COLS[i] for i in top8_idx]
                shap_top8  = shap_vals[:, top8_idx]
                corr_mat   = np.corrcoef(shap_top8.T)
                fig_hm = go.Figure(go.Heatmap(
                    z=corr_mat, x=top8_names, y=top8_names,
                    colorscale="RdBu", zmid=0,
                    text=[[f"{v:.2f}" for v in row] for row in corr_mat],
                    texttemplate="%{text}", textfont=dict(size=9),
                ))
                apply_layout(fig_hm, title="SHAP Value Correlation (Top 8 Features)", height=450)
                st.plotly_chart(fig_hm, use_container_width=True)

            except ImportError:
                st.error("SHAP not installed. Run: `pip install shap`")
            except Exception as e:
                st.error(f"SHAP failed: {e}")
        else:
            st.info("Click **🧮 Compute SHAP Values** to run SHAP analysis on XGBoost.")

    # ── TAB 3: Seasonality Decomposition ──────────────────────────────────────
    with tab_decomp:
        section_header("Time Series Decomposition")
        n_days_decomp = st.slider("Last N days for decomposition", 30, 180, 60, 10)
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            sample_dec = df_raw.set_index("timestamp")["consumption_mwh"].tail(24 * n_days_decomp)
            result_dec = seasonal_decompose(sample_dec, model="additive",
                                             period=24, extrapolate_trend="freq")
            components = {
                "Observed":      (result_dec.observed,  "#4a7fba"),
                "Trend":         (result_dec.trend,     "#e8a04d"),
                "Seasonal (24h)":(result_dec.seasonal,  "#3dd6c6"),
                "Residual":      (result_dec.resid,     "#a78bfa"),
            }
            for title, (comp, color) in components.items():
                fig_c = go.Figure(go.Scatter(
                    x=comp.index, y=comp.values,
                    mode="lines", line=dict(color=color, width=0.9),
                    fill="tozeroy", fillcolor=hex_to_rgba(color, 0.07), name=title,
                ))
                apply_layout(fig_c, title=title, yaxis_title="MWh", height=220)
                st.plotly_chart(fig_c, use_container_width=True)
        except Exception as e:
            st.error(f"Decomposition failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 5: ANOMALY DETECTION
# ──────────────────────────────────────────────────────────────────────────────
def page_anomaly_detection(ctx):
    df_raw = ctx["df_raw"]
    st.markdown("# 🔍 Anomaly Detection")

    col_a, col_b = st.columns(2)
    with col_a: Z_THRESH = st.slider("Z-Score Threshold (σ)", 1.5, 5.0, 3.0, 0.1)
    with col_b: WINDOW   = st.slider("Rolling Window (hours)", 24, 336, 168, 24)

    df_anom  = df_raw.copy()
    roll_mean = df_anom["consumption_mwh"].rolling(window=WINDOW, center=True).mean()
    roll_std  = df_anom["consumption_mwh"].rolling(window=WINDOW, center=True).std()
    df_anom["z_score"] = (df_anom["consumption_mwh"] - roll_mean) / (roll_std + 1e-8)
    df_anom["anomaly"] = df_anom["z_score"].abs() > Z_THRESH
    n_anom = int(df_anom["anomaly"].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Anomalies Detected", f"{n_anom:,}", f"{n_anom/len(df_anom)*100:.2f}% of data"), unsafe_allow_html=True)
    with c2: st.markdown(metric_card("Z-Score Threshold", f"±{Z_THRESH:.1f}σ", ""), unsafe_allow_html=True)
    with c3: st.markdown(metric_card("Rolling Window", f"{WINDOW}h", f"{WINDOW//24} days"), unsafe_allow_html=True)
    with c4:
        max_z = df_anom["z_score"].abs().max()
        st.markdown(metric_card("Max |Z-Score|", f"{max_z:.2f}", "σ"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section_header("Anomaly Detection — Time Series")
    sample  = st.slider("Downsample for rendering", 1, 8, 2, key="anom_sample")
    normal  = df_anom[~df_anom["anomaly"]].iloc[::sample]
    anomaly = df_anom[ df_anom["anomaly"]]

    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(
        x=normal.timestamp, y=normal.consumption_mwh,
        mode="lines", line=dict(color="#4a7fba", width=0.7), name="Normal",
    ))
    fig_a.add_trace(go.Scatter(
        x=anomaly.timestamp, y=anomaly.consumption_mwh,
        mode="markers", marker=dict(color=RED, size=5, symbol="x"),
        name=f"Anomaly (n={n_anom})",
    ))
    apply_layout(fig_a, title="Energy Consumption with Detected Anomalies",
                 xaxis_title="Date", yaxis_title="MWh")
    st.plotly_chart(fig_a, use_container_width=True)

    section_header("Rolling Z-Score")
    df_plot_z = df_anom.iloc[::sample]
    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=df_plot_z.timestamp, y=df_plot_z.z_score,
        mode="lines", line=dict(color=TEAL, width=0.7), name="Z-Score"))
    anom_z = df_anom[df_anom.anomaly].iloc[::max(1, sample//2)]
    fig_z.add_trace(go.Scatter(
        x=anom_z.timestamp, y=anom_z.z_score,
        mode="markers", marker=dict(color=RED, size=5), name="Anomaly"))
    fig_z.add_hline(y= Z_THRESH, line_dash="dash", line_color=RED)
    fig_z.add_hline(y=-Z_THRESH, line_dash="dash", line_color=RED)
    fig_z.add_hrect(y0=-Z_THRESH, y1=Z_THRESH,
                    fillcolor="rgba(61,214,198,0.04)", line_width=0)
    apply_layout(fig_z, title="Rolling Z-Score Over Time",
                 xaxis_title="Date", yaxis_title="Z-Score")
    st.plotly_chart(fig_z, use_container_width=True)

    section_header("Top Anomalous Records")
    top_anom = (df_anom[df_anom.anomaly][["timestamp","consumption_mwh","z_score"]]
                .sort_values("z_score", key=abs, ascending=False)
                .head(30).reset_index(drop=True))
    top_anom["z_score"]       = top_anom["z_score"].round(3)
    top_anom["consumption_mwh"] = top_anom["consumption_mwh"].round(1)
    st.dataframe(
        top_anom.style.background_gradient(subset=["z_score"], cmap="RdYlGn_r"),
        use_container_width=True, height=400,
    )

    section_header("Anomaly Distribution by Month / Hour")
    df_anom_only = df_anom[df_anom.anomaly].copy()
    df_anom_only["month"] = df_anom_only.timestamp.dt.month_name()
    df_anom_only["hour"]  = df_anom_only.timestamp.dt.hour
    col_m, col_h = st.columns(2)
    with col_m:
        month_cnt = df_anom_only.groupby("month").size().reset_index(name="count")
        fig_mc = px.bar(month_cnt, x="month", y="count", color="count",
                        color_continuous_scale="Reds", title="Anomalies per Month")
        apply_layout(fig_mc, coloraxis_showscale=False)
        st.plotly_chart(fig_mc, use_container_width=True)
    with col_h:
        hour_cnt = df_anom_only.groupby("hour").size().reset_index(name="count")
        fig_hc = px.bar(hour_cnt, x="hour", y="count", color="count",
                        color_continuous_scale="Reds", title="Anomalies per Hour of Day")
        apply_layout(fig_hc, coloraxis_showscale=False)
        st.plotly_chart(fig_hc, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 6: MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
def page_model_comparison(ctx):
    X_train = ctx["X_train"]; y_train = ctx["y_train"]
    X_val   = ctx["X_val"];   y_val   = ctx["y_val"]
    X_test  = ctx["X_test"];  y_test  = ctx["y_test"]
    HORIZON = ctx["HORIZON"]

    st.markdown("# 📈 Model Comparison")

    @st.cache_resource(show_spinner="Training XGBoost…")
    def train_xgb2(_X_train, _y_train, _X_val, _y_val):
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=30,
            eval_metric="rmse", random_state=42, verbosity=0, n_jobs=-1)
        model.fit(_X_train, _y_train, eval_set=[(_X_val, _y_val)], verbose=False)
        return model

    @st.cache_resource(show_spinner="Training Random Forest…")
    def train_rf2(_X_train, _y_train):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, max_depth=10,
                                      min_samples_split=5, random_state=42, n_jobs=-1).fit(_X_train, _y_train)

    results = {}
    with st.spinner("Running models…"):
        try:
            xgb   = train_xgb2(X_train, y_train, X_val, y_val)
            xgb_p = xgb.predict(X_test[:HORIZON])
            results["XGBoost"] = {"preds": xgb_p, "metrics": evaluate(y_test[:HORIZON], xgb_p), "color": AMBER}
        except Exception as e:
            st.warning(f"XGBoost: {e}")
        try:
            rf   = train_rf2(X_train, y_train)
            rf_p = rf.predict(X_test[:HORIZON])
            results["Random Forest"] = {"preds": rf_p, "metrics": evaluate(y_test[:HORIZON], rf_p), "color": TEAL}
        except Exception as e:
            st.warning(f"Random Forest: {e}")

    if not results:
        st.error("No models ran successfully.")
        st.stop()

    section_header("Metrics Table")
    rows = []
    for name, v in results.items():
        row = {"Model": name}
        row.update({k: round(val, 4) for k, val in v["metrics"].items()})
        rows.append(row)
    mdf = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        mdf.style
            .highlight_min(subset=["RMSE","MAE","MAPE"], color="#1a3a2a")
            .highlight_max(subset=["R²"], color="#1a3a2a")
            .highlight_max(subset=["RMSE","MAE","MAPE"], color="#3a1a1a")
            .format({"RMSE":"{:.2f}","MAE":"{:.2f}","MAPE":"{:.2f}%","R²":"{:.4f}"}),
        use_container_width=True,
    )

    section_header("Metric Comparison")
    metrics_to_plot = ["RMSE", "MAE", "MAPE"]
    fig_bar  = make_subplots(rows=1, cols=3, subplot_titles=metrics_to_plot)
    colors_bar = [AMBER, TEAL, RED, "#a78bfa"]
    for i, metric in enumerate(metrics_to_plot):
        vals = [(name, v["metrics"][metric]) for name, v in results.items()]
        vals.sort(key=lambda x: x[1])
        fig_bar.add_trace(go.Bar(
            y=[v[0] for v in vals], x=[v[1] for v in vals],
            orientation="h", marker_color=colors_bar[i % len(colors_bar)], name=metric,
        ), row=1, col=i+1)
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=300, title_text="Model Comparison by Metric")
    st.plotly_chart(fig_bar, use_container_width=True)

    section_header("Radar / Spider Chart")
    radar_metrics = ["RMSE", "MAE", "MAPE"]
    fig_radar = go.Figure()
    max_vals  = {m: max(v["metrics"][m] for v in results.values()) + 1e-6 for m in radar_metrics}
    for (name, v), color in zip(results.items(), colors_bar):
        norm = [1 - v["metrics"][m] / max_vals[m] for m in radar_metrics]
        norm.append(norm[0])
        cats = radar_metrics + [radar_metrics[0]]
        fill_color = hex_to_rgba(color, 0.12) if color.startswith("#") else color.replace(")", ",0.12)").replace("rgb(", "rgba(")
        fig_radar.add_trace(go.Scatterpolar(r=norm, theta=cats, name=name,
                                              line_color=color, fill="toself",
                                              fillcolor=fill_color))
    apply_layout(fig_radar, title="Normalized Performance (higher = better)",
                 polar=dict(bgcolor="#161b22",
                            radialaxis=dict(visible=True, gridcolor="#30363d"),
                            angularaxis=dict(gridcolor="#30363d")))
    st.plotly_chart(fig_radar, use_container_width=True)

    section_header("Prediction Error Distribution")
    fig_err = go.Figure()
    for (name, v), color in zip(results.items(), colors_bar):
        errors = y_test[:HORIZON] - v["preds"]
        fig_err.add_trace(go.Violin(
            y=errors, name=name, box_visible=True,
            meanline_visible=True, fillcolor=hex_to_rgba(color, 0.27),
            line_color=color,
        ))
    fig_err.add_hline(y=0, line_dash="dash", line_color=RED)
    apply_layout(fig_err, title="Error Distribution per Model", yaxis_title="Error (MWh)")
    st.plotly_chart(fig_err, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 7: SYNTHETIC DATA
# ──────────────────────────────────────────────────────────────────────────────
def page_synthetic_data(ctx):
    df_raw = ctx["df_raw"]
    st.markdown("# 🧬 Synthetic Data Generator")
    st.markdown("Generate realistic synthetic energy consumption data using Fourier reconstruction.")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1: n_samples     = st.slider("Number of synthetic hours", 720, 17520, 8760, 720)
    with col_s2: noise_factor  = st.slider("Noise factor", 0.01, 0.5, 0.05, 0.01)
    with col_s3: K_components  = st.slider("Fourier components (K)", 50, 500, 200, 50)

    @st.cache_data(show_spinner="Generating synthetic data…")
    def generate_fourier_synthetic(_df, n_samples, noise_factor, K):
        series = _df["consumption_mwh"].values.astype(float)
        n = len(series)
        fft_vals  = np.fft.rfft(series)
        magnitude = np.abs(fft_vals)
        K = min(K, len(fft_vals))
        top_k     = np.argpartition(magnitude, -K)[-K:]
        filtered  = np.zeros_like(fft_vals)
        filtered[top_k] = fft_vals[top_k]
        reconstructed = np.fft.irfft(filtered, n=n)
        scale = series.std() / (reconstructed.std() + 1e-8)
        reconstructed = reconstructed * scale + (series.mean() - reconstructed.mean() * scale)
        noise_std = (series - reconstructed).std() * (1 + noise_factor)
        t_orig = np.linspace(0, 1, n)
        t_new  = np.linspace(0, 1, n_samples)
        synthetic = np.interp(t_new, t_orig, reconstructed)
        np.random.seed(42)
        synthetic += np.random.normal(0, noise_std, n_samples)
        synthetic  = np.clip(synthetic, series.min() * 0.5, series.max() * 1.5)
        start      = _df["timestamp"].max() + pd.Timedelta(hours=1)
        timestamps = pd.date_range(start=start, periods=n_samples, freq="h")
        return pd.DataFrame({"timestamp": timestamps, "consumption_mwh": synthetic})

    if st.button("🧬 Generate Synthetic Data", use_container_width=True):
        synthetic_df = generate_fourier_synthetic(df_raw, n_samples, noise_factor, K_components)

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(metric_card("Generated Rows", f"{len(synthetic_df):,}", f"{n_samples//24:.0f} days"), unsafe_allow_html=True)
        with c2: st.markdown(metric_card("Synth Mean", f"{synthetic_df.consumption_mwh.mean():.0f}", "MWh"), unsafe_allow_html=True)
        with c3: st.markdown(metric_card("Orig Mean",  f"{df_raw.consumption_mwh.mean():.0f}", "MWh"), unsafe_allow_html=True)
        with c4:
            diff = abs(synthetic_df.consumption_mwh.mean() - df_raw.consumption_mwh.mean())
            st.markdown(metric_card("Mean Diff", f"{diff:.1f}", "MWh"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        section_header("Time Series Comparison")
        col_t1, col_t2 = st.columns(2)
        sample_orig = df_raw.tail(min(8760, len(df_raw)))
        with col_t1:
            fig_o = go.Figure()
            fig_o.add_trace(go.Scatter(x=sample_orig.timestamp.iloc[::4],
                                        y=sample_orig.consumption_mwh.iloc[::4],
                                        mode="lines", line=dict(color="#4a7fba", width=0.8),
                                        fill="tozeroy", fillcolor="rgba(74,127,186,0.08)"))
            apply_layout(fig_o, title="Original Data (last year)", yaxis_title="MWh")
            st.plotly_chart(fig_o, use_container_width=True)
        with col_t2:
            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=synthetic_df.timestamp.iloc[::4],
                                        y=synthetic_df.consumption_mwh.iloc[::4],
                                        mode="lines", line=dict(color=RED, width=0.8),
                                        fill="tozeroy", fillcolor="rgba(240,80,96,0.08)"))
            apply_layout(fig_s, title="Synthetic Data (Fourier)", yaxis_title="MWh")
            st.plotly_chart(fig_s, use_container_width=True)

        section_header("Distribution Comparison")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=sample_orig.consumption_mwh, nbinsx=60,
                                             name="Original", marker_color="#4a7fba",
                                             opacity=0.7, histnorm="probability density"))
            fig_dist.add_trace(go.Histogram(x=synthetic_df.consumption_mwh, nbinsx=60,
                                             name="Synthetic", marker_color=RED,
                                             opacity=0.6, histnorm="probability density"))
            fig_dist.update_layout(barmode="overlay")
            apply_layout(fig_dist, title="Distribution Comparison",
                         xaxis_title="MWh", yaxis_title="Density")
            st.plotly_chart(fig_dist, use_container_width=True)
        with col_d2:
            orig_profile  = sample_orig.copy()
            orig_profile["hour"]  = orig_profile.timestamp.dt.hour
            synth_profile = synthetic_df.copy()
            synth_profile["hour"] = synth_profile.timestamp.dt.hour
            oh = orig_profile.groupby("hour")["consumption_mwh"].mean()
            sh = synth_profile.groupby("hour")["consumption_mwh"].mean()
            fig_hp = go.Figure()
            fig_hp.add_trace(go.Scatter(x=oh.index, y=oh.values, mode="lines+markers",
                                         marker_size=5, line_color="#4a7fba", name="Original"))
            fig_hp.add_trace(go.Scatter(x=sh.index, y=sh.values, mode="lines+markers",
                                         marker_symbol="square", marker_size=5,
                                         line_color=RED, line_dash="dash", name="Synthetic"))
            apply_layout(fig_hp, title="Hourly Profile Comparison",
                         xaxis_title="Hour of Day", yaxis_title="Avg MWh")
            st.plotly_chart(fig_hp, use_container_width=True)

        section_header("Statistical Summary")
        stats = pd.DataFrame({
            "Original": df_raw["consumption_mwh"].describe(),
            "Synthetic": synthetic_df["consumption_mwh"].describe(),
        }).round(2)
        st.dataframe(stats, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = synthetic_df[["timestamp", "consumption_mwh"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Synthetic Data CSV",
            data=csv_bytes,
            file_name="synthetic_energy_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Configure parameters above and click **Generate Synthetic Data** to begin.")


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 8: AI CHATBOT
# ──────────────────────────────────────────────────────────────────────────────
def _local_answer(question: str, df) -> str:
    """Rule-based fallback answers when no Groq API key is set."""
    q     = question.lower()
    stats = df["consumption_mwh"].describe()

    if any(w in q for w in ["best model", "which model", "top model", "winner", "most accurate"]):
        return (
            "**XGBoost** is the best-performing model on this dataset: "
            "RMSE=93.7 MWh, MAPE=0.58%, R²=0.988. It outperforms Random Forest (RMSE=132 MWh), "
            "LSTM (RMSE≈690 MWh), ARIMA (RMSE≈4,624 MWh), and Prophet (RMSE≈5,078 MWh). "
            "The key advantage is its ability to exploit the 30 lag/rolling features that "
            "encode the strong 1-hour, 24-hour, and 168-hour autocorrelation in electricity data."
        )
    if any(w in q for w in ["seasonal", "pattern", "trend", "monthly", "hourly", "winter", "summer"]):
        return (
            "This dataset (2016–2021, European grid) shows three overlapping seasonalities:\n\n"
            "**Daily:** Peak at 16:00–17:00 (~10,020 MWh avg), trough at 04:00 (~7,800 MWh).\n\n"
            "**Weekly:** Weekdays avg 9,671 MWh vs weekends 9,055 MWh (6.2% lower).\n\n"
            "**Annual:** Jan/Feb are highest (~11,300–11,500 MWh); Jun/Jul lowest (~7,800–7,900 MWh).\n\n"
            "**Trend:** Flat 2016–2019, then 2020 drops to 8,910 MWh avg (COVID-19), recovering to 9,670 in 2021."
        )
    if any(w in q for w in ["shap", "explainab", "interpret", "feature importance"]):
        return (
            "SHAP assigns each feature a contribution value for every prediction. "
            "For this dataset, **lag_1** (previous hour) dominates. **lag_24** and **lag_168** capture "
            "daily and weekly repeat patterns. "
            "Go to the 🔬 Explainability page → SHAP Analysis tab to compute live SHAP values."
        )
    if any(w in q for w in ["lag", "feature", "roll", "engineering"]):
        return (
            "30 features are engineered: **Lag** (lag_1…lag_168), **Rolling stats** (mean/std over 6–168h), "
            "**Cyclical encodings** (sin/cos of hour, day-of-week, month), "
            "and **Calendar** fields (hour, day_of_week, month, quarter, is_weekend)."
        )
    if any(w in q for w in ["lstm", "deep learning", "neural", "pytorch"]):
        return (
            "The LSTM is a 2-layer PyTorch network. Current benchmark: RMSE≈690 MWh (vs XGBoost 93.7 MWh). "
            "To improve: increase hidden units to 256, train 50+ epochs, add dropout=0.3, use seq_len=336."
        )
    if any(w in q for w in ["arima", "prophet", "statistical"]):
        return (
            "**ARIMA(2,1,2):** RMSE≈4,624 MWh. Good for short horizons, degrades beyond 24h.\n\n"
            "**Prophet:** RMSE≈5,078 MWh. Better for trend analysis and long-range planning."
        )
    if any(w in q for w in ["mean", "average", "typical", "statistics", "stats"]):
        return (
            f"Mean: **{stats['mean']:.0f} MWh** | Std: **{stats['std']:.0f} MWh** | "
            f"Min: **{stats['min']:.0f}** | Max: **{stats['max']:.0f}** MWh"
        )
    if any(w in q for w in ["covid", "2020", "lockdown", "drop"]):
        return (
            "2020 shows a **6.4% decline** to 8,910 MWh avg (vs 9,523 in 2019), "
            "consistent with COVID-19 lockdown effects. 2021 partially recovered to 9,670 MWh."
        )
    if any(w in q for w in ["synthetic", "generate", "fourier", "artificial"]):
        return (
            "Synthetic data uses **Fourier reconstruction**: FFT → keep top-K components → "
            "inverse FFT → scale to match original stats → add calibrated Gaussian noise."
        )
    if any(w in q for w in ["anomaly", "outlier", "spike", "unusual"]):
        return (
            "Anomaly detection uses **rolling Z-score** (168h window). "
            "Lower the threshold to **2σ** on the 🔍 Anomaly Detection page to surface subtler deviations."
        )
    if any(w in q for w in ["improve", "better", "accuracy", "optimize", "tune"]):
        return (
            "Tips: Tune XGBoost with Optuna; train LSTM 50+ epochs with hidden=256; "
            "add temperature / holiday features; ensemble XGBoost + RF."
        )
    return (
        f"Your dataset has **{len(df):,} hourly records** "
        f"({df.timestamp.min().strftime('%Y-%m-%d')} → {df.timestamp.max().strftime('%Y-%m-%d')}, "
        f"mean: {float(df['consumption_mwh'].mean()):.0f} MWh).\n\n"
        "Add a **Groq API key** in the sidebar for full AI responses. "
        "Free at [console.groq.com](https://console.groq.com)."
    )


def _build_context(df_raw) -> str:
    stats       = df_raw["consumption_mwh"].describe()
    df2         = df_raw.copy()
    df2["hour"]  = df2.timestamp.dt.hour
    df2["month"] = df2.timestamp.dt.month
    peak_hour   = int(df2.groupby("hour")["consumption_mwh"].mean().idxmax())
    low_month   = int(df2.groupby("month")["consumption_mwh"].mean().idxmin())
    high_month  = int(df2.groupby("month")["consumption_mwh"].mean().idxmax())
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return f"""
You are an expert energy data analyst. You have full context about this dataset:

DATASET OVERVIEW:
- Source: European electricity grid hourly consumption data
- Time range: {df_raw.timestamp.min().strftime('%Y-%m-%d')} to {df_raw.timestamp.max().strftime('%Y-%m-%d')}
- Total records: {len(df_raw):,} hourly readings (~6 years)
- Energy metric: Electricity consumption in MWh

STATISTICAL SUMMARY:
- Mean: {stats['mean']:.1f} MWh | Std: {stats['std']:.1f} MWh
- Min: {stats['min']:.1f} MWh  | Max: {stats['max']:.1f} MWh
- 25th pct: {stats['25%']:.1f} | 75th pct: {stats['75%']:.1f} | Median: {float(df_raw['consumption_mwh'].median()):.1f}

SEASONAL PATTERNS:
- Peak hour: {peak_hour}:00 | Trough: ~4:00
- Highest month: {month_names.get(high_month)} (winter heating) | Lowest: {month_names.get(low_month)}
- Weekdays avg 9,671 MWh | Weekends 9,055 MWh | 2020 avg 8,910 MWh (COVID dip)

MODELS:
1. XGBoost   — RMSE 93.7 MWh, MAPE 0.58%, R²=0.988 (BEST)
2. Random Forest — RMSE 132.2 MWh, MAPE 0.78%, R²=0.977
3. LSTM (PyTorch) — RMSE ~690 MWh, MAPE ~4.4%, R²=0.37
4. ARIMA(2,1,2) — RMSE ~4,624 MWh
5. Prophet — RMSE ~5,078 MWh

FEATURES (30 total): lag_1/2/3/6/12/24/48/168, roll_mean/std over 6/12/24/48/168h,
hour/dow/month sin-cos encodings, hour, day_of_week, month, quarter, is_weekend.

Answer clearly. Provide code snippets when asked. Use specific numbers above.
"""


def page_ai_chatbot(ctx):
    df_raw = ctx["df_raw"]
    st.markdown("# 💬 AI Energy Analyst")
    st.markdown("Ask questions about the dataset, models, and forecasts.")

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🤖 AI Settings")
        groq_key   = st.text_input("Groq API Key", type="password",
                                    help="Get free key at console.groq.com")
        chat_model = st.selectbox("Model", [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
        ])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        section_header("💡 Suggested Questions")
        suggestions = [
            "Which model performs best and why?",
            "What are the main seasonal patterns in the data?",
            "How do I interpret the SHAP values?",
            "What does the lag_1 feature represent?",
            "Why does XGBoost outperform LSTM on this dataset?",
            "How is the synthetic data generated?",
            "What is the average consumption during winter vs summer?",
            "How can I improve the LSTM model's accuracy?",
        ]
        cols_sugg = st.columns(2)
        for i, sug in enumerate(suggestions):
            with cols_sugg[i % 2]:
                if st.button(sug, key=f"sugg_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": sug})
                    st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask anything about the energy data, models, or forecasts…")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if not groq_key:
                response = _local_answer(user_input, df_raw)
                st.markdown(response)
                st.caption("_⚠️ Add Groq API key in sidebar for full AI responses_")
            else:
                try:
                    import requests
                    messages_payload = [{"role": "system", "content": _build_context(df_raw)}]
                    for h in st.session_state.chat_history[-8:]:
                        messages_payload.append({"role": h["role"], "content": h["content"]})
                    with st.spinner("Thinking…"):
                        resp = requests.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {groq_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": chat_model,
                                "messages": messages_payload,
                                "max_tokens": 1024,
                                "temperature": 0.4,
                            },
                            timeout=30,
                        )
                        resp.raise_for_status()
                        response = resp.json()["choices"][0]["message"]["content"]
                    st.markdown(response)
                except Exception as e:
                    response = f"API error: {e}. Using local fallback.\n\n" + _local_answer(user_input, df_raw)
                    st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_ctrl2:
        if st.session_state.chat_history:
            chat_text = "\n\n".join(
                f"{'User' if m['role']=='user' else 'AI'}: {m['content']}"
                for m in st.session_state.chat_history
            )
            st.download_button("⬇️ Export Chat", chat_text.encode("utf-8"),
                                "energy_chat.txt", "text/plain",
                                use_container_width=True)
