"""
MarketMind MLOps & Model Health Component
Renders model tracking, Population Stability Index (PSI), Kolmogorov-Smirnov
drift metrics, prediction distribution stability, and MLflow registry telemetry.
"""

from __future__ import annotations
from typing import Any
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from dashboard.styles import TOKENS, badge_html, card_container


def render_mlops_monitor(drift_data: dict[str, Any], api_client: Any = None):
    """Render the MLOps, Model Monitoring & Experiment Tracking tab."""
    # Use session state drift report if retrained or updated
    if "active_drift_data" in st.session_state and st.session_state["active_drift_data"]:
        drift_data = st.session_state["active_drift_data"]

    st.markdown('<div class="mm-section-hdr">MLOps Telemetry & Distribution Drift Monitor</div>', unsafe_allow_html=True)

    if not drift_data:
        st.info("No drift monitoring data available.")
        return

    overall_status = drift_data.get("overall_status", "UNKNOWN")
    crit_count = drift_data.get("critical_features_count", 0)
    warn_count = drift_data.get("warning_features_count", 0)
    total_feat = drift_data.get("total_features_evaluated", 13)
    pred_drift = drift_data.get("prediction_drift", {})
    pred_psi = pred_drift.get("psi", 0.0)
    pred_level = pred_drift.get("drift_level", "STABLE")
    ref_split = drift_data.get("reference_split", "2023-2024")
    cur_split = drift_data.get("current_split", "2025-2026")
    timestamp = drift_data.get("timestamp", "")[:19].replace("T", " ")

    # 1. Top Telemetry Row
    col1, col2, col3, col4 = st.columns(4)

    status_badge_type = "bull" if overall_status == "HEALTHY" else ("warn" if overall_status == "WARNING" else "bear")
    pred_badge_type = "bull" if pred_level == "STABLE" else ("warn" if pred_level == "MODERATE" else "bear")

    with col1:
        st.markdown(
            card_container(
                f"""
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:{TOKENS['text_secondary']}; margin-bottom:4px;">
                    Model Health Status
                </div>
                <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                    {badge_html(overall_status, status_badge_type)}
                </div>
                <div style="font-size:11px; color:{TOKENS['text_muted']};">
                    Ref: {ref_split}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            card_container(
                f"""
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:{TOKENS['text_secondary']}; margin-bottom:4px;">
                    Drifted Features
                </div>
                <div class="mono" style="font-size:22px; font-weight:800; color:{TOKENS['bear'] if crit_count > 0 else TOKENS['warn']}; margin:2px 0;">
                    {crit_count + warn_count} <span style="font-size:13px; color:{TOKENS['text_muted']};">/ {total_feat}</span>
                </div>
                <div style="font-size:11px; color:{TOKENS['text_secondary']};">
                    {crit_count} Critical • {warn_count} Moderate
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            card_container(
                f"""
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:{TOKENS['text_secondary']}; margin-bottom:4px;">
                    Prediction Drift PSI
                </div>
                <div class="mono" style="font-size:22px; font-weight:800; color:{TOKENS['text_primary']}; margin:2px 0;">
                    {pred_psi:.4f}
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    {badge_html(f"Shift: {pred_level}", pred_badge_type)}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            card_container(
                f"""
                <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:{TOKENS['text_secondary']}; margin-bottom:4px;">
                    MLflow Registry
                </div>
                <div class="mono" style="font-size:13px; font-weight:700; color:{TOKENS['accent_primary']}; margin:6px 0 2px 0;">
                    sqlite:///mlflow.db
                </div>
                <div style="font-size:11px; color:{TOKENS['text_muted']};">
                    Experiment: <b>marketmind</b>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # 2. Main Visualizations Grid: Feature PSI Chart & Prediction Shift Chart
    feat_col, pred_col = st.columns([1.6, 1.0])

    features_dict = drift_data.get("feature_metrics", {})
    sorted_feats = sorted(features_dict.items(), key=lambda x: x[1]["psi"], reverse=True)

    with feat_col:
        st.markdown('<div class="mm-card-title">Population Stability Index (PSI) by Feature</div>', unsafe_allow_html=True)

        feat_names = [item[0] for item in sorted_feats]
        psi_vals = [item[1]["psi"] for item in sorted_feats]
        colors = [
            TOKENS["bear"] if v >= 0.25 else (TOKENS["warn"] if v >= 0.10 else TOKENS["bull"])
            for v in psi_vals
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=psi_vals,
            y=feat_names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="#1E293B", width=1),
            ),
            hovertemplate="<b>%{y}</b><br>PSI: %{x:.4f}<extra></extra>",
        ))

        # Add vertical benchmark lines for PSI thresholds
        fig.add_vline(x=0.10, line_dash="dash", line_color=TOKENS["warn"], annotation_text="Moderate (0.10)", annotation_position="top right")
        fig.add_vline(x=0.25, line_dash="dash", line_color=TOKENS["bear"], annotation_text="Critical (0.25)", annotation_position="top right")

        fig.update_layout(
            paper_bgcolor="#0D121A",
            plot_bgcolor="#0D121A",
            margin=dict(l=10, r=20, t=25, b=10),
            height=340,
            xaxis=dict(
                title=dict(text="Population Stability Index (PSI)", font=dict(color="#94A3B8", size=11)),
                showgrid=True,
                gridcolor="#1E293B",
                tickfont=dict(color="#94A3B8", size=10, family="monospace"),
            ),
            yaxis=dict(
                autorange="reversed",
                showgrid=False,
                tickfont=dict(color="#F1F5F9", size=11, family="monospace"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with pred_col:
        st.markdown('<div class="mm-card-title">Model Prediction Distribution Shift</div>', unsafe_allow_html=True)

        ref_dist = pred_drift.get("reference_distribution", {"NEGATIVE": 0.33, "NEUTRAL": 0.33, "POSITIVE": 0.34})
        cur_dist = pred_drift.get("current_distribution", {"NEGATIVE": 0.33, "NEUTRAL": 0.33, "POSITIVE": 0.34})

        classes = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        ref_shares = [ref_dist.get(c, 0.0) * 100 for c in classes]
        cur_shares = [cur_dist.get(c, 0.0) * 100 for c in classes]

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Bar(
            name=f"Baseline ({ref_split})",
            x=classes,
            y=ref_shares,
            marker_color="#64748B",
            hovertemplate="%{x}<br>Baseline: %{y:.1f}%<extra></extra>",
        ))
        fig_pred.add_trace(go.Bar(
            name=f"Serving ({cur_split})",
            x=classes,
            y=cur_shares,
            marker_color=TOKENS["accent_primary"],
            hovertemplate="%{x}<br>Serving: %{y:.1f}%<extra></extra>",
        ))

        fig_pred.update_layout(
            barmode="group",
            paper_bgcolor="#0D121A",
            plot_bgcolor="#0D121A",
            margin=dict(l=10, r=10, t=25, b=10),
            height=340,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#94A3B8", size=10),
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color="#F1F5F9", size=11, weight="bold"),
            ),
            yaxis=dict(
                title=dict(text="Class Share (%)", font=dict(color="#94A3B8", size=11)),
                showgrid=True,
                gridcolor="#1E293B",
                tickfont=dict(color="#94A3B8", size=10, family="monospace"),
            ),
        )
        st.plotly_chart(fig_pred, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    # 3. Detailed Forensic Drift Table
    st.markdown('<div class="mm-card-title">Comprehensive Feature Distribution & Statistical Shifts</div>', unsafe_allow_html=True)

    table_rows = []
    for feat, m in sorted_feats:
        level = m["drift_level"]
        badge_type = "bear" if level == "CRITICAL" else ("warn" if level == "MODERATE" else "bull")
        table_rows.append({
            "Feature": feat,
            "Drift Status": level,
            "PSI": f"{m['psi']:.4f}",
            "KS Stat": f"{m['ks_statistic']:.4f}",
            "p-value": f"{m['ks_pvalue']:.4f}",
            "Ref Mean": f"{m['ref_mean']:.3f}",
            "Cur Mean": f"{m['cur_mean']:.3f}",
            "Δ Mean": f"{m['delta_mean']:+.3f}",
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
    )

    # 4. Automated Recommendations & Recalculate Trigger
    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)
    rec_col, act_col = st.columns([2.0, 1.0])

    with rec_col:
        st.markdown('<div class="mm-card-title">Automated Forensic Recommendations</div>', unsafe_allow_html=True)
        recs = drift_data.get("recommendations", [])
        for r in recs:
            st.markdown(
                f"""
                <div style="background:#0D121A; border-left:3px solid {TOKENS['accent_primary']}; border-top:1px solid #1E293B; border-right:1px solid #1E293B; border-bottom:1px solid #1E293B; border-radius:4px; padding:10px 14px; margin-bottom:8px; font-size:12.5px; color:{TOKENS['text_secondary']};">
                    <b>Action Required:</b> {r}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with act_col:
        st.markdown('<div class="mm-card-title">Experiment Controls</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size:12px; color:{TOKENS['text_secondary']}; margin-bottom:12px;">
                Audit Timestamp: <span class="mono" style="color:{TOKENS['text_primary']};">{timestamp} UTC</span><br>
                Baseline: <span class="mono" style="color:{TOKENS['accent_primary']};">{ref_split}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("⚡ Retrain Walk-Forward Model (2023–2025)", key="btn_retrain_walk_forward", help="Incorporate 2025 calls to adapt model to higher macro volatility regime"):
            with st.spinner("Retraining LightGBM on 2023–2025 walk-forward split & logging to MLflow..."):
                res = None
                if api_client and hasattr(api_client, "retrain_model"):
                    res = api_client.retrain_model(train_years=[2023, 2024, 2025])
                else:
                    try:
                        from src.models.trainer import run_training
                        from src.models.monitoring import DriftMonitor
                        run_training(train_years=[2023, 2024, 2025], val_years=[2024], test_years=[2026], run_name="walk_forward_expanding_2023_2025")
                        monitor = DriftMonitor()
                        report = monitor.run_drift_analysis(ref_years=[2023, 2024, 2025], cur_years=[2026])
                        monitor.save_reports(report)
                        monitor.log_to_mlflow(report)
                        res = {"status": "success", "drift_report": report}
                    except (ImportError, ModuleNotFoundError):
                        from dashboard.utils.mock_data import get_mock_retrained_drift_data
                        res = {"status": "demo_simulated", "drift_report": get_mock_retrained_drift_data()}

                if res and "drift_report" in res:
                    st.session_state["active_drift_data"] = res["drift_report"]

                if res and res.get("status") == "demo_simulated":
                    st.info("⚡ Simulated walk-forward retraining completed for demo mode! (To run physical MLflow retraining in the cloud, ensure your live Render backend URL is configured in Streamlit secrets)")
                else:
                    st.success("Model successfully retrained and logged to MLflow!")
                st.rerun()

        if st.button("🔄 Recalculate Drift Metrics", key="btn_recalc_drift"):
            with st.spinner("Executing Kolmogorov-Smirnov & PSI tests across features..."):
                if api_client and hasattr(api_client, "recalculate_drift"):
                    res = api_client.recalculate_drift()
                    if res:
                        st.session_state["active_drift_data"] = res
                st.success("Drift metrics recalculated!")
                st.rerun()

        if "2025" in str(ref_split):
            if st.button("↩️ Reset to Baseline Window (2023–2024)", key="btn_reset_baseline"):
                with st.spinner("Resetting model to initial 2023–2024 baseline..."):
                    res = None
                    if api_client and hasattr(api_client, "retrain_model"):
                        res = api_client.retrain_model(train_years=[2023, 2024])
                    else:
                        from dashboard.utils.mock_data import get_mock_drift_data
                        res = {"status": "demo_simulated", "drift_report": get_mock_drift_data()}

                    if res and "drift_report" in res:
                        st.session_state["active_drift_data"] = res["drift_report"]
                    st.success("Reset back to 2023–2024 baseline!")
                    st.rerun()
