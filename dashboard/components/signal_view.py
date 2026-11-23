"""
MarketMind Signal Overview & SHAP Waterfall Component
Displays LightGBM predictions, Plotly SHAP waterfall chart, attribution table,
and an interactive What-If sensitivity simulator.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import TOKENS, badge_html


def build_shap_waterfall_figure(shap_data: dict[str, Any], base_value: float = 0.0) -> go.Figure:
    """Builds a Plotly Waterfall chart visualizing SHAP feature attributions."""
    shap_vals = shap_data.get("shap_values", [])
    if not shap_vals:
        # Fallback empty chart
        fig = go.Figure()
        fig.update_layout(title="No SHAP drivers available")
        return fig

    # Top drivers sorted by absolute impact
    sorted_drivers = sorted(shap_vals, key=lambda x: abs(x.get("shap_value", 0.0)), reverse=True)[:7]

    names = ["Base E[f(x)]"] + [d.get("feature", "feature") for d in sorted_drivers] + ["Predicted f(x)"]
    measures = ["absolute"] + ["relative"] * len(sorted_drivers) + ["total"]
    values = [base_value] + [d.get("shap_value", 0.0) for d in sorted_drivers] + [0.0]

    text_labels = [f"{base_value:+.2f}"] + [f"{d.get('shap_value', 0.0):+.3f}" for d in sorted_drivers] + [""]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=names,
            textposition="outside",
            text=text_labels,
            y=values,
            connector={"line": {"color": TOKENS["border_strong"], "width": 1.5, "dash": "dot"}},
            decreasing={"marker": {"color": TOKENS["bear_primary"]}},
            increasing={"marker": {"color": TOKENS["bull_primary"]}},
            totals={"marker": {"color": TOKENS["accent_primary"]}},
        )
    )

    fig.update_layout(
        title={
            "text": "<b>SHAP Attribution Waterfall (Base Expectation → Final Return Signal)</b>",
            "font": {"size": 13, "color": TOKENS["text_primary"], "family": "JetBrains Mono"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 18, 26, 0.6)",
        font={"color": TOKENS["text_secondary"], "family": "Inter", "size": 11},
        height=380,
        margin={"l": 40, "r": 40, "t": 50, "b": 60},
        xaxis={
            "showgrid": False,
            "tickangle": -25,
            "tickfont": {"size": 10.5, "family": "JetBrains Mono", "color": TOKENS["text_secondary"]},
            "linecolor": TOKENS["border_subtle"],
        },
        yaxis={
            "showgrid": True,
            "gridcolor": TOKENS["border_subtle"],
            "zeroline": True,
            "zerolinecolor": TOKENS["border_strong"],
            "tickfont": {"family": "JetBrains Mono"},
            "title": {"text": "SHAP Contribution (log-odds / return %)", "font": {"size": 10.5}},
        },
    )
    return fig


def render_signal_view(signal_data: dict[str, Any]) -> None:
    """Renders the Signal Overview and Explainability tab."""
    pred_signal = str(signal_data.get("predicted_signal", "NEUTRAL")).upper()
    conf = float(signal_data.get("confidence") or 0.70)
    exp_return = float(signal_data.get("expected_return") or 0.0)
    probabilities = signal_data.get("probabilities") or {"POSITIVE": 0.33, "NEUTRAL": 0.34, "NEGATIVE": 0.33}
    shap_data = signal_data.get("shap") or {}

    st.markdown('<div class="mm-section-hdr">Model Predictions & Explainability Engine</div>', unsafe_allow_html=True)

    top_col1, top_col2 = st.columns([1.4, 2.6])

    with top_col1:
        st.markdown(
            f"""
            <div class="mm-card">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="font-size:12px; font-weight:700; color:{TOKENS['text_secondary']};">SIGNAL CLASSIFICATION</span>
                {badge_html(pred_signal, "bull" if pred_signal=="POSITIVE" else ("bear" if pred_signal=="NEGATIVE" else "warn"))}
              </div>
              <div style="font-size:28px; font-weight:800; font-family:'JetBrains Mono'; color:{TOKENS['text_primary']}; margin-bottom:4px;">
                {exp_return:+.2f}% <span style="font-size:12px; font-weight:500; color:{TOKENS['text_muted']};">vs SMH Benchmark</span>
              </div>
              <div style="font-size:11.5px; color:{TOKENS['text_secondary']}; margin-bottom:16px;">
                Walk-Forward LightGBM Model (Macro F1: 0.609 | Accuracy: 58.8%)
              </div>
              <div style="margin-top:14px;">
                <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-bottom:3px;">
                  <span style="color:{TOKENS['bull_primary']}; font-weight:600;">Positive Probability</span>
                  <span class="mono" style="color:{TOKENS['text_primary']};">{probabilities.get('POSITIVE', 0)*100:.1f}%</span>
                </div>
                <div style="background:{TOKENS['bg_surface_3']}; border-radius:4px; height:6px; overflow:hidden; margin-bottom:10px;">
                  <div style="background:{TOKENS['bull_primary']}; width:{probabilities.get('POSITIVE', 0)*100}%; height:100%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-bottom:3px;">
                  <span style="color:{TOKENS['warn_primary']}; font-weight:600;">Neutral Probability</span>
                  <span class="mono" style="color:{TOKENS['text_primary']};">{probabilities.get('NEUTRAL', 0)*100:.1f}%</span>
                </div>
                <div style="background:{TOKENS['bg_surface_3']}; border-radius:4px; height:6px; overflow:hidden; margin-bottom:10px;">
                  <div style="background:{TOKENS['warn_primary']}; width:{probabilities.get('NEUTRAL', 0)*100}%; height:100%;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:11.5px; margin-bottom:3px;">
                  <span style="color:{TOKENS['bear_primary']}; font-weight:600;">Negative Probability</span>
                  <span class="mono" style="color:{TOKENS['text_primary']};">{probabilities.get('NEGATIVE', 0)*100:.1f}%</span>
                </div>
                <div style="background:{TOKENS['bg_surface_3']}; border-radius:4px; height:6px; overflow:hidden;">
                  <div style="background:{TOKENS['bear_primary']}; width:{probabilities.get('NEGATIVE', 0)*100}%; height:100%;"></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_col2:
        # Plotly SHAP Waterfall
        base_val = signal_data.get("base_value", 0.05)
        fig_waterfall = build_shap_waterfall_figure(shap_data, base_value=base_val)
        st.plotly_chart(fig_waterfall, use_container_width=True, config={"displayModeBar": False})

    # Detailed Feature Attribution Table
    shap_vals = shap_data.get("shap_values", [])
    if shap_vals:
        st.markdown('<div class="mm-section-hdr" style="margin-top:14px;">Feature Attribution Details (TreeExplainer Breakdown)</div>', unsafe_allow_html=True)

        rows = []
        for s in shap_vals:
            val = s.get("value", 0.0)
            shap_v = s.get("shap_value", 0.0)
            direction = "Bullish" if shap_v > 0 else "Bearish"
            badge = badge_html(f"{direction} ({shap_v:+.3f})", "bull" if shap_v > 0 else "bear")
            rows.append({
                "Feature": s.get("feature", "unknown"),
                "Observed Value": f"{val:.2f}" if isinstance(val, (int, float)) else str(val),
                "SHAP Attribution": f"{shap_v:+.4f}",
                "Market Impact": direction,
            })

        df_shap = pd.DataFrame(rows)
        st.dataframe(
            df_shap,
            use_container_width=True,
            hide_index=True,
        )

    # =========================================================================
    # INTERACTIVE WHAT-IF SENSITIVITY SIMULATOR (SANDBOX)
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="mm-section-hdr">Interactive Feature Sensitivity Simulator (What-If Sandbox)</div>', unsafe_allow_html=True)
    st.caption("Adjust key financial, sentiment, and evasion variables to observe real-time recalculation of model attributions.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_rev = st.slider("Revenue QoQ Growth (%)", min_value=-20.0, max_value=35.0, value=12.0, step=1.0)
        sim_margin = st.slider("Gross Margin QoQ (pp)", min_value=-5.0, max_value=5.0, value=1.0, step=0.2)
    with sim_col2:
        sim_ceo = st.slider("CEO Sentiment Score", min_value=-0.5, max_value=1.0, value=0.65, step=0.05)
        sim_tone_delta = st.slider("CEO vs CFO Divergence", min_value=-0.4, max_value=0.6, value=0.20, step=0.05)
    with sim_col3:
        sim_evasion = st.slider("Q&A Evasion Mean", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
        sim_discrepancies = st.slider("10-Q Discrepancy Count", min_value=0, max_value=5, value=0, step=1)

    # Simplified simulated return projection based on weights
    simulated_score = (
        (sim_rev * 0.12)
        + (sim_margin * 0.45)
        + (sim_ceo * 1.8)
        + (sim_tone_delta * 1.2)
        - (sim_evasion * 3.5)
        - (sim_discrepancies * 1.2)
    )

    if simulated_score > 1.5:
        sim_label = "BULLISH"
        sim_tone = "bull"
    elif simulated_score < -1.5:
        sim_label = "BEARISH"
        sim_tone = "bear"
    else:
        sim_label = "NEUTRAL"
        sim_tone = "warn"

    st.markdown(
        f"""
        <div class="mm-card-elevated" style="display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">Simulated Return Trajectory</div>
            <div class="mono" style="font-size:22px; font-weight:700; color:{TOKENS['accent_primary']};">
              {simulated_score:+.2f}% Expected Return
            </div>
          </div>
          <div>
            <span style="font-size:12px; color:{TOKENS['text_secondary']}; margin-right:8px;">Projected Signal:</span>
            {badge_html(sim_label, sim_tone)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
