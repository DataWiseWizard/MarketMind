"""
MarketMind Tone Dynamics & Financial Guidance Component
Visualizes speaker-separated FinBERT sentiment trajectory across the call timeline,
CEO vs. CFO divergence metrics, and extracted forward-looking guidance.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.styles import TOKENS, badge_html


def build_tone_trajectory_figure(segments: list[dict[str, Any]]) -> go.Figure:
    """Builds an interactive Plotly timeline chart of speaker sentiment."""
    if not segments:
        fig = go.Figure()
        fig.update_layout(title="No segment sentiment data available")
        return fig

    df = pd.DataFrame(segments)

    fig = go.Figure()

    # Distinct traces for CEO, CFO, Analyst
    for role, color, name in [
        ("CEO", TOKENS["bull_primary"], "CEO Remarks"),
        ("CFO", TOKENS["accent_primary"], "CFO Financials"),
        ("Analyst", TOKENS["warn_primary"], "Analyst Inquiries"),
    ]:
        role_df = df[df["role"] == role]
        if not role_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=role_df["segment_order"],
                    y=role_df["compound"],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2.5},
                    marker={"size": 7, "color": color},
                    text=[f"Topic: {t}<br>Compound: {c:+.2f}" for t, c in zip(role_df.get("topic", [""]), role_df["compound"])],
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        title={
            "text": "<b>FinBERT Sentiment Trajectory by Speaker Role (Call Timeline)</b>",
            "font": {"size": 13, "color": TOKENS["text_primary"], "family": "JetBrains Mono"},
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13, 18, 26, 0.6)",
        font={"color": TOKENS["text_secondary"], "family": "Inter", "size": 11},
        height=340,
        margin={"l": 40, "r": 20, "t": 45, "b": 40},
        xaxis={
            "showgrid": True,
            "gridcolor": TOKENS["border_subtle"],
            "title": {"text": "Segment Order (Prepared Remarks → Q&A)", "font": {"size": 10.5}},
            "tickfont": {"family": "JetBrains Mono"},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": TOKENS["border_subtle"],
            "zeroline": True,
            "zerolinecolor": TOKENS["border_strong"],
            "range": [-0.8, 1.0],
            "title": {"text": "Compound Sentiment Score (-1 to +1)", "font": {"size": 10.5}},
            "tickfont": {"family": "JetBrains Mono"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 10.5},
        },
    )
    return fig


def render_tone_dynamics(tone_data: dict[str, Any], metrics_data: list[dict[str, Any]]) -> None:
    """Renders the Tone Dynamics & Guidance Tracker tab."""
    st.markdown('<div class="mm-section-hdr">Speaker-Separated Tone Dynamics (FinBERT Pipeline)</div>', unsafe_allow_html=True)

    roles = tone_data.get("roles", {})
    ceo_mean = tone_data.get("ceo_mean", roles.get("ceo", roles.get("CEO", 0.65)))
    cfo_mean = tone_data.get("cfo_mean", roles.get("cfo", roles.get("CFO", 0.45)))
    analyst_mean = tone_data.get("analyst_mean", roles.get("analyst", roles.get("ANALYST", 0.0)))
    delta = tone_data.get("ceo_cfo_delta", round(ceo_mean - cfo_mean, 4))
    segments = tone_data.get("segments", [])

    # Overview Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="mm-card-elevated" style="text-align:center;">
              <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">CEO Mean Sentiment</div>
              <div class="mono" style="font-size:20px; font-weight:700; color:{TOKENS['bull_primary']};">{ceo_mean:+.2f}</div>
              <div style="font-size:10px; color:{TOKENS['text_muted']};">Visionary / Strategic</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="mm-card-elevated" style="text-align:center;">
              <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">CFO Mean Sentiment</div>
              <div class="mono" style="font-size:20px; font-weight:700; color:{TOKENS['accent_primary']};">{cfo_mean:+.2f}</div>
              <div style="font-size:10px; color:{TOKENS['text_muted']};">Margins / Cost Reality</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="mm-card-elevated" style="text-align:center;">
              <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">CEO vs CFO Divergence</div>
              <div class="mono" style="font-size:20px; font-weight:700; color:{TOKENS['warn_primary'] if delta > 0.25 else TOKENS['bull_primary']};">{delta:+.2f} Δ</div>
              <div style="font-size:10px; color:{TOKENS['text_muted']};">Spread > 0.25 flags hedge</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="mm-card-elevated" style="text-align:center;">
              <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">Analyst Inquiry Tone</div>
              <div class="mono" style="font-size:20px; font-weight:700; color:{TOKENS['text_primary']};">{analyst_mean:+.2f}</div>
              <div style="font-size:10px; color:{TOKENS['text_muted']};">Skeptical vs Supportive</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Plotly Trajectory Chart
    fig = build_tone_trajectory_figure(segments)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =========================================================================
    # FINANCIAL METRIC & FORWARD GUIDANCE EXTRACTOR
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="mm-section-hdr">Extracted Financial Guidance & Reported Metrics (Regex + NER Engine)</div>', unsafe_allow_html=True)

    if not metrics_data:
        st.info("No structured financial metrics extracted for this call.")
        return

    m_rows = []
    for m in metrics_data:
        is_guidance = m.get("is_guidance", False)
        badge = badge_html("FORWARD GUIDANCE", "warn") if is_guidance else badge_html("REPORTED", "info")
        val = m.get("value", 0.0)
        unit = m.get("unit", "")
        conf = m.get("confidence", 0.90)

        m_rows.append({
            "Metric": m.get("metric_name", ""),
            "Value": f"{val} {unit}".strip(),
            "Period": m.get("period", ""),
            "Type": "Forward Guidance" if is_guidance else "Reported Result",
            "Confidence": f"{conf*100:.0f}%",
            "Verbatim Quote": m.get("quote", ""),
        })

    df_metrics = pd.DataFrame(m_rows)
    st.dataframe(
        df_metrics[["Metric", "Value", "Period", "Type", "Confidence"]],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("🔎 View Verbatim Excerpts for Extracted Financial Guidance"):
        for m in metrics_data:
            st.markdown(
                f"""
                <div class="mm-quote-box" style="margin-bottom:8px;">
                  <b>{m.get('metric_name')}:</b> "{m.get('quote')}"
                  <span style="font-size:10.5px; color:{TOKENS['text_muted']}; margin-left:8px;">(Confidence: {m.get('confidence', 0.9)*100:.0f}%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
