"""
MarketMind Hero KPI Cards
Renders 4 top institutional metrics: Signal, Confidence, Executive Tone, and Evasion Rate.
"""

from __future__ import annotations
from typing import Any
import streamlit as st

from dashboard.styles import kpi_card_html, badge_html, TOKENS


def render_hero_kpis(signal_data: dict[str, Any], tone_data: dict[str, Any], evasion_data: list[dict[str, Any]]) -> None:
    """Renders the top row of 4 KPI cards."""
    pred_signal = signal_data.get("predicted_signal", "NEUTRAL").upper()
    conf = float(signal_data.get("confidence") or 0.70)
    exp_return = float(signal_data.get("expected_return") or 0.0)

    # Tone calculations
    roles = tone_data.get("roles", {})
    ceo_mean = tone_data.get("ceo_mean", roles.get("ceo", roles.get("CEO", 0.65)))
    cfo_mean = tone_data.get("cfo_mean", roles.get("cfo", roles.get("CFO", 0.45)))
    tone_delta = tone_data.get("ceo_cfo_delta", round(ceo_mean - cfo_mean, 4))

    # Evasion stats
    total_q = len(evasion_data) if evasion_data else 1
    evasive_cnt = sum(1 for e in evasion_data if e.get("is_evasive", False))
    evasion_pct = (evasive_cnt / total_q) * 100

    # Determine badge tones
    if pred_signal == "POSITIVE":
        sig_tone = "bull"
        sig_label = f"BULLISH (+{exp_return:.2f}%)"
    elif pred_signal == "NEGATIVE":
        sig_tone = "bear"
        sig_label = f"BEARISH ({exp_return:.2f}%)"
    else:
        sig_tone = "warn"
        sig_label = f"NEUTRAL ({exp_return:+.2f}%)"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            kpi_card_html(
                label="Predicted Market Signal",
                value=sig_label,
                sub="LightGBM 3-Class Classifier • Return vs Benchmark",
                tone=sig_tone,
            ),
            unsafe_allow_html=True,
        )

    with col2:
        conf_tone = "bull" if conf >= 0.75 else ("warn" if conf >= 0.60 else "bear")
        st.markdown(
            kpi_card_html(
                label="Model Confidence",
                value=f"{conf * 100:.1f}%",
                sub=f"Prob: Pos {signal_data.get('probabilities', {}).get('POSITIVE', 0):.2f} | Neg {signal_data.get('probabilities', {}).get('NEGATIVE', 0):.2f}",
                tone=conf_tone,
            ),
            unsafe_allow_html=True,
        )

    with col3:
        tone_style = "bull" if tone_delta > 0.15 else ("warn" if tone_delta >= -0.10 else "bear")
        st.markdown(
            kpi_card_html(
                label="Executive Tone Divergence",
                value=f"{tone_delta:+.2f} Δ",
                sub=f"CEO Score: {ceo_mean:.2f} vs CFO Score: {cfo_mean:.2f}",
                tone=tone_style,
            ),
            unsafe_allow_html=True,
        )

    with col4:
        ev_tone = "bear" if evasion_pct >= 40 else ("warn" if evasion_pct >= 20 else "bull")
        st.markdown(
            kpi_card_html(
                label="Q&A Evasion Index",
                value=f"{evasion_pct:.0f}%",
                sub=f"{evasive_cnt} of {total_q} analyst questions deflected",
                tone=ev_tone,
            ),
            unsafe_allow_html=True,
        )
