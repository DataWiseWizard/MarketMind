"""
MarketMind | Financial Earnings Call Analysis & Market Signal Terminal
Interactive Streamlit application integrating FinBERT tone dynamics,
Q&A evasion detection, LightGBM return prediction with SHAP explainability,
and grounded Gemini RAG forensic auditing.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.styles import (
    TOKENS,
    inject_design_system_css,
    badge_html,
)
from dashboard.api_client import get_client
from dashboard.components.hero_kpis import render_hero_kpis
from dashboard.components.signal_view import render_signal_view
from dashboard.components.evasion_matrix import render_evasion_matrix
from dashboard.components.tone_dynamics import render_tone_dynamics
from dashboard.components.forensic_chat import render_forensic_chat
from dashboard.components.peer_comparison import render_peer_comparison
from dashboard.components.mlops_monitor import render_mlops_monitor

SEMICONDUCTOR_TICKERS = ["NVDA", "AMD", "INTC", "TSM", "ASML", "MRVL", "QCOM"]


def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="MarketMind | Semiconductor Earnings & Signal Terminal",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 2. Inject Fintech Dark Theme CSS
    inject_design_system_css()

    # 3. Initialize API Client
    client = get_client()

    # =========================================================================
    # SIDEBAR CONTROLS
    # =========================================================================
    with st.sidebar:
        # Brand Header
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
              <div style="width:32px; height:32px; border-radius:8px; background:{TOKENS['accent_primary']}; display:flex; align-items:center; justify-content:center; flex-shrink:0;">
                <span style="color:#070A0F; font-size:18px; font-weight:900;">M</span>
              </div>
              <div>
                <div class="mm-brand-title">MarketMind</div>
                <div class="mm-brand-sub">Forensic Signal Terminal</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # API Connection Health Badge & Refresh Button
        if client.is_live:
            st.markdown(badge_html("🟢 API: LIVE (HTTP://LOCALHOST:8000)", "bull"), unsafe_allow_html=True)
        else:
            st.markdown(badge_html("🟡 MODE: LOCAL DEMO FALLBACK", "warn"), unsafe_allow_html=True)

        if st.button("🔄 Reconnect Backend", key="btn_reconnect_sidebar", use_container_width=True, help="Click to test live FastAPI server connection"):
            if client.check_health():
                st.success("Connected to live FastAPI backend!")
            else:
                st.warning("Backend not reachable at http://localhost:8000. Start it with: uvicorn src.api.main:app")
            st.rerun()

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="mm-section-hdr">Company Selection</div>', unsafe_allow_html=True)

        # Ticker Filter
        selected_ticker = st.selectbox(
            "Select Ticker",
            SEMICONDUCTOR_TICKERS,
            index=0,
            help="Focus on one of the 7 tracked semiconductor leaders.",
        )

        # Earnings Call Selector
        calls = client.get_calls(ticker=selected_ticker)
        if not calls:
            # Fallback to any call
            calls = client.get_calls()

        call_options = {
            f"{c.get('ticker')} {c.get('quarter')} FY{c.get('fiscal_year')} ({c.get('call_date', '')})": c["call_id"]
            for c in calls
        }

        selected_call_label = st.selectbox(
            "Select Earnings Call",
            list(call_options.keys()),
            index=0,
        )
        selected_call_id = call_options[selected_call_label]

        # Selected Call Metadata
        curr_call = next((c for c in calls if c.get("call_id") == selected_call_id), calls[0] if calls else {})

        st.markdown("---")
        st.markdown('<div class="mm-section-hdr">Pipeline Architecture</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="font-size:11.5px; line-height:1.7; color:{TOKENS['text_secondary']};">
              <div style="display:flex; justify-content:space-between;">
                <span>Total Calls Analyzed</span>
                <span class="mono" style="color:{TOKENS['text_primary']}; font-weight:700;">95 Calls</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span>RAG Vector Chunks</span>
                <span class="mono" style="color:{TOKENS['text_primary']}; font-weight:700;">65,301 Chunks</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span>Feature Vector</span>
                <span class="mono" style="color:{TOKENS['text_primary']}; font-weight:700;">13 Features</span>
              </div>
              <div style="display:flex; justify-content:space-between;">
                <span>Benchmark</span>
                <span class="mono" style="color:{TOKENS['accent_primary']}; font-weight:700;">VanEck SMH</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:10px; color:{TOKENS['text_muted']}; line-height:1.4;">
              <b>Disclaimer:</b> For research and educational purposes only. Not financial advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # MAIN WORKSPACE
    # =========================================================================

    # Fetch Data for selected call
    signal_data = client.get_signal(selected_call_id)
    evasion_data = client.get_evasion(selected_call_id)
    tone_data = client.get_tone(selected_call_id)
    audit_data = client.get_audit(selected_call_id)
    metrics_data = client.get_metrics(selected_call_id)

    # Main Header
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:14px;">
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <h2 style="margin:0; font-weight:800; font-size:22px; color:{TOKENS['text_primary']};">
                {curr_call.get('company_name', selected_ticker)} ({selected_ticker})
              </h2>
              {badge_html(f"{curr_call.get('quarter')} FY{curr_call.get('fiscal_year')}", "info")}
            </div>
            <div style="font-size:12px; color:{TOKENS['text_secondary']}; margin-top:2px;">
              Earnings Call Date: <b>{curr_call.get('call_date')}</b> • Sector: <b>Semiconductors</b> • Target Horizon: <b>T+30 Days Alpha</b>
            </div>
          </div>
          <div>
            {badge_html(f"STATUS: {curr_call.get('status', 'COMPLETED').upper()}", "bull")}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Top Hero KPI Cards Row
    render_hero_kpis(signal_data=signal_data, tone_data=tone_data, evasion_data=evasion_data)

    st.markdown('<div style="height:14px;"></div>', unsafe_allow_html=True)

    # 2. Main Tab Views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Signal & SHAP Waterfall",
        "🔍 Evasion & Forensic Audit",
        "📈 Tone Dynamics & Guidance",
        "🤖 AI Forensic Analyst Chat",
        "🌐 Peer Benchmark",
        "🛡️ MLOps & Drift Monitor",
    ])

    with tab1:
        render_signal_view(signal_data=signal_data)

    with tab2:
        render_evasion_matrix(evasion_data=evasion_data, audit_data=audit_data)

    with tab3:
        render_tone_dynamics(tone_data=tone_data, metrics_data=metrics_data)

    with tab4:
        render_forensic_chat(call_id=selected_call_id, api_client=client)

    with tab5:
        all_calls = client.get_calls()
        render_peer_comparison(all_calls=all_calls)

    with tab6:
        drift_data = client.get_drift_report()
        render_mlops_monitor(drift_data=drift_data, api_client=client)


if __name__ == "__main__":
    main()
