"""
MarketMind Streamlit Web Dashboard

Interactive visual dashboard for earnings call analysis:
- Prediction Signal & Confidence Summary
- Top Feature Drivers (SHAP Explainability)
- Financial Metrics & Forward Guidance
- Speaker Sentiment Breakdown (CEO, CFO, Analysts)
- Q&A Evasion Detection
- Interactive Report Viewer & Download
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from loguru import logger

from src.db.connection import get_session, init_database
from src.db.models import (
    Company, EarningsCall, TranscriptSegment, ExtractedMetric,
    SentimentScore, EvasionScore, Signal,
    PredictedDirection, SpeakerRole
)
from src.pipeline import MarketMindPipeline
from src.reporting.report_generator import ReportGenerator
from src.config import RAW_DATA_DIR, DISCLAIMER

# Page Configuration
st.set_page_config(
    page_title="MarketMind | Financial Intelligence Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
    }
    .badge-positive {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    init_database()
    return MarketMindPipeline(), ReportGenerator()

pipeline, report_gen = get_pipeline()

# Title
st.markdown('<div class="main-title">🧠 MarketMind Financial Intelligence Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Automated Financial Earnings Call Analysis & Market Signal Generation Pipeline</div>', unsafe_allow_html=True)

# Academic Disclaimer
st.info(f"⚠️ **Notice:** {DISCLAIMER}")

# Sidebar
st.sidebar.header("📁 Call Selection & Ingestion")

session = get_session()
try:
    calls = session.query(EarningsCall).order_by(EarningsCall.call_date.desc()).all()
    if not calls:
        st.warning("No earnings calls found in the database. Please run the pipeline first.")
        st.stop()

    call_options = {
        f"NVDA {c.quarter} FY{c.fiscal_year} ({c.call_date})": c.call_id
        for c in calls
    }

    selected_label = st.sidebar.selectbox("Select Earnings Call:", list(call_options.keys()))
    selected_call_id = call_options[selected_label]

    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload New Transcript")
    uploaded_file = st.sidebar.file_uploader("Upload PDF Transcript", type=["pdf"])
    if uploaded_file is not None:
        if st.sidebar.button("Run MarketMind Pipeline", type="primary"):
            with st.spinner("Processing PDF through all 6 pipeline stages..."):
                save_dir = RAW_DATA_DIR / "transcripts" / "uploaded"
                save_dir.mkdir(parents=True, exist_ok=True)
                target_path = save_dir / uploaded_file.name
                target_path.write_bytes(uploaded_file.getbuffer())

                res = pipeline.process_pdf(target_path)
                st.sidebar.success(f"✅ Processed {res['ticker']} {res['quarter']} FY{res['fiscal_year']}!")
                st.rerun()

    # Query Data for Selected Call
    call = session.query(EarningsCall).filter_by(call_id=selected_call_id).first()
    company = session.query(Company).filter_by(company_id=call.company_id).first()
    signal = session.query(Signal).filter_by(call_id=selected_call_id).first()
    metrics = session.query(ExtractedMetric).filter_by(call_id=selected_call_id).all()
    sentiments = session.query(SentimentScore).filter_by(call_id=selected_call_id).all()
    evasions = session.query(EvasionScore).filter_by(call_id=selected_call_id).all()

finally:
    session.close()

# Top Summary Banner
col1, col2, col3, col4 = st.columns([1.5, 1, 1, 2.5])

with col1:
    st.markdown("### Research Signal")
    if signal:
        sig_val = signal.predicted_direction.value.upper()
        if sig_val == "POSITIVE":
            st.markdown(f'<div class="badge-positive">🟢 {sig_val}</div>', unsafe_allow_html=True)
        elif sig_val == "NEGATIVE":
            st.markdown(f'<div class="badge-negative">🔴 {sig_val}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="badge-neutral">🟡 {sig_val}</div>', unsafe_allow_html=True)
    else:
        st.write("Not generated")

with col2:
    st.markdown("### Confidence")
    conf = signal.confidence if signal else 0.50
    st.metric(label="Model Confidence", value=f"{conf*100:.0f}%")

with col3:
    st.markdown("### Expected Bias")
    exp_ret = signal.expected_return if signal else 0.0
    st.metric(label="Short-Term Bias", value=f"{exp_ret:+.1f}%")

with col4:
    st.markdown("### 🔍 Top Feature Drivers (SHAP)")
    if signal and signal.top_features:
        for driver in signal.top_features:
            st.markdown(f"• **{driver}**")
    else:
        st.write("Standard market variables within normal boundaries.")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Financial Metrics & Guidance",
    "🎭 Sentiment Breakdown",
    "🕵️ Evasion Detection (Q&A)",
    "📑 Full Research Report"
])

# ---- TAB 1: Financial Metrics ----
with tab1:
    st.subheader(f"Extracted Financial Metrics — {company.company_name} ({call.quarter} FY{call.fiscal_year})")
    if metrics:
        metric_data = []
        for m in metrics:
            val_display = (
                f"${m.value:.2f} Billion" if m.unit == "USD_B"
                else (f"{m.value:.1f}%" if m.unit == "percent"
                      else (f"${m.value:.2f}" if m.unit == "USD" else f"{m.value}"))
            )
            metric_data.append({
                "Metric Name": m.metric_name,
                "Value": val_display,
                "Type": "🔮 Forward Guidance" if m.is_guidance else "📊 Reported Historical",
                "Speaker": m.speaker_role.value.upper() if m.speaker_role else "UNKNOWN",
                "Confidence": f"{m.confidence*100:.0f}%",
                "Exact Quote": m.quote,
            })
        df_metrics = pd.DataFrame(metric_data)
        st.dataframe(df_metrics[["Metric Name", "Value", "Type", "Speaker", "Confidence"]], use_container_width=True)

        with st.expander("🔍 View Verifiable Source Quotes"):
            for idx, row in df_metrics.iterrows():
                st.markdown(f"**{row['Metric Name']} ({row['Value']}):**")
                st.caption(f'"{row["Exact Quote"]}"')
    else:
        st.info("No structured financial metrics extracted for this call.")

# ---- TAB 2: Sentiment Analysis ----
with tab2:
    st.subheader("Tone & Sentiment by Speaker Role")
    if sentiments:
        # Role Breakdown
        role_scores = {}
        for s in sentiments:
            role = s.speaker_role.value.upper() if s.speaker_role else "UNKNOWN"
            role_scores.setdefault(role, []).append(s.compound_score)

        avg_roles = {r: sum(vals)/len(vals) for r, vals in role_scores.items()}
        df_roles = pd.DataFrame({
            "Speaker Role": list(avg_roles.keys()),
            "Average Compound Score": [round(v, 3) for v in avg_roles.values()]
        }).sort_values(by="Average Compound Score", ascending=False)

        col_chart, col_stats = st.columns([2, 1])
        with col_chart:
            st.bar_chart(df_roles.set_index("Speaker Role"), use_container_width=True)

        with col_stats:
            st.markdown("#### Summary Insights")
            overall_comp = sum(s.compound_score for s in sentiments) / len(sentiments)
            st.metric("Overall Call Sentiment", f"{overall_comp:+.3f}")
            st.write(f"Total scored text segments: **{len(sentiments)}**")

            if "CEO" in avg_roles:
                st.write(f"• **CEO Tone:** `{avg_roles['CEO']:+.3f}`")
            if "CFO" in avg_roles:
                st.write(f"• **CFO Tone:** `{avg_roles['CFO']:+.3f}`")
            if "ANALYST" in avg_roles:
                st.write(f"• **Analyst Tone:** `{avg_roles['ANALYST']:+.3f}`")
    else:
        st.info("No sentiment scores recorded for this call.")

# ---- TAB 3: Evasion Detection ----
with tab3:
    st.subheader("Executive Evasion & Vagueness Detection in Q&A")
    if evasions:
        avg_evasion = sum(e.evasion_score for e in evasions) / len(evasions)
        evasive_count = sum(1 for e in evasions if e.is_evasive)

        ev_c1, ev_c2, ev_c3 = st.columns(3)
        ev_c1.metric("Average Evasion Index", f"{avg_evasion:.3f}", help="0.00 = Completely Direct, 1.00 = Highly Evasive")
        ev_c2.metric("Total Q&A Pairs", len(evasions))
        ev_c3.metric("Flagged Evasive Answers", evasive_count)

        st.markdown("### Q&A Exchanges & Explanations")
        session = get_session()
        try:
            for i, e in enumerate(evasions, 1):
                q = session.query(TranscriptSegment).filter_by(segment_id=e.question_segment_id).first()
                a = session.query(TranscriptSegment).filter_by(segment_id=e.answer_segment_id).first()

                with st.container():
                    status_badge = "🔴 Flagged as Evasive" if e.is_evasive else "🟢 Direct Answer"
                    st.markdown(f"**Exchange {i}: {status_badge} (Score: `{e.evasion_score:.2f}`)**")
                    st.markdown(f"**Q ({q.speaker_label if q else 'Analyst'}):** {q.text if q else 'N/A'}")
                    st.markdown(f"**A ({a.speaker_label if a else 'Executive'}):** {a.text if a else 'N/A'}")
                    st.caption(f"💡 **Analysis:** {e.reason}")
                    st.markdown("---")
        finally:
            session.close()
    else:
        st.info("No Q&A evasion records for this call.")

# ---- TAB 4: Full Report ----
with tab4:
    st.subheader("Downloadable Intelligence Report")
    session = get_session()
    try:
        report_md, report_path = report_gen.generate_report_markdown(selected_call_id, session=session)
        st.download_button(
            label="📥 Download Full Report (Markdown)",
            data=report_md,
            file_name=report_path.name,
            mime="text/markdown",
            type="primary"
        )
        st.markdown(report_md)
    finally:
        session.close()
