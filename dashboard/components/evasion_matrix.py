"""
MarketMind Q&A Evasion Matrix & Forensic Audit Component
Interactive viewer for question-answer evasion detection, deflection taxonomy,
and adversarial audit discrepancies with SEC 10-Q filings.
"""

from __future__ import annotations
from typing import Any
import pandas as pd
import streamlit as st

from dashboard.styles import TOKENS, badge_html


def render_evasion_matrix(evasion_data: list[dict[str, Any]], audit_data: dict[str, Any]) -> None:
    """Renders the Evasion & Forensic Audit tab."""
    st.markdown('<div class="mm-section-hdr">Q&A Evasion Matrix & Deflection Taxonomy</div>', unsafe_allow_html=True)

    if not evasion_data:
        st.info("No Q&A evasion records found for this call.")
        return

    # Filter & Search Controls
    filter_col1, filter_col2, filter_col3 = st.columns([1.5, 1.2, 1.3])
    with filter_col1:
        search_kw = st.text_input("🔍 Search Q&A Excerpts", placeholder="Filter by keyword (e.g. margin, Blackwell, sovereign)...")
    with filter_col2:
        evasion_filter = st.selectbox("Evasion Level", ["All Questions", "Evasive Only (Score ≥ 0.50)", "Direct Only (< 0.50)"])
    with filter_col3:
        sort_by = st.selectbox("Sort By", ["Highest Evasion First", "Natural Call Order"])

    # Apply filters
    filtered = list(evasion_data)
    if evasion_filter == "Evasive Only (Score ≥ 0.50)":
        filtered = [e for e in filtered if e.get("evasion_score", 0.0) >= 0.50]
    elif evasion_filter == "Direct Only (< 0.50)":
        filtered = [e for e in filtered if e.get("evasion_score", 0.0) < 0.50]

    if search_kw.strip():
        kw = search_kw.lower()
        filtered = [
            e for e in filtered
            if kw in e.get("question_text", "").lower()
            or kw in e.get("answer_text", "").lower()
            or kw in e.get("question_speaker", "").lower()
        ]

    if sort_by == "Highest Evasion First":
        filtered = sorted(filtered, key=lambda x: x.get("evasion_score", 0.0), reverse=True)

    # Summary metric bar
    total_q = len(evasion_data)
    evasive_cnt = sum(1 for e in evasion_data if e.get("is_evasive", False))
    avg_score = sum(e.get("evasion_score", 0.0) for e in evasion_data) / max(total_q, 1)

    st.markdown(
        f"""
        <div class="mm-card-elevated" style="display:flex; justify-content:space-around; align-items:center; text-align:center;">
          <div>
            <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">Total Questions Analyzed</div>
            <div class="mono" style="font-size:18px; font-weight:700; color:{TOKENS['text_primary']};">{total_q}</div>
          </div>
          <div style="border-left:1px solid {TOKENS['border_subtle']}; height:30px;"></div>
          <div>
            <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">Deflected / Evasive</div>
            <div class="mono" style="font-size:18px; font-weight:700; color:{TOKENS['bear_primary']};">{evasive_cnt} ({evasive_cnt/max(total_q,1)*100:.0f}%)</div>
          </div>
          <div style="border-left:1px solid {TOKENS['border_subtle']}; height:30px;"></div>
          <div>
            <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['text_secondary']};">Mean Evasion Score</div>
            <div class="mono" style="font-size:18px; font-weight:700; color:{TOKENS['warn_primary']};">{avg_score:.2f} / 1.00</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Q&A Cards List
    st.markdown(f'<div style="font-size:12px; font-weight:600; color:{TOKENS["text_secondary"]}; margin:14px 0 8px 0;">Showing {len(filtered)} Exchange(s):</div>', unsafe_allow_html=True)

    for item in filtered:
        q_speaker = item.get("question_speaker", "Analyst")
        q_text = item.get("question_text", "")
        a_speaker = item.get("answer_speaker", "Executive")
        a_text = item.get("answer_text", "")
        score = float(item.get("evasion_score") or 0.0)
        is_evasive = item.get("is_evasive", False)
        category = item.get("deflection_category", "General Inquiry")
        reason = item.get("reason", "")

        tone = "bear" if is_evasive else "bull"
        status_text = f"EVASIVE ({score:.2f})" if is_evasive else f"DIRECT ({score:.2f})"
        quote_style = "mm-quote-evasion" if is_evasive else ""

        with st.container():
            st.markdown(
                f"""
                <div class="mm-card" style="margin-bottom:12px;">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
                    <div>
                      <span style="font-weight:700; font-size:13px; color:{TOKENS['accent_primary']};">{q_speaker}</span>
                    </div>
                    <div style="display:flex; gap:6px;">
                      {badge_html(category, 'ai')}
                      {badge_html(status_text, tone)}
                    </div>
                  </div>
                  <div style="font-size:12.5px; color:{TOKENS['text_secondary']}; margin-bottom:10px;">
                    <b>Q:</b> <i>"{q_text}"</i>
                  </div>
                  <div class="mm-quote-box {quote_style}">
                    <div style="font-weight:700; font-size:12px; color:{TOKENS['text_primary']}; margin-bottom:4px;">
                      <b>A ({a_speaker}):</b>
                    </div>
                    "{a_text}"
                  </div>
                  {f'<div style="font-size:11.5px; color:{TOKENS["warn_primary"]}; margin-top:8px;"><b>Forensic Rationale:</b> {reason}</div>' if reason else ''}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # FORENSIC AUDIT DISCREPANCY FINDINGS
    # =========================================================================
    st.markdown("---")
    st.markdown('<div class="mm-section-hdr">Adversarial Forensic Audit (Discrepancies vs SEC 10-Q Filings)</div>', unsafe_allow_html=True)

    findings = audit_data.get("findings", [])
    if not findings:
        st.success("No critical contradictions or accountability gaps detected between call and SEC filings.")
        return

    for f in findings:
        ftype = f.get("finding_type", f.get("type", "Discrepancy"))
        sev = f.get("severity", "MEDIUM").upper()
        curr_stmt = f.get("current_statement", f.get("title", ""))
        hist_stmt = f.get("historical_statement", f.get("evidence", ""))
        hist_src = f.get("historical_source", "SEC 10-Q Filing")
        explanation = f.get("explanation", f.get("description", ""))

        sev_tone = "bear" if sev in ["HIGH", "CRITICAL"] else ("warn" if sev in ["MEDIUM", "WARNING"] else "info")

        st.markdown(
            f"""
            <div class="mm-card" style="border-left: 4px solid {TOKENS['bear_primary'] if sev=='HIGH' else TOKENS['warn_primary']};">
              <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-weight:700; font-size:13px; color:{TOKENS['text_primary']};">{ftype}</span>
                {badge_html(f"{sev} SEVERITY", sev_tone)}
              </div>
              <div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:10px;">
                <div style="background:{TOKENS['bg_surface_2']}; padding:10px; border-radius:6px; font-size:12px;">
                  <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['accent_primary']}; font-weight:600; margin-bottom:4px;">
                    Claim on Earnings Call
                  </div>
                  "{curr_stmt}"
                </div>
                <div style="background:{TOKENS['bg_surface_2']}; padding:10px; border-radius:6px; font-size:12px;">
                  <div style="font-size:11px; text-transform:uppercase; color:{TOKENS['warn_primary']}; font-weight:600; margin-bottom:4px;">
                    Contradiction in {hist_src}
                  </div>
                  "{hist_stmt}"
                </div>
              </div>
              <div style="font-size:12px; color:{TOKENS['text_secondary']}; line-height:1.45;">
                <b style="color:{TOKENS['text_primary']};">Forensic Analysis:</b> {explanation}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    recommendation = audit_data.get("recommendation")
    if recommendation:
        st.markdown(
            f"""
            <div class="mm-card-elevated" style="border-left: 3px solid {TOKENS['accent_secondary']};">
              <div style="font-weight:700; font-size:12px; color:{TOKENS['accent_secondary']}; margin-bottom:4px;">
                FORENSIC AUDIT RECOMMENDATION
              </div>
              <div style="font-size:12.5px; color:{TOKENS['text_primary']};">
                {recommendation}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
