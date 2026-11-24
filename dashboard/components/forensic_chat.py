"""
MarketMind Forensic AI Analyst Chat Component
Interactive RAG assistant grounded in earnings call transcripts and SEC 10-Q/10-K filings
with verifiable citations and expandable source excerpts.
"""

from __future__ import annotations
from typing import Any
import streamlit as st

from dashboard.styles import TOKENS, badge_html


PRESET_PROMPTS = [
    "Did management provide hard quantification on Blackwell production yields and margin drag?",
    "What were the primary gross margin headwinds disclosed by the CFO?",
    "Check hyperscaler customer concentration against 10-Q risk factor disclosures.",
    "Did executives dodge questions on sovereign AI or datacenter power bottlenecks?",
]


def _format_citation_html(c: dict[str, Any]) -> str:
    """Safely normalizes both backend sources and mock citations."""
    meta = c.get("metadata", {})
    c_source = (c.get("source") or meta.get("doc_type", "TRANSCRIPT")).upper()
    c_label = c.get("citation") or f"[{meta.get('ticker', '')} {meta.get('quarter', '')} {c_source}]"
    c_speaker = c.get("speaker") or meta.get("speaker_role", "Verified Source")
    c_text = c.get("text", "")
    return f"""
    <div class="mm-quote-box" style="margin-bottom:6px;">
      <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
        <span class="mono" style="color:{TOKENS['accent_secondary']}; font-weight:700;">{c_label}</span>
        <span style="font-size:10.5px; color:{TOKENS['accent_primary']}; font-weight:600;">{c_source} • {c_speaker}</span>
      </div>
      <div style="font-size:12px; color:{TOKENS['text_secondary']};">
        "{c_text}"
      </div>
    </div>
    """


def render_forensic_chat(call_id: int, api_client: Any) -> None:
    """Renders the AI Forensic Analyst RAG Chat interface."""
    st.markdown('<div class="mm-section-hdr">Grounded Forensic AI Analyst (RAG: Gemini 3.5 Flash + ChromaDB)</div>', unsafe_allow_html=True)
    st.caption("Ask adversarial forensic questions grounded in full transcript segments and SEC Form 10-Q/10-K filings with inline verifiable citations.")

    session_key = f"chat_messages_{call_id}"
    if session_key not in st.session_state:
        # Default welcoming message
        st.session_state[session_key] = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to the **MarketMind Forensic Audit Agent**. I have indexed all transcript segments "
                    "and relevant SEC 10-Q/10-K filings in ChromaDB. Ask me anything about executive disclosures, "
                    "accounting divergence, forward-looking guidance, or evasion patterns."
                ),
                "citations": [],
            }
        ]

    # Preset Prompt Chips
    st.markdown('<div style="font-size:11px; text-transform:uppercase; color:' + TOKENS['text_secondary'] + '; margin-bottom:6px;">Suggested Forensic Inquiries:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(PRESET_PROMPTS))
    selected_prompt = None

    for i, p in enumerate(PRESET_PROMPTS):
        with chip_cols[i]:
            if st.button(p, key=f"chip_{call_id}_{i}"):
                selected_prompt = p

    # Render Chat History
    for msg in st.session_state[session_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            citations = msg.get("citations", []) or msg.get("sources", [])
            if citations:
                with st.expander(f"📚 View {len(citations)} Verified Grounding Citations", expanded=False):
                    for c in citations:
                        st.markdown(_format_citation_html(c), unsafe_allow_html=True)

    # Chat Input Box
    user_input = st.chat_input("Enter forensic inquiry (e.g. check margin trajectory against prior guidance)...")
    prompt_to_send = selected_prompt or user_input

    if prompt_to_send:
        # Append User Message
        st.session_state[session_key].append({"role": "user", "content": prompt_to_send, "citations": []})
        with st.chat_message("user"):
            st.markdown(prompt_to_send)

        # Call RAG Agent
        with st.chat_message("assistant"):
            with st.spinner("Retrieving semantic chunks and running adversarial forensic audit..."):
                response_data = api_client.chat(call_id=call_id, question=prompt_to_send, use_rag=True)
                ans_text = response_data.get("answer", "Unable to generate forensic audit response.")
                citations = response_data.get("citations", []) or response_data.get("sources", [])

                st.markdown(ans_text)
                if citations:
                    with st.expander(f"📚 View {len(citations)} Verified Grounding Citations", expanded=True):
                        for c in citations:
                            st.markdown(_format_citation_html(c), unsafe_allow_html=True)

            st.session_state[session_key].append({
                "role": "assistant",
                "content": ans_text,
                "citations": citations,
            })
            st.rerun()
