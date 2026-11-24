"""
MarketMind Design System & Styling Tokens
Institutional Fintech Dark Terminal theme tokens, CSS injection, and HTML card/badge builders.
"""

from __future__ import annotations
import streamlit as st

# =============================================================================
# DESIGN SYSTEM TOKENS
# =============================================================================

TOKENS: dict[str, str] = {
    # Surfaces & Canvas
    "bg_canvas": "#070A0F",
    "bg_surface_1": "#0D121A",
    "bg_surface_2": "#131B26",
    "bg_surface_3": "#192332",
    # Borders
    "border_subtle": "#1E293B",
    "border_strong": "#2E3F56",
    # Typography
    "text_primary": "#F1F5F9",
    "text_secondary": "#94A3B8",
    "text_muted": "#64748B",
    # Semantic & Brand
    "accent_primary": "#38BDF8",     # Electric Sky Blue
    "accent_secondary": "#818CF8",   # Forensic Indigo
    "bull_primary": "#10B981",       # Emerald Green
    "bull_muted": "rgba(16, 185, 129, 0.14)",
    "bear_primary": "#F43F5E",       # Crimson Red
    "bear_muted": "rgba(244, 63, 94, 0.14)",
    "warn_primary": "#F59E0B",       # Amber Warning
    "warn_muted": "rgba(245, 158, 11, 0.14)",
    # Convenience Aliases
    "bull": "#10B981",
    "bear": "#F43F5E",
    "warn": "#F59E0B",
    "border": "#1E293B",
    "bg_card": "#0D121A",
}


def inject_design_system_css() -> None:
    """Injects high-contrast Institutional Fintech Dark Terminal CSS into Streamlit."""
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

      :root {{
        --bg-canvas: {TOKENS["bg_canvas"]};
        --bg-surface-1: {TOKENS["bg_surface_1"]};
        --bg-surface-2: {TOKENS["bg_surface_2"]};
        --bg-surface-3: {TOKENS["bg_surface_3"]};
        --border-subtle: {TOKENS["border_subtle"]};
        --border-strong: {TOKENS["border_strong"]};
        --text-primary: {TOKENS["text_primary"]};
        --text-secondary: {TOKENS["text_secondary"]};
        --text-muted: {TOKENS["text_muted"]};
        --accent-primary: {TOKENS["accent_primary"]};
        --accent-secondary: {TOKENS["accent_secondary"]};
        --bull-primary: {TOKENS["bull_primary"]};
        --bull-muted: {TOKENS["bull_muted"]};
        --bear-primary: {TOKENS["bear_primary"]};
        --bear-muted: {TOKENS["bear_muted"]};
        --warn-primary: {TOKENS["warn_primary"]};
        --warn-muted: {TOKENS["warn_muted"]};
      }}

      /* Base App Styling */
      html, body, [data-testid="stAppViewContainer"] {{
        background-color: var(--bg-canvas) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 13.5px !important;
      }}

      /* Top Header & Toolbar */
      [data-testid="stHeader"] {{
        background-color: var(--bg-canvas) !important;
        border-bottom: 1px solid var(--border-subtle) !important;
      }}

      /* Left Sidebar Styling */
      [data-testid="stSidebar"] {{
        background-color: var(--bg-surface-1) !important;
        border-right: 1px solid var(--border-subtle) !important;
      }}
      [data-testid="stSidebar"] .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
      }}

      /* Main Content Container */
      .main .block-container {{
        padding-top: 1.0rem;
        padding-bottom: 2.5rem;
        max-width: 98% !important;
      }}

      /* Monospace Tabular Figures */
      .mono, code, pre, [data-testid="stMetricValue"], .stDataFrame, .tab-nums {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
      }}

      /* Custom Card Surfaces */
      .mm-card {{
        background-color: var(--bg-surface-1);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
      }}

      .mm-card-elevated {{
        background-color: var(--bg-surface-2);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
      }}

      .mm-card-title {{
        font-size: 13px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
      }}

      /* KPI Cards */
      .mm-kpi-card {{
        background-color: var(--bg-surface-1);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 14px 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
      }}
      .mm-kpi-label {{
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary);
        margin-bottom: 4px;
      }}
      .mm-kpi-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
        color: var(--text-primary);
      }}
      .mm-kpi-sub {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: var(--text-secondary);
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
      }}

      /* Badges */
      .mm-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 9px;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}
      .mm-badge-bull {{
        background: var(--bull-muted);
        color: var(--bull-primary);
        border: 1px solid rgba(16, 185, 129, 0.4);
      }}
      .mm-badge-bear {{
        background: var(--bear-muted);
        color: var(--bear-primary);
        border: 1px solid rgba(244, 63, 94, 0.4);
      }}
      .mm-badge-warn {{
        background: var(--warn-muted);
        color: var(--warn-primary);
        border: 1px solid rgba(245, 158, 11, 0.4);
      }}
      .mm-badge-info {{
        background: rgba(56, 189, 248, 0.14);
        color: var(--accent-primary);
        border: 1px solid rgba(56, 189, 248, 0.35);
      }}
      .mm-badge-ai {{
        background: rgba(129, 140, 248, 0.16);
        color: var(--accent-secondary);
        border: 1px solid rgba(129, 140, 248, 0.35);
      }}

      /* Citation Pill */
      .mm-cite-tag {{
        display: inline-block;
        background: rgba(129, 140, 248, 0.18);
        color: var(--accent-secondary);
        border: 1px solid rgba(129, 140, 248, 0.35);
        border-radius: 4px;
        padding: 1px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        margin: 0 2px;
      }}

      /* Quote Boxes */
      .mm-quote-box {{
        background: var(--bg-surface-2);
        border-left: 3px solid var(--accent-primary);
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        font-size: 12.5px;
        line-height: 1.5;
        color: var(--text-primary);
        margin: 6px 0;
      }}
      .mm-quote-evasion {{
        border-left-color: var(--bear-primary) !important;
        background: rgba(244, 63, 94, 0.06);
      }}

      /* Tabs Customization */
      .stTabs [data-baseweb="tab-list"] {{
        background-color: var(--bg-surface-1);
        border-radius: 8px;
        padding: 4px;
        gap: 6px;
        border: 1px solid var(--border-subtle);
      }}
      .stTabs [data-baseweb="tab"] {{
        height: 38px;
        border-radius: 6px;
        color: var(--text-secondary);
        font-weight: 600;
        font-size: 12.5px;
        padding: 0 16px;
      }}
      .stTabs [aria-selected="true"] {{
        background-color: var(--bg-surface-3) !important;
        color: var(--text-primary) !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def badge_html(text: str, tone: str = "info") -> str:
    """Generates styled inline HTML badge."""
    tone_map = {
        "bull": "mm-badge-bull",
        "bear": "mm-badge-bear",
        "warn": "mm-badge-warn",
        "ai": "mm-badge-ai",
        "info": "mm-badge-info",
    }
    css_cls = tone_map.get(tone, "mm-badge-info")
    return f'<span class="mm-badge {css_cls}">{text}</span>'


def kpi_card_html(label: str, value: str, sub: str, tone: str = "info") -> str:
    """Generates an institutional KPI metric card."""
    accent_color = {
        "bull": TOKENS["bull_primary"],
        "bear": TOKENS["bear_primary"],
        "warn": TOKENS["warn_primary"],
        "ai": TOKENS["accent_secondary"],
        "info": TOKENS["accent_primary"],
    }.get(tone, TOKENS["accent_primary"])

    return f"""
    <div class="mm-kpi-card">
      <div>
        <div class="mm-kpi-label">{label}</div>
        <div class="mm-kpi-value" style="color: {accent_color};">{value}</div>
      </div>
      <div class="mm-kpi-sub">{sub}</div>
    </div>
    """


def card_container(content: str, elevated: bool = False) -> str:
    """Wraps HTML content inside an institutional styled card."""
    cls = "mm-card-elevated" if elevated else "mm-card"
    return f'<div class="{cls}">{content}</div>'
