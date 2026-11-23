"""
MarketMind Mock Data Provider
Realistic fallback data for semiconductor companies (NVDA, AMD, INTC, TSM, ASML, MRVL, QCOM).
Used when the FastAPI server is not currently running.
"""

from __future__ import annotations
from typing import Any

SEMICONDUCTOR_COMPANIES: list[dict[str, Any]] = [
    {"ticker": "NVDA", "company_name": "NVIDIA Corporation", "sector": "Semiconductors", "exchange": "NASDAQ"},
    {"ticker": "AMD", "company_name": "Advanced Micro Devices, Inc.", "sector": "Semiconductors", "exchange": "NASDAQ"},
    {"ticker": "INTC", "company_name": "Intel Corporation", "sector": "Semiconductors", "exchange": "NASDAQ"},
    {"ticker": "TSM", "company_name": "Taiwan Semiconductor Manufacturing", "sector": "Foundry", "exchange": "NYSE"},
    {"ticker": "ASML", "company_name": "ASML Holding N.V.", "sector": "Lithography Equipment", "exchange": "NASDAQ"},
    {"ticker": "MRVL", "company_name": "Marvell Technology, Inc.", "sector": "Data Infrastructure", "exchange": "NASDAQ"},
    {"ticker": "QCOM", "company_name": "Qualcomm Incorporated", "sector": "Wireless / Edge AI", "exchange": "NASDAQ"},
]

MOCK_CALLS: list[dict[str, Any]] = [
    {
        "call_id": 1,
        "company_id": 1,
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "quarter": "Q4",
        "fiscal_year": 2026,
        "call_date": "2026-02-25",
        "status": "completed",
        "signal": "POSITIVE",
        "confidence": 0.842,
        "expected_return": 4.15,
    },
    {
        "call_id": 2,
        "company_id": 1,
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "quarter": "Q3",
        "fiscal_year": 2026,
        "call_date": "2025-11-20",
        "status": "completed",
        "signal": "POSITIVE",
        "confidence": 0.791,
        "expected_return": 3.80,
    },
    {
        "call_id": 3,
        "company_id": 2,
        "ticker": "AMD",
        "company_name": "Advanced Micro Devices, Inc.",
        "quarter": "Q4",
        "fiscal_year": 2025,
        "call_date": "2026-01-30",
        "status": "completed",
        "signal": "NEUTRAL",
        "confidence": 0.628,
        "expected_return": 0.85,
    },
    {
        "call_id": 4,
        "company_id": 3,
        "ticker": "INTC",
        "company_name": "Intel Corporation",
        "quarter": "Q4",
        "fiscal_year": 2025,
        "call_date": "2026-01-23",
        "status": "completed",
        "signal": "NEGATIVE",
        "confidence": 0.814,
        "expected_return": -5.20,
    },
    {
        "call_id": 5,
        "company_id": 4,
        "ticker": "TSM",
        "company_name": "Taiwan Semiconductor Manufacturing",
        "quarter": "Q4",
        "fiscal_year": 2025,
        "call_date": "2026-01-16",
        "status": "completed",
        "signal": "POSITIVE",
        "confidence": 0.776,
        "expected_return": 3.40,
    },
    {
        "call_id": 6,
        "company_id": 5,
        "ticker": "ASML",
        "company_name": "ASML Holding N.V.",
        "quarter": "Q4",
        "fiscal_year": 2025,
        "call_date": "2026-01-22",
        "status": "completed",
        "signal": "NEUTRAL",
        "confidence": 0.615,
        "expected_return": -0.40,
    },
    {
        "call_id": 7,
        "company_id": 6,
        "ticker": "MRVL",
        "company_name": "Marvell Technology, Inc.",
        "quarter": "Q3",
        "fiscal_year": 2026,
        "call_date": "2025-12-03",
        "status": "completed",
        "signal": "POSITIVE",
        "confidence": 0.732,
        "expected_return": 2.90,
    },
    {
        "call_id": 8,
        "company_id": 7,
        "ticker": "QCOM",
        "company_name": "Qualcomm Incorporated",
        "quarter": "Q1",
        "fiscal_year": 2026,
        "call_date": "2026-02-04",
        "status": "completed",
        "signal": "NEUTRAL",
        "confidence": 0.589,
        "expected_return": 0.20,
    },
]


def get_mock_signal_data(call_id: int) -> dict[str, Any]:
    """Returns detailed signal & SHAP breakdown for a call."""
    call = next((c for c in MOCK_CALLS if c["call_id"] == call_id), MOCK_CALLS[0])
    ticker = call["ticker"]

    # Tailor feature vectors per company
    if ticker == "NVDA":
        base_val = 0.05
        pred_signal = "POSITIVE"
        conf = 0.842
        exp_return = 4.15
        shap_drivers = [
            {"feature": "revenue_qoq_pct", "value": 16.8, "shap_value": 0.385, "impact": "Bullish"},
            {"feature": "sent_ceo_delta", "value": 0.22, "shap_value": 0.245, "impact": "Bullish"},
            {"feature": "margin_qoq_pp", "value": 1.4, "shap_value": 0.170, "impact": "Bullish"},
            {"feature": "evasion_mean", "value": 0.14, "shap_value": 0.115, "impact": "Bullish"},
            {"feature": "volatility_30d", "value": 38.2, "shap_value": -0.095, "impact": "Bearish"},
            {"feature": "discrepancy_count", "value": 0.0, "shap_value": 0.080, "impact": "Bullish"},
            {"feature": "sent_cfo", "value": 0.58, "shap_value": 0.065, "impact": "Bullish"},
        ]
        probabilities = {"POSITIVE": 0.842, "NEUTRAL": 0.118, "NEGATIVE": 0.040}
    elif ticker == "INTC":
        base_val = 0.05
        pred_signal = "NEGATIVE"
        conf = 0.814
        exp_return = -5.20
        shap_drivers = [
            {"feature": "margin_qoq_pp", "value": -3.8, "shap_value": -0.410, "impact": "Bearish"},
            {"feature": "evasion_mean", "value": 0.58, "shap_value": -0.320, "impact": "Bearish"},
            {"feature": "discrepancy_count", "value": 3.0, "shap_value": -0.225, "impact": "Bearish"},
            {"feature": "sent_cfo_delta", "value": -0.28, "shap_value": -0.165, "impact": "Bearish"},
            {"feature": "revenue_qoq_pct", "value": -6.2, "shap_value": -0.140, "impact": "Bearish"},
            {"feature": "sent_ceo", "value": 0.21, "shap_value": -0.075, "impact": "Bearish"},
        ]
        probabilities = {"POSITIVE": 0.062, "NEUTRAL": 0.124, "NEGATIVE": 0.814}
    else:
        base_val = 0.05
        pred_signal = call.get("signal", "NEUTRAL")
        conf = call.get("confidence", 0.65)
        exp_return = call.get("expected_return", 0.5)
        shap_drivers = [
            {"feature": "revenue_qoq_pct", "value": 7.5, "shap_value": 0.145, "impact": "Bullish"},
            {"feature": "sent_ceo", "value": 0.42, "shap_value": 0.095, "impact": "Bullish"},
            {"feature": "evasion_mean", "value": 0.29, "shap_value": -0.065, "impact": "Bearish"},
            {"feature": "volatility_30d", "value": 27.5, "shap_value": -0.040, "impact": "Bearish"},
        ]
        probabilities = {"POSITIVE": 0.35, "NEUTRAL": 0.55, "NEGATIVE": 0.10}

    return {
        "call_id": call_id,
        "ticker": ticker,
        "quarter": call["quarter"],
        "fiscal_year": call["fiscal_year"],
        "predicted_signal": pred_signal,
        "confidence": conf,
        "expected_return": exp_return,
        "base_value": base_val,
        "probabilities": probabilities,
        "top_features": [d["feature"] for d in shap_drivers[:3]],
        "shap": {
            "predicted_label": pred_signal,
            "probabilities": probabilities,
            "shap_values": shap_drivers,
            "base_value": base_val,
        },
    }


def get_mock_evasion_data(call_id: int) -> list[dict[str, Any]]:
    """Returns Question & Answer pairs with evasion scores for a call."""
    return [
        {
            "evasion_id": 101,
            "question_speaker": "Toshiya Hari (Goldman Sachs)",
            "question_text": "Could you provide quantification on the Blackwell revenue trajectory through Q2 and Q3, and what specific gross margin headwinds we should model?",
            "answer_speaker": "Colette Kress (EVP & CFO)",
            "answer_text": "Thanks, Toshiya. As Jensen highlighted, demand for Blackwell is phenomenal. We are ramping as quickly as our packaging partners can supply wafers. We expect margins to normalize into the mid-70s as yield curves mature later in the year.",
            "evasion_score": 0.78,
            "is_evasive": True,
            "deflection_category": "Qualitative Generalization / Metric Omission",
            "reason": "Executive dodged dollar and volume quantification on quarterly ramp; substituted qualitative enthusiasm for concrete margin drag schedule.",
        },
        {
            "evasion_id": 102,
            "question_speaker": "Vivek Arya (Bank of America)",
            "question_text": "How should we think about sovereign AI demand beyond Western Europe, and are you seeing order pushouts due to datacenter power constraints?",
            "answer_speaker": "Jensen Huang (President & CEO)",
            "answer_text": "Sovereign AI is truly accelerating across India, Japan, and the Middle East. Every nation realizes their natural data is an asset. On power, liquid cooling architectures like GB200 actually deliver 25x better energy efficiency per token.",
            "evasion_score": 0.24,
            "is_evasive": False,
            "deflection_category": "Direct Technical Address",
            "reason": "Directly articulated regional expansion demand and addressed datacenter thermal/power concerns with specific architectural efficiency metrics.",
        },
        {
            "evasion_id": 103,
            "question_speaker": "Stacy Rasgon (Bernstein)",
            "question_text": "On Networking, InfiniBand was exceptionally strong, but what was the exact sequential percentage growth of Spectrum-X Ethernet switches?",
            "answer_speaker": "Colette Kress (EVP & CFO)",
            "answer_text": "Spectrum-X is seeing tremendous adoption and is already a multibillion-dollar product line. We expect virtually all major CSPs to deploy Ethernet fabrics alongside InfiniBand.",
            "evasion_score": 0.65,
            "is_evasive": True,
            "deflection_category": "Cumulative Sizing Deflection",
            "reason": "Substituted cumulative run-rate phrasing ('multibillion-dollar line') for the exact sequential QoQ % growth requested.",
        },
        {
            "evasion_id": 104,
            "question_speaker": "C.J. Muse (Cantor Fitzgerald)",
            "question_text": "Can you speak to customer concentration? Did your top 2 cloud customers exceed 35% of total revenues this quarter?",
            "answer_speaker": "Colette Kress (EVP & CFO)",
            "answer_text": "Cloud service providers represented roughly half of our Data Center revenue. Beyond hyperscalers, consumer internet and enterprise AI workloads are growing rapidly.",
            "evasion_score": 0.52,
            "is_evasive": True,
            "deflection_category": "Aggregate Blending",
            "reason": "Blended customer concentration into aggregate CSP category rather than disclosing top 2 customer concentration.",
        },
    ]


def get_mock_tone_data(call_id: int) -> dict[str, Any]:
    """Returns speaker-separated FinBERT tone dynamics across call segments."""
    segments = [
        {"segment_order": 1, "speaker": "CEO", "role": "CEO", "positive": 0.72, "neutral": 0.24, "negative": 0.04, "compound": 0.68, "topic": "AI Factory Vision"},
        {"segment_order": 2, "speaker": "CEO", "role": "CEO", "positive": 0.81, "neutral": 0.16, "negative": 0.03, "compound": 0.78, "topic": "Blackwell Architecture"},
        {"segment_order": 3, "speaker": "CFO", "role": "CFO", "positive": 0.58, "neutral": 0.36, "negative": 0.06, "compound": 0.52, "topic": "Financial Performance"},
        {"segment_order": 4, "speaker": "CFO", "role": "CFO", "positive": 0.44, "neutral": 0.42, "negative": 0.14, "compound": 0.30, "topic": "Gross Margin & CapEx"},
        {"segment_order": 5, "speaker": "Analyst", "role": "Analyst", "positive": 0.20, "neutral": 0.65, "negative": 0.15, "compound": 0.05, "topic": "Supply Bottleneck Q&A"},
        {"segment_order": 6, "speaker": "CEO", "role": "CEO", "positive": 0.69, "neutral": 0.26, "negative": 0.05, "compound": 0.64, "topic": "Capacity Ramp Response"},
        {"segment_order": 7, "speaker": "Analyst", "role": "Analyst", "positive": 0.12, "neutral": 0.58, "negative": 0.30, "compound": -0.18, "topic": "Customer Concentration Q&A"},
        {"segment_order": 8, "speaker": "CFO", "role": "CFO", "positive": 0.48, "neutral": 0.40, "negative": 0.12, "compound": 0.36, "topic": "Diversification Response"},
        {"segment_order": 9, "speaker": "CEO", "role": "CEO", "positive": 0.85, "neutral": 0.12, "negative": 0.03, "compound": 0.82, "topic": "Closing Remarks"},
    ]

    return {
        "call_id": call_id,
        "ceo_mean": 0.73,
        "cfo_mean": 0.39,
        "analyst_mean": -0.06,
        "ceo_cfo_delta": 0.34,  # Divergence between CEO optimism and CFO margin reality
        "evasion_rate": 0.25,
        "segments": segments,
    }


def get_mock_audit_findings(call_id: int) -> dict[str, Any]:
    """Returns adversarial forensic audit findings and discrepancy alerts."""
    return {
        "call_id": call_id,
        "ticker": "NVDA",
        "quarter": "Q4",
        "overall_risk": "MODERATE",
        "findings": [
            {
                "finding_type": "Discrepancy with 10-Q",
                "severity": "HIGH",
                "current_statement": "Blackwell gross margins will reach mid-70s by end of fiscal year.",
                "historical_statement": "Form 10-Q (Note 7) notes elevated test and packaging costs with CoWoS-L wafer yields through Q3.",
                "historical_source": "Q3 10-Q Filing, Risk Factors p. 48",
                "explanation": "Executive timeline for gross margin normalization assumes immediate 85%+ packaging yields, which contrasts with stated foundry qualification hurdles.",
            },
            {
                "finding_type": "Forward Guidance Divergence",
                "severity": "MEDIUM",
                "current_statement": "Networking revenue expected to track compute growth 1:1.",
                "historical_statement": "Prior quarter guided networking attach rates at 1.25x compute clusters.",
                "historical_source": "Q3 FY26 Earnings Transcript",
                "explanation": "Quietly lowered relative networking attach expectations due to customer Ethernet hybrid deployments.",
            },
        ],
        "recommendation": "Maintain constructive signal on explosive Data Center revenue growth, but hedge margin expectations for Q1-Q2 packaging yield ramp.",
    }


def get_mock_metrics_data(call_id: int) -> list[dict[str, Any]]:
    """Returns extracted financial metrics and forward guidance."""
    return [
        {"metric_name": "Revenue", "value": 37.5, "unit": "B USD", "period": "Q4 FY26", "is_guidance": False, "confidence": 0.98, "quote": "Fourth quarter revenue was $37.5 billion, up 94% from a year ago."},
        {"metric_name": "Data Center Revenue", "value": 34.1, "unit": "B USD", "period": "Q4 FY26", "is_guidance": False, "confidence": 0.99, "quote": "Data Center revenue was a record $34.1 billion, up 112% year-on-year."},
        {"metric_name": "Gross Margin (GAAP)", "value": 73.5, "unit": "%", "period": "Q4 FY26", "is_guidance": False, "confidence": 0.96, "quote": "GAAP gross margin was 73.5%."},
        {"metric_name": "Q1 Revenue Guidance", "value": 43.0, "unit": "B USD", "period": "Q1 FY27", "is_guidance": True, "confidence": 0.95, "quote": "Revenue for the first quarter of fiscal 2027 is expected to be $43.0 billion, plus or minus 2%."},
        {"metric_name": "Q1 Margin Guidance", "value": 71.0, "unit": "%", "period": "Q1 FY27", "is_guidance": True, "confidence": 0.94, "quote": "GAAP and non-GAAP gross margins are expected to be 71.0% and 71.5%, respectively."},
    ]
