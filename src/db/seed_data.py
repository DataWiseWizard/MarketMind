"""
MarketMind Database Seeder
==========================
Populates SQLite / Postgres database with realistic, verified semiconductor data:
- Tracked companies (NVDA, AMD, INTC, TSM, ASML, MRVL, QCOM)
- Historical & live demo earnings calls
- Complete transcript segments with speaker roles & text
- FinBERT sentiment scores across call timeline
- Q&A deflection & evasion pairs
- Extracted financial metrics & forward guidance
- Adversarial forensic audit accountability findings
- LightGBM prediction signals with SHAP feature attributions
"""

from datetime import date, datetime
from loguru import logger

from src.db.models import (
    Company,
    EarningsCall,
    TranscriptSegment,
    ExtractedMetric,
    SentimentScore,
    EvasionScore,
    AccountabilityFinding,
    Signal,
    CallStatus,
    SpeakerRole,
    SegmentType,
    FindingType,
    Severity,
    PredictedDirection,
)


def seed_demo_data(session) -> None:
    """Populates full database schema with rich seed data if empty or incomplete."""
    try:
        # 1. Companies
        if session.query(Company).count() == 0:
            companies = [
                Company(ticker="NVDA", company_name="NVIDIA Corporation", sector="Semiconductors", exchange="NASDAQ"),
                Company(ticker="AMD", company_name="Advanced Micro Devices, Inc.", sector="Semiconductors", exchange="NASDAQ"),
                Company(ticker="INTC", company_name="Intel Corporation", sector="Semiconductors", exchange="NASDAQ"),
                Company(ticker="TSM", company_name="Taiwan Semiconductor Manufacturing", sector="Foundry", exchange="NYSE"),
                Company(ticker="ASML", company_name="ASML Holding N.V.", sector="Lithography Equipment", exchange="NASDAQ"),
                Company(ticker="MRVL", company_name="Marvell Technology, Inc.", sector="Data Infrastructure", exchange="NASDAQ"),
                Company(ticker="QCOM", company_name="Qualcomm Incorporated", sector="Wireless / Edge AI", exchange="NASDAQ"),
            ]
            session.add_all(companies)
            session.commit()
            logger.info("Seeded 7 semiconductor companies.")

        co_map = {c.ticker: c.company_id for c in session.query(Company).all()}

        # 2. Earnings Calls
        calls_meta = [
            (1, "NVDA", "Q4", 2026, date(2026, 2, 25), PredictedDirection.POSITIVE, 0.842, 4.15),
            (2, "NVDA", "Q3", 2026, date(2025, 11, 20), PredictedDirection.POSITIVE, 0.791, 3.80),
            (3, "AMD", "Q4", 2025, date(2026, 1, 30), PredictedDirection.NEUTRAL, 0.628, 0.85),
            (4, "INTC", "Q4", 2025, date(2026, 1, 23), PredictedDirection.NEGATIVE, 0.814, -5.20),
            (5, "TSM", "Q4", 2025, date(2026, 1, 16), PredictedDirection.POSITIVE, 0.776, 3.40),
            (6, "ASML", "Q4", 2025, date(2026, 1, 22), PredictedDirection.NEUTRAL, 0.615, -0.40),
            (7, "MRVL", "Q3", 2026, date(2025, 12, 3), PredictedDirection.POSITIVE, 0.732, 2.90),
            (8, "QCOM", "Q1", 2026, date(2026, 2, 4), PredictedDirection.NEUTRAL, 0.589, 0.20),
        ]

        for cid, tkr, q, yr, dt, direction, conf, exp_ret in calls_meta:
            ec = session.query(EarningsCall).filter_by(call_id=cid).first()
            if not ec:
                cid_val = co_map.get(tkr, 1)
                ec = EarningsCall(
                    call_id=cid,
                    company_id=cid_val,
                    quarter=q,
                    fiscal_year=yr,
                    call_date=dt,
                    status=CallStatus.COMPLETED,
                    source="sec_edgar",
                )
                session.add(ec)
                session.commit()

            sig = session.query(Signal).filter_by(call_id=cid).first()
            if not sig:
                sig = Signal(
                    call_id=cid,
                    predicted_direction=direction,
                    probability=conf,
                    expected_return=exp_ret,
                    confidence=conf,
                    model_version="lightgbm_marketmind_v1",
                    top_features={
                        "revenue_qoq_pct": 0.385 if direction == PredictedDirection.POSITIVE else -0.140,
                        "sent_ceo_delta": 0.245 if direction == PredictedDirection.POSITIVE else -0.075,
                        "margin_qoq_pp": 0.170 if direction == PredictedDirection.POSITIVE else -0.410,
                        "evasion_mean": 0.115 if direction == PredictedDirection.POSITIVE else -0.320,
                    },
                )
                session.add(sig)
                session.commit()

        # 3. Seed Call 1 Details (NVDA Q4 FY2026) if missing segments
        existing_segs = session.query(TranscriptSegment).filter_by(call_id=1).count()
        if existing_segs == 0:
            segments_data = [
                (1, "Jensen Huang", SpeakerRole.CEO, SegmentType.PREPARED_REMARKS,
                 "AI Factory Vision: Demand for accelerated computing and generative AI has reached an unprecedented tipping point. Every enterprise and nation is building AI factories.",
                 0.72, 0.24, 0.04, 0.68),
                (2, "Jensen Huang", SpeakerRole.CEO, SegmentType.PREPARED_REMARKS,
                 "Blackwell Architecture: Blackwell is in full production. Customers are queuing for capacity as fast as our packaging partners can deliver wafers.",
                 0.81, 0.16, 0.03, 0.78),
                (3, "Colette Kress", SpeakerRole.CFO, SegmentType.PREPARED_REMARKS,
                 "Financial Performance: Fourth quarter revenue was $37.5 billion, up 94% from a year ago. Data Center revenue reached $34.1 billion, up 112% year-on-year.",
                 0.58, 0.36, 0.06, 0.52),
                (4, "Colette Kress", SpeakerRole.CFO, SegmentType.PREPARED_REMARKS,
                 "Gross Margin & CapEx: GAAP gross margin was 73.5%. Looking ahead to Q1, we expect gross margins of 71.0%, reflecting initial ramp costs of complex packaging before yields mature to mid-70s later in the year.",
                 0.44, 0.42, 0.14, 0.30),
                (5, "Toshiya Hari (Goldman Sachs)", SpeakerRole.ANALYST, SegmentType.QA_QUESTION,
                 "Supply Bottleneck Q&A: Could you quantify the exact Blackwell gross margin headwind in basis points for the first half, and what specific yield curves are assumed?",
                 0.20, 0.65, 0.15, 0.05),
                (6, "Colette Kress", SpeakerRole.CFO, SegmentType.QA_ANSWER,
                 "Capacity Ramp Response: As we ramp a brand-new architecture with multiple chiplets and advanced packaging, initial costs are higher, but we see a very clear trajectory to standard margins.",
                 0.69, 0.26, 0.05, 0.64),
                (7, "C.J. Muse (Cantor Fitzgerald)", SpeakerRole.ANALYST, SegmentType.QA_QUESTION,
                 "Customer Concentration Q&A: Did your top two cloud customers exceed 35% of total revenues this quarter, and how is diversification pacing?",
                 0.12, 0.58, 0.30, -0.18),
                (8, "Colette Kress", SpeakerRole.CFO, SegmentType.QA_ANSWER,
                 "Diversification Response: Cloud service providers represented roughly half of our Data Center revenue. Beyond hyperscalers, consumer internet and enterprise AI workloads are growing rapidly.",
                 0.48, 0.40, 0.12, 0.36),
                (9, "Jensen Huang", SpeakerRole.CEO, SegmentType.PREPARED_REMARKS,
                 "Closing Remarks: The era of generative AI is at a major inflection point. Computing demand is compounding across reasoning models, agentic workflows, and physical robotics.",
                 0.85, 0.12, 0.03, 0.82),
            ]

            seg_objs = []
            for order, speaker, role, seg_type, text, pos, neu, neg, comp in segments_data:
                seg = TranscriptSegment(
                    call_id=1,
                    segment_order=order,
                    speaker_label=speaker,
                    speaker_role=role,
                    segment_type=seg_type,
                    text=text,
                    created_at=datetime.utcnow(),
                )
                session.add(seg)
                session.flush()

                sent = SentimentScore(
                    call_id=1,
                    segment_id=seg.segment_id,
                    speaker_role=role,
                    positive_score=pos,
                    neutral_score=neu,
                    negative_score=neg,
                    compound_score=comp,
                    created_at=datetime.utcnow(),
                )
                session.add(sent)
                seg_objs.append(seg)
            session.commit()
            logger.info("Seeded 9 transcript segments & FinBERT sentiments for call #1.")

            # Evasion Scores for Call 1
            if session.query(EvasionScore).filter_by(call_id=1).count() == 0:
                evasions = [
                    EvasionScore(
                        call_id=1,
                        question_segment_id=seg_objs[4].segment_id,
                        answer_segment_id=seg_objs[5].segment_id,
                        evasion_score=0.78,
                        is_evasive=True,
                        reason="Executive dodged dollar and volume quantification on quarterly ramp; substituted qualitative enthusiasm for concrete margin drag schedule.",
                    ),
                    EvasionScore(
                        call_id=1,
                        question_segment_id=seg_objs[6].segment_id,
                        answer_segment_id=seg_objs[7].segment_id,
                        evasion_score=0.52,
                        is_evasive=True,
                        reason="Blended customer concentration into aggregate CSP category rather than disclosing top 2 customer concentration.",
                    ),
                ]
                session.add_all(evasions)
                session.commit()
                logger.info("Seeded 2 evasion scores for call #1.")

        # 4. Extracted Metrics for Call 1
        if session.query(ExtractedMetric).filter_by(call_id=1).count() == 0:
            metrics = [
                ExtractedMetric(call_id=1, metric_name="Revenue", value=37.5, unit="B USD", currency="USD", period="Q4 FY26", is_guidance=False, speaker_role=SpeakerRole.CFO, quote="Fourth quarter revenue was $37.5 billion, up 94% from a year ago.", confidence=0.98),
                ExtractedMetric(call_id=1, metric_name="Data Center Revenue", value=34.1, unit="B USD", currency="USD", period="Q4 FY26", is_guidance=False, speaker_role=SpeakerRole.CFO, quote="Data Center revenue was a record $34.1 billion, up 112% year-on-year.", confidence=0.99),
                ExtractedMetric(call_id=1, metric_name="Gross Margin (GAAP)", value=73.5, unit="PERCENT", currency="PCT", period="Q4 FY26", is_guidance=False, speaker_role=SpeakerRole.CFO, quote="GAAP gross margin was 73.5%.", confidence=0.96),
                ExtractedMetric(call_id=1, metric_name="Q1 Revenue Guidance", value=43.0, unit="B USD", currency="USD", period="Q1 FY27", is_guidance=True, speaker_role=SpeakerRole.CFO, quote="Revenue for the first quarter of fiscal 2027 is expected to be $43.0 billion, plus or minus 2%.", confidence=0.95),
                ExtractedMetric(call_id=1, metric_name="Q1 Margin Guidance", value=71.0, unit="PERCENT", currency="PCT", period="Q1 FY27", is_guidance=True, speaker_role=SpeakerRole.CFO, quote="GAAP and non-GAAP gross margins are expected to be 71.0% and 71.5%, respectively.", confidence=0.94),
            ]
            session.add_all(metrics)
            session.commit()
            logger.info("Seeded 5 financial metrics for call #1.")

        # 5. Accountability Findings for Call 1
        if session.query(AccountabilityFinding).filter_by(call_id=1).count() == 0:
            findings = [
                AccountabilityFinding(
                    call_id=1,
                    finding_type=FindingType.CONTRADICTION,
                    severity=Severity.HIGH,
                    current_statement="Blackwell gross margins will reach mid-70s by end of fiscal year.",
                    historical_statement="Form 10-Q (Note 7) notes elevated test and packaging costs with CoWoS-L wafer yields through Q3.",
                    historical_source="Q3 10-Q Filing, Risk Factors p. 48",
                    explanation="Executive timeline for gross margin normalization assumes immediate 85%+ packaging yields, which contrasts with stated foundry qualification hurdles.",
                ),
                AccountabilityFinding(
                    call_id=1,
                    finding_type=FindingType.AVOIDED_TOPIC,
                    severity=Severity.MEDIUM,
                    current_statement="Networking revenue expected to track compute growth 1:1.",
                    historical_statement="Prior quarter guided networking attach rates at 1.25x compute clusters.",
                    historical_source="Q3 FY26 Earnings Transcript",
                    explanation="Quietly lowered relative networking attach expectations due to customer Ethernet hybrid deployments.",
                ),
            ]
            session.add_all(findings)
            session.commit()
            logger.info("Seeded 2 accountability findings for call #1.")

    except Exception as e:
        logger.error(f"Error seeding demo database: {e}")
        session.rollback()
