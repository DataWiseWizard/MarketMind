"""
MarketMind — Stage 2C: Feature Matrix Builder
==============================================
Assembles the ML training matrix by joining:
  • market_labels.parquet (ground-truth labels from yfinance)
  • sentiment_scores table (FinBERT per role)
  • evasion_scores table
  • extracted_metrics table (revenue, margin surprises)
  • yfinance pre-call momentum + volatility

Output: data/processed/features/train_matrix.parquet
         data/processed/features/train_matrix.csv   (human-readable)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.connection import get_session
from src.db.models import (
    Company, EarningsCall, SentimentScore, EvasionScore,
    ExtractedMetric, AccountabilityFinding, Signal
)

LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "market_labels.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "features" / "train_matrix.parquet"
OUT_CSV     = PROJECT_ROOT / "data" / "processed" / "features" / "train_matrix.csv"

TICKERS = ["NVDA", "AMD", "INTC", "QCOM", "TSM", "ASML", "MRVL"]


# ── DB feature extractors ─────────────────────────────────────────────────────

def _get_sentiment_features(session) -> pd.DataFrame:
    """Pivot FinBERT compound scores by speaker role per call."""
    rows = session.query(SentimentScore).all()
    records = []
    for s in rows:
        role = s.speaker_role.value.upper() if s.speaker_role else "UNKNOWN"
        records.append({
            "call_id"      : s.call_id,
            "role"         : role,
            "compound"     : s.compound_score,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    agg = df.groupby(["call_id", "role"])["compound"].mean().unstack(fill_value=0)
    agg.columns = [f"sent_{c.lower()}" for c in agg.columns]
    agg = agg.reset_index()
    # Ensure expected columns exist even if role not present
    for role in ["ceo", "cfo", "analyst", "unknown"]:
        col = f"sent_{role}"
        if col not in agg.columns:
            agg[col] = 0.0
    return agg


def _get_evasion_features(session) -> pd.DataFrame:
    """Aggregate evasion scores per call."""
    rows = session.query(EvasionScore).all()
    if not rows:
        return pd.DataFrame(columns=["call_id", "evasion_mean", "evasion_max", "evasive_count"])

    records = [{"call_id": e.call_id, "evasion_score": e.evasion_score, "is_evasive": e.is_evasive}
               for e in rows]
    df = pd.DataFrame(records)
    agg = df.groupby("call_id").agg(
        evasion_mean  = ("evasion_score", "mean"),
        evasion_max   = ("evasion_score", "max"),
        evasive_count = ("is_evasive",    "sum"),
    ).reset_index()
    return agg


def _get_metric_features(session) -> pd.DataFrame:
    """
    Compute revenue and gross-margin 'surprises' as QoQ delta.
    Returns one row per call with revenue_qoq_pct, margin_qoq_pp.
    """
    rows = session.query(ExtractedMetric).all()
    if not rows:
        return pd.DataFrame(columns=["call_id", "revenue_qoq_pct", "margin_qoq_pp"])

    records = []
    for m in rows:
        name_lower = (m.metric_name or "").lower()
        records.append({
            "call_id"   : m.call_id,
            "metric_name": name_lower,
            "value"     : m.value,
            "is_guidance": m.is_guidance,
        })

    df = pd.DataFrame(records)

    # Pull revenue and margin per call (actual, not guidance)
    rev = (df[df["metric_name"].str.contains("revenue", na=False) & ~df["is_guidance"]]
           .groupby("call_id")["value"].mean().rename("revenue"))
    mgn = (df[df["metric_name"].str.contains("margin|gross", na=False) & ~df["is_guidance"]]
           .groupby("call_id")["value"].mean().rename("margin"))

    combo = pd.concat([rev, mgn], axis=1).reset_index()
    combo = combo.sort_values("call_id")

    # QoQ delta (shifted by 1 within same ticker — approximation)
    combo["revenue_qoq_pct"] = combo["revenue"].pct_change().fillna(0) * 100
    combo["margin_qoq_pp"]   = combo["margin"].diff().fillna(0)

    return combo[["call_id", "revenue_qoq_pct", "margin_qoq_pp"]]


def _get_accountability_features(session) -> pd.DataFrame:
    """Count accountability / discrepancy findings per call."""
    rows = session.query(AccountabilityFinding).all()
    if not rows:
        return pd.DataFrame(columns=["call_id", "discrepancy_count"])

    df = pd.DataFrame([{"call_id": r.call_id} for r in rows])
    agg = df.groupby("call_id").size().reset_index(name="discrepancy_count")
    return agg


# ── Pre-call market features from yfinance ────────────────────────────────────

def _get_pre_call_features(call_dates: pd.DataFrame) -> pd.DataFrame:
    """
    For each (ticker, call_date) pull 30 trading days prior and compute:
      - momentum_30d  : total return over prior 30 days
      - volatility_30d: std of daily returns (annualised)
    """
    from datetime import timedelta

    all_syms   = TICKERS
    all_dates  = call_dates["call_date"].tolist()
    start_bulk = min(all_dates) - timedelta(days=50)
    end_bulk   = max(all_dates) + timedelta(days=5)

    logger.info("Downloading pre-call price history for momentum/volatility...")
    prices_raw = yf.download(
        all_syms,
        start=start_bulk.strftime("%Y-%m-%d"),
        end=end_bulk.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices = prices_raw["Close"]
    else:
        prices = prices_raw

    daily_ret = prices.pct_change()

    records = []
    for _, r in call_dates.iterrows():
        ticker    = r["ticker"]
        call_date = r["call_date"]
        call_id   = r["call_id"]

        if ticker not in daily_ret.columns:
            records.append({"call_id": call_id, "momentum_30d": 0.0, "volatility_30d": 0.0})
            continue

        series = daily_ret[ticker].dropna()
        before = series[series.index.date < call_date]

        if len(before) >= 20:
            window = before.iloc[-30:]
            mom30  = round((1 + window).prod() - 1, 6) * 100
            vol30  = round(window.std() * (252 ** 0.5), 6) * 100
        else:
            mom30 = vol30 = 0.0

        records.append({"call_id": call_id, "momentum_30d": mom30, "volatility_30d": vol30})

    return pd.DataFrame(records)


# ── Sentiment QoQ delta ───────────────────────────────────────────────────────

def _add_sentiment_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add QoQ delta for CEO sentiment (most predictive role)."""
    df = df.sort_values(["ticker", "call_date"]).copy()
    df["sent_ceo_delta"] = df.groupby("ticker")["sent_ceo"].diff().fillna(0)
    df["sent_cfo_delta"] = df.groupby("ticker")["sent_cfo"].diff().fillna(0)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def build_feature_matrix() -> pd.DataFrame:
    if not LABELS_PATH.exists():
        logger.error(f"Labels file not found: {LABELS_PATH}")
        logger.error("Run market_returns.py first: python -m src.features.market_returns")
        return pd.DataFrame()

    labels = pd.read_parquet(LABELS_PATH)
    logger.info(f"Loaded {len(labels)} labelled calls from {LABELS_PATH.name}")

    session = get_session()
    try:
        sent_df  = _get_sentiment_features(session)
        evas_df  = _get_evasion_features(session)
        metr_df  = _get_metric_features(session)
        acct_df  = _get_accountability_features(session)
    finally:
        session.close()

    pre_df = _get_pre_call_features(labels[["call_id", "ticker", "call_date"]])

    # ── Merge all features onto labels ────────────────────────────────────────
    df = labels.copy()
    for feature_df in [sent_df, evas_df, metr_df, acct_df, pre_df]:
        if not feature_df.empty and "call_id" in feature_df.columns:
            df = df.merge(feature_df, on="call_id", how="left")

    df = _add_sentiment_deltas(df)

    # Fill remaining NaNs with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # ── Feature list summary ──────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in
                    ["call_id", "ticker", "quarter", "fiscal_year", "call_date", "label",
                     "return_1d", "bench_return_1d", "return_3d", "bench_return_3d",
                     "abnormal_return_1d", "abnormal_return_3d"]]
    logger.info(f"Features assembled ({len(feature_cols)}): {feature_cols}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    df.to_csv(OUT_CSV, index=False)
    logger.success(f"Train matrix saved → {OUT_PARQUET}  ({df.shape[0]} rows × {df.shape[1]} cols)")

    return df


if __name__ == "__main__":
    df = build_feature_matrix()
    if not df.empty:
        print(df.head(10).to_string(index=False))
        print(f"\nShape: {df.shape}")
        print(f"\nLabel counts:\n{df['label'].value_counts()}")
