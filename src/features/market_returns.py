"""
MarketMind — Stage 2B: Market Return Labels
============================================
Pulls post-earnings stock returns via yfinance for all 7 tracked tickers
and the SMH ETF as the sector benchmark.

Sector-adjusted abnormal return:
    AR = R_stock - R_SMH

Label thresholds:
    AR > +1.5%  → POSITIVE
    AR < -1.5%  → NEGATIVE
    else        → NEUTRAL

Output: data/processed/features/market_labels.parquet
"""

import sys
import warnings
from datetime import timedelta, date
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.connection import get_session
from src.db.models import Company, EarningsCall

# ── Constants ────────────────────────────────────────────────────────────────
TICKERS        = ["NVDA", "AMD", "INTC", "QCOM", "TSM", "ASML", "MRVL"]
BENCHMARK      = "SMH"          # VanEck Semiconductor ETF
THRESHOLD_POS  = 0.015          # +1.5%
THRESHOLD_NEG  = -0.015         # -1.5%
WINDOWS        = [1, 3]         # post-earnings trading days
OUT_PATH       = PROJECT_ROOT / "data" / "processed" / "features" / "market_labels.parquet"


# ── Price helpers ─────────────────────────────────────────────────────────────

def _next_trading_days(ref_date: date, n: int) -> list[date]:
    """Return n calendar days after ref_date (simple; yfinance handles non-trading-day gaps)."""
    return [ref_date + timedelta(days=d) for d in range(1, n + 15)]   # pull wide window


def _get_return(ticker: str, call_date: date, window: int, prices: pd.DataFrame) -> float | None:
    """Compute the cumulative log-return for `ticker` over `window` trading days after call_date."""
    try:
        col = ticker if ticker in prices.columns else None
        if col is None:
            return None

        series = prices[col].dropna()
        # Find the first valid close ON or AFTER call_date
        after = series[series.index.date >= call_date]
        if len(after) < window + 1:
            return None

        p_open  = after.iloc[0]
        p_close = after.iloc[window]
        return (p_close - p_open) / p_open
    except Exception as e:
        logger.warning(f"Return calc failed for {ticker} on {call_date}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def build_market_labels() -> pd.DataFrame:
    """
    Query every earnings call from the DB, fetch prices, compute returns,
    assign labels, and save to parquet.
    """
    session = get_session()
    try:
        rows = (
            session.query(EarningsCall, Company)
            .join(Company, Company.company_id == EarningsCall.company_id)
            .order_by(EarningsCall.call_date)
            .all()
        )
    finally:
        session.close()

    if not rows:
        logger.error("No earnings calls found in DB. Run the ingestion pipeline first.")
        return pd.DataFrame()

    logger.info(f"Found {len(rows)} earnings calls to label.")

    # ── Determine date range for bulk price download ──────────────────────────
    all_dates   = [row.call_date for _, row in rows]  # EarningsCall attr
    # Unpack correctly
    call_records = [(ec, co) for ec, co in rows]
    all_dates    = [ec.call_date for ec, co in call_records]

    start_dt = min(all_dates) - timedelta(days=5)
    end_dt   = max(all_dates) + timedelta(days=20)

    all_syms = TICKERS + [BENCHMARK]
    logger.info(f"Downloading prices for {all_syms} from {start_dt} to {end_dt} ...")

    prices_raw = yf.download(
        all_syms,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    # Flatten multi-level columns → {ticker: close_series}
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices = prices_raw["Close"]
    else:
        prices = prices_raw[["Close"]].rename(columns={"Close": BENCHMARK})

    # ── Build label rows ──────────────────────────────────────────────────────
    records = []
    for ec, co in call_records:
        ticker    = co.ticker
        call_date = ec.call_date

        row = {
            "call_id"    : ec.call_id,
            "ticker"     : ticker,
            "quarter"    : ec.quarter,
            "fiscal_year": ec.fiscal_year,
            "call_date"  : call_date,
        }

        for w in WINDOWS:
            stock_ret  = _get_return(ticker,    call_date, w, prices)
            bench_ret  = _get_return(BENCHMARK, call_date, w, prices)

            if stock_ret is not None and bench_ret is not None:
                ar = stock_ret - bench_ret
            else:
                ar = None

            row[f"return_{w}d"]          = round(stock_ret * 100, 4) if stock_ret is not None else None
            row[f"bench_return_{w}d"]    = round(bench_ret * 100, 4) if bench_ret is not None else None
            row[f"abnormal_return_{w}d"] = round(ar * 100, 4) if ar is not None else None

            # Label based on 1-day abnormal return (primary)
            if w == 1:
                if ar is None:
                    label = "NEUTRAL"
                elif ar > THRESHOLD_POS:
                    label = "POSITIVE"
                elif ar < THRESHOLD_NEG:
                    label = "NEGATIVE"
                else:
                    label = "NEUTRAL"
                row["label"] = label

        records.append(row)

    df = pd.DataFrame(records)
    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    logger.info(f"Label distribution: {label_counts}")
    logger.info(f"Calls with 1-day return: {df['return_1d'].notna().sum()} / {len(df)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    logger.success(f"Saved market labels → {OUT_PATH}")

    return df


if __name__ == "__main__":
    df = build_market_labels()
    if not df.empty:
        print(df[["ticker", "quarter", "fiscal_year", "call_date",
                   "return_1d", "abnormal_return_1d", "label"]].to_string(index=False))
