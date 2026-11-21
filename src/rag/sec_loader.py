"""
MarketMind — Stage 2E: SEC EDGAR Loader
========================================
Fetches 10-Q and 10-K filings from the SEC EDGAR public REST API.
No API key required. Rate-limited to 10 req/sec per EDGAR fair-use policy.

Output: data/raw/sec_filings/{ticker}/{form_type}_{period}.txt

CIK numbers for tracked companies:
    NVDA  → 0001045810
    AMD   → 0000002488
    INTC  → 0000050863
    QCOM  → 0000804328
    TSM   → 0001046179
    ASML  → 0000937556
    MRVL  → 0001058057
"""

import sys
import time
import re
from pathlib import Path
from typing import Optional

import requests
from loguru import logger
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "sec_filings"

# EDGAR requires a user-agent identifying yourself
HEADERS = {
    "User-Agent": "MarketMind Academic Research Project rudy@marketmind.edu",
    "Accept-Encoding": "gzip, deflate",
}

COMPANY_CIKS = {
    "NVDA" : "0001045810",
    "AMD"  : "0000002488",
    "INTC" : "0000050863",
    "QCOM" : "0000804328",
    "TSM"  : "0001046179",
    "ASML" : "0000937556",
    "MRVL" : "0001058057",
}

FORM_TYPES = ["10-Q", "10-K"]
MAX_FILINGS_PER_TYPE = 8   # last 2 years of 10-Qs + last 2 10-Ks


def _get(url: str, retries: int = 3) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 429:
                logger.warning("Rate limited — sleeping 10s...")
                time.sleep(10)
            else:
                logger.warning(f"HTTP {resp.status_code} for {url}")
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{retries}): {e}")
        time.sleep(1)
    return None


def get_filing_list(cik: str, form_type: str) -> list[dict]:
    """Return list of recent filings for a CIK + form_type."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _get(url)
    if not resp:
        logger.error(f"Could not fetch submission list for CIK {cik}")
        return []

    data    = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    accns   = filings.get("accessionNumber", [])
    primary = filings.get("primaryDocument", [])

    results = []
    for form, date_, accn, prim in zip(forms, dates, accns, primary):
        if form == form_type:
            results.append({
                "form_type"       : form,
                "filing_date"     : date_,
                "accession_number": accn,
                "primary_doc"     : prim,
            })

    # Return most recent N filings
    return results[:MAX_FILINGS_PER_TYPE]


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags, returning clean prose text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script / style / table-of-contents noise
    for tag in soup(["script", "style", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Collapse excessive whitespace
    lines  = [ln.strip() for ln in text.splitlines()]
    lines  = [ln for ln in lines if len(ln) > 20]   # drop one-word lines
    text   = "\n".join(lines)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_filing(ticker: str, cik: str, filing: dict) -> Optional[str]:
    """Download a single filing document and return plain text."""
    accn_no = filing["accession_number"].replace("-", "")
    prim    = filing["primary_doc"]

    # Construct document URL
    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accn_no}/{prim}"
    )

    resp = _get(doc_url)
    if not resp:
        logger.warning(f"Could not download {filing['form_type']} for {ticker}: {doc_url}")
        return None

    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type or prim.endswith(".htm") or prim.endswith(".html"):
        return _extract_text_from_html(resp.text)
    else:
        return resp.text


def fetch_all_filings(tickers: Optional[list] = None):
    """Fetch 10-Q and 10-K filings for all tracked companies."""
    if tickers is None:
        tickers = list(COMPANY_CIKS.keys())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for ticker in tickers:
        cik = COMPANY_CIKS.get(ticker)
        if not cik:
            logger.warning(f"No CIK for {ticker} — skipping.")
            continue

        ticker_dir = OUT_DIR / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        summary[ticker] = {}

        for form_type in FORM_TYPES:
            logger.info(f"Fetching {form_type} list for {ticker} (CIK {cik})...")
            filings = get_filing_list(cik, form_type)
            logger.info(f"  → {len(filings)} filings found")

            saved = 0
            for filing in filings:
                date_   = filing["filing_date"]
                # Sanitise filename
                safe_dt = date_.replace("-", "")
                out_file = ticker_dir / f"{form_type.replace('-','')}_{safe_dt}.txt"

                if out_file.exists():
                    logger.debug(f"  Skip (exists): {out_file.name}")
                    saved += 1
                    continue

                logger.info(f"  Downloading {form_type} filed {date_}...")
                text = download_filing(ticker, cik, filing)
                time.sleep(0.15)   # ~6–7 req/s, well under EDGAR limit

                if text:
                    out_file.write_text(text, encoding="utf-8")
                    saved += 1
                    logger.success(f"  Saved → {out_file.name}  ({len(text):,} chars)")

            summary[ticker][form_type] = saved

    logger.success("SEC filing download complete.")
    for ticker, counts in summary.items():
        logger.info(f"  {ticker}: {counts}")

    return summary


if __name__ == "__main__":
    fetch_all_filings()
