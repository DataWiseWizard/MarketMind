# MarketMind

**Automated Financial Earnings Call Analysis and Market Signal Generation Pipeline**

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. No real-money trading is involved.

## Overview

MarketMind is an end-to-end AI/ML pipeline that automatically processes company earnings call transcripts, extracts financial insights, analyzes sentiment, detects evasive answers, compares statements with historical data, and generates market reaction signals.

## Features

- **PDF Transcript Ingestion** — Parse earnings call transcripts from PDF files
- **Financial Metric Extraction** — Extract revenue, EPS, margins, guidance, capex using local LLM
- **Sentiment Analysis** — FinBERT-powered financial sentiment scoring
- **Evasion Detection** — Identify vague or evasive executive responses
- **Historical Accountability (RAG)** — Compare current statements with past promises
- **Market Reaction Prediction** — LightGBM model with SHAP explainability
- **Report Generation** — Automated PDF reports per earnings call
- **Dashboard** — Interactive Streamlit interface

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.14 |
| Database | PostgreSQL 18 + SQLAlchemy + Alembic |
| NLP/ML | PyTorch, Transformers (FinBERT), spaCy, scikit-learn |
| LLM | Ollama (local — Mistral / Llama) |
| RAG | LangChain + Qdrant |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| MLOps | MLflow |
| Prediction | LightGBM + SHAP |

## Project Structure

```
MarketMind/
├── data/
│   ├── transcripts/          # Raw PDF transcripts (by year)
│   ├── raw/                  # Other raw data (audio, filings, market data)
│   ├── processed/            # Processed outputs
│   └── external/             # External datasets
├── src/
│   ├── ingestion/            # Data loading and PDF parsing
│   ├── processing/           # Transcript segmentation, sentiment, evasion
│   ├── rag/                  # Historical accountability (vector DB)
│   ├── features/             # Feature engineering, metric extraction
│   ├── models/               # ML model training and prediction
│   ├── backtesting/          # Strategy backtesting
│   ├── api/                  # FastAPI backend
│   ├── reporting/            # PDF report generation
│   ├── db/                   # Database models and session
│   └── utils/                # Config, logging, constants
├── dashboard/                # Streamlit app
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Test suite
├── alembic/                  # Database migrations
├── scripts/                  # Utility scripts
├── mlflow/                   # MLflow tracking
├── reports/                  # Generated reports
├── docker/                   # Docker configuration
├── requirements.txt
├── alembic.ini
├── .env.example
└── README.md
```

## Setup

### Prerequisites

- Python 3.14+
- PostgreSQL 18+
- Ollama (for local LLM)
- NVIDIA GPU with CUDA support (recommended)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # Linux/Mac
   ```

2. **Install PyTorch with CUDA:**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   copy .env.example .env          # Windows
   cp .env.example .env            # Linux/Mac
   ```
   Edit `.env` with your PostgreSQL credentials.

5. **Create database:**
   ```bash
   python scripts/create_db.py
   ```

6. **Run migrations:**
   ```bash
   alembic upgrade head
   ```

7. **Test connection:**
   ```bash
   python scripts/test_db.py
   ```

## Current Data

| Company | Ticker | Transcripts |
|---|---|---|
| NVIDIA Corporation | NVDA | Q1–Q4 2023, Q1–Q4 2026, Q1–Q2 2027 |

## License

This project is for academic use only (final-year project).
