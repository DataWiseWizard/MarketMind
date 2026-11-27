---
title: MarketMind Financial Intelligence API
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 MarketMind

**Automated Financial Earnings Call Analysis and Market Signal Generation Pipeline**

[![MarketMind CI](https://github.com/DataWiseWizard/MarketMind/actions/workflows/ci.yml/badge.svg)](https://github.com/DataWiseWizard/MarketMind/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Pipeline-9C27B0)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is not financial advice. Do not use for real-money trading.

---

## 📖 What is MarketMind?

MarketMind is an end-to-end AI/ML pipeline that automatically processes company earnings calls, extracts financial insights, analyzes sentiment, detects evasive answers, compares current statements with past promises, generates market reaction signals, and produces easy-to-understand reports.

### What it does (in simple terms):

| Step | What Happens |
|------|-------------|
| 1. **Listening** | Reads the transcript or listens to the audio |
| 2. **Identifying speakers** | Figures out who is speaking: CEO, CFO, analyst |
| 3. **Finding numbers** | Extracts revenue, EPS, margins, guidance |
| 4. **Understanding mood** | Checks if language is positive, negative, or neutral |
| 5. **Detecting evasion** | Flags vague answers to direct questions |
| 6. **Checking past promises** | Compares current vs. previous quarter statements |
| 7. **Creating a signal** | Produces a research signal (positive/negative/neutral) |
| 8. **Creating a report** | Writes a clean, readable report |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.14+ (tested on Python 3.14.8)
- PostgreSQL 14+ (or built-in SQLite for zero-config development)
- Git

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/DataWiseWizard/MarketMind.git
cd MarketMind

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
copy .env.example .env
# Edit .env with your database credentials

# 6. Verify database connection & schema
python scripts/verify_db.py

# 7. Run the end-to-end analysis pipeline
python src/pipeline.py --all

# 8. Launch the API & Dashboard (in separate terminals)
uvicorn src.api.main:app --reload
streamlit run dashboard/app.py

# 9. Run test suite
pytest -q
```

---

## 📁 Project Structure

```
MarketMind/
├── data/
│   ├── raw/                    # Raw input data (transcripts, filings, audio, market data)
│   ├── processed/              # Cleaned transcripts & ML feature tables
│   ├── vector_store/           # ChromaDB persistent vector index
│   └── external/               # Third-party datasets
├── src/
│   ├── ingestion/              # Transcript PDF parser & speaker diarization
│   ├── processing/             # Metric extraction, FinBERT/Lexicon sentiment, evasion detection
│   ├── rag/                    # ChromaDB vector store, SEC 10-Q loader & Gemini AI agent
│   ├── features/               # Feature engineering & market returns
│   ├── models/                 # LightGBM trainer, SHAP explainer & drift monitoring
│   ├── reporting/              # Signal generation & Markdown reports
│   ├── api/                    # FastAPI backend (main.py)
│   ├── db/                     # SQLAlchemy ORM models, connection & seeding
│   ├── config.py               # Central configuration
│   └── pipeline.py             # Master pipeline runner
├── dashboard/                  # Streamlit workstation UI & components
├── models/                     # Trained LightGBM model, feature schema, drift reports
├── scripts/                    # Utility scripts (verify_db.py)
├── notebooks/                  # Jupyter notebooks for exploration
├── tests/                      # Unit & integration tests (pytest)
├── mlflow/                     # MLflow experiment tracking
├── reports/                    # Generated quarter-by-quarter reports
├── docker/                     # Service-specific Dockerfiles & compose config
├── Dockerfile                  # Production container image
├── render.yaml                 # Render Blueprint deployment config
├── pyproject.toml              # Project metadata & tool config
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

---

## ☁️ Cloud Deployment Architecture ($0 Free Tier)

MarketMind is configured for zero-cost cloud deployment using **Render** for the backend and **Streamlit Community Cloud** for the frontend:

```
┌─────────────────────────────────┐        REST JSON        ┌────────────────────────────────────┐
│   Streamlit Community Cloud     │ ──────────────────────> │           Render Web Service       │
│   (Frontend Terminal UI)        │ <────────────────────── │         (FastAPI Docker API)       │
│   • Entry: dashboard/app.py     │        HTTPS / TLS      │   • Port: 10000 (Dynamic $PORT)    │
│   • Free Hosting (1 GB RAM)     │                         │   • 100% Free Tier (512 MB RAM)    │
│   • Installs lightweight reqs   │                         │   • Built via render.yaml or Docker│
└─────────────────────────────────┘                         └────────────────────────────────────┘
```

### 1. Backend: Render (FastAPI Web Service - 100% Free)
1. Go to [dashboard.render.com](https://dashboard.render.com/) and click **New** $\rightarrow$ **Blueprint** (or **Web Service**).
2. Connect your GitHub repository: `DataWiseWizard/MarketMind`.
3. Render automatically detects [render.yaml](render.yaml) or your `Dockerfile`.
4. Enter your environment secret:
   - `GEMINI_API_KEY`: *(From [Google AI Studio](https://aistudio.google.com/))*
5. Click **Apply**. Render builds the service and generates your live URL:
   `https://marketmind-api.onrender.com`

### 2. Frontend: Streamlit Community Cloud (100% Free)
1. Deploy on [share.streamlit.io](https://share.streamlit.io) from your GitHub repo.
2. Set **Main file path** to `dashboard/app.py`.
3. Under **Advanced settings -> Secrets**, set:
   ```toml
   MARKETMIND_API_URL = "https://marketmind-api.onrender.com"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
4. Streamlit installs `dashboard/requirements.txt` instantly and links to your live Render backend.

---

## 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python |
| Data | pandas, NumPy |
| Database | SQLite (default), PostgreSQL, SQLAlchemy, ChromaDB |
| NLP/ML | Transformers, FinBERT, spaCy, Sentence-Transformers |
| Prediction | LightGBM, scikit-learn, SHAP |
| Market Data | yfinance |
| API | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit |
| Reports | ReportLab, Jinja2 |
| Testing | pytest |

---

## 📊 Pipeline

```
Input → Ingestion → Transcription → Speaker ID → Segmentation
→ Metric Extraction → Sentiment Analysis → Evasion Detection
→ Historical RAG → Feature Engineering → Prediction
→ Report Generation → Dashboard → Monitoring
```

---

## 👥 Team

- **Rudraksha ([DataWiseWizard](https://github.com/DataWiseWizard))** - Core Contributor
- **Gourab ([gourab007-ai](https://github.com/gourab007-ai))** - Core Contributor

---

## 📄 License

This project is for academic use only.

---

> **Research output only.** Signals are based on historical patterns. Not financial advice.