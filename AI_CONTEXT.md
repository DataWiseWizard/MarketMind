PROJECT CONTEXT: MARKETMIND

You are now helping me build a final-year AI/ML major project called MarketMind.

PROJECT NAME:
MarketMind

PROJECT TITLE:
Automated Financial Earnings Call Analysis and Market Signal Generation Pipeline

PROJECT TYPE:
End-to-end AI/ML pipeline project

PROJECT GOAL:
Build a complete end-to-end system that automatically processes company earnings calls, extracts financial insights, analyzes sentiment, detects evasive answers, compares current statements with past promises, generates market reaction signals, and produces easy-to-understand reports.

IMPORTANT DISCLAIMER:
This project is for educational and research purposes only. It is not financial advice. It should not use real-money trading. If trading-related features are included, they must be paper trading only.

--------------------------------------------------
1. SIMPLE PROJECT SUMMARY
--------------------------------------------------

MarketMind takes an earnings call transcript or audio file and automatically turns it into structured financial intelligence.

The system should be able to:

1. Accept earnings call audio or transcript.
2. Convert audio to text if audio is provided.
3. Identify speakers such as CEO, CFO, analyst, and operator.
4. Extract financial metrics such as revenue, EPS, margins, guidance, and capex.
5. Analyze sentiment of executives and analysts.
6. Detect evasive or vague answers from management.
7. Compare current statements with historical earnings calls and filings.
8. Generate a research signal for possible market reaction.
9. Generate a simple report.
10. Show results on a dashboard.

--------------------------------------------------
2. CORE PROBLEM BEING SOLVED
--------------------------------------------------

Earnings calls are long and contain important financial information, but analyzing them manually is slow and difficult.

Problems MarketMind solves:

1. Earnings calls are too long to read or listen to fully.
2. Important financial details are scattered throughout the call.
3. It is hard to remember what management promised in previous quarters.
4. It is hard to detect evasive answers manually.
5. Manual comparison of current statements with past statements is time-consuming.
6. Investors and analysts need structured insights quickly.

--------------------------------------------------
3. MAIN FEATURES
--------------------------------------------------

MarketMind should include the following features:

A. Data Ingestion
- Upload or fetch earnings call transcript.
- Upload or fetch earnings call audio.
- Store company metadata.
- Store call date, quarter, fiscal year, ticker, and source.

B. Audio Processing
- Convert earnings call audio into timestamped text using Whisper or faster-whisper.
- Optional: speaker diarization using pyannote-audio or WhisperX.

C. Transcript Processing
- Clean transcript text.
- Split transcript into segments.
- Identify Q&A sections.
- Map speakers to roles such as CEO, CFO, analyst, operator.

D. Financial Metric Extraction
- Extract metrics such as:
  - revenue
  - revenue growth
  - EPS
  - gross margin
  - operating margin
  - net income
  - free cash flow
  - capital expenditure
  - guidance
- Store extracted values in structured format.
- Keep the original quote from the transcript for verification.

E. Sentiment Analysis
- Use FinBERT or another financial sentiment model.
- Calculate sentiment for each segment.
- Aggregate sentiment by speaker role.
- Compare sentiment with previous calls.

F. Evasion Detection
- Identify analyst questions.
- Identify executive answers.
- Detect vague or evasive responses.
- Generate an evasion score.

G. Historical Accountability / RAG
- Store historical transcripts and filings.
- Embed text chunks into a vector database.
- Retrieve relevant historical statements.
- Compare current statements with historical statements.
- Detect:
  - guidance raised
  - guidance lowered
  - broken promises
  - contradictions
  - avoided topics

H. Market Reaction Model
- Create features from sentiment, evasion, financial metrics, guidance changes, and market data.
- Train a model such as LightGBM or XGBoost.
- Predict next-day or short-term market reaction.
- Use time-based validation only.
- Do not claim guaranteed stock prediction.

I. Report Generation
- Generate a simple report containing:
  - call summary
  - key financial metrics
  - sentiment analysis
  - evasion analysis
  - guidance changes
  - accountability findings
  - prediction signal
  - risk disclaimer

J. Dashboard
- Build a simple dashboard using Streamlit.
- Show processed calls.
- Show sentiment and evasion scores.
- Show accountability findings.
- Show prediction signal.
- Allow report download.

--------------------------------------------------
4. TECH STACK
--------------------------------------------------

Use the following technology stack unless there is a strong reason to change it:

Programming:
- Python

Data Processing:
- pandas
- NumPy

Database:
- PostgreSQL
- SQLAlchemy
- Alembic

Audio Transcription:
- Whisper or faster-whisper
- Optional: WhisperX

Speaker Diarization:
- pyannote-audio
- Optional: WhisperX

NLP / ML:
- Hugging Face Transformers
- spaCy
- FinBERT
- Sentence-Transformers

LLM:
- Ollama or vLLM
- Optional local models: Mistral, Llama 3, Qwen

RAG:
- LangChain or LlamaIndex
- Qdrant or Chroma for local development
- Optional: Pinecone

Orchestration:
- Apache Airflow

Prediction Model:
- LightGBM or XGBoost
- scikit-learn
- SHAP for explainability

Backtesting:
- pandas-based custom backtester
- Optional: vectorbt or backtrader

Market Data:
- yfinance
- Optional: Alpha Vantage, Finnhub, Polygon

API:
- FastAPI
- Pydantic
- Uvicorn

Dashboard:
- Streamlit

Report Generation:
- ReportLab or WeasyPrint
- Jinja2

MLOps:
- MLflow
- Evidently AI
- Prometheus
- Grafana

Testing:
- pytest

Containerization:
- Docker
- Docker Compose

CI/CD:
- GitHub Actions

--------------------------------------------------
5. END-TO-END PIPELINE
--------------------------------------------------

The full pipeline should be:

Input
→ Data ingestion
→ Audio transcription
→ Speaker diarization
→ Speaker role mapping
→ Transcript segmentation
→ Financial metric extraction
→ Sentiment analysis
→ Evasion detection
→ Historical RAG accountability
→ Feature engineering
→ Market reaction prediction
→ Report generation
→ Dashboard display
→ Monitoring and logging

--------------------------------------------------
6. MVP SCOPE
--------------------------------------------------

The MVP should be simple and achievable.

MVP includes:

1. Transcript upload only.
2. Transcript segmentation.
3. Financial metric extraction.
4. Sentiment analysis.
5. Evasion detection.
6. Simple prediction signal.
7. FastAPI backend.
8. Streamlit dashboard.
9. Basic report generation.

MVP does not require:

1. Real-time processing.
2. Full audio diarization.
3. Live trading.
4. Complex deep learning models.
5. Large-scale distributed deployment.

After MVP works, add:

1. Audio transcription.
2. Speaker diarization.
3. Historical RAG.
4. Backtesting.
5. MLflow monitoring.
6. Automated retraining.
7. Docker deployment.

--------------------------------------------------
7. DATABASE DESIGN
--------------------------------------------------

Main tables:

companies
- company_id
- ticker
- company_name
- sector
- exchange

earnings_calls
- call_id
- company_id
- quarter
- fiscal_year
- call_date
- audio_path
- transcript_path
- status

transcript_segments
- segment_id
- call_id
- start_time
- end_time
- speaker_label
- speaker_role
- segment_type
- text

extracted_metrics
- metric_id
- call_id
- metric_name
- value
- unit
- currency
- period
- is_guidance
- speaker_role
- quote
- confidence

sentiment_scores
- sentiment_id
- call_id
- segment_id
- speaker_role
- positive_score
- neutral_score
- negative_score
- compound_score

evasion_scores
- evasion_id
- call_id
- question_segment_id
- answer_segment_id
- evasion_score
- is_evasive
- reason

accountability_findings
- finding_id
- call_id
- finding_type
- severity
- current_statement
- historical_statement
- historical_source
- explanation

signals
- signal_id
- call_id
- predicted_direction
- probability
- expected_return
- confidence
- model_version

model_runs
- run_id
- model_name
- version
- metrics
- created_at

--------------------------------------------------
8. PROJECT FOLDER STRUCTURE
--------------------------------------------------

Use this folder structure:

marketmind/
├── data/
│   ├── raw/
│   │   ├── audio/
│   │   ├── transcripts/
│   │   ├── filings/
│   │   └── market_data/
│   ├── processed/
│   │   ├── transcripts/
│   │   ├── features/
│   │   └── embeddings/
│   └── external/
├── src/
│   ├── ingestion/
│   ├── processing/
│   ├── rag/
│   ├── features/
│   ├── models/
│   ├── backtesting/
│   ├── api/
│   ├── reporting/
│   ├── db/
│   └── utils/
├── dashboard/
├── notebooks/
├── tests/
├── mlflow/
├── reports/
├── docker/
├── requirements.txt
├── README.md
└── .env.example

--------------------------------------------------
9. EXPECTED FINAL DELIVERABLES
--------------------------------------------------

By the end of the project, I need:

1. Working end-to-end pipeline.
2. Transcript ingestion module.
3. Optional audio transcription module.
4. Financial metric extraction module.
5. Sentiment analysis module.
6. Evasion detection module.
7. Historical RAG accountability module.
8. Market reaction prediction model.
9. FastAPI backend.
10. Streamlit dashboard.
11. Report generator.
12. MLflow experiment tracking.
13. Basic monitoring.
14. README documentation.
15. Final-year project report.
16. Presentation content.
17. Demo script.

--------------------------------------------------
10. DESIGN RULES
--------------------------------------------------

When helping me build this project, follow these rules:

1. Keep explanations simple enough for a non-technical person to understand.
2. Build the MVP first before advanced features.
3. Do not use real trading.
4. Always include disclaimers that this is not financial advice.
5. Prefer open-source and local tools where possible.
6. Avoid paid APIs unless necessary.
7. Provide runnable code examples.
8. Provide file paths and folder structure.
9. Provide commands to run each step.
10. Use modular code.
11. Use validation for all extracted data.
12. Avoid data leakage in time-series models.
13. Use walk-forward validation for market prediction.
14. Keep the project realistic for a final-year student.
15. If a task is complex, break it into small steps.

--------------------------------------------------
11. CURRENT STATUS
--------------------------------------------------

I am starting this project fresh.

Current status:

- No code has been written yet.
- No repository has been created yet.
- No database has been created yet.
- No dataset has been fully collected yet.
- The project is at the planning and setup stage.

--------------------------------------------------
12. WHAT I WANT FROM YOU
--------------------------------------------------

Help me build MarketMind step by step.

When I ask for implementation, provide:

1. Clear explanation.
2. Required folder structure.
3. Required files.
4. Complete code.
5. Commands to run.
6. Expected output.
7. Common errors and fixes.
8. Next step.

When I ask for documentation, provide:

1. Simple language.
2. Copy-paste-ready format.
3. Clear headings.
4. Non-technical explanations.
5. Tables where useful.
6. Glossary if needed.

--------------------------------------------------
13. FIRST TASK
--------------------------------------------------

Your first task is to help me start the project properly.

Begin by giving me:

1. A simple project setup checklist.
2. The exact requirements.txt file.
3. The exact folder structure to create.
4. A README.md file for the project.
5. A .env.example file.
6. Instructions to create a Python virtual environment.
7. Instructions to install dependencies.
8. Instructions to create the PostgreSQL database.
9. A simple script to test the database connection.
10. The next step after setup is complete.

If anything is unclear, ask me clarifying questions before generating the full setup.

If you understand this project context, reply:

"MarketMind project context received. I am ready to help you build the project step by step."