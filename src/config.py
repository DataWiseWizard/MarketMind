"""
MarketMind Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---- Project Paths ----
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ---- Database ----
# Defaults to SQLite in data/marketmind.db if DATABASE_URL is not set
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{PROJECT_ROOT / 'data' / 'marketmind.db'}"
)

# ---- API ----
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ---- MLflow ----
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "marketmind")

# ---- Models ----
FINBERT_MODEL = "ProsusAI/finbert"
SENTENCE_MODEL = "all-MiniLM-L6-v2"

# ---- Disclaimer ----
DISCLAIMER = (
    "This project is for educational and research purposes only. "
    "Not financial advice. Do not use for real trading."
)
