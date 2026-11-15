"""
MarketMind Configuration

Centralized configuration loaded from .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root directory (3 levels up from this file)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env from project root
load_dotenv(ROOT_DIR / ".env")


class Settings:
    """Application settings loaded from environment variables."""

    # Project
    PROJECT_NAME: str = "MarketMind"
    ROOT_DIR: Path = ROOT_DIR

    # Database — individual components (used by create_db.py)
    PG_USER: str = os.getenv("PG_USER", "postgres")
    PG_PASSWORD: str = os.getenv("PG_PASSWORD", "password")
    PG_HOST: str = os.getenv("PG_HOST", "localhost")
    PG_PORT: str = os.getenv("PG_PORT", "5432")
    PG_DATABASE: str = os.getenv("PG_DATABASE", "marketmind_db")

    # Database — full connection URL (used by SQLAlchemy / Alembic)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"postgresql://{os.getenv('PG_USER', 'postgres')}:"
        f"{os.getenv('PG_PASSWORD', 'password')}@"
        f"{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/"
        f"{os.getenv('PG_DATABASE', 'marketmind_db')}",
    )

    # Ollama (Local LLM)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

    # Data Paths
    TRANSCRIPT_DIR: Path = ROOT_DIR / os.getenv("TRANSCRIPT_DIR", "data/transcripts")
    PROCESSED_DIR: Path = ROOT_DIR / os.getenv("PROCESSED_DIR", "data/processed")
    REPORTS_DIR: Path = ROOT_DIR / "reports"

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))

    # Dashboard
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))

    # MLflow
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{ROOT_DIR / 'mlflow' / 'mlflow.db'}",
    )

    # Company defaults (NVIDIA single-company MVP)
    DEFAULT_TICKER: str = "NVDA"
    DEFAULT_COMPANY: str = "NVIDIA Corporation"
    DEFAULT_SECTOR: str = "Technology"
    DEFAULT_EXCHANGE: str = "NASDAQ"


# Singleton instance — import this everywhere
settings = Settings()
