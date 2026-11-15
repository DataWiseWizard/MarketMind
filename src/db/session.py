"""
MarketMind Database Session

Provides the SQLAlchemy engine and session factory.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.utils.config import settings

# Database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


def get_db():
    """
    Get a database session.

    Yields a session and ensures it is closed after use.
    Use as a FastAPI dependency or context manager.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
