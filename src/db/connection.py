"""
MarketMind Database Connection & Session Management
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL
from src.db.models import Base
from loguru import logger


def get_engine(db_url: str = None):
    """Create and return a SQLAlchemy engine."""
    url = db_url or DATABASE_URL
    kwargs = {
        "echo": False,
    }
    if not url.startswith("sqlite"):
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)
    return engine


def get_session(engine=None):
    """Create and return a new database session."""
    if engine is None:
        engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database(engine=None):
    """Create all tables defined in models.py."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("✅ All database tables created successfully.")


def test_connection(db_url: str = None):
    """Test database connectivity."""
    try:
        engine = get_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                logger.info("✅ Database connection successful!")
                return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False
