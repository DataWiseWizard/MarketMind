"""
Test database connection and verify schema.

Usage:
    python scripts/test_db.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import inspect, text
from src.db.models import Base
from src.db.session import engine


def test_connection() -> bool:
    """Test the database connection and list existing tables."""
    print("=" * 50)
    print("MarketMind — Database Connection Test")
    print("=" * 50)

    try:
        with engine.connect() as conn:
            # Basic connection test
            result = conn.execute(text("SELECT version()"))
            pg_version = result.scalar()
            print(f"\n  Connected to PostgreSQL")
            print(f"  Version : {pg_version}")

            # Current database
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.scalar()
            print(f"  Database: {db_name}")

            # List tables
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            if tables:
                print(f"\n  Found {len(tables)} table(s):")
                for table in sorted(tables):
                    columns = inspector.get_columns(table)
                    print(f"    - {table} ({len(columns)} columns)")
            else:
                print("\n  No tables found yet.")
                print("  Run:  alembic upgrade head")

            print("\n" + "=" * 50)
            print("  Connection test PASSED")
            print("=" * 50)
            return True

    except Exception as e:
        print(f"\n  Connection test FAILED: {e}")
        print("\n  Troubleshooting:")
        print("    1. Is PostgreSQL running?")
        print("    2. Does the database exist? Run: python scripts/create_db.py")
        print("    3. Is .env configured correctly?")
        print("=" * 50)
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
