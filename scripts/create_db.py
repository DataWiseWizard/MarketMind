"""
Create the marketmind_db database in PostgreSQL.

Usage:
    python scripts/create_db.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def create_database() -> bool:
    """Create the MarketMind database if it doesn't exist."""
    pg_user = os.getenv("PG_USER", "postgres")
    pg_password = os.getenv("PG_PASSWORD", "password")
    pg_host = os.getenv("PG_HOST", "localhost")
    pg_port = os.getenv("PG_PORT", "5432")
    db_name = os.getenv("PG_DATABASE", "marketmind_db")

    print(f"Connecting to PostgreSQL at {pg_host}:{pg_port} as '{pg_user}'...")

    try:
        # Connect to the default 'postgres' database
        conn = psycopg2.connect(
            dbname="postgres",
            user=pg_user,
            password=pg_password,
            host=pg_host,
            port=pg_port,
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if database already exists
        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        )

        if cur.fetchone():
            print(f"Database '{db_name}' already exists. Skipping creation.")
        else:
            cur.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully.")

        cur.close()
        conn.close()
        return True

    except psycopg2.OperationalError as e:
        print(f"\nConnection FAILED: {e}")
        print("Make sure:")
        print("  1. PostgreSQL service is running")
        print("  2. Your .env has the correct PG_USER and PG_PASSWORD")
        print("  3. The user has permission to create databases")
        return False

    except Exception as e:
        print(f"\nError: {e}")
        return False


if __name__ == "__main__":
    success = create_database()
    sys.exit(0 if success else 1)
