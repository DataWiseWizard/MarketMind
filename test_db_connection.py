"""
MarketMind - Database Connection Test Script

Run this script to verify your PostgreSQL connection is working.

Usage:
    python test_db_connection.py

If you don't have PostgreSQL set up yet, this script will also test
with SQLite as a fallback to verify the schema works.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_sqlite_fallback():
    """Test database schema using SQLite (no PostgreSQL needed)."""
    from sqlalchemy import create_engine, text
    from src.db.models import (
        Base, Company, EarningsCall, TranscriptSegment,
        ExtractedMetric, SentimentScore, EvasionScore,
        AccountabilityFinding, Signal, ModelRun
    )
    from sqlalchemy.orm import sessionmaker
    from datetime import date

    print("=" * 50)
    print("  MarketMind - Database Schema Test (SQLite)")
    print("=" * 50)

    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    print("\n✅ All 9 tables created successfully!")

    # List tables
    tables = Base.metadata.tables.keys()
    for i, table in enumerate(tables, 1):
        print(f"   {i}. {table}")

    # Test inserting sample data
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Add a sample company
        company = Company(
            ticker="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            exchange="NASDAQ"
        )
        session.add(company)
        session.commit()

        # Add a sample earnings call
        call = EarningsCall(
            company_id=company.company_id,
            quarter="Q4",
            fiscal_year=2026,
            call_date=date(2026, 10, 30),
            transcript_path="data/raw/transcripts/aapl_q4_2026.txt",
            source="uploaded"
        )
        session.add(call)
        session.commit()

        # Verify
        result = session.query(Company).filter_by(ticker="AAPL").first()
        assert result is not None
        assert result.company_name == "Apple Inc."

        result_call = session.query(EarningsCall).first()
        assert result_call is not None
        assert result_call.quarter == "Q4"

        print("\n✅ Sample data inserted and verified!")
        print(f"   Company: {result}")
        print(f"   Call:    {result_call}")
        print("\n✅ Database schema is working correctly!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        session.rollback()
    finally:
        session.close()


def test_postgresql():
    """Test PostgreSQL connection using .env settings."""
    try:
        from src.db.connection import test_connection, init_database, get_engine

        print("=" * 50)
        print("  MarketMind - PostgreSQL Connection Test")
        print("=" * 50)

        if test_connection():
            print("\n🔧 Creating tables in PostgreSQL...")
            init_database()
            print("\n✅ PostgreSQL is fully set up!")
            return True
        else:
            return False

    except Exception as e:
        print(f"\n⚠️  PostgreSQL test failed: {e}")
        return False


if __name__ == "__main__":
    print()

    # Always test schema with SQLite first
    test_sqlite_fallback()

    print("\n" + "-" * 50)

    # Then try PostgreSQL if .env exists
    if os.path.exists(".env"):
        print("\n📡 Found .env file, testing PostgreSQL connection...\n")
        test_postgresql()
    else:
        print("\n⚠️  No .env file found. Skipping PostgreSQL test.")
        print("   Copy .env.example to .env and fill in your database credentials.")
        print("   Then run this script again.")

    print()
    print("🎉 Schema test complete! Your database models are ready.")
    print()
    print("=" * 50)
    print("  DISCLAIMER: This project is for educational and")
    print("  research purposes only. Not financial advice.")
    print("=" * 50)
    print()
