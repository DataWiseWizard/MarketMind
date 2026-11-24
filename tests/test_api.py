"""
Tests for FastAPI REST Endpoints
"""

import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app


class TestMarketMindAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.db.connection import get_session, init_database
        from src.db.models import Company, EarningsCall, CallStatus
        from datetime import date

        init_database()
        session = get_session()
        try:
            # Seed minimum test fixtures if DB is empty (e.g. fresh CI environment)
            if session.query(Company).count() == 0:
                sample_co = Company(
                    ticker="NVDA",
                    company_name="NVIDIA Corporation",
                    sector="Semiconductors",
                    exchange="NASDAQ",
                )
                session.add(sample_co)
                session.commit()
                session.refresh(sample_co)

                sample_call = EarningsCall(
                    company_id=sample_co.company_id,
                    quarter="Q4",
                    fiscal_year=2024,
                    call_date=date(2024, 11, 20),
                    status=CallStatus.COMPLETED,
                    source="test",
                )
                session.add(sample_call)
                session.commit()
        finally:
            session.close()

        cls.client = TestClient(app)

    def test_health_check(self):
        """GET /health should return 200 with service information."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("disclaimer", data)

    def test_list_companies(self):
        """GET /api/companies should return list of tracked companies."""
        resp = self.client.get("/api/companies")
        self.assertEqual(resp.status_code, 200)
        companies = resp.json()
        self.assertIsInstance(companies, list)
        self.assertGreater(len(companies), 0)
        tickers = {c["ticker"] for c in companies}
        self.assertIn("NVDA", tickers)

    def test_list_calls(self):
        """GET /api/calls should return list of earnings call summaries."""
        resp = self.client.get("/api/calls")
        self.assertEqual(resp.status_code, 200)
        calls = resp.json()
        self.assertIsInstance(calls, list)
        self.assertGreater(len(calls), 0)
        first_call = calls[0]
        self.assertIn("call_id", first_call)
        self.assertIn("quarter", first_call)
        self.assertIn("fiscal_year", first_call)

    def test_predictions_endpoint(self):
        """GET /api/predictions should return LightGBM batch predictions."""
        resp = self.client.get("/api/predictions")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("predictions", data)
        if len(data["predictions"]) > 0:
            sample = data["predictions"][0]
            self.assertIn("predicted_label", sample)
            self.assertIn(sample["predicted_label"], ["POSITIVE", "NEUTRAL", "NEGATIVE"])

    def test_call_signal_with_shap(self):
        """GET /api/signals/{call_id} should return signal and SHAP breakdown."""
        resp = self.client.get("/api/signals/1")
        if resp.status_code == 200:
            data = resp.json()
            self.assertIn("predicted_signal", data)
            self.assertIn("shap", data)
        else:
            self.assertEqual(resp.status_code, 404)

    def test_monitoring_drift_endpoint(self):
        """GET /api/monitoring/drift should return drift metrics."""
        resp = self.client.get("/api/monitoring/drift")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("overall_status", data)
        self.assertIn("feature_metrics", data)
        self.assertIn("prediction_drift", data)

    def test_monitoring_mlflow_endpoint(self):
        """GET /api/monitoring/mlflow should return tracking registry metadata."""
        resp = self.client.get("/api/monitoring/mlflow")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("tracking_uri", data)
        self.assertIn("experiment_name", data)
        self.assertEqual(data["experiment_name"], "marketmind")

    @patch("src.models.trainer.run_training")
    @patch("src.models.monitoring.DriftMonitor")
    def test_monitoring_retrain_endpoint(self, mock_drift_cls, mock_run_train):
        """POST /api/monitoring/retrain should trigger walk-forward retraining."""
        mock_run_train.return_value = (None, [], {"accuracy": 0.85})
        mock_inst = mock_drift_cls.return_value
        mock_inst.run_drift_analysis.return_value = {"overall_status": "HEALTHY"}

        resp = self.client.post("/api/monitoring/retrain", json={"train_years": [2023, 2024]})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("drift_report", data)
        self.assertIn("metrics", data)


if __name__ == "__main__":
    unittest.main()
