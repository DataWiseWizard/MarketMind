"""
Tests for FastAPI REST Endpoints
"""

import unittest
from fastapi.testclient import TestClient
from src.api.main import app


class TestMarketMindAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        self.assertGreaterEqual(len(companies), 7)
        tickers = {c["ticker"] for c in companies}
        self.assertIn("NVDA", tickers)
        self.assertIn("AMD", tickers)

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
        self.assertGreater(len(data["predictions"]), 0)
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


if __name__ == "__main__":
    unittest.main()
