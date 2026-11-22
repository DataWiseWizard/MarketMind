"""
Tests for ML Model (LightGBM) & SHAP Explainer Modules
"""

import json
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.explainer import predict_with_shap, predict_batch, _load_model


class TestModelAndExplainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = Path("models/lightgbm_marketmind_v1.pkl")
        cls.features_path = Path("models/feature_columns.json")
        cls.metrics_path = Path("models/model_metrics.json")

    def test_model_artifacts_exist(self):
        """Assert that trained model and metadata artifacts exist."""
        self.assertTrue(self.model_path.exists(), "Model pickle file missing")
        self.assertTrue(self.features_path.exists(), "Feature columns JSON missing")
        self.assertTrue(self.metrics_path.exists(), "Model metrics JSON missing")

    def test_feature_columns_contract(self):
        """Assert exactly 13 feature columns are expected by the trained model."""
        with open(self.features_path) as f:
            features = json.load(f)
        self.assertEqual(len(features), 13)
        expected = [
            "sent_ceo", "sent_cfo", "sent_analyst", "sent_ceo_delta", "sent_cfo_delta",
            "evasion_mean", "evasion_max", "evasive_count", "revenue_qoq_pct",
            "margin_qoq_pp", "discrepancy_count", "momentum_30d", "volatility_30d"
        ]
        self.assertEqual(sorted(features), sorted(expected))

    def test_single_prediction_with_shap(self):
        """Verify single prediction outputs correct schema and valid probabilities."""
        sample_features = {
            "sent_ceo": 0.65,
            "sent_cfo": 0.50,
            "sent_analyst": 0.20,
            "sent_ceo_delta": 0.10,
            "sent_cfo_delta": 0.05,
            "evasion_mean": 0.25,
            "evasion_max": 0.40,
            "evasive_count": 1,
            "revenue_qoq_pct": 8.5,
            "margin_qoq_pp": 1.2,
            "discrepancy_count": 0,
            "momentum_30d": 5.0,
            "volatility_30d": 25.0,
        }
        res = predict_with_shap(sample_features)

        self.assertIn("predicted_label", res)
        self.assertIn(res["predicted_label"], ["POSITIVE", "NEUTRAL", "NEGATIVE"])

        # Check probabilities sum to ~1.0
        probs = res["probabilities"]
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        # Check SHAP values structure
        shap_vals = res["shap_values"]
        self.assertEqual(len(shap_vals), 13)
        self.assertTrue(all("feature" in s and "shap_value" in s for s in shap_vals))

    def test_batch_prediction_with_missing_columns(self):
        """Assert predict_batch handles dataframes with missing columns without crashing."""
        df_incomplete = pd.DataFrame([
            {"sent_ceo": 0.70, "revenue_qoq_pct": 12.0},
            {"sent_ceo": -0.40, "revenue_qoq_pct": -5.0},
        ])
        res = predict_batch(df_incomplete)

        self.assertEqual(len(res), 2)
        self.assertIn("predicted_label", res.columns)
        self.assertIn("confidence", res.columns)
        self.assertTrue(all(p in ["POSITIVE", "NEUTRAL", "NEGATIVE"] for p in res["predicted_label"]))


if __name__ == "__main__":
    unittest.main()
