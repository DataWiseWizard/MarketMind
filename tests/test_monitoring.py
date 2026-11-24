"""
Tests for Stage 4: Feature & Prediction Drift Monitoring Engine
"""

import json
from pathlib import Path
import unittest
import numpy as np

from src.models.monitoring import (
    calculate_psi,
    calculate_ks,
    DriftMonitor,
    PSI_STABLE_THRESHOLD,
    PSI_CRITICAL_THRESHOLD,
)


class TestDriftMonitoring(unittest.TestCase):
    def test_psi_identical_distributions(self):
        """PSI of identical distributions should be ~0.0."""
        np.random.seed(42)
        ref = np.random.normal(loc=0.0, scale=1.0, size=500)
        cur = np.random.normal(loc=0.0, scale=1.0, size=500)
        psi = calculate_psi(ref, cur)
        self.assertLess(psi, PSI_STABLE_THRESHOLD)

    def test_psi_drifted_distribution(self):
        """PSI of significantly shifted distribution should exceed critical threshold."""
        np.random.seed(42)
        ref = np.random.normal(loc=0.0, scale=1.0, size=500)
        cur = np.random.normal(loc=2.5, scale=1.0, size=500)
        psi = calculate_psi(ref, cur)
        self.assertGreater(psi, PSI_CRITICAL_THRESHOLD)

    def test_psi_edge_cases(self):
        """PSI handles constant values, empty arrays, and NaNs without crashing."""
        self.assertEqual(calculate_psi(np.array([]), np.array([])), 0.0)
        self.assertEqual(calculate_psi(np.array([5.0, 5.0]), np.array([5.0, 5.0])), 0.0)
        self.assertEqual(calculate_psi(np.array([np.nan, 1.0]), np.array([1.0, np.nan])), 0.0)

    def test_ks_statistic_and_pvalue(self):
        """Kolmogorov-Smirnov test detects significant distributional difference."""
        np.random.seed(42)
        ref = np.random.normal(loc=0.0, scale=1.0, size=200)
        cur = np.random.normal(loc=3.0, scale=1.0, size=200)
        res = calculate_ks(ref, cur)
        self.assertIn("statistic", res)
        self.assertIn("p_value", res)
        self.assertTrue(res["drift_detected"])
        self.assertLess(res["p_value"], 0.05)

    def test_drift_monitor_analysis_schema(self):
        """DriftMonitor returns complete structured report dictionary."""
        monitor = DriftMonitor()
        report = monitor.run_drift_analysis()

        self.assertIn("overall_status", report)
        self.assertIn(report["overall_status"], ["HEALTHY", "WARNING", "CRITICAL"])
        self.assertIn("feature_metrics", report)
        self.assertIn("prediction_drift", report)
        self.assertIn("recommendations", report)

        # Ensure all 13 features are evaluated
        self.assertEqual(len(report["feature_metrics"]), 13)
        for feat, metrics in report["feature_metrics"].items():
            self.assertIn("psi", metrics)
            self.assertIn("ks_statistic", metrics)
            self.assertIn("drift_level", metrics)
            self.assertIn(metrics["drift_level"], ["STABLE", "MODERATE", "CRITICAL"])

    def test_drift_report_generation(self):
        """DriftMonitor saves valid JSON and HTML reports."""
        monitor = DriftMonitor()
        report = monitor.run_drift_analysis()
        json_path, html_path = monitor.save_reports(report)

        self.assertTrue(json_path.exists())
        self.assertTrue(html_path.exists())

        with open(json_path) as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["total_features_evaluated"], 13)

        html_text = html_path.read_text(encoding="utf-8")
        self.assertIn("MarketMind — Feature & Prediction Drift Monitor", html_text)


if __name__ == "__main__":
    unittest.main()
