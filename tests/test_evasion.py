"""
Tests for Evasion Detection Module
"""

import unittest
from src.processing.evasion_detector import EvasionDetector


class TestEvasionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EvasionDetector(evasion_threshold=0.50)

    def test_evasive_quantification_refusal(self):
        question = "Can you quantify how much the export control affected profit and what percentage was lost?"
        answer = "We are monitoring the situation closely and remain focused on long-term growth opportunities."

        res = self.detector.evaluate_pair(question, answer)
        self.assertTrue(res["is_evasive"])
        self.assertGreaterEqual(res["evasion_score"], 0.50)
        self.assertIn("quantification", res["reason"].lower())

    def test_direct_numerical_response(self):
        question = "Can you tell us what revenue was for Data Center this quarter?"
        answer = "Data Center revenue was $39 billion, up 73% year-on-year."

        res = self.detector.evaluate_pair(question, answer)
        self.assertFalse(res["is_evasive"])
        self.assertLess(res["evasion_score"], 0.40)


if __name__ == "__main__":
    unittest.main()
