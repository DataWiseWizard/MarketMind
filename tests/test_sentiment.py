"""
Tests for Financial Sentiment Analysis Module
"""

import unittest
from src.processing.sentiment_analyzer import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_financial_sentiment(self):
        text = "We achieved record revenue, strong momentum, and outstanding profit growth across all segments."
        res = self.analyzer.analyze_text(text)
        self.assertGreater(res["positive"], res["negative"])
        self.assertGreater(res["compound"], 0.2)

    def test_negative_financial_sentiment(self):
        text = "We faced severe headwinds, declining margins, significant impairment charges, and supply chain disruptions."
        res = self.analyzer.analyze_text(text)
        self.assertGreater(res["negative"], res["positive"])
        self.assertLess(res["compound"], -0.2)


if __name__ == "__main__":
    unittest.main()
