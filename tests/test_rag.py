"""
Tests for ChromaDB Vector Store & Retrieval
"""

import unittest
from pathlib import Path
from src.rag.vector_store import query_similar, _get_collection


class TestVectorStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collection = _get_collection()

    def test_vector_store_collection_exists(self):
        """Assert ChromaDB collection is created and populated with chunks."""
        count = self.collection.count()
        self.assertGreater(count, 0, "ChromaDB collection should not be empty")

    def test_semantic_query_retrieval(self):
        """Verify semantic search returns relevant documents with expected metadata."""
        query = "revenue growth data center guidance"
        results = query_similar(query, n=3, ticker="NVDA")

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        top_match = results[0]
        self.assertIn("text", top_match)
        self.assertIn("metadata", top_match)
        self.assertIn("distance", top_match)

        # Distance should be a reasonable cosine distance
        self.assertGreaterEqual(top_match["distance"], 0.0)
        self.assertLess(top_match["distance"], 1.5)

        # Metadata should contain ticker NVDA
        self.assertEqual(top_match["metadata"].get("ticker"), "NVDA")


if __name__ == "__main__":
    unittest.main()
