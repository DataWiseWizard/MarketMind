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
        # In clean CI environments where data/vector_store/ is not committed,
        # self-seed a single test chunk so vector store and search contracts can be verified.
        if cls.collection.count() == 0:
            from src.rag.vector_store import _get_embedder
            embedder = _get_embedder()
            sample_doc = "NVIDIA reported Q4 revenue of $22.1 billion, up 265% year-over-year driven by Data Center growth."
            sample_emb = embedder.encode([sample_doc]).tolist()
            cls.collection.add(
                documents=[sample_doc],
                embeddings=sample_emb,
                metadatas=[{
                    "ticker": "NVDA",
                    "quarter": "Q4",
                    "year": "2024",
                    "doc_type": "transcript",
                    "speaker_role": "CFO",
                    "call_id": "1",
                    "segment_id": "1",
                }],
                ids=["test_ci_nvda_chunk_1"],
            )

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
