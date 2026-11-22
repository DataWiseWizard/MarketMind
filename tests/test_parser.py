"""
Tests for Transcript Ingestion & Parser
"""

import unittest
from pathlib import Path
from src.ingestion.transcript_parser import TranscriptParser
from src.db.models import SpeakerRole, SegmentType


class TestTranscriptParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = TranscriptParser()
        # Find first available transcript PDF dynamically across all company directories
        cls.sample_pdf = next(Path("data/raw/transcripts").rglob("*.pdf"), None)

    def test_directory_ticker_resolution(self):
        """Verify directory-aware ticker resolution for tracked companies."""
        test_cases = [
            (Path("data/raw/transcripts/NVDA/2026/call.pdf"), "NVDA", "NVIDIA Corporation"),
            (Path("data/raw/transcripts/AMD/2024/call.pdf"), "AMD", "Advanced Micro Devices, Inc."),
            (Path("data/raw/transcripts/INTC/2025/call.pdf"), "INTC", "Intel Corporation"),
            (Path("data/raw/transcripts/TSMC/2024/call.pdf"), "TSM", "Taiwan Semiconductor Manufacturing Company Limited"),
            (Path("data/raw/transcripts/ASML/2024/call.pdf"), "ASML", "ASML Holding N.V."),
        ]
        for path, expected_ticker, expected_name in test_cases:
            meta = self.parser.extract_metadata(path, "Sample text")
            self.assertEqual(meta["ticker"], expected_ticker)
            self.assertEqual(meta["company_name"], expected_name)

    def test_metadata_extraction(self):
        if not self.sample_pdf or not self.sample_pdf.exists():
            self.skipTest("No sample PDF found in data/raw/transcripts")
        raw_text = self.parser.extract_raw_text(self.sample_pdf)
        meta = self.parser.extract_metadata(self.sample_pdf, raw_text)

        self.assertIn("ticker", meta)
        self.assertIn("quarter", meta)
        self.assertIn("fiscal_year", meta)
        self.assertIsNotNone(meta["call_date"])
        self.assertNotEqual(meta["ticker"], "UNKNOWN")

    def test_section_splitting(self):
        if not self.sample_pdf or not self.sample_pdf.exists():
            self.skipTest("No sample PDF found in data/raw/transcripts")
        raw_text = self.parser.extract_raw_text(self.sample_pdf)
        mgmt_text, qa_text, mgmt_idx, qa_idx = self.parser.split_sections(raw_text)

        self.assertGreaterEqual(mgmt_idx, 0)
        self.assertGreater(len(mgmt_text), 200)

    def test_transcript_parsing(self):
        if not self.sample_pdf or not self.sample_pdf.exists():
            self.skipTest("No sample PDF found in data/raw/transcripts")
        parsed = self.parser.parse(self.sample_pdf)

        self.assertIn("metadata", parsed)
        self.assertIn("participants", parsed)
        self.assertIn("segments", parsed)

        segments = parsed["segments"]
        self.assertGreater(len(segments), 0)

        # Ensure speaker roles exist
        roles = {s["speaker_role"] for s in segments}
        self.assertTrue(SpeakerRole.CEO in roles or SpeakerRole.CFO in roles or SpeakerRole.UNKNOWN in roles)

        # Ensure segment types exist
        types = {s["segment_type"] for s in segments}
        self.assertIn(SegmentType.PREPARED_REMARKS, types)


if __name__ == "__main__":
    unittest.main()
