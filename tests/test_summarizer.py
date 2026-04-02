"""Tests for background summarization."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from engram.summarizer import BackgroundSummarizer, summarize_content, _check_model_available


class TestSummarizeContent:
    def test_returns_summary_on_success(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "A concise summary."}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = summarize_content("Some long memory content here.")

        assert result == "A concise summary."

    def test_returns_none_on_http_error(self):
        with patch("httpx.post", side_effect=Exception("connection refused")):
            result = summarize_content("Some content.")

        assert result is None

    def test_returns_none_on_empty_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": ""}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            result = summarize_content("Some content.")

        assert result is None


class TestBackgroundSummarizer:
    def test_start_skips_when_model_unavailable(self):
        db = MagicMock()
        summarizer = BackgroundSummarizer(db=db, project="test")

        with patch("engram.summarizer._check_model_available", return_value=False):
            summarizer.start()

        assert not summarizer._thread.is_alive()

    def test_stop_signals_event(self):
        db = MagicMock()
        db.get_memories_pending_summary.return_value = []
        summarizer = BackgroundSummarizer(db=db, project="test")

        with patch("engram.summarizer._check_model_available", return_value=True):
            with patch("engram.summarizer.SUMMARIZE_ENABLED", True):
                summarizer.start()
                time.sleep(0.05)
                summarizer.stop()

        assert summarizer._stop_event.is_set()

    def test_processes_pending_memories(self):
        db = MagicMock()
        db.get_memories_pending_summary.return_value = [("id1", "content1")]
        db.store_summary = MagicMock()
        summarizer = BackgroundSummarizer(db=db, project="test")

        with patch("engram.summarizer._check_model_available", return_value=True):
            with patch("engram.summarizer.SUMMARIZE_ENABLED", True):
                with patch("engram.summarizer.summarize_content", return_value="Summary text"):
                    summarizer.start()
                    time.sleep(0.1)
                    summarizer.stop()

        db.store_summary.assert_called_once_with("id1", "Summary text")
