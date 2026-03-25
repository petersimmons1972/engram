"""
Verify embedder factory resolves correctly for production Docker config.

These tests catch silent NullEmbedder fallback caused by missing dependencies.
The critical bug: httpx missing from requirements.txt causes Docker deployments
to silently fall back to BM25-only mode with no error message.

Related issue: https://github.com/petersimmons1972/engram/issues/42
"""

import pytest
from engram.embeddings import NullEmbedder, OllamaEmbedder, create_embedder


class TestEmbedderResolution:
    """Verify create_embedder returns the correct provider, not a silent fallback."""

    def test_create_embedder_ollama_returns_ollama_type(self):
        """
        create_embedder("ollama") MUST return OllamaEmbedder, never NullEmbedder.

        If this fails with ImportError, httpx is missing from requirements.txt.
        Docker deployments with ENGRAM_EMBEDDER=ollama would silently fall back
        to NullEmbedder (BM25-only mode) with no error, breaking the default use case.
        """
        embedder = create_embedder("ollama", ollama_url="http://localhost:11434")

        assert isinstance(
            embedder, OllamaEmbedder
        ), (
            f"Expected OllamaEmbedder but got {type(embedder).__name__}. "
            "Likely cause: httpx is missing from requirements.txt. "
            "Docker deployments default to ENGRAM_EMBEDDER=ollama but Ollama requires httpx. "
            "See https://github.com/petersimmons1972/engram/issues/42"
        )

    def test_null_embedder_not_returned_for_named_provider(self):
        """
        NullEmbedder should ONLY be returned when provider='none', never as a fallback.

        If this fails, a required dependency is missing (httpx for ollama).
        """
        embedder = create_embedder("ollama", ollama_url="http://localhost:11434")

        assert not isinstance(
            embedder, NullEmbedder
        ), (
            "Got NullEmbedder when requesting provider='ollama'. "
            "This indicates a silent fallback due to missing dependencies. "
            "Check that httpx>=0.24 is in requirements.txt."
        )

    def test_null_embedder_only_for_none_provider(self):
        """NullEmbedder is the correct response for provider='none'."""
        embedder = create_embedder("none")
        assert isinstance(embedder, NullEmbedder)

    def test_openai_embedder_creation(self):
        """create_embedder('openai') should succeed if openai is installed."""
        try:
            from engram.embeddings import OpenAIEmbedder

            # This will fail without OPENAI_API_KEY, but we're testing the type, not the call
            embedder = create_embedder("openai", api_key="sk-test-key")
            assert isinstance(embedder, OpenAIEmbedder)
        except ImportError:
            # openai not installed, skip this test
            pytest.skip("openai package not installed")

    def test_ollama_url_validation(self):
        """OllamaEmbedder should validate the URL is reachable (or at least parse correctly)."""
        # This test verifies the URL is accepted; actual reachability is tested in integration tests
        embedder = create_embedder("ollama", ollama_url="http://ollama:11434")
        assert isinstance(embedder, OllamaEmbedder)
        # Verify the embedder has the internal base_url attribute
        assert hasattr(embedder, "_base_url")
