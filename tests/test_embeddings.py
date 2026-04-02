"""Tests for engram.embeddings -- provider pattern and metadata enforcement."""

from __future__ import annotations

import os

import numpy as np
import pytest

from engram.db_postgres import PostgresBackend
from engram.embeddings import NullEmbedder, OllamaEmbedder, cosine_similarity, create_embedder, from_blob, to_blob
from engram.errors import EmbeddingConfigMismatchError
from engram.search import SearchEngine
from engram.types import Memory
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)


class TestNullEmbedder:
    def test_embed_returns_empty(self):
        emb = NullEmbedder()
        vec = emb.embed("any text")
        assert len(vec) == 0

    def test_embed_batch_returns_empty_list(self):
        emb = NullEmbedder()
        results = emb.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 0 for v in results)

    def test_name_and_dimensions(self):
        emb = NullEmbedder()
        assert emb.name == "none"
        assert emb.dimensions == 0


class TestFakeEmbedder:
    def test_deterministic(self):
        emb = FakeEmbedder()
        v1 = emb.embed("hello world")
        v2 = emb.embed("hello world")
        assert np.array_equal(v1, v2)

    def test_similar_texts_high_similarity(self):
        emb = FakeEmbedder()
        v1 = emb.embed("database PostgreSQL performance tuning optimization")
        v2 = emb.embed("database PostgreSQL query optimization performance")
        sim = cosine_similarity(v1, v2)
        assert sim >= 0.5

    def test_different_texts_lower_similarity(self):
        emb = FakeEmbedder()
        v1 = emb.embed("database PostgreSQL performance")
        v2 = emb.embed("frontend React TypeScript components")
        sim = cosine_similarity(v1, v2)
        assert sim < 0.5

    def test_has_protocol_fields(self):
        emb = FakeEmbedder()
        assert hasattr(emb, "name")
        assert hasattr(emb, "dimensions")
        assert emb.dimensions == 64


class TestCreateEmbedder:
    def test_none_provider(self):
        emb = create_embedder(provider="none")
        assert isinstance(emb, NullEmbedder)

    def test_openai_without_key_falls_back(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        emb = create_embedder(provider="openai", api_key=None)
        assert isinstance(emb, NullEmbedder)

    def test_env_var_selects_none(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_EMBEDDER", "none")
        emb = create_embedder()
        assert isinstance(emb, NullEmbedder)


class TestBlobSerialization:
    def test_round_trip(self):
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        blob = to_blob(vec)
        restored = from_blob(blob)
        assert np.allclose(vec, restored)

    def test_empty_round_trip(self):
        vec = np.array([], dtype=np.float32)
        blob = to_blob(vec)
        restored = from_blob(blob)
        assert len(restored) == 0

    def test_cosine_empty_returns_zero(self):
        empty = np.array([], dtype=np.float32)
        full = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(empty, full) == 0.0
        assert cosine_similarity(full, empty) == 0.0


class TestCosineSimilarityDimensionMismatch:
    def test_mismatched_dimensions_returns_zero(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


def _make_db(project: str) -> PostgresBackend:
    return PostgresBackend(project=project, dsn=os.environ["TEST_DATABASE_URL"])


def _cleanup(db: PostgresBackend, project: str) -> None:
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM chunks WHERE project = %s", (project,))
        conn.execute("DELETE FROM memories WHERE project = %s", (project,))
        conn.execute("DELETE FROM project_meta WHERE project = %s", (project,))
        conn.commit()
    db.close()


class TestMetadataEnforcement:
    def test_first_store_sets_metadata(self):
        db = _make_db("meta-test")
        try:
            emb = FakeEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="First memory"))
            assert db.get_meta("embedder_name") == "fake/test-embedder"
            assert db.get_meta("embedder_dimensions") == "64"
        finally:
            _cleanup(db, "meta-test")

    def test_same_embedder_succeeds(self):
        db = _make_db("meta-test2")
        try:
            emb = FakeEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="First"))
            engine.store(Memory(content="Second"))
        finally:
            _cleanup(db, "meta-test2")

    def test_different_embedder_raises_error(self):
        db = _make_db("meta-test3")
        try:
            emb1 = FakeEmbedder()
            engine1 = SearchEngine(db=db, embedder=emb1)
            engine1.store(Memory(content="Stored with fake embedder"))

            class DifferentEmbedder:
                name = "other/model"
                dimensions = 128
                def embed(self, text):
                    return np.zeros(128, dtype=np.float32)
                def embed_batch(self, texts, batch_size=64):
                    return [self.embed(t) for t in texts]

            engine2 = SearchEngine(db=db, embedder=DifferentEmbedder())

            with pytest.raises(EmbeddingConfigMismatchError) as exc_info:
                engine2.store(Memory(content="This should fail"))

            assert "fake/test-embedder" in str(exc_info.value)
            assert "other/model" in str(exc_info.value)
        finally:
            _cleanup(db, "meta-test3")

    def test_null_embedder_skips_metadata(self):
        db = _make_db("meta-null")
        try:
            emb = NullEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="BM25 only mode"))
            assert db.get_meta("embedder_name") is None
            assert db.get_meta("embedder_dimensions") is None
        finally:
            _cleanup(db, "meta-null")


class TestNullEmbedderSearch:
    """Verify the full store/recall cycle works in BM25-only mode."""

    def test_store_and_recall_bm25_only(self):
        db = _make_db("bm25-test")
        try:
            emb = NullEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="PostgreSQL is our main database"))
            results = engine.recall("PostgreSQL database")
            assert len(results) >= 1
            assert "PostgreSQL" in results[0].memory.content
        finally:
            _cleanup(db, "bm25-test")

    def test_no_vector_score_in_bm25_mode(self):
        db = _make_db("bm25-score")
        try:
            emb = NullEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="Authentication uses JWT tokens"))
            results = engine.recall("JWT authentication")
            if results:
                assert results[0].score_breakdown["vector"] == 0.0
        finally:
            _cleanup(db, "bm25-score")

    def test_bm25_dedup_works_without_embeddings(self):
        """Bug fix: dedup must work in NullEmbedder mode (no embeddings on chunks)."""
        db = _make_db("bm25-dedup")
        try:
            emb = NullEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="Exact same content stored twice"))
            engine.store(Memory(content="Exact same content stored twice"))
            stats = db.get_stats()
            assert stats.total_chunks == 1
        finally:
            _cleanup(db, "bm25-dedup")


class TestAutoDetectOllamaUrl:
    def test_auto_detect_reads_env_var(self, monkeypatch):
        """Bug fix: auto-detect must respect OLLAMA_URL env var."""
        monkeypatch.setenv("OLLAMA_URL", "http://custom-host:11434")
        monkeypatch.delenv("ENGRAM_EMBEDDER", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        emb = create_embedder()
        assert isinstance(emb, NullEmbedder)


class TestSSRFProtection:
    def test_localhost_allowed(self):
        from engram.embeddings import _validate_ollama_url
        assert _validate_ollama_url("http://localhost:11434") is True

    def test_normal_ip_allowed(self):
        from engram.embeddings import _validate_ollama_url
        assert _validate_ollama_url("http://192.168.1.100:11434") is True

    def test_link_local_blocked(self):
        from engram.embeddings import _validate_ollama_url
        assert _validate_ollama_url("http://169.254.169.254/latest/meta-data") is False

    def test_metadata_hostname_blocked(self):
        from engram.embeddings import _validate_ollama_url
        assert _validate_ollama_url("http://metadata.google.internal") is False


class TestMetadataVersion:
    def test_version_stored_on_first_embed(self):
        db = _make_db("version-test")
        try:
            emb = FakeEmbedder()
            engine = SearchEngine(db=db, embedder=emb)
            engine.store(Memory(content="Test version storage"))
            assert db.get_meta("embedder_version") == "v1-test"
        finally:
            _cleanup(db, "version-test")


class TestOllamaAuth:
    def test_ollama_embedder_accepts_api_key(self):
        """OllamaEmbedder should accept an optional api_key parameter."""
        emb = OllamaEmbedder(base_url="http://localhost:11434", api_key="sk-test-123")
        assert emb._headers.get("Authorization") == "Bearer sk-test-123"

    def test_ollama_embedder_no_auth_header_without_key(self):
        """OllamaEmbedder should not include auth header when no key is provided."""
        emb = OllamaEmbedder(base_url="http://localhost:11434")
        assert "Authorization" not in emb._headers

    def test_create_embedder_passes_ollama_api_key(self, monkeypatch):
        """create_embedder should pass OLLAMA_API_KEY to OllamaEmbedder."""
        monkeypatch.setenv("OLLAMA_API_KEY", "sk-from-env")
        emb = create_embedder(provider="ollama", ollama_url="http://localhost:11434")
        assert isinstance(emb, OllamaEmbedder)
        assert emb._headers.get("Authorization") == "Bearer sk-from-env"

    def test_create_embedder_no_ollama_api_key(self, monkeypatch):
        """create_embedder should work without OLLAMA_API_KEY."""
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        emb = create_embedder(provider="ollama", ollama_url="http://localhost:11434")
        assert isinstance(emb, OllamaEmbedder)
        assert "Authorization" not in emb._headers


class TestOllamaReachableErrorHandling:
    def test_returns_false_when_httpx_missing(self, monkeypatch):
        """When httpx is not available, _ollama_reachable returns False."""
        import engram.embeddings as emb
        monkeypatch.setattr(emb, "_httpx", None)
        assert emb._ollama_reachable("http://localhost:11434") is False
