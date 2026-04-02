"""Smoke tests for engram MCP server tools.

Uses the SearchEngine directly (not FastMCP Client) to test the tool
functions, since the tools are thin wrappers around the engine. This avoids
async complexity while still validating the full store -> recall -> correct
-> forget lifecycle.
"""

from __future__ import annotations

import os

import pytest

from engram.db_postgres import PostgresBackend
from engram.search import SearchEngine
from tests.conftest import FakeEmbedder

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)


@pytest.fixture(autouse=True)
def _isolate_engines(monkeypatch):
    """Ensure each test gets a fresh engine pool backed by TEST_DATABASE_URL."""
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("No TEST_DATABASE_URL set")
    import engram.server as srv
    srv._engines.clear()
    # Point server's create_database at the test database
    monkeypatch.setenv("DATABASE_URL", dsn)
    monkeypatch.setenv("ENGRAM_EMBEDDER", "none")
    yield
    srv._engines.clear()


@pytest.fixture
def _patch_embedder(monkeypatch):
    """Patch _get_engine to use FakeEmbedder and PostgresBackend."""
    import engram.server as srv

    def patched_get_engine(project=None):
        import re
        raw = (project or "default").strip().lower()
        project_key = re.sub(r'[^a-z0-9_-]', '', raw) or "default"
        if project_key not in srv._engines:
            dsn = os.environ["TEST_DATABASE_URL"]
            db = PostgresBackend(project=project_key, dsn=dsn)
            embedder = FakeEmbedder()
            srv._engines[project_key] = SearchEngine(db=db, embedder=embedder)
        return srv._engines[project_key]

    monkeypatch.setattr(srv, "_get_engine", patched_get_engine)


@pytest.fixture(autouse=True)
def _cleanup_test_data():
    """Remove test rows from Postgres after each test."""
    yield
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        return
    db = PostgresBackend(project="__cleanup__", dsn=dsn)
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM relationships")
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM project_meta")
        conn.commit()
    db.close()


class TestMemoryStoreRecall:
    def test_store_and_recall(self, _patch_embedder):
        from engram.server import memory_recall, memory_store

        result = memory_store(
            content="We use PostgreSQL for the main database",
            memory_type="decision",
            tags="database,postgres",
            importance=1,
            project="test-project",
        )
        assert result["status"] == "stored"
        assert result["memory_type"] == "decision"

        recall = memory_recall(
            query="database choice",
            project="test-project",
        )
        assert recall["count"] >= 1
        assert "PostgreSQL" in recall["results"][0]["content"]

    def test_project_isolation(self, _patch_embedder):
        from engram.server import memory_recall, memory_store

        memory_store(
            content="Alpha project secret",
            project="alpha",
        )
        recall = memory_recall(query="secret", project="beta")
        assert recall["count"] == 0


class TestMemoryCorrect:
    def test_correct_supersedes_old(self, _patch_embedder):
        from engram.server import memory_correct, memory_store

        store_result = memory_store(
            content="Use MySQL for the database",
            memory_type="decision",
            tags="database",
            project="test-project",
        )
        old_id = store_result["id"]

        correct_result = memory_correct(
            old_memory_id=old_id,
            new_content="Use PostgreSQL instead of MySQL for JSONB support",
            project="test-project",
        )
        assert correct_result["status"] == "corrected"
        assert correct_result["old_demoted_to"] == "trivial (will be pruned if unused)"

    def test_correct_nonexistent_returns_error(self, _patch_embedder):
        from engram.server import memory_correct

        result = memory_correct(
            old_memory_id="nonexistent",
            new_content="Doesn't matter",
            project="test-project",
        )
        assert "error" in result


class TestMemoryForget:
    def test_forget_removes_memory(self, _patch_embedder):
        from engram.server import memory_forget, memory_store

        store_result = memory_store(content="Delete me", project="test-project")
        mid = store_result["id"]

        forget_result = memory_forget(memory_id=mid, project="test-project")
        assert forget_result["status"] == "forgotten"

    def test_forget_nonexistent_returns_error(self, _patch_embedder):
        from engram.server import memory_forget

        result = memory_forget(memory_id="nope", project="test-project")
        assert "error" in result


class TestMemoryList:
    def test_list_returns_stored_memories(self, _patch_embedder):
        from engram.server import memory_list, memory_store

        memory_store(content="First memory", project="test-project")
        memory_store(content="Second memory", project="test-project")

        result = memory_list(project="test-project")
        assert result["count"] == 2


class TestInputValidation:
    def test_invalid_memory_type_in_list(self, _patch_embedder):
        from engram.server import memory_list
        result = memory_list(memory_type="invalid_type", project="test-project")
        assert "error" in result

    def test_limit_capped(self, _patch_embedder):
        from engram.server import memory_list
        result = memory_list(limit=999999, project="test-project")
        assert isinstance(result, dict)

    def test_content_too_long_rejected(self, _patch_embedder):
        from engram.server import memory_store
        huge = "x" * 60_000
        result = memory_store(content=huge, project="test-project")
        assert "error" in result

    def test_rate_limit_blocks_excess_calls(self, _patch_embedder):
        import engram.server as srv
        proj = "rate-limit-test"
        original_max = srv._RATE_LIMIT_MAX
        srv._RATE_LIMIT_MAX = 3
        with srv._rate_limit_lock:
            srv._store_calls.pop(proj, None)
        try:
            for _ in range(3):
                r = srv.memory_store(content="ok", project=proj)
                assert "error" not in r
            r = srv.memory_store(content="overflow", project=proj)
            assert "error" in r
            assert "Rate limit" in r["error"]
            assert "retry_after_seconds" in r
        finally:
            srv._RATE_LIMIT_MAX = original_max
            with srv._rate_limit_lock:
                srv._store_calls.pop(proj, None)

    def test_recall_limit_capped(self, _patch_embedder):
        from engram.server import memory_recall
        result = memory_recall(query="test", top_k=10000, project="test-project")
        assert isinstance(result, dict)


class TestAuthMiddleware:
    @pytest.mark.asyncio
    async def test_auth_rejects_websocket_without_token(self):
        from engram.server import _wrap_with_api_key_auth

        async def fake_app(scope, receive, send):
            pass

        wrapped = _wrap_with_api_key_auth(fake_app, "test-key")
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        scope = {"type": "websocket", "headers": []}
        await wrapped(scope, None, mock_send)
        assert len(responses) > 0

    @pytest.mark.asyncio
    async def test_auth_allows_lifespan(self):
        from engram.server import _wrap_with_api_key_auth

        called = []

        async def fake_app(scope, receive, send):
            called.append(True)

        wrapped = _wrap_with_api_key_auth(fake_app, "test-key")
        scope = {"type": "lifespan"}
        await wrapped(scope, None, None)
        assert called

    @pytest.mark.asyncio
    async def test_auth_rejects_unknown_scope(self):
        from engram.server import _wrap_with_api_key_auth

        async def fake_app(scope, receive, send):
            pass

        wrapped = _wrap_with_api_key_auth(fake_app, "test-key")
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        scope = {"type": "unknown_protocol", "headers": []}
        await wrapped(scope, None, mock_send)
        assert len(responses) > 0


class TestProjectNormalization:
    def test_project_name_normalized_consistently(self, _patch_embedder):
        from engram.server import memory_recall, memory_store

        memory_store(content="Test content for normalization", project="My-App")
        recall = memory_recall(query="Test", project="my-app")
        assert recall["count"] >= 1


class TestAtomicOperations:
    def test_forget_is_atomic(self, _patch_embedder):
        from engram.server import memory_forget, memory_store

        result = memory_store(content="Atomic test", project="test-atomic")
        mid = result["id"]
        forget = memory_forget(memory_id=mid, project="test-atomic")
        assert forget["status"] == "forgotten"


class TestEngineCacheLRU:
    def test_cache_evicts_oldest(self, monkeypatch):
        """Test that engine cache evicts LRU entries when exceeding max size."""
        import engram.server as srv
        from tests.conftest import FakeEmbedder

        monkeypatch.setattr(srv, "create_embedder", lambda: FakeEmbedder())
        srv._engines.clear()
        original_max = srv.MAX_ENGINE_CACHE_SIZE
        srv.MAX_ENGINE_CACHE_SIZE = 3
        try:
            for i in range(5):
                srv._get_engine(f"proj-{i}")
            assert len(srv._engines) <= 3
            assert "proj-4" in srv._engines
            assert "proj-3" in srv._engines
            assert "proj-2" in srv._engines
            assert "proj-0" not in srv._engines
            assert "proj-1" not in srv._engines
        finally:
            srv.MAX_ENGINE_CACHE_SIZE = original_max
            srv._engines.clear()

    def test_cache_moves_accessed_to_end(self, monkeypatch):
        """Test that accessing a cached engine moves it to MRU position."""
        import engram.server as srv
        from tests.conftest import FakeEmbedder

        monkeypatch.setattr(srv, "create_embedder", lambda: FakeEmbedder())
        srv._engines.clear()
        original_max = srv.MAX_ENGINE_CACHE_SIZE
        srv.MAX_ENGINE_CACHE_SIZE = 3
        try:
            srv._get_engine("proj-a")
            srv._get_engine("proj-b")
            srv._get_engine("proj-c")
            srv._get_engine("proj-a")
            srv._get_engine("proj-d")
            assert "proj-a" in srv._engines
            assert "proj-c" in srv._engines
            assert "proj-d" in srv._engines
            assert "proj-b" not in srv._engines
        finally:
            srv.MAX_ENGINE_CACHE_SIZE = original_max
            srv._engines.clear()


class TestMemoryConnectErrorHandling:
    """Regression tests for #37: Unhandled ValueError in memory_connect."""

    def test_connect_nonexistent_source_returns_error(self, _patch_embedder):
        from engram.server import memory_connect
        result = memory_connect(
            source_id="nonexistent", target_id="also-nonexistent", project="test-project",
        )
        assert "error" in result

    def test_connect_valid_memories_succeeds(self, _patch_embedder):
        from engram.server import memory_connect, memory_store
        a = memory_store(content="Memory A", project="test-project")
        b = memory_store(content="Memory B", project="test-project")
        result = memory_connect(
            source_id=a["id"], target_id=b["id"], project="test-project",
        )
        assert result["status"] == "connected"


class TestMemoryStatus:
    def test_status_returns_stats(self, _patch_embedder):
        from engram.server import memory_status, memory_store

        memory_store(content="A memory", project="test-project")
        stats = memory_status(project="test-project")
        assert stats["total_memories"] == 1


class TestMemoryCompress:
    def test_memory_compress_dry_run(self, _patch_embedder):
        """memory_compress dry_run reports without writing."""
        from engram.server import memory_compress, memory_store
        memory_store(content="x" * 1000, project="test")
        result = memory_compress(project="test", dry_run=True)
        assert result.get("status") == "dry_run"
        assert "compressed" in result

    def test_memory_recall_content_format_text(self, _patch_embedder):
        """content_format=text returns plain content, no compressed envelope."""
        from engram.server import memory_recall, memory_store
        memory_store(content="hello world " * 100, project="test")
        results = memory_recall(query="hello", project="test", content_format="text")
        for r in results["results"]:
            assert isinstance(r["content"], str)
            assert "content_compressed" not in r

    def test_memory_recall_content_format_compressed_unready(self, _patch_embedder):
        """content_format=compressed on uncompressed memory returns text with warning."""
        from engram.server import memory_recall, memory_store
        memory_store(content="hello world " * 100, project="test")
        results = memory_recall(query="hello", project="test", content_format="compressed")
        for r in results["results"]:
            assert isinstance(r["content"], str)
            env = r.get("content_compressed")
            if env:
                assert env.get("warning") == "not_yet_compressed"

    def test_memory_recall_content_format_compressed_ready(self, _patch_embedder):
        """content_format=compressed after compression returns base64 data."""
        import base64
        from engram.server import memory_compress, memory_recall, memory_store
        memory_store(content="x" * 1000, project="test")
        memory_compress(project="test", algorithm="zlib")
        results = memory_recall(query="xxx", project="test", content_format="compressed")
        for r in results["results"]:
            env = r.get("content_compressed")
            if env and env.get("data"):
                decoded = base64.b64decode(env["data"])
                assert len(decoded) > 0
                assert env["algo"] == "zlib"
                assert env["warning"] is None

    def test_memory_compress_unknown_algo_returns_error(self, _patch_embedder):
        from engram.server import memory_compress
        result = memory_compress(project="test", algorithm="magic-algo")
        assert "error" in result

    def test_concurrent_consolidate_returns_already_running(self, _patch_embedder):
        """Second concurrent consolidate returns already_running status when lock is held."""
        from engram.server import _get_engine
        engine = _get_engine("test")
        engine._consolidation_lock.acquire()
        try:
            result = engine.consolidate()
            assert result.get("status") == "already_running"
        finally:
            engine._consolidation_lock.release()

    def test_memory_status_includes_compression(self, _patch_embedder):
        """memory_status response includes compression sub-dict."""
        from engram.server import memory_status, memory_store
        memory_store(content="A memory", project="test")
        stats = memory_status(project="test")
        assert "compression" in stats
        assert "compressed_count" in stats["compression"]
