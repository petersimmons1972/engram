"""Smoke tests for engram MCP server tools.

Uses the SearchEngine directly (not FastMCP Client) to test the tool
functions, since the tools are thin wrappers around the engine. This avoids
async complexity while still validating the full store -> recall -> correct
-> forget lifecycle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engram.db import MemoryDB
from engram.search import SearchEngine
from tests.conftest import FakeEmbedder


@pytest.fixture(autouse=True)
def _isolate_engines(tmp_path: Path, monkeypatch):
    """Ensure each test gets a fresh engine pool with temp DB dir."""
    import engram.server as srv
    srv._engines.clear()
    monkeypatch.setenv("ENGRAM_DIR", str(tmp_path))
    monkeypatch.setenv("ENGRAM_EMBEDDER", "none")
    yield
    srv._engines.clear()


@pytest.fixture
def _patch_embedder(monkeypatch):
    """Patch _get_engine to use FakeEmbedder instead of real one."""
    import engram.server as srv

    def patched_get_engine(project=None):
        project = (project or "default").strip().lower()
        if project not in srv._engines:
            import os
            db_dir = os.environ.get("ENGRAM_DIR", None)
            db = MemoryDB(project=project, db_dir=db_dir)
            embedder = FakeEmbedder()
            srv._engines[project] = SearchEngine(db=db, embedder=embedder)
        return srv._engines[project]

    monkeypatch.setattr(srv, "_get_engine", patched_get_engine)


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

    def test_recall_limit_capped(self, _patch_embedder):
        from engram.server import memory_recall
        result = memory_recall(query="test", top_k=10000, project="test-project")
        assert isinstance(result, dict)


class TestMemoryStatus:
    def test_status_returns_stats(self, _patch_embedder):
        from engram.server import memory_status, memory_store

        memory_store(content="A memory", project="test-project")
        stats = memory_status(project="test-project")
        assert stats["total_memories"] == 1
