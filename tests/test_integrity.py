"""Tests for memory content hash integrity (issue #67)."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)

from engram.db_postgres import PostgresBackend, _content_hash
from engram.types import Memory


class TestContentHash:
    def test_hash_stored_on_store_memory(self, db):
        mem = db.store_memory(Memory(content="test content for hashing"))
        assert mem.content_hash is not None
        assert mem.content_hash == _content_hash("test content for hashing")

    def test_hash_updated_on_content_update(self, db):
        mem = db.store_memory(Memory(content="original content"))
        db.update_memory(mem.id, content="updated content")
        updated = db.get_memory(mem.id)
        assert updated.content_hash == _content_hash("updated content")
        assert updated.content_hash != _content_hash("original content")

    def test_hash_not_changed_on_importance_update(self, db):
        mem = db.store_memory(Memory(content="important memory"))
        original_hash = mem.content_hash
        db.update_memory(mem.id, importance=1)
        updated = db.get_memory(mem.id)
        assert updated.content_hash == original_hash

    def test_mismatch_logged_not_raised(self, db, caplog):
        import logging
        mem = db.store_memory(Memory(content="original"))
        # Corrupt the hash directly in the DB
        with db.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET content_hash = %s WHERE id = %s",
                ("deadbeef" * 8, mem.id),
            )
            conn.commit()
        with caplog.at_level(logging.WARNING):
            retrieved = db.get_memory(mem.id)
        assert retrieved is not None  # did not raise
        assert "INTEGRITY" in caplog.text
        assert "mismatch" in caplog.text

    def test_no_warning_for_valid_hash(self, db, caplog):
        import logging
        mem = db.store_memory(Memory(content="clean memory"))
        with caplog.at_level(logging.WARNING):
            db.get_memory(mem.id)
        assert "INTEGRITY" not in caplog.text


class TestIntegrityStats:
    def test_get_integrity_stats_all_hashed(self, db):
        db.store_memory(Memory(content="memory one"))
        db.store_memory(Memory(content="memory two"))
        stats = db.get_integrity_stats(db.project)
        assert stats["total"] >= 2
        assert stats["hashed"] == stats["total"]
        assert stats["corrupt"] == 0

    def test_get_memories_missing_hash(self, db):
        mem = db.store_memory(Memory(content="will lose its hash"))
        # Null out the hash to simulate pre-v7 row
        with db.pool.connection() as conn:
            conn.execute("UPDATE memories SET content_hash = NULL WHERE id = %s", (mem.id,))
            conn.commit()
        missing = db.get_memories_missing_hash(db.project)
        ids = [m_id for m_id, _ in missing]
        assert mem.id in ids

    def test_update_memory_hash(self, db):
        mem = db.store_memory(Memory(content="needs hash"))
        new_hash = _content_hash("needs hash")
        db.update_memory_hash(mem.id, new_hash)
        updated = db.get_memory(mem.id)
        assert updated.content_hash == new_hash
