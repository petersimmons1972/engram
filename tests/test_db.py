"""Tests for engram.db.MemoryDB -- CRUD, FTS, relationships, project isolation."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.db import MemoryDB
from engram.types import Memory, MemoryType, Relationship, RelationType


class TestMemoryCRUD:
    def test_store_and_retrieve(self, db: MemoryDB):
        mem = Memory(content="PostgreSQL chosen for the main database")
        stored = db.store_memory(mem)

        retrieved = db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.content == "PostgreSQL chosen for the main database"
        assert retrieved.project == "test"

    def test_get_nonexistent_returns_none(self, db: MemoryDB):
        assert db.get_memory("does-not-exist") is None

    def test_update_memory(self, db: MemoryDB):
        mem = Memory(content="Old content", tags=["old"])
        stored = db.store_memory(mem)

        updated = db.update_memory(stored.id, content="New content", tags=["new"])
        assert updated is not None
        assert updated.content == "New content"
        assert updated.tags == ["new"]

    def test_delete_memory(self, db: MemoryDB):
        mem = Memory(content="To be deleted")
        stored = db.store_memory(mem)

        assert db.delete_memory(stored.id) is True
        assert db.get_memory(stored.id) is None

    def test_delete_nonexistent_returns_false(self, db: MemoryDB):
        assert db.delete_memory("nope") is False

    def test_touch_increments_access(self, db: MemoryDB):
        mem = Memory(content="Touch me")
        stored = db.store_memory(mem)

        db.touch_memory(stored.id)
        db.touch_memory(stored.id)
        retrieved = db.get_memory(stored.id)
        assert retrieved.access_count == 2

    def test_list_memories_filters_by_type(self, db: MemoryDB):
        db.store_memory(Memory(content="A decision", memory_type=MemoryType.DECISION))
        db.store_memory(Memory(content="An error", memory_type=MemoryType.ERROR))
        db.store_memory(Memory(content="A pattern", memory_type=MemoryType.PATTERN))

        decisions = db.list_memories(memory_type=MemoryType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].memory_type == MemoryType.DECISION

    def test_list_memories_filters_by_importance(self, db: MemoryDB):
        db.store_memory(Memory(content="Critical", importance=0))
        db.store_memory(Memory(content="Trivial", importance=4))

        critical = db.list_memories(min_importance=0)
        assert len(critical) == 1
        assert critical[0].content == "Critical"

    def test_list_memories_filters_by_tags(self, db: MemoryDB):
        db.store_memory(Memory(content="Auth stuff", tags=["auth", "jwt"]))
        db.store_memory(Memory(content="DB stuff", tags=["postgres", "sql"]))

        auth = db.list_memories(tags=["auth"])
        assert len(auth) == 1
        assert "auth" in auth[0].tags


class TestProjectIsolation:
    def test_separate_db_files(self, tmp_db_dir: Path):
        db_a = MemoryDB(project="alpha", db_dir=tmp_db_dir)
        db_b = MemoryDB(project="beta", db_dir=tmp_db_dir)

        assert db_a.db_path != db_b.db_path
        assert db_a.db_path.name == "alpha.db"
        assert db_b.db_path.name == "beta.db"

    def test_memories_do_not_leak(self, tmp_db_dir: Path):
        db_a = MemoryDB(project="alpha", db_dir=tmp_db_dir)
        db_b = MemoryDB(project="beta", db_dir=tmp_db_dir)

        db_a.store_memory(Memory(content="Alpha secret"))
        db_b.store_memory(Memory(content="Beta secret"))

        alpha_mems = db_a.list_memories()
        beta_mems = db_b.list_memories()

        assert len(alpha_mems) == 1
        assert alpha_mems[0].content == "Alpha secret"
        assert len(beta_mems) == 1
        assert beta_mems[0].content == "Beta secret"

    def test_fts_isolated_between_projects(self, tmp_db_dir: Path):
        db_a = MemoryDB(project="alpha", db_dir=tmp_db_dir)
        db_b = MemoryDB(project="beta", db_dir=tmp_db_dir)

        db_a.store_memory(Memory(content="Alpha uses PostgreSQL for everything"))

        results = db_b.fts_search("PostgreSQL")
        assert len(results) == 0


class TestFTSRebuildStaleCleanup:
    def test_rebuild_fts_removes_stale_entries(self, db: MemoryDB):
        mem = db.store_memory(Memory(content="Ephemeral data to be deleted"))
        assert len(db.fts_search("Ephemeral")) > 0
        # Delete the memory directly (bypassing triggers won't fire for manual FTS cleanup)
        conn = db._get_conn()
        conn.execute("DELETE FROM memories WHERE id = ?", (mem.id,))
        conn.commit()
        # Rebuild should clean up stale FTS entries
        db.rebuild_fts()
        assert len(db.fts_search("Ephemeral")) == 0


class TestFTSSearch:
    def test_basic_search(self, db: MemoryDB):
        db.store_memory(Memory(content="JWT authentication with refresh tokens"))
        db.store_memory(Memory(content="Database migration using alembic"))

        results = db.fts_search("JWT authentication")
        assert len(results) >= 1
        assert "JWT" in results[0][0].content

    def test_empty_query_returns_empty(self, db: MemoryDB):
        db.store_memory(Memory(content="Some content"))
        results = db.fts_search("")
        assert results == []

    def test_no_match_returns_empty(self, db: MemoryDB):
        db.store_memory(Memory(content="Python web framework"))
        results = db.fts_search("quantum entanglement")
        assert len(results) == 0


class TestRelationships:
    def test_store_and_get_connected(self, db: MemoryDB):
        m1 = db.store_memory(Memory(content="Memory A"))
        m2 = db.store_memory(Memory(content="Memory B"))

        rel = Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO, strength=0.8,
        )
        db.store_relationship(rel)

        connected = db.get_connected(m1.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][0].id == m2.id
        assert connected[0][1] == "relates_to"

    def test_supersedes_relationship(self, db: MemoryDB):
        old = db.store_memory(Memory(content="Use MySQL"))
        new = db.store_memory(Memory(content="Use PostgreSQL instead"))

        rel = Relationship(
            source_id=new.id, target_id=old.id,
            rel_type=RelationType.SUPERSEDES,
        )
        db.store_relationship(rel)

        connected = db.get_connected(old.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][1] == "supersedes"

    def test_boost_and_decay_edges(self, db: MemoryDB):
        m1 = db.store_memory(Memory(content="A"))
        m2 = db.store_memory(Memory(content="B"))

        rel = Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO, strength=0.5,
        )
        db.store_relationship(rel)

        db.boost_edges_for_memory(m1.id, factor=0.2)
        connected = db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.7, abs=0.01)

        db.decay_edges_for_memory(m1.id, factor=0.3)
        connected = db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.4, abs=0.01)

    def test_delete_relationships_for_memory(self, db: MemoryDB):
        m1 = db.store_memory(Memory(content="A"))
        m2 = db.store_memory(Memory(content="B"))

        rel = Relationship(source_id=m1.id, target_id=m2.id)
        db.store_relationship(rel)

        db.delete_relationships_for_memory(m1.id)
        assert db.get_connection_count(m1.id) == 0


class TestFTSSanitization:
    def test_column_filter_stripped(self, db):
        db.store_memory(Memory(content="Test content for FTS"))
        results = db.fts_search("content:Test")
        assert isinstance(results, list)

    def test_prefix_operator_stripped(self, db):
        db.store_memory(Memory(content="Testing prefix operators"))
        results = db.fts_search("test*")
        assert isinstance(results, list)


class TestTagFilterBeforeLimit:
    def test_tag_filter_respects_limit(self, db):
        for i in range(30):
            tags = ["target"] if i % 3 == 0 else ["other"]
            db.store_memory(Memory(content=f"Memory {i} about things", tags=tags))
        results = db.list_memories(tags=["target"], limit=10)
        assert len(results) == 10


class TestBFSPhantomNodes:
    def test_deleted_memory_not_in_frontier(self, db):
        m1 = db.store_memory(Memory(content="Existing"))
        m2 = db.store_memory(Memory(content="Will be deleted"))
        rel = Relationship(source_id=m1.id, target_id=m2.id)
        db.store_relationship(rel)
        # Delete the memory but relationship persists (bypass CASCADE with FK off)
        conn = db._get_conn()
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM memories WHERE id = ?", (m2.id,))
        conn.execute("PRAGMA foreign_keys=ON")
        conn.commit()
        connected = db.get_connected(m1.id, max_hops=2)
        for mem, *_ in connected:
            assert mem is not None


class TestStats:
    def test_stats_reflect_stored_data(self, db: MemoryDB):
        db.store_memory(Memory(content="Decision 1", memory_type=MemoryType.DECISION))
        db.store_memory(Memory(content="Error 1", memory_type=MemoryType.ERROR))

        stats = db.get_stats()
        assert stats.total_memories == 2
        assert stats.by_type.get("decision") == 1
        assert stats.by_type.get("error") == 1


class TestStatsPerProject:
    def test_chunks_counted_per_project(self, tmp_db_dir):
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        db_a = MemoryDB(project="alpha", db_dir=tmp_db_dir)
        db_b = MemoryDB(project="beta", db_dir=tmp_db_dir)
        eng_a = SearchEngine(db=db_a, embedder=FakeEmbedder())
        eng_b = SearchEngine(db=db_b, embedder=FakeEmbedder())
        eng_a.store(Memory(content="Alpha memory about databases"))
        eng_b.store(Memory(content="Beta memory about APIs"))
        eng_b.store(Memory(content="Beta memory about auth"))
        stats_a = db_a.get_stats()
        stats_b = db_b.get_stats()
        assert stats_a.total_chunks >= 1
        assert stats_b.total_chunks >= 2


class TestFTSRebuild:
    def test_rebuild_fts_restores_search(self, db):
        db.store_memory(Memory(content="PostgreSQL database"))
        # Verify search works
        assert len(db.fts_search("PostgreSQL")) >= 1

        # Clear FTS index content (simulates index corruption/staleness)
        conn = db._get_conn()
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('delete-all')")
        conn.commit()

        # Search should return nothing now
        assert db.fts_search("PostgreSQL") == []

        # Rebuild should fix it
        db.rebuild_fts()
        results = db.fts_search("PostgreSQL")
        assert len(results) >= 1


class TestTransactionSafety:
    def test_store_relationship_rejects_invalid_memory_id(self, db):
        rel = Relationship(source_id="nonexistent", target_id="also-fake")
        with pytest.raises(ValueError):
            db.store_relationship(rel)

    def test_delete_memory_atomic(self, db):
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        engine = SearchEngine(db=db, embedder=FakeEmbedder())
        mem = engine.store(Memory(content="To be atomically deleted"))
        mid = mem.id
        chunks_before = db.get_chunks_for_memory(mid)
        assert len(chunks_before) >= 1
        result = db.delete_memory_atomic(mid)
        assert result is True
        assert db.get_memory(mid) is None
        assert db.get_chunks_for_memory(mid) == []

    def test_delete_memory_atomic_nonexistent(self, db):
        result = db.delete_memory_atomic("nonexistent")
        assert result is False


class TestPruning:
    def test_prune_stale_memories(self, db: MemoryDB):
        old = Memory(content="Old and forgotten", importance=4)
        stored = db.store_memory(old)

        conn = db._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
        conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id = ?",
            (cutoff, stored.id),
        )
        conn.commit()

        pruned = db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 1
        assert db.get_memory(stored.id) is None

    def test_important_memories_survive_pruning(self, db: MemoryDB):
        important = Memory(content="Critical decision", importance=0)
        stored = db.store_memory(important)

        conn = db._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        conn.execute(
            "UPDATE memories SET last_accessed = ? WHERE id = ?",
            (cutoff, stored.id),
        )
        conn.commit()

        pruned = db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 0
        assert db.get_memory(stored.id) is not None


class TestThreadSafety:
    def test_concurrent_stores(self, tmp_db_dir):
        db = MemoryDB(project="threadsafe", db_dir=tmp_db_dir)
        errors = []

        def store_batch(start):
            try:
                for i in range(20):
                    db.store_memory(Memory(content=f"Thread {start} memory {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Errors: {errors}"
        mems = db.list_memories(limit=200)
        assert len(mems) == 100

    def test_close_releases_connection(self, tmp_db_dir):
        db = MemoryDB(project="closetest", db_dir=tmp_db_dir)
        db.store_memory(Memory(content="test"))
        db.close()
        assert db._conn is None

    def test_context_manager(self, tmp_db_dir):
        with MemoryDB(project="ctxtest", db_dir=tmp_db_dir) as db:
            db.store_memory(Memory(content="test"))
        assert db._conn is None


class TestSchemaMigration:
    def test_schema_version_stored(self, tmp_db_dir):
        db = MemoryDB(project="migration", db_dir=tmp_db_dir)
        version = db.get_meta("schema_version")
        assert version is not None
        assert int(version) >= 2

    def test_date_indexes_exist(self, tmp_db_dir):
        db = MemoryDB(project="indexes", db_dir=tmp_db_dir)
        conn = db._get_conn()
        indexes = conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
        index_names = {r["name"] for r in indexes}
        assert "idx_memories_last_accessed" in index_names
        assert "idx_memories_updated_at" in index_names
        assert "idx_memories_project" in index_names

    def test_old_db_gets_migrated(self, tmp_db_dir):
        db = MemoryDB(project="olddb", db_dir=tmp_db_dir)
        conn = db._get_conn()
        conn.execute("DELETE FROM project_meta WHERE key = 'schema_version'")
        conn.commit()
        db.close()
        db2 = MemoryDB(project="olddb", db_dir=tmp_db_dir)
        version = db2.get_meta("schema_version")
        assert version == "2"
