"""Tests for engram.db_postgres.PostgresBackend.

All tests skip when TEST_DATABASE_URL is not set, so the suite remains green
without a running PostgreSQL instance.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)


@pytest.fixture
def pg_db():
    from engram.db_postgres import PostgresBackend

    dsn = os.environ["TEST_DATABASE_URL"]
    db = PostgresBackend(project="test_pg", dsn=dsn)
    yield db
    # Cleanup all test data (including other_project from isolation tests)
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM relationships")
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM project_meta")
        conn.commit()
    db.close()


# ── Connection & Schema ──────────────────────────────────────────────────


class TestConnection:
    def test_creates_schema_on_init(self, pg_db):
        """PostgresBackend should create all tables on first connection."""
        with pg_db.pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM information_schema.tables "
                "WHERE table_name IN ('memories', 'chunks', 'relationships', 'project_meta')"
            ).fetchone()
            assert row["c"] == 4

    def test_context_manager(self, pg_db):
        """Context manager should work without error."""
        from engram.db_postgres import PostgresBackend

        dsn = os.environ["TEST_DATABASE_URL"]
        with PostgresBackend(project="test_pg", dsn=dsn) as db:
            assert db.project == "test_pg"


# ── Project Metadata ─────────────────────────────────────────────────────


class TestMeta:
    def test_get_missing_key_returns_none(self, pg_db):
        assert pg_db.get_meta("nonexistent") is None

    def test_set_and_get_roundtrip(self, pg_db):
        pg_db.set_meta("version", "42")
        assert pg_db.get_meta("version") == "42"

    def test_set_overwrites(self, pg_db):
        pg_db.set_meta("k", "v1")
        pg_db.set_meta("k", "v2")
        assert pg_db.get_meta("k") == "v2"


# ── Memory CRUD ──────────────────────────────────────────────────────────

from engram.types import Chunk, Memory, MemoryType, Relationship, RelationType


class TestMemoryCRUD:
    def test_store_and_retrieve(self, pg_db):
        mem = Memory(content="PostgreSQL chosen for the main database")
        stored = pg_db.store_memory(mem)
        assert stored.project == "test_pg"

        retrieved = pg_db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.content == "PostgreSQL chosen for the main database"
        assert retrieved.project == "test_pg"

    def test_get_nonexistent_returns_none(self, pg_db):
        assert pg_db.get_memory("does-not-exist") is None

    def test_update_content(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Old content", tags=["old"]))
        updated = pg_db.update_memory(mem.id, content="New content", tags=["new"])
        assert updated is not None
        assert updated.content == "New content"
        assert updated.tags == ["new"]

    def test_update_nonexistent_returns_none(self, pg_db):
        assert pg_db.update_memory("fake-id", content="x") is None

    def test_delete_memory(self, pg_db):
        mem = pg_db.store_memory(Memory(content="To be deleted"))
        assert pg_db.delete_memory(mem.id) is True
        assert pg_db.get_memory(mem.id) is None

    def test_delete_nonexistent_returns_false(self, pg_db):
        assert pg_db.delete_memory("nope") is False

    def test_delete_memory_atomic(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Atomic delete target"))
        # Store a chunk for this memory
        chunk = Chunk(memory_id=mem.id, chunk_text="chunk", chunk_index=0, chunk_hash="h1")
        pg_db.store_chunks([chunk])
        # Create a second memory and a relationship
        m2 = pg_db.store_memory(Memory(content="Related"))
        rel = Relationship(source_id=mem.id, target_id=m2.id)
        pg_db.store_relationship(rel)

        result = pg_db.delete_memory_atomic(mem.id)
        assert result is True
        assert pg_db.get_memory(mem.id) is None
        assert pg_db.get_chunks_for_memory(mem.id) == []
        assert pg_db.get_connection_count(mem.id) == 0

    def test_delete_memory_atomic_nonexistent(self, pg_db):
        assert pg_db.delete_memory_atomic("nonexistent") is False

    def test_touch_increments_access(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Touch me"))
        pg_db.touch_memory(mem.id)
        pg_db.touch_memory(mem.id)
        retrieved = pg_db.get_memory(mem.id)
        assert retrieved.access_count == 2

    def test_list_memories_default(self, pg_db):
        pg_db.store_memory(Memory(content="A"))
        pg_db.store_memory(Memory(content="B"))
        mems = pg_db.list_memories()
        assert len(mems) == 2

    def test_list_memories_filters_by_type(self, pg_db):
        pg_db.store_memory(Memory(content="Decision", memory_type=MemoryType.DECISION))
        pg_db.store_memory(Memory(content="Error", memory_type=MemoryType.ERROR))
        decisions = pg_db.list_memories(memory_type=MemoryType.DECISION)
        assert len(decisions) == 1
        assert decisions[0].memory_type == MemoryType.DECISION

    def test_list_memories_filters_by_importance(self, pg_db):
        pg_db.store_memory(Memory(content="Critical", importance=0))
        pg_db.store_memory(Memory(content="Trivial", importance=4))
        critical = pg_db.list_memories(min_importance=0)
        assert len(critical) == 1
        assert critical[0].content == "Critical"

    def test_list_memories_filters_by_tags(self, pg_db):
        pg_db.store_memory(Memory(content="Auth stuff", tags=["auth", "jwt"]))
        pg_db.store_memory(Memory(content="DB stuff", tags=["postgres", "sql"]))
        auth = pg_db.list_memories(tags=["auth"])
        assert len(auth) == 1
        assert "auth" in auth[0].tags

    def test_list_memories_limit_offset(self, pg_db):
        for i in range(5):
            pg_db.store_memory(Memory(content=f"Mem {i}"))
        page1 = pg_db.list_memories(limit=2, offset=0)
        page2 = pg_db.list_memories(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {m.id for m in page1}
        ids2 = {m.id for m in page2}
        assert ids1.isdisjoint(ids2)


# ── Chunk CRUD ───────────────────────────────────────────────────────────


class TestChunkCRUD:
    def test_store_and_get_chunks(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        chunks = [
            Chunk(memory_id=mem.id, chunk_text="First chunk", chunk_index=0, chunk_hash="h0"),
            Chunk(memory_id=mem.id, chunk_text="Second chunk", chunk_index=1, chunk_hash="h1"),
        ]
        pg_db.store_chunks(chunks)
        retrieved = pg_db.get_chunks_for_memory(mem.id)
        assert len(retrieved) == 2
        assert retrieved[0].chunk_index == 0
        assert retrieved[1].chunk_index == 1

    def test_chunk_hash_exists(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        pg_db.store_chunks([
            Chunk(memory_id=mem.id, chunk_text="data", chunk_index=0, chunk_hash="unique_hash"),
        ])
        assert pg_db.chunk_hash_exists("unique_hash") is True
        assert pg_db.chunk_hash_exists("nonexistent") is False

    def test_delete_chunks_for_memory(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        pg_db.store_chunks([
            Chunk(memory_id=mem.id, chunk_text="data", chunk_index=0, chunk_hash="dh"),
        ])
        pg_db.delete_chunks_for_memory(mem.id)
        assert pg_db.get_chunks_for_memory(mem.id) == []

    def test_delete_chunks_by_ids(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        c1 = Chunk(memory_id=mem.id, chunk_text="a", chunk_index=0, chunk_hash="da")
        c2 = Chunk(memory_id=mem.id, chunk_text="b", chunk_index=1, chunk_hash="db")
        pg_db.store_chunks([c1, c2])
        deleted = pg_db.delete_chunks_by_ids([c1.id])
        assert deleted == 1
        remaining = pg_db.get_chunks_for_memory(mem.id)
        assert len(remaining) == 1
        assert remaining[0].id == c2.id

    def test_delete_chunks_by_ids_empty_list(self, pg_db):
        assert pg_db.delete_chunks_by_ids([]) == 0

    def test_store_chunks_with_embedding(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        embedding = b"\x00\x01\x02\x03"
        pg_db.store_chunks([
            Chunk(memory_id=mem.id, chunk_text="embed", chunk_index=0,
                  chunk_hash="eh", embedding=embedding),
        ])
        chunks = pg_db.get_chunks_for_memory(mem.id)
        assert chunks[0].embedding == embedding

    def test_get_all_chunks_with_embeddings(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        pg_db.store_chunks([
            Chunk(memory_id=mem.id, chunk_text="has embed", chunk_index=0,
                  chunk_hash="e1", embedding=b"\x01"),
            Chunk(memory_id=mem.id, chunk_text="no embed", chunk_index=1,
                  chunk_hash="e2", embedding=None),
        ])
        with_emb = pg_db.get_all_chunks_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0].chunk_text == "has embed"

    def test_get_all_chunk_texts(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Parent"))
        pg_db.store_chunks([
            Chunk(memory_id=mem.id, chunk_text="text alpha", chunk_index=0, chunk_hash="ta"),
            Chunk(memory_id=mem.id, chunk_text="text beta", chunk_index=1, chunk_hash="tb"),
        ])
        texts = pg_db.get_all_chunk_texts()
        assert "text alpha" in texts
        assert "text beta" in texts

    def test_duplicate_chunk_ignored(self, pg_db):
        """Storing a chunk with the same ID twice should not error (ON CONFLICT DO NOTHING)."""
        mem = pg_db.store_memory(Memory(content="Parent"))
        c = Chunk(memory_id=mem.id, chunk_text="dup", chunk_index=0, chunk_hash="ddup")
        pg_db.store_chunks([c])
        pg_db.store_chunks([c])  # should not raise
        assert len(pg_db.get_chunks_for_memory(mem.id)) == 1


# ── Relationship CRUD ────────────────────────────────────────────────────


class TestRelationships:
    def test_store_and_get_connected(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="Memory A"))
        m2 = pg_db.store_memory(Memory(content="Memory B"))
        rel = Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO, strength=0.8,
        )
        pg_db.store_relationship(rel)
        connected = pg_db.get_connected(m1.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][0].id == m2.id
        assert connected[0][1] == "relates_to"
        assert connected[0][2] == "outgoing"

    def test_boost_edges(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id, strength=0.5,
        ))
        count = pg_db.boost_edges_for_memory(m1.id, factor=0.2)
        assert count == 1
        connected = pg_db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.7, abs=0.01)

    def test_decay_edges(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id, strength=0.5,
        ))
        count = pg_db.decay_edges_for_memory(m1.id, factor=0.3)
        assert count == 1
        connected = pg_db.get_connected(m1.id)
        assert connected[0][3] == pytest.approx(0.2, abs=0.01)

    def test_connection_count(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        m3 = pg_db.store_memory(Memory(content="C"))
        pg_db.store_relationship(Relationship(source_id=m1.id, target_id=m2.id))
        pg_db.store_relationship(Relationship(source_id=m3.id, target_id=m1.id))
        assert pg_db.get_connection_count(m1.id) == 2

    def test_reject_invalid_source(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="Real"))
        with pytest.raises(ValueError, match="Source memory"):
            pg_db.store_relationship(Relationship(source_id="fake", target_id=m1.id))

    def test_reject_invalid_target(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="Real"))
        with pytest.raises(ValueError, match="Target memory"):
            pg_db.store_relationship(Relationship(source_id=m1.id, target_id="fake"))

    def test_decay_all_edges(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id, strength=0.5,
        ))
        decayed, pruned = pg_db.decay_all_edges(decay_factor=0.02, min_strength=0.1)
        assert decayed >= 1

    def test_decay_all_edges_prunes_weak(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id, strength=0.05,
        ))
        _decayed, pruned = pg_db.decay_all_edges(decay_factor=0.02, min_strength=0.1)
        assert pruned >= 1

    def test_delete_relationships_for_memory(self, pg_db):
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(source_id=m1.id, target_id=m2.id))
        pg_db.delete_relationships_for_memory(m1.id)
        assert pg_db.get_connection_count(m1.id) == 0

    def test_upsert_on_duplicate_pair(self, pg_db):
        """Storing a duplicate (source, target, rel_type) should update strength."""
        m1 = pg_db.store_memory(Memory(content="A"))
        m2 = pg_db.store_memory(Memory(content="B"))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO, strength=0.5,
        ))
        pg_db.store_relationship(Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO, strength=0.9,
        ))
        connected = pg_db.get_connected(m1.id)
        assert len(connected) == 1
        assert connected[0][3] == pytest.approx(0.9, abs=0.01)


# ── FTS Search ───────────────────────────────────────────────────────────


class TestFTSSearch:
    def test_basic_search(self, pg_db):
        pg_db.store_memory(Memory(content="JWT authentication with refresh tokens"))
        pg_db.store_memory(Memory(content="Database migration using alembic"))
        results = pg_db.fts_search("JWT authentication")
        assert len(results) >= 1
        assert "JWT" in results[0][0].content

    def test_empty_query_returns_empty(self, pg_db):
        pg_db.store_memory(Memory(content="Some content"))
        assert pg_db.fts_search("") == []

    def test_no_match_returns_empty(self, pg_db):
        pg_db.store_memory(Memory(content="Python web framework"))
        results = pg_db.fts_search("quantum entanglement")
        assert len(results) == 0

    def test_fts_positive_scores(self, pg_db):
        pg_db.store_memory(Memory(content="Search scoring should be positive"))
        results = pg_db.fts_search("scoring positive")
        if results:
            assert results[0][1] > 0

    def test_rebuild_fts(self, pg_db):
        """rebuild_fts should not error (it runs REINDEX)."""
        pg_db.store_memory(Memory(content="Rebuild test"))
        pg_db.rebuild_fts()  # should not raise


# ── Stats ────────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_reflect_stored_data(self, pg_db):
        pg_db.store_memory(Memory(content="Decision 1", memory_type=MemoryType.DECISION))
        pg_db.store_memory(Memory(content="Error 1", memory_type=MemoryType.ERROR))
        stats = pg_db.get_stats()
        assert stats.total_memories == 2
        assert stats.by_type.get("decision") == 1
        assert stats.by_type.get("error") == 1
        assert stats.db_size_bytes == 0  # Postgres: no single file

    def test_stats_empty(self, pg_db):
        stats = pg_db.get_stats()
        assert stats.total_memories == 0


# ── Project Isolation ────────────────────────────────────────────────────


class TestProjectIsolation:
    def test_memories_filtered_by_project(self, pg_db):
        """Memories from another project should not appear in list_memories."""
        pg_db.store_memory(Memory(content="test_pg project memory"))
        # Insert a memory for a different project directly
        with pg_db.pool.connection() as conn:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            conn.execute(
                "INSERT INTO memories (id, content, memory_type, project, tags, "
                "importance, access_count, last_accessed, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                ("other_proj_id", "other project mem", "context", "other_project",
                 "[]", 2, 0, now, now, now),
            )
            conn.commit()
        mems = pg_db.list_memories()
        assert all(m.project == "test_pg" for m in mems)
        assert len(mems) == 1

    def test_fts_isolated_by_project(self, pg_db):
        """FTS results should be filtered by the backend's project."""
        with pg_db.pool.connection() as conn:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            conn.execute(
                "INSERT INTO memories (id, content, memory_type, project, tags, "
                "importance, access_count, last_accessed, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                ("iso_id", "unique xylophone content", "context", "other_project",
                 "[]", 2, 0, now, now, now),
            )
            conn.commit()
        results = pg_db.fts_search("xylophone")
        assert len(results) == 0


# ── Pruning ──────────────────────────────────────────────────────────────


class TestPruning:
    def test_prune_stale_memories(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Old forgotten", importance=4))
        # Backdate last_accessed
        cutoff = datetime.now(timezone.utc) - timedelta(days=31)
        with pg_db.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = %s WHERE id = %s",
                (cutoff, mem.id),
            )
            conn.commit()
        pruned = pg_db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 1
        assert pg_db.get_memory(mem.id) is None

    def test_important_memories_survive_pruning(self, pg_db):
        mem = pg_db.store_memory(Memory(content="Critical decision", importance=0))
        cutoff = datetime.now(timezone.utc) - timedelta(days=60)
        with pg_db.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = %s WHERE id = %s",
                (cutoff, mem.id),
            )
            conn.commit()
        pruned = pg_db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 0
        assert pg_db.get_memory(mem.id) is not None
