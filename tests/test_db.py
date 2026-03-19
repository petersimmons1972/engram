"""Tests for engram.db.MemoryDB -- CRUD, FTS, relationships, project isolation."""

from __future__ import annotations

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


class TestStats:
    def test_stats_reflect_stored_data(self, db: MemoryDB):
        db.store_memory(Memory(content="Decision 1", memory_type=MemoryType.DECISION))
        db.store_memory(Memory(content="Error 1", memory_type=MemoryType.ERROR))

        stats = db.get_stats()
        assert stats.total_memories == 2
        assert stats.by_type.get("decision") == 1
        assert stats.by_type.get("error") == 1


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
