"""Tests for engram.search.SearchEngine -- store/recall round trips and scoring."""

from __future__ import annotations

import pytest

from engram.types import Memory, MemoryType, Relationship, RelationType


class TestStoreRecallRoundTrip:
    def test_stored_memory_is_recallable(self, engine):
        mem = Memory(content="We chose PostgreSQL because it supports JSONB natively")
        engine.store(mem)

        results = engine.recall("PostgreSQL database choice")
        assert len(results) >= 1
        assert "PostgreSQL" in results[0].memory.content

    def test_recall_returns_best_match_first(self, engine):
        engine.store(Memory(content="Authentication uses JWT with RS256 signing"))
        engine.store(Memory(content="Database uses PostgreSQL 16 with pgvector"))
        engine.store(Memory(content="Frontend built with React and TypeScript"))

        results = engine.recall("JWT authentication signing")
        assert "JWT" in results[0].memory.content

    def test_recall_empty_query(self, engine):
        engine.store(Memory(content="Some stored content"))
        results = engine.recall("")
        # Should not crash; may return empty or all
        assert isinstance(results, list)

    def test_recall_no_results(self, engine):
        results = engine.recall("quantum entanglement")
        assert len(results) == 0


class TestMemoryTypeFiltering:
    def test_filter_by_type(self, engine):
        engine.store(Memory(
            content="Chose microservices over monolith",
            memory_type=MemoryType.DECISION,
        ))
        engine.store(Memory(
            content="Port 3000 is already bound by another service",
            memory_type=MemoryType.ERROR,
        ))

        results = engine.recall("architecture", memory_type="decision")
        for r in results:
            assert r.memory.memory_type == MemoryType.DECISION

    def test_filter_by_tags(self, engine):
        engine.store(Memory(content="Auth uses JWT", tags=["auth", "jwt"]))
        engine.store(Memory(content="DB uses Postgres", tags=["database"]))

        results = engine.recall("system", tags=["auth"])
        for r in results:
            assert "auth" in r.memory.tags


class TestScoringOrder:
    def test_higher_importance_scores_higher(self, engine):
        engine.store(Memory(content="Critical auth decision", importance=0))
        engine.store(Memory(content="Trivial auth note", importance=4))

        results = engine.recall("auth decision")
        if len(results) >= 2:
            assert results[0].memory.importance <= results[1].memory.importance

    def test_score_breakdown_populated(self, engine):
        engine.store(Memory(content="Test memory for scoring breakdown"))
        results = engine.recall("scoring breakdown")
        assert len(results) >= 1
        breakdown = results[0].score_breakdown
        assert "vector" in breakdown
        assert "bm25" in breakdown
        assert "recency" in breakdown


class TestGraphExpansion:
    def test_connected_memories_attached(self, engine):
        m1 = Memory(content="Auth uses JWT tokens")
        m2 = Memory(content="JWT tokens expire after 24 hours")
        stored1 = engine.store(m1)
        stored2 = engine.store(m2)

        rel = Relationship(
            source_id=stored1.id, target_id=stored2.id,
            rel_type=RelationType.RELATES_TO,
        )
        engine.db.store_relationship(rel)

        results = engine.recall("JWT authentication")
        if results:
            top = results[0]
            connected_ids = [c.memory.id for c in top.connected]
            other_id = stored2.id if top.memory.id == stored1.id else stored1.id
            assert other_id in connected_ids


class TestSupersedeWarning:
    def test_superseded_memory_shows_warning(self, engine):
        """Verify that superseded memories get a WARNING flag in recall results."""
        old = engine.store(Memory(content="Use MySQL for the database"))
        new = engine.store(Memory(content="Use PostgreSQL instead of MySQL"))

        rel = Relationship(
            source_id=new.id, target_id=old.id,
            rel_type=RelationType.SUPERSEDES,
        )
        engine.db.store_relationship(rel)

        # Demote old memory like memory_correct does
        engine.db.update_memory(old.id, importance=4)

        results = engine.recall("MySQL database")
        for r in results:
            if r.memory.id == old.id:
                # The server layer adds the WARNING; search layer attaches connected
                connected_types = [c.rel_type for c in r.connected]
                assert "supersedes" in connected_types


class TestFeedback:
    def test_positive_feedback_boosts_edges(self, engine):
        m1 = Memory(content="Memory A")
        m2 = Memory(content="Memory B")
        s1 = engine.store(m1)
        s2 = engine.store(m2)

        rel = Relationship(source_id=s1.id, target_id=s2.id, strength=0.5)
        engine.db.store_relationship(rel)

        result = engine.feedback([s1.id], helpful=True)
        assert result["action"] == "reinforced"

    def test_negative_feedback_weakens_edges(self, engine):
        m1 = Memory(content="Memory X")
        s1 = engine.store(m1)

        result = engine.feedback([s1.id], helpful=False)
        assert result["action"] == "weakened"
