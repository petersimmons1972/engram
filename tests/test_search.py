"""Tests for engram.search.SearchEngine -- store/recall round trips and scoring."""

from __future__ import annotations

import os

import pytest

from engram.search import SearchEngine
from engram.types import Memory, MemoryType, Relationship, RelationType

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)


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


class TestBM25Normalization:
    def test_no_negative_bm25_scores(self, engine):
        engine.store(Memory(content="Alpha beta gamma delta epsilon"))
        engine.store(Memory(content="Zeta eta theta iota kappa"))
        results = engine.recall("alpha")
        for r in results:
            assert r.score_breakdown["bm25"] >= 0.0


class TestBM25OnlyWeights:
    def test_bm25_only_redistributes_vector_weight(self):
        from engram.db_postgres import PostgresBackend
        from engram.embeddings import NullEmbedder
        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="bm25weights", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=NullEmbedder())
            engine.store(Memory(content="Test BM25 weight redistribution"))
            results = engine.recall("BM25 weight")
            if results:
                assert results[0].score_breakdown["vector"] == 0.0
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("bm25weights",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("bm25weights",))
                conn.commit()
            db.close()


class TestStoreDedup:
    def test_hash_dedup_prevents_duplicate_chunks(self, engine):
        engine.store(Memory(content="Exact duplicate content for hash test"))
        engine.store(Memory(content="Exact duplicate content for hash test"))
        stats = engine.db.get_stats()
        assert stats.total_chunks == 1


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


class TestSimplifiedScoring:
    """Tests for B3: simplified scoring — graph removed, importance bounded, BM25 fixes."""

    def test_no_graph_in_score_breakdown(self, engine):
        """score_breakdown must not have a 'graph' key."""
        engine.store(Memory(content="Test memory for score breakdown check"))
        results = engine.recall("score breakdown check")
        assert len(results) >= 1
        assert "graph" not in results[0].score_breakdown

    def test_weights_sum_to_one(self):
        """New scoring weights must sum to 1.0."""
        from engram.search import WEIGHT_BM25, WEIGHT_RECENCY, WEIGHT_VECTOR
        total = WEIGHT_BM25 + WEIGHT_VECTOR + WEIGHT_RECENCY
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_importance_multiplier_range_bounded(self, engine):
        """Importance 0 multiplier <= 1.5x, importance 4 >= 0.7x. Max variance <= 2x."""
        engine.store(Memory(content="Critical auth decision about tokens", importance=0))
        engine.store(Memory(content="Trivial auth note about tokens", importance=4))

        results = engine.recall("auth tokens")
        assert len(results) >= 2

        mults = {}
        for r in results:
            mults[r.memory.importance] = r.score_breakdown["importance_mult"]

        assert mults[0] <= 1.5, f"Importance 0 multiplier {mults[0]} > 1.5"
        assert mults[4] >= 0.7, f"Importance 4 multiplier {mults[4]} < 0.7"
        variance = mults[0] / mults[4]
        assert variance <= 2.0, f"Importance variance {variance} > 2.0"

    def test_bm25_single_result_normalization(self):
        """Single result must have valid BM25 score (not NaN/Inf)."""
        import math

        from engram.db_postgres import PostgresBackend
        from engram.embeddings import NullEmbedder
        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="bm25single", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=NullEmbedder())
            engine.store(Memory(content="Unique xylophone testing content"))

            results = engine.recall("xylophone")
            assert len(results) == 1
            bm25_score = results[0].score_breakdown["bm25"]
            assert not math.isnan(bm25_score), "BM25 score is NaN"
            assert not math.isinf(bm25_score), "BM25 score is Inf"
            assert bm25_score == 1.0, f"Single result BM25 should be 1.0, got {bm25_score}"
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("bm25single",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("bm25single",))
                conn.commit()
            db.close()

    def test_connected_memories_still_attached(self, engine):
        """Graph enrichment must survive — connected list populated."""
        m1 = engine.store(Memory(content="Auth uses JWT tokens for API"))
        m2 = engine.store(Memory(content="JWT tokens expire after 24 hours"))

        rel = Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.RELATES_TO,
        )
        engine.db.store_relationship(rel)

        results = engine.recall("JWT authentication tokens")
        assert len(results) >= 1
        top = results[0]
        connected_ids = [c.memory.id for c in top.connected]
        other_id = m2.id if top.memory.id == m1.id else m1.id
        assert other_id in connected_ids, "Connected memories not attached"

    def test_null_embedder_weight_redistribution(self):
        """With NullEmbedder, BM25 must get full non-recency weight."""
        from engram.db_postgres import PostgresBackend
        from engram.embeddings import NullEmbedder
        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="nullredist", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=NullEmbedder())
            engine.store(Memory(content="Test null embedder weight redistribution"))

            results = engine.recall("null embedder weight")
            assert len(results) >= 1
            bd = results[0].score_breakdown
            assert bd["vector"] == 0.0, "Vector should be 0 with NullEmbedder"
            assert "graph" not in bd, "graph should not be in score_breakdown"
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("nullredist",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("nullredist",))
                conn.commit()
            db.close()


class TestTransactionalStore:
    """B6: Transactional store — atomic memory+chunks+embeddings with rollback."""

    def test_failed_embedding_leaves_no_orphan(self):
        """Mock embedder to raise. Assert no orphan memory or chunks in DB."""
        from unittest.mock import MagicMock

        from engram.db_postgres import PostgresBackend
        from engram.search import SearchEngine

        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="txntest1", dsn=dsn)
        try:
            embedder = MagicMock()
            embedder.name = "mock/embedder"
            embedder.dimensions = 64
            embedder.version = "v1-mock"
            engine = SearchEngine(db=db, embedder=embedder)

            embedder.embed_batch.side_effect = RuntimeError("Embedding service down")

            mem = Memory(content="This should not persist if embedding fails")
            with pytest.raises(ValueError, match="Failed to store memory"):
                engine.store(mem)

            stats = db.get_stats()
            assert stats.total_memories == 0, f"Orphan memory found: {stats.total_memories}"
            assert stats.total_chunks == 0, f"Orphan chunks found: {stats.total_chunks}"
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("txntest1",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("txntest1",))
                conn.commit()
            db.close()

    def test_failed_chunk_store_rolls_back_memory(self):
        """Mock db.store_chunks to raise. Assert memory is also gone."""
        from unittest.mock import patch

        from engram.db_postgres import PostgresBackend
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="txntest2", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=FakeEmbedder())

            with patch.object(db, "store_chunks", side_effect=RuntimeError("DB write error")):
                with pytest.raises((RuntimeError, ValueError)):
                    engine.store(Memory(content="This should not persist if chunk store fails"))

            stats = db.get_stats()
            assert stats.total_memories == 0, f"Orphan memory found: {stats.total_memories}"
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("txntest2",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("txntest2",))
                conn.commit()
            db.close()

    def test_successful_store_is_atomic(self):
        """Store succeeds: both memory and chunks appear together."""
        from engram.db_postgres import PostgresBackend
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="txntest3", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=FakeEmbedder())

            stored = engine.store(Memory(content="Atomic store test memory"))
            stats = db.get_stats()
            assert stats.total_memories == 1
            assert stats.total_chunks >= 1

            db.delete_memory_atomic(stored.id)
            stats2 = db.get_stats()
            assert stats2.total_memories == 0
            assert stats2.total_chunks == 0
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("txntest3",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("txntest3",))
                conn.commit()
            db.close()


class TestGoldenQueryRegression:
    """Golden query regression tests — capture ranking baseline and verify after refactor."""

    def _setup_golden_memories(self, engine):
        """Store 10 representative memories covering different types."""
        memories = [
            Memory(
                content="We decided to use PostgreSQL because it supports JSONB",
                memory_type=MemoryType.DECISION, importance=1,
                tags=["database", "architecture"],
            ),
            Memory(
                content="The auth flow uses JWT tokens with RS256 signing",
                memory_type=MemoryType.ARCHITECTURE, importance=1,
                tags=["auth", "security"],
            ),
            Memory(
                content="Port 3000 conflict error running the dev server",
                memory_type=MemoryType.ERROR, importance=2,
                tags=["dev", "error"],
            ),
            Memory(
                content="Always use TypeScript strict mode for frontend",
                memory_type=MemoryType.PREFERENCE, importance=0,
                tags=["frontend", "typescript"],
            ),
            Memory(
                content="Codebase follows repository pattern for DB access",
                memory_type=MemoryType.PATTERN, importance=1,
                tags=["architecture", "database"],
            ),
            Memory(
                content="Docker containers deployed on Kubernetes with Helm",
                memory_type=MemoryType.CONTEXT, importance=2,
                tags=["deployment", "k8s"],
            ),
            Memory(
                content="React components use functional hooks, no classes",
                memory_type=MemoryType.PATTERN, importance=1,
                tags=["frontend", "react"],
            ),
            Memory(
                content="API rate limiting 100 requests per minute per user",
                memory_type=MemoryType.DECISION, importance=2,
                tags=["api", "security"],
            ),
            Memory(
                content="CI pipeline runs pytest then ruff then Docker build",
                memory_type=MemoryType.CONTEXT, importance=3,
                tags=["ci", "testing"],
            ),
            Memory(
                content="User prefers dark mode and monospace fonts",
                memory_type=MemoryType.PREFERENCE, importance=0,
                tags=["preferences", "ui"],
            ),
        ]
        stored = []
        for m in memories:
            stored.append(engine.store(m))
        return stored

    def test_golden_query_database_decision(self, engine):
        """Query about database decisions should return PostgreSQL memory in top 3."""
        self._setup_golden_memories(engine)
        results = engine.recall("PostgreSQL JSONB database")
        assert len(results) >= 1
        top_3 = " ".join(r.memory.content for r in results[:3])
        assert "PostgreSQL" in top_3

    def test_golden_query_auth_architecture(self, engine):
        """Query about authentication should return JWT memory first."""
        self._setup_golden_memories(engine)
        results = engine.recall("JWT authentication RS256 signing")
        assert len(results) >= 1
        assert "JWT" in results[0].memory.content

    def test_golden_query_deployment(self, engine):
        """Query about deployment should return Kubernetes memory in top 3."""
        self._setup_golden_memories(engine)
        results = engine.recall("deployment containers Kubernetes")
        assert len(results) >= 1
        top_3_content = " ".join(r.memory.content for r in results[:3])
        assert "Kubernetes" in top_3_content

    def test_golden_query_frontend_patterns(self, engine):
        """Query about frontend should return React/TypeScript memories."""
        self._setup_golden_memories(engine)
        results = engine.recall("frontend React TypeScript")
        assert len(results) >= 1
        top_3_content = " ".join(r.memory.content for r in results[:3])
        assert "React" in top_3_content or "TypeScript" in top_3_content

    def test_golden_query_error_resolution(self, engine):
        """Query about port errors should return port 3000 memory."""
        self._setup_golden_memories(engine)
        results = engine.recall("port 3000 error conflict dev server")
        assert len(results) >= 1
        assert "3000" in results[0].memory.content


class TestFTSFirst:
    """B4: FTS-first retrieval — vector search scoped to FTS candidates only."""

    def test_fts_candidates_limit_vector_scope(self):
        """Store many memories. Recall should NOT load all embeddings."""
        from unittest.mock import patch

        from engram.db_postgres import PostgresBackend
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="ftsscope", dsn=dsn)
        try:
            embedder = FakeEmbedder()
            engine = SearchEngine(db=db, embedder=embedder)

            for i in range(30):
                engine.store(Memory(content=f"Memory number {i} about topic alpha bravo charlie"))

            original = db.get_all_chunks_with_embeddings
            with patch.object(db, "get_all_chunks_with_embeddings", wraps=original) as spy:
                results = engine.recall("alpha bravo charlie", top_k=5)
                assert len(results) >= 1
                spy.assert_not_called()
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("ftsscope",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("ftsscope",))
                conn.commit()
            db.close()

    def test_fts_first_still_returns_semantic_matches(self, engine):
        """Vector re-rank must still work on FTS candidates."""
        engine.store(Memory(content="PostgreSQL is our database"))
        engine.store(Memory(content="We use Postgres for persistence"))

        results = engine.recall("relational database choice")
        assert len(results) >= 1
        top_content = " ".join(r.memory.content for r in results)
        assert "database" in top_content.lower() or "Postgres" in top_content

    def test_bm25_only_mode_unchanged(self):
        """NullEmbedder path must still work with FTS-first."""
        from engram.db_postgres import PostgresBackend
        from engram.embeddings import NullEmbedder
        from engram.search import SearchEngine

        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="bm25only", dsn=dsn)
        try:
            engine = SearchEngine(db=db, embedder=NullEmbedder())
            engine.store(Memory(content="BM25 only mode testing with unique words"))
            results = engine.recall("BM25 testing unique")
            assert len(results) >= 1
            assert "BM25" in results[0].memory.content
        finally:
            with db.pool.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE project = %s", ("bm25only",))
                conn.execute("DELETE FROM memories WHERE project = %s", ("bm25only",))
                conn.commit()
            db.close()

    def test_vector_fallback_when_fts_insufficient(self, engine):
        """When FTS returns < top_k, vector fills remaining slots."""
        engine.store(Memory(content="The cat sat on the mat quietly"))
        engine.store(Memory(content="Dogs are loyal animals and great pets"))

        results = engine.recall("loyal pets animals", top_k=5)
        assert len(results) >= 1
        assert any("loyal" in r.memory.content for r in results)

    def test_get_chunks_for_memories_method_exists(self):
        """DB backend must have get_chunks_for_memories method."""
        from engram.db_postgres import PostgresBackend
        dsn = os.environ["TEST_DATABASE_URL"]
        db = PostgresBackend(project="methodcheck", dsn=dsn)
        assert hasattr(db, "get_chunks_for_memories"), (
            "PostgresBackend must implement get_chunks_for_memories(memory_ids)"
        )
        db.close()
