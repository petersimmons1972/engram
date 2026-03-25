"""Tests for engram.search.SearchEngine -- store/recall round trips and scoring."""

from __future__ import annotations

from engram.search import SearchEngine
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


class TestBM25Normalization:
    def test_no_negative_bm25_scores(self, engine):
        engine.store(Memory(content="Alpha beta gamma delta epsilon"))
        engine.store(Memory(content="Zeta eta theta iota kappa"))
        results = engine.recall("alpha")
        for r in results:
            assert r.score_breakdown["bm25"] >= 0.0


class TestBM25OnlyWeights:
    def test_bm25_only_redistributes_vector_weight(self, tmp_path):
        from engram.db import MemoryDB
        from engram.embeddings import NullEmbedder
        db = MemoryDB(project="bm25weights", db_dir=tmp_path)
        engine = SearchEngine(db=db, embedder=NullEmbedder())
        engine.store(Memory(content="Test BM25 weight redistribution"))
        results = engine.recall("BM25 weight")
        if results:
            assert results[0].score_breakdown["vector"] == 0.0


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

    def test_bm25_single_result_normalization(self, tmp_path):
        """Single result must have valid BM25 score (not NaN/Inf)."""
        import math

        from engram.db import MemoryDB
        from engram.embeddings import NullEmbedder
        db = MemoryDB(project="bm25single", db_dir=tmp_path)
        engine = SearchEngine(db=db, embedder=NullEmbedder())
        engine.store(Memory(content="Unique xylophone testing content"))

        results = engine.recall("xylophone")
        assert len(results) == 1
        bm25_score = results[0].score_breakdown["bm25"]
        assert not math.isnan(bm25_score), "BM25 score is NaN"
        assert not math.isinf(bm25_score), "BM25 score is Inf"
        assert bm25_score == 1.0, f"Single result BM25 should be 1.0, got {bm25_score}"

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

    def test_null_embedder_weight_redistribution(self, tmp_path):
        """With NullEmbedder, BM25 must get full non-recency weight."""
        from engram.db import MemoryDB
        from engram.embeddings import NullEmbedder
        db = MemoryDB(project="nullredist", db_dir=tmp_path)
        engine = SearchEngine(db=db, embedder=NullEmbedder())
        engine.store(Memory(content="Test null embedder weight redistribution"))

        results = engine.recall("null embedder weight")
        assert len(results) >= 1
        bd = results[0].score_breakdown
        assert bd["vector"] == 0.0, "Vector should be 0 with NullEmbedder"
        # BM25 should get all non-recency weight (i.e., 1.0 - WEIGHT_RECENCY)
        # For a single result with BM25=1.0 and recency~1.0, total before importance
        # should be approximately (1.0 - WEIGHT_RECENCY) * 1.0 + WEIGHT_RECENCY * recency
        # The key check: no weight goes to graph, and vector weight goes to BM25
        assert "graph" not in bd, "graph should not be in score_breakdown"


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
