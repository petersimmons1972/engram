"""Stress tests for memory_consolidate (engram's memify system).

Validates all three consolidation stages under accumulated load:
deduplication, edge decay/pruning, and stale memory pruning.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engram.db import MemoryDB
from engram.search import SearchEngine
from engram.types import Memory, MemoryType, Relationship, RelationType
from tests.conftest import FakeEmbedder


@pytest.fixture
def stress_engine(tmp_path: Path) -> SearchEngine:
    """Engine wired to a temp DB for consolidation stress tests."""
    db = MemoryDB(project="stress", db_dir=tmp_path)
    return SearchEngine(db=db, embedder=FakeEmbedder())


class TestChunkDeduplication:
    def test_dedup_removes_duplicate_chunks(self, stress_engine: SearchEngine):
        """Bypass store-level dedup by inserting chunks with embeddings directly."""
        from engram.chunker import chunk_hash
        from engram.embeddings import to_blob
        from engram.types import Chunk

        m = stress_engine.store(Memory(content="Base memory for dedup test"))
        fake_emb = to_blob(stress_engine.embedder.embed("dummy"))

        conn = stress_engine.db._get_conn()
        the_hash = chunk_hash("This exact chunk appears many times")
        for i in range(20):
            chunk = Chunk(
                memory_id=m.id,
                chunk_text="This exact chunk appears many times",
                chunk_index=i + 10,
                chunk_hash=the_hash,
                embedding=fake_emb,
            )
            conn.execute(
                "INSERT INTO chunks (id, memory_id, chunk_text, chunk_index, chunk_hash, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (chunk.id, chunk.memory_id, chunk.chunk_text, chunk.chunk_index, chunk.chunk_hash, chunk.embedding),
            )
        conn.commit()

        result = stress_engine.memify()
        assert result["chunks_deduped"] >= 19

    def test_dedup_preserves_unique_chunks(self, stress_engine: SearchEngine):
        """Store 100 unique memories -> consolidate should not remove any chunks."""
        for i in range(100):
            stress_engine.store(Memory(content=f"Unique memory number {i} about topic {i * 7}"))

        result = stress_engine.memify()
        stats_after = stress_engine.db.get_stats()

        assert stats_after.total_chunks >= 100


class TestEdgeDecayAndPruning:
    def test_weak_edges_pruned(self, stress_engine: SearchEngine):
        """Create edges with varying strengths -> weak ones should be pruned."""
        memories = []
        for i in range(20):
            m = Memory(content=f"Memory {i} for edge testing")
            memories.append(stress_engine.store(m))

        for i in range(0, 18, 2):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                rel_type=RelationType.RELATES_TO,
                strength=0.05 if i < 10 else 0.8,
            )
            stress_engine.db.store_relationship(rel)

        result = stress_engine.memify()

        assert result["edges_decayed"] >= 9
        assert result["edges_pruned"] >= 1

    def test_strong_edges_survive(self, stress_engine: SearchEngine):
        """Edges with strength > 0.1 should survive decay."""
        m1 = stress_engine.store(Memory(content="Strong edge source"))
        m2 = stress_engine.store(Memory(content="Strong edge target"))

        rel = Relationship(
            source_id=m1.id, target_id=m2.id,
            rel_type=RelationType.DEPENDS_ON,
            strength=1.0,
        )
        stress_engine.db.store_relationship(rel)

        stress_engine.memify()

        connected = stress_engine.db.get_connected(m1.id)
        assert len(connected) == 1
        assert connected[0][3] > 0.1

    def test_50_edges_decay_correctly(self, stress_engine: SearchEngine):
        """Create 50 edges, verify decay math is applied to all."""
        memories = []
        for i in range(51):
            memories.append(stress_engine.store(Memory(content=f"Node {i}")))

        for i in range(50):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                strength=0.5,
            )
            stress_engine.db.store_relationship(rel)

        result = stress_engine.memify()
        assert result["edges_decayed"] == 50


class TestStalePruning:
    def test_old_unaccessed_trivial_pruned(self, stress_engine: SearchEngine):
        """Old, never-accessed, low-importance memories should be pruned."""
        conn = stress_engine.db._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()

        for i in range(30):
            m = Memory(content=f"Stale memory {i}", importance=4)
            stored = stress_engine.db.store_memory(m)
            conn.execute(
                "UPDATE memories SET last_accessed = ? WHERE id = ?",
                (old_date, stored.id),
            )

        conn.commit()

        result = stress_engine.memify()
        assert result["stale_memories_pruned"] >= 25

    def test_important_old_memories_survive(self, stress_engine: SearchEngine):
        """High-importance memories should never be pruned regardless of age."""
        conn = stress_engine.db._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()

        for i in range(10):
            m = Memory(content=f"Critical decision {i}", importance=0)
            stored = stress_engine.db.store_memory(m)
            conn.execute(
                "UPDATE memories SET last_accessed = ? WHERE id = ?",
                (old_date, stored.id),
            )

        conn.commit()

        result = stress_engine.memify()
        assert result["stale_memories_pruned"] == 0

        stats = stress_engine.db.get_stats()
        assert stats.total_memories == 10

    def test_accessed_memories_survive(self, stress_engine: SearchEngine):
        """Memories that have been accessed should survive even if old and low-importance."""
        conn = stress_engine.db._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()

        for i in range(10):
            m = Memory(content=f"Accessed memory {i}", importance=4)
            stored = stress_engine.db.store_memory(m)
            stress_engine.db.touch_memory(stored.id)
            conn.execute(
                "UPDATE memories SET last_accessed = ? WHERE id = ?",
                (old_date, stored.id),
            )

        conn.commit()

        result = stress_engine.memify()
        assert result["stale_memories_pruned"] == 0


class TestIdempotency:
    def test_double_consolidation(self, stress_engine: SearchEngine):
        """Running consolidation twice in a row -> second run should report zero changes."""
        for i in range(50):
            stress_engine.store(Memory(content=f"Memory {i} for idempotency test"))

        result1 = stress_engine.memify()
        result2 = stress_engine.memify()

        assert result2["chunks_deduped"] == 0
        assert result2["edges_pruned"] == 0
        assert result2["stale_memories_pruned"] == 0


class TestPerformanceBenchmark:
    @pytest.mark.slow
    def test_consolidation_timing(self, stress_engine: SearchEngine):
        """Measure consolidation time on 200+ memories. Reports timing, does not assert."""
        conn = stress_engine.db._get_conn()
        old_date = (datetime.now(timezone.utc) - timedelta(days=45)).isoformat()

        for i in range(200):
            m = Memory(
                content=f"Benchmark memory {i} about various topics like auth databases and APIs",
                importance=3 if i % 3 == 0 else 2,
            )
            stored = stress_engine.db.store_memory(m)
            if i % 3 == 0:
                conn.execute(
                    "UPDATE memories SET last_accessed = ? WHERE id = ?",
                    (old_date, stored.id),
                )

        conn.commit()

        memories = stress_engine.db.list_memories(limit=200)
        for i in range(0, min(len(memories) - 1, 60), 2):
            rel = Relationship(
                source_id=memories[i].id,
                target_id=memories[i + 1].id,
                strength=0.3 if i < 30 else 0.8,
            )
            stress_engine.db.store_relationship(rel)

        start = time.perf_counter()
        result = stress_engine.memify()
        elapsed = time.perf_counter() - start

        print(f"\n=== Consolidation Benchmark ===")
        print(f"  Memories: 200")
        print(f"  Edges: ~30")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Chunks deduped: {result['chunks_deduped']}")
        print(f"  Edges decayed: {result['edges_decayed']}")
        print(f"  Edges pruned: {result['edges_pruned']}")
        print(f"  Stale pruned: {result['stale_memories_pruned']}")
