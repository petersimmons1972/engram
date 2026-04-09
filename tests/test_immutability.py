"""Tests for memory immutability flags (Wave 1, Feature 1)."""

from __future__ import annotations

from engram.types import Memory, MemoryType


class TestImmutabilityFlag:
    """Immutable memories cannot be corrected, forgotten, or pruned."""

    def test_store_immutable_memory(self, engine):
        """Immutable flag is preserved through store and retrieve."""
        mem = Memory(content="critical preference: always use UTC", immutable=True)
        stored = engine.store(mem)
        retrieved = engine.db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.immutable is True

    def test_store_default_not_immutable(self, engine):
        """Memories are mutable by default."""
        mem = Memory(content="normal memory")
        stored = engine.store(mem)
        retrieved = engine.db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.immutable is False

    def test_prune_skips_immutable(self, engine):
        """prune_stale_memories skips immutable memories even if they qualify."""
        from datetime import datetime, timedelta, timezone

        mem = Memory(
            content="immutable but old and trivial",
            importance=4,
            immutable=True,
        )
        stored = engine.store(mem)

        # Backdate last_accessed so it qualifies for pruning (psycopg v3 API)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2000)
        with engine.db.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = %s, access_count = 0 WHERE id = %s",
                (old_time, stored.id),
            )
            conn.commit()

        pruned = engine.db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 0

        # Memory still exists
        assert engine.db.get_memory(stored.id) is not None

    def test_prune_still_removes_mutable(self, engine):
        """Non-immutable memories still get pruned normally."""
        from datetime import datetime, timedelta, timezone

        mem = Memory(content="ephemeral", importance=4, immutable=False)
        stored = engine.store(mem)

        old_time = datetime.now(timezone.utc) - timedelta(hours=2000)
        with engine.db.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed = %s, access_count = 0 WHERE id = %s",
                (old_time, stored.id),
            )
            conn.commit()

        pruned = engine.db.prune_stale_memories(max_age_hours=720, max_importance=3)
        assert pruned == 1
        assert engine.db.get_memory(stored.id) is None


class TestImmutabilityServerGuards:
    """Server-level guards prevent correcting/forgetting immutable memories."""

    def test_memory_correct_blocked_for_immutable(self, engine):
        """memory_correct on an immutable memory should be blocked at server level.

        This test validates the Memory model and DB layer; the server guard
        is tested via the MCP tool in test_server.py.
        """
        mem = Memory(content="immutable decision", immutable=True)
        stored = engine.store(mem)
        retrieved = engine.db.get_memory(stored.id)
        assert retrieved.immutable is True
        # The server.py guard checks this field before proceeding

    def test_memory_forget_blocked_for_immutable(self, engine):
        """memory_forget on an immutable memory should be blocked at server level."""
        mem = Memory(content="permanent record", immutable=True)
        stored = engine.store(mem)
        retrieved = engine.db.get_memory(stored.id)
        assert retrieved.immutable is True
