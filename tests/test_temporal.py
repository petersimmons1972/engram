"""Tests for temporal query operators (since/before) on memory_recall."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from engram.types import Memory


class TestTemporalFilters:
    """Recall with since/before parameters filters by created_at timestamp."""

    def _store_and_backdate(self, engine, content: str, hours_ago: float) -> Memory:
        """Store a memory then backdate its created_at by hours_ago hours."""
        mem = Memory(content=content)
        stored = engine.store(mem)
        target_time = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
        conn = engine.db._get_conn()
        conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            (target_time, stored.id),
        )
        conn.commit()
        # Re-read to get the updated created_at
        return engine.db.get_memory(stored.id)

    def test_recall_since_filters_old_memories(self, engine):
        """Recall with since should only return memories created after that time."""
        old = self._store_and_backdate(engine, "old memory about deployment", hours_ago=48)
        new = self._store_and_backdate(engine, "new memory about deployment", hours_ago=1)

        since = datetime.now(timezone.utc) - timedelta(hours=24)
        results = engine.recall("deployment", since=since)

        result_ids = [r.memory.id for r in results]
        assert new.id in result_ids
        assert old.id not in result_ids

    def test_recall_before_filters_new_memories(self, engine):
        """Recall with before should only return memories created before that time."""
        old = self._store_and_backdate(engine, "old memory about authentication", hours_ago=48)
        new = self._store_and_backdate(engine, "new memory about authentication", hours_ago=1)

        before = datetime.now(timezone.utc) - timedelta(hours=24)
        results = engine.recall("authentication", before=before)

        result_ids = [r.memory.id for r in results]
        assert old.id in result_ids
        assert new.id not in result_ids

    def test_recall_since_and_before_window(self, engine):
        """Recall with both since and before creates a time window filter."""
        ancient = self._store_and_backdate(engine, "ancient memory about database migration", hours_ago=96)
        middle = self._store_and_backdate(engine, "middle memory about database migration", hours_ago=48)
        recent = self._store_and_backdate(engine, "recent memory about database migration", hours_ago=1)

        since = datetime.now(timezone.utc) - timedelta(hours=72)
        before = datetime.now(timezone.utc) - timedelta(hours=24)
        results = engine.recall("database migration", since=since, before=before)

        result_ids = [r.memory.id for r in results]
        assert middle.id in result_ids
        assert ancient.id not in result_ids
        assert recent.id not in result_ids

    def test_recall_without_temporal_filters_unchanged(self, engine):
        """Recall with no since/before returns all matching memories (unchanged behavior)."""
        old = self._store_and_backdate(engine, "old memory about testing patterns", hours_ago=48)
        new = self._store_and_backdate(engine, "new memory about testing patterns", hours_ago=1)

        results = engine.recall("testing patterns")

        result_ids = [r.memory.id for r in results]
        assert old.id in result_ids
        assert new.id in result_ids
