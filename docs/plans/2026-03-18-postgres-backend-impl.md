# Postgres Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a PostgreSQL backend to engram's database layer, selectable at runtime via `DATABASE_URL`, while keeping SQLite as the default.

**Architecture:** Extract a `DatabaseBackend` protocol from the current `MemoryDB` class. Move SQLite code to `db_sqlite.py` unchanged. Build `db_postgres.py` with psycopg v3 + connection pool. Factory function in `db.py` picks backend based on `DATABASE_URL` env var.

**Tech Stack:** psycopg v3 (sync), psycopg.pool.ConnectionPool, PostgreSQL `tsvector`+GIN for FTS, `bytea` for embeddings.

**Design doc:** `docs/plans/2026-03-18-postgres-backend-design.md`

---

### Task 1: Extract Protocol and Factory into db.py

**Files:**
- Create: `src/engram/db_protocol.py` (temporary, will become `db.py`)
- Rename: `src/engram/db.py` → `src/engram/db_sqlite.py`
- Create: `src/engram/db.py` (new — protocol + factory + re-exports)

**Step 1: Rename db.py to db_sqlite.py**

```bash
cd ~/projects/engram
git mv src/engram/db.py src/engram/db_sqlite.py
```

**Step 2: Edit db_sqlite.py — rename class and fix imports**

In `src/engram/db_sqlite.py`, change:
- Class name `MemoryDB` → `SqliteBackend`
- No other changes needed — all methods stay identical

**Step 3: Create new db.py with Protocol, factory, and re-exports**

Write `src/engram/db.py`:

```python
"""Database abstraction layer for engram.

Provides a DatabaseBackend protocol and a factory function that selects
between SQLite (default, zero-config) and PostgreSQL (via DATABASE_URL).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .types import Chunk, Memory, MemoryStats, MemoryType, Relationship


def _normalize_project(project: str) -> str:
    """Sanitize project name to safe characters."""
    return re.sub(r'[^a-zA-Z0-9_-]', '', project) or "default"


@runtime_checkable
class DatabaseBackend(Protocol):
    """Protocol that both SQLite and Postgres backends implement."""

    project: str

    # Metadata
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...

    # Memory CRUD
    def store_memory(self, memory: Memory) -> Memory: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory(self, memory_id: str, content: str | None = None,
                      tags: list[str] | None = None,
                      importance: int | None = None) -> Memory | None: ...
    def delete_memory(self, memory_id: str) -> bool: ...
    def delete_memory_atomic(self, memory_id: str) -> bool: ...
    def list_memories(self, memory_type: MemoryType | None = None,
                      tags: list[str] | None = None,
                      min_importance: int | None = None,
                      limit: int = 20, offset: int = 0) -> list[Memory]: ...
    def touch_memory(self, memory_id: str) -> None: ...

    # Chunks
    def store_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunks_for_memory(self, memory_id: str) -> list[Chunk]: ...
    def get_all_chunks_with_embeddings(self, limit: int = 10_000) -> list[Chunk]: ...
    def get_all_chunk_texts(self, limit: int = 5000) -> list[str]: ...
    def chunk_hash_exists(self, chunk_hash: str) -> bool: ...
    def delete_chunks_for_memory(self, memory_id: str) -> None: ...
    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int: ...

    # Relationships
    def store_relationship(self, rel: Relationship) -> Relationship: ...
    def get_connected(self, memory_id: str, max_hops: int = 2) -> list[tuple[Memory, str, str, float]]: ...
    def boost_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int: ...
    def decay_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int: ...
    def get_connection_count(self, memory_id: str) -> int: ...
    def decay_all_edges(self, decay_factor: float = 0.02, min_strength: float = 0.1) -> tuple[int, int]: ...
    def delete_relationships_for_memory(self, memory_id: str) -> None: ...

    # FTS
    def fts_search(self, query: str, limit: int = 20) -> list[tuple[Memory, float]]: ...
    def rebuild_fts(self) -> None: ...

    # Maintenance
    def prune_stale_memories(self, max_age_hours: float = 720, max_importance: int = 3) -> int: ...
    def get_stats(self) -> MemoryStats: ...
    def close(self) -> None: ...


def create_database(project: str = "default", db_dir: str | Path | None = None) -> DatabaseBackend:
    """Factory: create the appropriate database backend.

    If DATABASE_URL is set and starts with 'postgresql', uses PostgresBackend.
    Otherwise uses SqliteBackend (default, zero-config).
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("postgresql"):
        from .db_postgres import PostgresBackend
        return PostgresBackend(project=project, dsn=database_url)
    from .db_sqlite import SqliteBackend
    return SqliteBackend(project=project, db_dir=db_dir)


# Backwards compatibility: re-export SqliteBackend as MemoryDB
from .db_sqlite import SqliteBackend as MemoryDB  # noqa: E402, F401

__all__ = ["DatabaseBackend", "MemoryDB", "create_database"]
```

**Step 4: Run tests to verify nothing broke**

```bash
cd ~/projects/engram && source .venv/bin/activate
python -m pytest tests/ -v --tb=short
```

Expected: All 136 tests PASS. The `MemoryDB` re-export keeps all existing imports working.

**Step 5: Commit**

```bash
git add src/engram/db.py src/engram/db_sqlite.py
git commit -m "refactor: extract DatabaseBackend protocol, move SQLite to db_sqlite.py

MemoryDB is re-exported from db.py for backwards compatibility.
No behavior changes — all 136 tests pass."
```

---

### Task 2: Update Callers to Use Factory

**Files:**
- Modify: `src/engram/server.py` (lines 15, 141)
- Modify: `src/engram/search.py` (lines 8, 29)
- Modify: `tests/conftest.py` (lines 15, 51-53)

**Step 1: Update server.py imports and _get_engine**

In `src/engram/server.py`:

Line 15 — change:
```python
from .db import MemoryDB
```
to:
```python
from .db import create_database
```

Line 141 — change:
```python
db = MemoryDB(project=project, db_dir=db_dir)
```
to:
```python
db = create_database(project=project, db_dir=db_dir)
```

**Step 2: Update search.py type hint**

In `src/engram/search.py`:

Line 8 — change:
```python
from .db import MemoryDB
```
to:
```python
from .db import DatabaseBackend
```

Line 29 — change:
```python
def __init__(self, db: MemoryDB, embedder: EmbeddingProvider):
```
to:
```python
def __init__(self, db: DatabaseBackend, embedder: EmbeddingProvider):
```

**Step 3: Update tests/conftest.py**

Line 15 — change:
```python
from engram.db import MemoryDB
```
to:
```python
from engram.db import MemoryDB  # SqliteBackend re-export
```

No other changes needed — `MemoryDB` still works via the re-export.

**Step 4: Run tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All 136 tests PASS.

**Step 5: Commit**

```bash
git add src/engram/server.py src/engram/search.py tests/conftest.py
git commit -m "refactor: use create_database factory in server, DatabaseBackend type in search"
```

---

### Task 3: Add psycopg Dependency

**Files:**
- Modify: `src/engram/pyproject.toml`

**Step 1: Add postgres optional dependency**

In `pyproject.toml`, in `[project.optional-dependencies]`, add:

```toml
postgres = ["psycopg[pool]>=3.1.0"]
```

Also update the `all` extra to include it:

```toml
all = ["openai>=1.30.0", "httpx>=0.24", "numpy>=1.26.0", "psycopg[pool]>=3.1.0"]
```

And add to `dev`:

```toml
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "numpy>=1.26.0",
    "psycopg[pool]>=3.1.0",
]
```

**Step 2: Install the new dependency**

```bash
cd ~/projects/engram && source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Step 3: Verify psycopg imports**

```bash
python -c "import psycopg; print(psycopg.__version__)"
```

Expected: Version number printed (3.x.x).

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add psycopg[pool] as optional postgres dependency"
```

---

### Task 4: Build PostgresBackend — Schema and Connection

**Files:**
- Create: `src/engram/db_postgres.py`
- Test: `tests/test_db_postgres.py`

**Step 1: Write the basic connection and schema test**

Create `tests/test_db_postgres.py`:

```python
"""Tests for PostgresBackend — skipped when TEST_DATABASE_URL is not set."""

from __future__ import annotations

import os

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
    # Clean up all test data
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM memories WHERE project = 'test_pg'")
        conn.execute("DELETE FROM project_meta")
        conn.commit()
    db.close()


class TestPostgresConnection:
    def test_creates_tables(self, pg_db):
        """Schema init should create all required tables."""
        with pg_db.pool.connection() as conn:
            tables = conn.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ).fetchall()
            table_names = {r[0] for r in tables}
            assert "memories" in table_names
            assert "chunks" in table_names
            assert "relationships" in table_names
            assert "project_meta" in table_names

    def test_meta_roundtrip(self, pg_db):
        pg_db.set_meta("test_key", "test_value")
        assert pg_db.get_meta("test_key") == "test_value"

    def test_meta_missing_returns_none(self, pg_db):
        assert pg_db.get_meta("nonexistent") is None
```

**Step 2: Run test to verify it fails (no db_postgres module yet)**

```bash
python -m pytest tests/test_db_postgres.py -v --tb=short
```

Expected: SKIP if no TEST_DATABASE_URL, or ImportError if TEST_DATABASE_URL is set.

**Step 3: Write PostgresBackend — schema, connection pool, metadata methods**

Create `src/engram/db_postgres.py`:

```python
"""PostgreSQL backend for engram using psycopg v3 + connection pooling."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone

from .types import (
    Chunk,
    Memory,
    MemoryStats,
    MemoryType,
    Relationship,
)

logger = logging.getLogger(__name__)

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
except ImportError as exc:
    raise ImportError(
        "PostgreSQL backend requires psycopg. Install with: pip install engram[postgres]"
    ) from exc


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'context',
    project TEXT NOT NULL DEFAULT 'default',
    tags JSONB NOT NULL DEFAULT '[]'::jsonb,
    importance INTEGER NOT NULL DEFAULT 2,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL DEFAULT '',
    embedding BYTEA,
    FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_memory ON chunks(memory_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(chunk_hash);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    rel_type TEXT NOT NULL DEFAULT 'relates_to',
    strength REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE (source_id, target_id, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);

CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _normalize_project(project: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '', project) or "default"


class PostgresBackend:
    """PostgreSQL database backend using psycopg v3 + connection pooling."""

    def __init__(self, project: str = "default", dsn: str | None = None):
        self.project = _normalize_project(project)
        dsn = dsn or os.environ.get("DATABASE_URL", "")
        if not dsn:
            raise ValueError("PostgresBackend requires a DSN (DATABASE_URL)")
        self.pool = ConnectionPool(dsn, min_size=2, max_size=10, open=True,
                                   kwargs={"row_factory": dict_row})
        self._init_db()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _init_db(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()

    def close(self) -> None:
        self.pool.close()

    # ── Project Metadata ─────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM project_meta WHERE key = %s", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO project_meta (key, value) VALUES (%s, %s)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                (key, value),
            )
            conn.commit()

    # ── Memory CRUD ──────────────────────────────────────────────

    def store_memory(self, memory: Memory) -> Memory:
        with self.pool.connection() as conn:
            now = datetime.now(timezone.utc)
            memory.created_at = now
            memory.updated_at = now
            memory.last_accessed = now
            memory.project = self.project

            conn.execute(
                """INSERT INTO memories (id, content, memory_type, project, tags,
                   importance, access_count, last_accessed, created_at, updated_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    memory.id,
                    memory.content,
                    memory.memory_type.value,
                    memory.project,
                    json.dumps(memory.tags),
                    memory.importance,
                    memory.access_count,
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
            return memory

    def get_memory(self, memory_id: str) -> Memory | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = %s", (memory_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    def update_memory(self, memory_id: str, content: str | None = None,
                      tags: list[str] | None = None,
                      importance: int | None = None) -> Memory | None:
        mem = self.get_memory(memory_id)
        if not mem:
            return None

        now = datetime.now(timezone.utc)
        if content is not None:
            mem.content = content
        if tags is not None:
            mem.tags = tags
        if importance is not None:
            mem.importance = importance

        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE memories SET content=%s, tags=%s, importance=%s, updated_at=%s
                   WHERE id=%s""",
                (mem.content, json.dumps(mem.tags), mem.importance, now, memory_id),
            )
            conn.commit()
        mem.updated_at = now
        return mem

    def delete_memory(self, memory_id: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,)
            ).fetchone()
            conn.commit()
            return row is not None

    def delete_memory_atomic(self, memory_id: str) -> bool:
        """Delete memory + cascaded chunks/relationships in one transaction."""
        with self.pool.connection() as conn:
            with conn.transaction():
                # FK ON DELETE CASCADE handles chunks + relationships
                row = conn.execute(
                    "DELETE FROM memories WHERE id = %s RETURNING id", (memory_id,)
                ).fetchone()
                return row is not None

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        with self.pool.connection() as conn:
            query = "SELECT * FROM memories WHERE project = %s"
            params: list = [self.project]

            if memory_type:
                query += " AND memory_type = %s"
                params.append(memory_type.value)
            if min_importance is not None:
                query += " AND importance <= %s"
                params.append(min_importance)
            if tags:
                # JSONB tag containment: check if any tag is in the array
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags @> %s::jsonb")
                    params.append(json.dumps([tag]))
                query += " AND (" + " OR ".join(tag_conditions) + ")"

            query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def touch_memory(self, memory_id: str) -> None:
        with self.pool.connection() as conn:
            now = datetime.now(timezone.utc)
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = %s WHERE id = %s",
                (now, memory_id),
            )
            conn.commit()

    # ── Chunk CRUD ───────────────────────────────────────────────

    def store_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    cur.execute(
                        """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index,
                           chunk_hash, embedding) VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT DO NOTHING""",
                        (c.id, c.memory_id, c.chunk_text, c.chunk_index,
                         c.chunk_hash, c.embedding),
                    )
            conn.commit()

    def get_chunks_for_memory(self, memory_id: str) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE memory_id = %s ORDER BY chunk_index",
                (memory_id,),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_chunks_with_embeddings(self, limit: int = 10_000) -> list[Chunk]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT c.* FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE c.embedding IS NOT NULL
                   AND m.project = %s
                   ORDER BY m.last_accessed DESC
                   LIMIT %s""",
                (self.project, limit),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_chunk_texts(self, limit: int = 5000) -> list[str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT c.chunk_text FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE m.project = %s
                   LIMIT %s""",
                (self.project, limit),
            ).fetchall()
            return [r["chunk_text"] for r in rows]

    def chunk_hash_exists(self, chunk_hash: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_hash = %s LIMIT 1", (chunk_hash,)
            ).fetchone()
            return row is not None

    def delete_chunks_for_memory(self, memory_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE memory_id = %s", (memory_id,))
            conn.commit()

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        with self.pool.connection() as conn:
            placeholders = ",".join(["%s"] * len(chunk_ids))
            row = conn.execute(
                f"WITH deleted AS (DELETE FROM chunks WHERE id IN ({placeholders}) RETURNING id) SELECT count(*) as c FROM deleted",
                chunk_ids,
            ).fetchone()
            conn.commit()
            return row["c"] if row else 0

    # ── Relationship CRUD ────────────────────────────────────────

    def store_relationship(self, rel: Relationship) -> Relationship:
        with self.pool.connection() as conn:
            src = conn.execute("SELECT 1 FROM memories WHERE id = %s", (rel.source_id,)).fetchone()
            tgt = conn.execute("SELECT 1 FROM memories WHERE id = %s", (rel.target_id,)).fetchone()
            if not src:
                raise ValueError(f"Source memory '{rel.source_id}' does not exist")
            if not tgt:
                raise ValueError(f"Target memory '{rel.target_id}' does not exist")

            conn.execute(
                """INSERT INTO relationships (id, source_id, target_id, rel_type,
                   strength, created_at) VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source_id, target_id, rel_type)
                   DO UPDATE SET strength = EXCLUDED.strength""",
                (
                    rel.id,
                    rel.source_id,
                    rel.target_id,
                    rel.rel_type.value,
                    rel.strength,
                    rel.created_at,
                ),
            )
            conn.commit()
            return rel

    def get_connected(
        self, memory_id: str, max_hops: int = 2,
    ) -> list[tuple[Memory, str, str, float]]:
        results: list[tuple[Memory, str, str, float]] = []
        visited: set[str] = {memory_id}
        frontier = [memory_id]

        with self.pool.connection() as conn:
            for _ in range(max_hops):
                if not frontier:
                    break
                next_frontier: list[str] = []
                placeholders = ",".join(["%s"] * len(frontier))

                outgoing = conn.execute(
                    f"""SELECT r.target_id, r.rel_type, r.strength
                        FROM relationships r WHERE r.source_id IN ({placeholders})""",
                    frontier,
                ).fetchall()

                incoming = conn.execute(
                    f"""SELECT r.source_id, r.rel_type, r.strength
                        FROM relationships r WHERE r.target_id IN ({placeholders})""",
                    frontier,
                ).fetchall()

                for row in outgoing:
                    nid = row["target_id"]
                    if nid not in visited:
                        visited.add(nid)
                        mem = self.get_memory(nid)
                        if mem:
                            results.append((mem, row["rel_type"], "outgoing", row["strength"]))
                            next_frontier.append(nid)

                for row in incoming:
                    nid = row["source_id"]
                    if nid not in visited:
                        visited.add(nid)
                        mem = self.get_memory(nid)
                        if mem:
                            results.append((mem, row["rel_type"], "incoming", row["strength"]))
                            next_frontier.append(nid)

                frontier = next_frontier

        return results

    def boost_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                """WITH updated AS (
                       UPDATE relationships
                       SET strength = LEAST(1.0, strength + %s)
                       WHERE source_id = %s OR target_id = %s
                       RETURNING id
                   ) SELECT count(*) as c FROM updated""",
                (factor, memory_id, memory_id),
            ).fetchone()
            conn.commit()
            return row["c"] if row else 0

    def decay_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                """WITH updated AS (
                       UPDATE relationships
                       SET strength = GREATEST(0.0, strength - %s)
                       WHERE source_id = %s OR target_id = %s
                       RETURNING id
                   ) SELECT count(*) as c FROM updated""",
                (factor, memory_id, memory_id),
            ).fetchone()
            conn.commit()
            return row["c"] if row else 0

    def get_connection_count(self, memory_id: str) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as c FROM relationships
                   WHERE source_id = %s OR target_id = %s""",
                (memory_id, memory_id),
            ).fetchone()
            return row["c"]

    def decay_all_edges(
        self, decay_factor: float = 0.02, min_strength: float = 0.1,
    ) -> tuple[int, int]:
        with self.pool.connection() as conn:
            decayed_row = conn.execute(
                """WITH updated AS (
                       UPDATE relationships SET strength = GREATEST(0.0, strength - %s)
                       RETURNING id
                   ) SELECT count(*) as c FROM updated""",
                (decay_factor,),
            ).fetchone()
            decayed = decayed_row["c"] if decayed_row else 0

            pruned_row = conn.execute(
                """WITH deleted AS (
                       DELETE FROM relationships WHERE strength < %s
                       RETURNING id
                   ) SELECT count(*) as c FROM deleted""",
                (min_strength,),
            ).fetchone()
            pruned = pruned_row["c"] if pruned_row else 0

            conn.commit()
            return decayed, pruned

    def delete_relationships_for_memory(self, memory_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM relationships WHERE source_id = %s OR target_id = %s",
                (memory_id, memory_id),
            )
            conn.commit()

    # ── FTS (tsvector) ───────────────────────────────────────────

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[Memory, float]]:
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []

        with self.pool.connection() as conn:
            try:
                rows = conn.execute(
                    """SELECT m.*, ts_rank(m.search_vector, plainto_tsquery('english', %s)) AS rank
                       FROM memories m
                       WHERE m.project = %s
                       AND m.search_vector @@ plainto_tsquery('english', %s)
                       ORDER BY rank DESC
                       LIMIT %s""",
                    (safe_query, self.project, safe_query, limit),
                ).fetchall()
            except psycopg.errors.SyntaxError:
                logger.debug("FTS query failed for %r", safe_query)
                return []

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            results.append((mem, row["rank"]))
        return results

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip special characters for safe use with plainto_tsquery."""
        import re
        sanitized = re.sub(r'[*^"():<>!&|]', ' ', query)
        sanitized = re.sub(r'\b(AND|OR|NOT|NEAR)\b', '', sanitized, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', sanitized).strip()

    def rebuild_fts(self) -> None:
        """Reindex the GIN search index. Generated column auto-maintains tsvector."""
        with self.pool.connection() as conn:
            conn.execute("REINDEX INDEX idx_memories_search")
            conn.commit()
        logger.info("Rebuilt FTS index for project %s", self.project)

    # ── Maintenance ──────────────────────────────────────────────

    def prune_stale_memories(self, max_age_hours: float = 720, max_importance: int = 3) -> int:
        with self.pool.connection() as conn:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            row = conn.execute(
                """WITH deleted AS (
                       DELETE FROM memories
                       WHERE project = %s AND importance >= %s
                       AND last_accessed < %s AND access_count = 0
                       RETURNING id
                   ) SELECT count(*) as c FROM deleted""",
                (self.project, max_importance, cutoff),
            ).fetchone()
            conn.commit()
            return row["c"] if row else 0

    def get_stats(self) -> MemoryStats:
        with self.pool.connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE project = %s", (self.project,)
            ).fetchone()["c"]

            total_chunks = conn.execute(
                """SELECT COUNT(*) as c FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE m.project = %s""",
                (self.project,),
            ).fetchone()["c"]

            total_rels = conn.execute(
                """SELECT COUNT(*) as c FROM relationships r
                   WHERE r.source_id IN (SELECT id FROM memories WHERE project = %s)
                   OR r.target_id IN (SELECT id FROM memories WHERE project = %s)""",
                (self.project, self.project),
            ).fetchone()["c"]

            type_rows = conn.execute(
                "SELECT memory_type, COUNT(*) as c FROM memories"
                " WHERE project = %s GROUP BY memory_type",
                (self.project,),
            ).fetchall()
            by_type = {r["memory_type"]: r["c"] for r in type_rows}

            imp_rows = conn.execute(
                "SELECT importance, COUNT(*) as c FROM memories WHERE project = %s GROUP BY importance",
                (self.project,),
            ).fetchall()
            by_importance = {str(r["importance"]): r["c"] for r in imp_rows}

            oldest_row = conn.execute(
                "SELECT MIN(created_at) as v FROM memories WHERE project = %s", (self.project,)
            ).fetchone()
            newest_row = conn.execute(
                "SELECT MAX(created_at) as v FROM memories WHERE project = %s", (self.project,)
            ).fetchone()

            # Postgres doesn't have a single file; report 0 for db_size_bytes
            return MemoryStats(
                total_memories=total,
                total_chunks=total_chunks,
                total_relationships=total_rels,
                by_type=by_type,
                by_importance=by_importance,
                oldest=oldest_row["v"].isoformat() if oldest_row and oldest_row["v"] else None,
                newest=newest_row["v"].isoformat() if newest_row and newest_row["v"] else None,
                db_size_bytes=0,
            )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        tags = row["tags"]
        if isinstance(tags, str):
            tags = json.loads(tags)
        last_accessed = row["last_accessed"]
        created_at = row["created_at"]
        updated_at = row["updated_at"]
        # psycopg returns datetime objects for TIMESTAMPTZ
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            project=row["project"],
            tags=tags,
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=last_accessed,
            created_at=created_at,
            updated_at=updated_at,
        )

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        embedding = row["embedding"]
        # psycopg returns memoryview for bytea; convert to bytes
        if isinstance(embedding, memoryview):
            embedding = bytes(embedding)
        return Chunk(
            id=row["id"],
            memory_id=row["memory_id"],
            chunk_text=row["chunk_text"],
            chunk_index=row["chunk_index"],
            chunk_hash=row["chunk_hash"],
            embedding=embedding,
        )
```

**Step 4: Run tests**

```bash
# SQLite tests still pass:
python -m pytest tests/ -v --tb=short -k "not postgres"

# If Postgres is available locally:
# docker run -d --name engram-test-pg -e POSTGRES_DB=engram_test -e POSTGRES_USER=engram -e POSTGRES_PASSWORD=test -p 5433:5432 postgres:16-alpine
# TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test python -m pytest tests/test_db_postgres.py -v --tb=short
```

Expected: All SQLite tests PASS. Postgres tests pass when TEST_DATABASE_URL is set.

**Step 5: Commit**

```bash
git add src/engram/db_postgres.py tests/test_db_postgres.py
git commit -m "feat: add PostgresBackend with psycopg v3, connection pool, tsvector FTS (#38 design)"
```

---

### Task 5: Add Full CRUD Tests for PostgresBackend

**Files:**
- Modify: `tests/test_db_postgres.py`

**Step 1: Add comprehensive CRUD + FTS + relationship tests**

Append to `tests/test_db_postgres.py`:

```python
from engram.types import Memory, MemoryType, Relationship, RelationType


class TestPostgresMemoryCRUD:
    def test_store_and_retrieve(self, pg_db):
        mem = Memory(content="PostgreSQL is the database")
        stored = pg_db.store_memory(mem)
        retrieved = pg_db.get_memory(stored.id)
        assert retrieved is not None
        assert retrieved.content == "PostgreSQL is the database"
        assert retrieved.project == "test_pg"

    def test_get_nonexistent_returns_none(self, pg_db):
        assert pg_db.get_memory("does-not-exist") is None

    def test_update_memory(self, pg_db):
        mem = Memory(content="Old content", tags=["old"])
        stored = pg_db.store_memory(mem)
        updated = pg_db.update_memory(stored.id, content="New content", tags=["new"])
        assert updated is not None
        assert updated.content == "New content"
        assert updated.tags == ["new"]

    def test_delete_memory(self, pg_db):
        mem = Memory(content="To be deleted")
        stored = pg_db.store_memory(mem)
        assert pg_db.delete_memory(stored.id) is True
        assert pg_db.get_memory(stored.id) is None

    def test_delete_memory_atomic(self, pg_db):
        mem = Memory(content="Atomic delete test")
        stored = pg_db.store_memory(mem)
        assert pg_db.delete_memory_atomic(stored.id) is True
        assert pg_db.get_memory(stored.id) is None

    def test_list_memories(self, pg_db):
        pg_db.store_memory(Memory(content="Memory A"))
        pg_db.store_memory(Memory(content="Memory B"))
        memories = pg_db.list_memories(limit=10)
        assert len(memories) >= 2

    def test_touch_increments_access(self, pg_db):
        mem = Memory(content="Touch test")
        stored = pg_db.store_memory(mem)
        pg_db.touch_memory(stored.id)
        retrieved = pg_db.get_memory(stored.id)
        assert retrieved.access_count == 1


class TestPostgresFTS:
    def test_basic_search(self, pg_db):
        pg_db.store_memory(Memory(content="PostgreSQL database optimization techniques"))
        results = pg_db.fts_search("PostgreSQL optimization", limit=5)
        assert len(results) >= 1
        assert results[0][1] > 0  # positive rank score

    def test_empty_query_returns_empty(self, pg_db):
        assert pg_db.fts_search("") == []

    def test_no_match_returns_empty(self, pg_db):
        pg_db.store_memory(Memory(content="Alpha beta gamma"))
        results = pg_db.fts_search("xylophone", limit=5)
        assert len(results) == 0


class TestPostgresRelationships:
    def test_store_and_get_connected(self, pg_db):
        a = pg_db.store_memory(Memory(content="Memory A"))
        b = pg_db.store_memory(Memory(content="Memory B"))
        rel = Relationship(source_id=a.id, target_id=b.id)
        pg_db.store_relationship(rel)
        connected = pg_db.get_connected(a.id, max_hops=1)
        assert len(connected) == 1
        assert connected[0][0].id == b.id

    def test_store_relationship_rejects_invalid(self, pg_db):
        rel = Relationship(source_id="nonexistent", target_id="also-nonexistent")
        with pytest.raises(ValueError):
            pg_db.store_relationship(rel)

    def test_boost_and_decay_edges(self, pg_db):
        a = pg_db.store_memory(Memory(content="Memory A"))
        b = pg_db.store_memory(Memory(content="Memory B"))
        rel = Relationship(source_id=a.id, target_id=b.id, strength=0.5)
        pg_db.store_relationship(rel)
        boosted = pg_db.boost_edges_for_memory(a.id, factor=0.1)
        assert boosted == 1
        decayed = pg_db.decay_edges_for_memory(a.id, factor=0.1)
        assert decayed == 1

    def test_connection_count(self, pg_db):
        a = pg_db.store_memory(Memory(content="Memory A"))
        b = pg_db.store_memory(Memory(content="Memory B"))
        pg_db.store_relationship(Relationship(source_id=a.id, target_id=b.id))
        assert pg_db.get_connection_count(a.id) == 1
```

**Step 2: Run tests**

```bash
# With Postgres available:
TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test python -m pytest tests/test_db_postgres.py -v --tb=short
```

Expected: All Postgres tests PASS.

**Step 3: Commit**

```bash
git add tests/test_db_postgres.py
git commit -m "test: add comprehensive CRUD, FTS, and relationship tests for PostgresBackend"
```

---

### Task 6: Add Docker Test Profile

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Add test-postgres service with test profile**

In `docker-compose.yml`, add before the `volumes:` section:

```yaml
  test-postgres:
    image: postgres:16-alpine
    profiles: ["test"]
    environment:
      - POSTGRES_DB=engram_test
      - POSTGRES_USER=engram
      - POSTGRES_PASSWORD=test
    ports:
      - "5433:5432"
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "engram"]
      interval: 3s
      timeout: 2s
      retries: 5
```

**Step 2: Verify it works**

```bash
cd ~/projects/engram
docker compose --profile test up -d test-postgres
# Wait for healthy
docker compose --profile test ps
# Run postgres tests
TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test python -m pytest tests/test_db_postgres.py -v --tb=short
# Clean up
docker compose --profile test down
```

**Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add docker-compose test profile for Postgres integration tests"
```

---

### Task 7: Verify Full Suite — Both Backends

**Step 1: Run SQLite tests (no DATABASE_URL)**

```bash
unset DATABASE_URL
unset TEST_DATABASE_URL
python -m pytest tests/ -v --tb=short
```

Expected: All 136+ tests PASS. Postgres tests SKIP.

**Step 2: Run with Postgres**

```bash
docker compose --profile test up -d test-postgres
TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test python -m pytest tests/ -v --tb=short
docker compose --profile test down
```

Expected: All tests PASS including Postgres-specific tests.

**Step 3: Verify factory selects correctly**

```bash
# SQLite (default):
python -c "from engram.db import create_database; db = create_database('test'); print(type(db).__name__)"
# Expected: SqliteBackend

# Postgres:
DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test python -c "from engram.db import create_database; db = create_database('test'); print(type(db).__name__); db.close()"
# Expected: PostgresBackend
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: dual SQLite/Postgres database backend — complete implementation

- DatabaseBackend protocol in db.py with factory function
- SqliteBackend (extracted from original db.py, unchanged behavior)
- PostgresBackend with psycopg v3, connection pool, tsvector FTS
- Runtime selection via DATABASE_URL env var
- Full test suite for both backends
- Docker test profile for Postgres integration tests

Closes design: docs/plans/2026-03-18-postgres-backend-design.md"
```
