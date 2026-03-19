# Postgres Backend Design — Dual Database Layer

**Date:** 2026-03-18
**Status:** Approved
**Scope:** Migrate engram's database layer to support both SQLite and PostgreSQL

---

## Architecture

Dual-backend database layer with a shared protocol and two implementations:

```
DatabaseBackend (Protocol)
├── SqliteBackend    — current db.py, extracted unchanged
└── PostgresBackend  — new, psycopg v3
```

**Backend selection** at runtime:
- `DATABASE_URL` set and starts with `postgresql` → `PostgresBackend`
- Otherwise → `SqliteBackend` (file-based, current behavior)

### File Structure

```
src/engram/
├── db.py              → Protocol + factory (create_database)
├── db_sqlite.py       → SqliteBackend (extracted from current db.py)
├── db_postgres.py     → PostgresBackend (new)
```

---

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project isolation | Single DB, project column filter | Queries already filter by project; avoids connection pool complexity |
| Backend strategy | Dual (SQLite + Postgres) | Preserves zero-config local use; Postgres for containers/multi-machine |
| Postgres driver | psycopg v3 (sync) | Modern, supports sync+async, pool built-in; works with current sync MCP tools |
| Credentials | `DATABASE_URL` env var | Already in docker-compose.yml; Infisical injects env vars |
| Full-text search | `tsvector` + GIN index | Direct FTS5 equivalent; built-in, no extensions |
| Embeddings storage | `bytea` column | Drop-in BLOB replacement; pgvector is a future upgrade |

---

## DatabaseBackend Protocol

```python
class DatabaseBackend(Protocol):
    project: str

    # Metadata
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...

    # Memory CRUD
    def store_memory(self, memory: Memory) -> Memory: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory(self, memory_id: str, ...) -> Memory | None: ...
    def delete_memory(self, memory_id: str) -> bool: ...
    def delete_memory_atomic(self, memory_id: str) -> bool: ...
    def list_memories(self, ...) -> list[Memory]: ...
    def touch_memory(self, memory_id: str) -> None: ...

    # Chunks
    def store_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunks_for_memory(self, memory_id: str) -> list[Chunk]: ...
    def get_all_chunks_with_embeddings(self, limit: int) -> list[Chunk]: ...
    def get_all_chunk_texts(self, limit: int) -> list[str]: ...
    def chunk_hash_exists(self, hash: str) -> bool: ...
    def delete_chunks_for_memory(self, memory_id: str) -> int: ...
    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int: ...

    # Relationships
    def store_relationship(self, rel: Relationship) -> Relationship: ...
    def get_connected(self, memory_id: str, max_hops: int) -> list[Memory]: ...
    def boost_edges_for_memory(self, memory_id: str, factor: float) -> int: ...
    def decay_edges_for_memory(self, memory_id: str, factor: float) -> int: ...
    def get_connection_count(self, memory_id: str) -> int: ...
    def decay_all_edges(self, decay_factor: float, min_strength: float) -> tuple[int, int]: ...
    def delete_relationships_for_memory(self, memory_id: str) -> int: ...

    # FTS
    def fts_search(self, query: str, limit: int) -> list[tuple[Memory, float]]: ...
    def rebuild_fts(self) -> None: ...

    # Maintenance
    def prune_stale_memories(self, max_age_hours: int, max_importance: int) -> int: ...
    def get_stats(self) -> MemoryStats: ...
    def close(self) -> None: ...
```

### Factory Function

```python
def create_database(project: str, db_dir: str | Path | None = None) -> DatabaseBackend:
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("postgresql"):
        from .db_postgres import PostgresBackend
        return PostgresBackend(project=project, dsn=database_url)
    from .db_sqlite import SqliteBackend
    return SqliteBackend(project=project, db_dir=db_dir)
```

---

## Postgres Schema

```sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'context',
    project TEXT NOT NULL DEFAULT 'default',
    tags JSONB NOT NULL DEFAULT '[]',
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
```

### SQLite → Postgres Type Mapping

| Aspect | SQLite | Postgres |
|--------|--------|----------|
| Tags | `TEXT` (JSON string) | `JSONB` (native) |
| Timestamps | `TEXT` (ISO strings) | `TIMESTAMPTZ` (native) |
| FTS | FTS5 virtual table + 3 triggers | `GENERATED` tsvector + GIN index |
| FTS query | `MATCH ?` with `bm25()` | `@@ plainto_tsquery()` with `ts_rank()` |
| Embeddings | `BLOB` | `bytea` |
| Upsert | `INSERT OR IGNORE`/`REPLACE` | `ON CONFLICT DO NOTHING`/`UPDATE` |
| Placeholders | `?` | `%s` |

### FTS Query

```sql
SELECT m.*, ts_rank(m.search_vector, plainto_tsquery('english', %s)) AS rank
FROM memories m
WHERE m.project = %s AND m.search_vector @@ plainto_tsquery('english', %s)
ORDER BY rank DESC LIMIT %s
```

`rebuild_fts()` becomes `REINDEX INDEX idx_memories_search` (generated column auto-maintains the vector).

---

## Connection Management

| Aspect | SqliteBackend | PostgresBackend |
|--------|---------------|-----------------|
| Concurrency | RLock, single connection | Connection pool (psycopg.pool) |
| Connection | Cached in `self._conn` | Borrowed per-operation from pool |
| Transactions | `BEGIN IMMEDIATE` + manual commit | `with conn.transaction():` |
| Row access | `sqlite3.Row` | `row_factory=dict_row` |

```python
class PostgresBackend:
    def __init__(self, project: str, dsn: str):
        self.project = _normalize_project(project)
        self.pool = ConnectionPool(dsn, min_size=2, max_size=10, open=True)
        self._init_db()

    def close(self):
        self.pool.close()
```

No application-level RLock — Postgres handles concurrency natively. `delete_memory_atomic` simplifies to a single `DELETE ... RETURNING` (FK CASCADE handles chunks + relationships).

---

## Testing Strategy

**Parameterized fixtures** run the same test suite against both backends:

```python
@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path):
    if request.param == "sqlite":
        return SqliteBackend(project="test", db_dir=tmp_path)
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("No TEST_DATABASE_URL")
    db = PostgresBackend(project="test", dsn=dsn)
    yield db
    # cleanup test data
    with db.pool.connection() as conn:
        conn.execute("DELETE FROM memories WHERE project = 'test'")
    db.close()
```

**Local Postgres for testing** via docker-compose test profile:

```yaml
services:
  test-postgres:
    image: postgres:16-alpine
    profiles: ["test"]
    environment:
      POSTGRES_DB: engram_test
      POSTGRES_USER: engram
      POSTGRES_PASSWORD: test
    ports:
      - "5433:5432"
```

Run: `docker compose --profile test up -d && TEST_DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test pytest`

**Dependencies** added to `pyproject.toml`:

```toml
[project.optional-dependencies]
postgres = ["psycopg[pool]>=3.1.0"]
```

---

## Implementation Order

1. Extract protocol + factory into `db.py` (rename current `db.py` → `db_sqlite.py`)
2. Update imports in `server.py`, `search.py` — all existing tests pass unchanged
3. Build `PostgresBackend` in `db_postgres.py`
4. Add parameterized test fixtures
5. Verify both backends pass the full suite
6. Update `docker-compose.yml` with test profile

## Files Changed

| File | Change |
|------|--------|
| `db.py` | Replaced with Protocol + factory |
| `db_sqlite.py` | Current `MemoryDB` extracted as `SqliteBackend` |
| `db_postgres.py` | New `PostgresBackend` implementation |
| `server.py` | `MemoryDB(...)` → `create_database(...)` |
| `search.py` | Type hint `db: DatabaseBackend` |
| `pyproject.toml` | Add `postgres` optional dependency |
| `docker-compose.yml` | Add test profile |
| `tests/conftest.py` | Parameterized backend fixtures |
