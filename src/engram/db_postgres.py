"""PostgreSQL backend for engram using psycopg v3 with connection pooling.

Implements the same ``DatabaseBackend`` protocol as ``SqliteBackend`` but
targets PostgreSQL.  Key differences from the SQLite backend:

- **psycopg v3** ``ConnectionPool`` instead of a single ``sqlite3.Connection``
- **JSONB** for tags, **TIMESTAMPTZ** for timestamps, **bytea** for embeddings
- **Generated tsvector column** + GIN index for FTS (no triggers, no virtual table)
- ``plainto_tsquery`` / ``ts_rank`` for full-text search scoring
- ``%s`` parameter placeholders (not ``?``)
- No RLock — Postgres handles concurrency natively via connection pool
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone

from psycopg.errors import DuplicateColumn
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .types import (
    Chunk,
    Memory,
    MemoryStats,
    MemoryType,
    Relationship,
)

logger = logging.getLogger(__name__)

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
    immutable BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at TIMESTAMPTZ,
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_project_type ON memories(project, memory_type);

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
    strength DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    project TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_project ON relationships(project);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_pair
    ON relationships(source_id, target_id, rel_type);

CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CURRENT_SCHEMA_VERSION = 6


class PostgresBackend:
    """PostgreSQL storage backend for engram."""

    def __init__(self, project: str = "default", dsn: str = ""):
        project = re.sub(r"[^a-zA-Z0-9_-]", "", project) or "default"
        self.project = project
        self.pool = ConnectionPool(
            conninfo=dsn,
            min_size=2,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
        self._validate_connection()
        self._init_db()

    def _validate_connection(self) -> None:
        """Verify the database is reachable before running schema migrations.

        Raises ConnectionError immediately if the DSN is wrong or the server
        is unreachable, instead of letting _init_db() fail mid-migration.
        """
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            self.pool.close()
            raise ConnectionError(
                f"Cannot connect to PostgreSQL — check DATABASE_URL and server status. "
                f"Underlying error: {exc}"
            ) from exc

    def __enter__(self) -> PostgresBackend:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.pool.close()

    # ── Initialisation ────────────────────────────────────────────

    def _init_db(self) -> None:
        with self.pool.connection() as conn:
            conn.execute(SCHEMA_SQL)
            conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM project_meta WHERE key = 'schema_version'"
            ).fetchone()
            current = int(row["value"]) if row else 1
            if current < 3:
                # Add project column to relationships table for cross-project isolation (#75)
                # Use a savepoint so a DuplicateColumn exception doesn't abort the outer
                # transaction (bare try/except pass in psycopg3 leaves the connection in
                # an error state and poisons the pool — see issue #84).
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE relationships ADD COLUMN project TEXT NOT NULL DEFAULT ''"
                        )
                except DuplicateColumn:
                    pass  # Column already exists (fresh DB created with new schema)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_rel_project ON relationships(project)"
                )
                # Backfill existing relationships with the project from their source memory
                conn.execute(
                    """UPDATE relationships SET project = (
                           SELECT m.project FROM memories m WHERE m.id = relationships.source_id
                       ) WHERE project = ''"""
                )
                conn.commit()
            if current < 4:
                # Add immutability and TTL columns (Wave 1 enhancements)
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE memories ADD COLUMN immutable BOOLEAN NOT NULL DEFAULT FALSE"
                        )
                except DuplicateColumn:
                    pass  # Column already exists (fresh DB created with new schema)
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE memories ADD COLUMN expires_at TIMESTAMPTZ"
                        )
                except DuplicateColumn:
                    pass  # Column already exists (fresh DB created with new schema)
                conn.commit()
            if current < 5:
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE memories ADD COLUMN content_compressed BYTEA"
                        )
                except DuplicateColumn:
                    pass
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE memories ADD COLUMN compression_algo TEXT"
                        )
                except DuplicateColumn:
                    pass
                try:
                    with conn.transaction():
                        conn.execute(
                            "ALTER TABLE memories ADD COLUMN compressed_at TIMESTAMPTZ"
                        )
                except DuplicateColumn:
                    pass
                conn.commit()
            if current < 6:
                try:
                    with conn.transaction():
                        conn.execute("ALTER TABLE memories ADD COLUMN summary TEXT")
                except DuplicateColumn:
                    pass
                try:
                    with conn.transaction():
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_memories_summary_null ON memories(id) WHERE summary IS NULL"
                        )
                except Exception:
                    pass
                conn.execute(
                    "UPDATE project_meta SET value=%s WHERE key='schema_version'", ("6",)
                )
                conn.commit()
            conn.execute(
                "INSERT INTO project_meta (key, value) VALUES ('schema_version', %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (str(CURRENT_SCHEMA_VERSION),),
            )
            conn.commit()

    # ── Project Metadata ──────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT value FROM project_meta WHERE key = %s", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO project_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )
            conn.commit()

    # ── Memory CRUD ───────────────────────────────────────────────

    def store_memory(self, memory: Memory) -> Memory:
        now = datetime.now(timezone.utc)
        memory.created_at = now
        memory.updated_at = now
        memory.last_accessed = now
        memory.project = self.project

        with self.pool.connection() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, content, memory_type, project, tags,
                    importance, access_count, last_accessed, created_at, updated_at,
                    immutable, expires_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
                    memory.immutable,
                    memory.expires_at,
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

    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
    ) -> Memory | None:
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
                """UPDATE memories
                   SET content = %s, tags = %s, importance = %s, updated_at = %s,
                   content_compressed = NULL, compression_algo = NULL, compressed_at = NULL
                   WHERE id = %s""",
                (mem.content, json.dumps(mem.tags), mem.importance, now, memory_id),
            )
            conn.commit()
        mem.updated_at = now
        return mem

    def delete_memory(self, memory_id: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "WITH deleted AS (DELETE FROM memories WHERE id = %s RETURNING id) "
                "SELECT count(*) AS c FROM deleted",
                (memory_id,),
            ).fetchone()
            conn.commit()
            return row["c"] > 0

    def delete_memory_atomic(self, memory_id: str) -> bool:
        with self.pool.connection() as conn:
            with conn.transaction():
                conn.execute(
                    "DELETE FROM chunks WHERE memory_id = %s", (memory_id,)
                )
                conn.execute(
                    "DELETE FROM relationships WHERE source_id = %s OR target_id = %s",
                    (memory_id, memory_id),
                )
                row = conn.execute(
                    "WITH deleted AS (DELETE FROM memories WHERE id = %s RETURNING id) "
                    "SELECT count(*) AS c FROM deleted",
                    (memory_id,),
                ).fetchone()
                return row["c"] > 0

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        query = "SELECT * FROM memories WHERE project = %s"
        params: list = [self.project]

        if memory_type:
            query += " AND memory_type = %s"
            params.append(memory_type.value)
        if min_importance is not None:
            query += " AND importance <= %s"
            params.append(min_importance)
        if tags:
            # JSONB containment with OR logic (match ANY tag, same as SQLite)
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags @> %s::jsonb")
                params.append(json.dumps([tag]))
            query += " AND (" + " OR ".join(tag_conditions) + ")"

        query += " ORDER BY updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with self.pool.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def touch_memory(self, memory_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = %s "
                "WHERE id = %s",
                (now, memory_id),
            )
            conn.commit()

    # ── Chunk CRUD ────────────────────────────────────────────────

    def store_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (id, memory_id, chunk_text, chunk_index,
                       chunk_hash, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    [
                        (c.id, c.memory_id, c.chunk_text, c.chunk_index,
                         c.chunk_hash, c.embedding)
                        for c in chunks
                    ],
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

    def get_chunks_for_memories(self, memory_ids: list[str]) -> list[Chunk]:
        """Load chunks with embeddings for specific memory IDs only."""
        if not memory_ids:
            return []
        with self.pool.connection() as conn:
            rows = conn.execute(
                """SELECT c.* FROM chunks c
                   WHERE c.memory_id = ANY(%s)
                   AND c.embedding IS NOT NULL
                   ORDER BY c.chunk_index""",
                (memory_ids,),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def chunk_hash_exists(self, chunk_hash: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_hash = %s LIMIT 1", (chunk_hash,)
            ).fetchone()
            return row is not None

    def delete_chunks_for_memory(self, memory_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM chunks WHERE memory_id = %s", (memory_id,)
            )
            conn.commit()

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        with self.pool.connection() as conn:
            # Use ANY(%s) with a list parameter for variable-length IN clauses
            row = conn.execute(
                "WITH deleted AS (DELETE FROM chunks WHERE id = ANY(%s) RETURNING id) "
                "SELECT count(*) AS c FROM deleted",
                (chunk_ids,),
            ).fetchone()
            conn.commit()
            return row["c"]

    # ── Relationship CRUD ─────────────────────────────────────────

    def store_relationship(self, rel: Relationship) -> Relationship:
        with self.pool.connection() as conn:
            src = conn.execute(
                "SELECT 1 FROM memories WHERE id = %s", (rel.source_id,)
            ).fetchone()
            tgt = conn.execute(
                "SELECT 1 FROM memories WHERE id = %s", (rel.target_id,)
            ).fetchone()
            if not src:
                raise ValueError(f"Source memory '{rel.source_id}' does not exist")
            if not tgt:
                raise ValueError(f"Target memory '{rel.target_id}' does not exist")
            rel.project = self.project

            conn.execute(
                """INSERT INTO relationships
                   (id, source_id, target_id, rel_type, strength, project, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source_id, target_id, rel_type)
                   DO UPDATE SET strength = EXCLUDED.strength""",
                (
                    rel.id,
                    rel.source_id,
                    rel.target_id,
                    rel.rel_type.value,
                    rel.strength,
                    rel.project,
                    rel.created_at,
                ),
            )
            conn.commit()
        return rel

    def get_connected(
        self, memory_id: str, max_hops: int = 2,
    ) -> list[tuple[Memory, str, str, float]]:
        visited: set[str] = {memory_id}
        results: list[tuple[Memory, str, str, float]] = []
        frontier = [memory_id]

        with self.pool.connection() as conn:
            for _ in range(max_hops):
                if not frontier:
                    break
                next_frontier: list[str] = []

                outgoing = conn.execute(
                    "SELECT target_id, rel_type, strength "
                    "FROM relationships WHERE source_id = ANY(%s) AND project = %s",
                    (frontier, self.project),
                ).fetchall()

                incoming = conn.execute(
                    "SELECT source_id, rel_type, strength "
                    "FROM relationships WHERE target_id = ANY(%s) AND project = %s",
                    (frontier, self.project),
                ).fetchall()

                for row in outgoing:
                    nid = row["target_id"]
                    if nid not in visited:
                        visited.add(nid)
                        mem = self.get_memory(nid)
                        if mem:
                            results.append(
                                (mem, row["rel_type"], "outgoing", row["strength"])
                            )
                            next_frontier.append(nid)

                for row in incoming:
                    nid = row["source_id"]
                    if nid not in visited:
                        visited.add(nid)
                        mem = self.get_memory(nid)
                        if mem:
                            results.append(
                                (mem, row["rel_type"], "incoming", row["strength"])
                            )
                            next_frontier.append(nid)

                frontier = next_frontier

        return results

    def boost_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "WITH updated AS ("
                "  UPDATE relationships"
                "  SET strength = LEAST(1.0, strength + %s)"
                "  WHERE source_id = %s OR target_id = %s"
                "  RETURNING id"
                ") SELECT count(*) AS c FROM updated",
                (factor, memory_id, memory_id),
            ).fetchone()
            conn.commit()
            return row["c"]

    def decay_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "WITH updated AS ("
                "  UPDATE relationships"
                "  SET strength = GREATEST(0.0, strength - %s)"
                "  WHERE source_id = %s OR target_id = %s"
                "  RETURNING id"
                ") SELECT count(*) AS c FROM updated",
                (factor, memory_id, memory_id),
            ).fetchone()
            conn.commit()
            return row["c"]

    def get_connection_count(self, memory_id: str) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM relationships "
                "WHERE (source_id = %s OR target_id = %s) AND project = %s",
                (memory_id, memory_id, self.project),
            ).fetchone()
            return row["c"]

    def decay_all_edges(
        self, decay_factor: float = 0.02, min_strength: float = 0.1,
    ) -> tuple[int, int]:
        with self.pool.connection() as conn:
            decayed_row = conn.execute(
                "WITH updated AS ("
                "  UPDATE relationships SET strength = GREATEST(0.0, strength - %s)"
                "  WHERE project = %s RETURNING id"
                ") SELECT count(*) AS c FROM updated",
                (decay_factor, self.project),
            ).fetchone()
            pruned_row = conn.execute(
                "WITH deleted AS ("
                "  DELETE FROM relationships WHERE strength < %s AND project = %s RETURNING id"
                ") SELECT count(*) AS c FROM deleted",
                (min_strength, self.project),
            ).fetchone()
            conn.commit()
            return decayed_row["c"], pruned_row["c"]

    def prune_stale_memories(
        self, max_age_hours: float = 720, max_importance: int = 3,
    ) -> int:
        """Remove low-importance memories that haven't been accessed in max_age_hours,
        plus any expired memories. Never prunes immutable memories."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        with self.pool.connection() as conn:
            row = conn.execute(
                "WITH deleted AS ("
                "  DELETE FROM memories"
                "  WHERE project = %s AND NOT immutable AND ("
                "    (importance >= %s AND last_accessed < %s AND access_count = 0)"
                "    OR (expires_at IS NOT NULL AND expires_at < NOW())"
                "  )"
                "  RETURNING id"
                ") SELECT count(*) AS c FROM deleted",
                (self.project, max_importance, cutoff),
            ).fetchone()
            conn.commit()
            return row["c"]

    def delete_relationships_for_memory(self, memory_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM relationships WHERE source_id = %s OR target_id = %s",
                (memory_id, memory_id),
            )
            conn.commit()

    # ── FTS Search ────────────────────────────────────────────────

    def fts_search(
        self, query: str, limit: int = 20,
        since: datetime | None = None, before: datetime | None = None,
    ) -> list[tuple[Memory, float]]:
        query = query.strip()
        if not query:
            return []

        with self.pool.connection() as conn:
            try:
                sql = """SELECT m.*, ts_rank(m.search_vector,
                              plainto_tsquery('english', %s)) AS rank
                       FROM memories m
                       WHERE m.search_vector @@ plainto_tsquery('english', %s)
                       AND m.project = %s"""
                params: list = [query, query, self.project]
                if since:
                    sql += " AND m.created_at >= %s"
                    params.append(since)
                if before:
                    sql += " AND m.created_at <= %s"
                    params.append(before)
                sql += " ORDER BY rank DESC LIMIT %s"
                params.append(limit)
                rows = conn.execute(sql, params).fetchall()
            except Exception as exc:
                logger.debug("FTS query failed for %r: %s", query, exc)
                return []

        results = []
        for row in rows:
            mem = self._row_to_memory(row)
            score = float(row["rank"])
            results.append((mem, score))
        return results

    def rebuild_fts(self) -> None:
        """Reindex the GIN index on the search_vector column."""
        with self.pool.connection() as conn:
            conn.execute("REINDEX INDEX idx_memories_search")
            conn.commit()
        logger.info("Reindexed FTS GIN index for project %s", self.project)

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> MemoryStats:
        with self.pool.connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) AS c FROM memories WHERE project = %s",
                (self.project,),
            ).fetchone()["c"]

            total_chunks = conn.execute(
                """SELECT COUNT(*) AS c FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE m.project = %s""",
                (self.project,),
            ).fetchone()["c"]

            total_rels = conn.execute(
                """SELECT COUNT(*) AS c FROM relationships r
                   WHERE r.source_id IN (SELECT id FROM memories WHERE project = %s)
                   OR r.target_id IN (SELECT id FROM memories WHERE project = %s)""",
                (self.project, self.project),
            ).fetchone()["c"]

            type_rows = conn.execute(
                "SELECT memory_type, COUNT(*) AS c FROM memories "
                "WHERE project = %s GROUP BY memory_type",
                (self.project,),
            ).fetchall()
            by_type = {r["memory_type"]: r["c"] for r in type_rows}

            imp_rows = conn.execute(
                "SELECT importance, COUNT(*) AS c FROM memories "
                "WHERE project = %s GROUP BY importance",
                (self.project,),
            ).fetchall()
            by_importance = {str(r["importance"]): r["c"] for r in imp_rows}

            oldest_row = conn.execute(
                "SELECT MIN(created_at) AS v FROM memories WHERE project = %s",
                (self.project,),
            ).fetchone()
            newest_row = conn.execute(
                "SELECT MAX(created_at) AS v FROM memories WHERE project = %s",
                (self.project,),
            ).fetchone()

            oldest = oldest_row["v"].isoformat() if oldest_row and oldest_row["v"] else None
            newest = newest_row["v"].isoformat() if newest_row and newest_row["v"] else None

        compression_stats = self.get_compression_stats(self.project)

        return MemoryStats(
            total_memories=total,
            total_chunks=total_chunks,
            total_relationships=total_rels,
            by_type=by_type,
            by_importance=by_importance,
            oldest=oldest,
            newest=newest,
            db_size_bytes=0,  # No single file for Postgres
            compression=compression_stats,
        )

    # ── Compression ───────────────────────────────────────────────

    def get_uncompressed_memories(
        self,
        project: str,
        min_size_chars: int = 500,
        recompress: bool = False,
        skip_importance_zero: bool = True,
        skip_expiring_within_days: int = 7,
    ) -> list[tuple[str, str]]:
        """Return list of (id, content) for memories that need compression."""
        conditions = ["project = %s", "char_length(content) >= %s"]
        params: list = [project, min_size_chars]

        if not recompress:
            conditions.append("content_compressed IS NULL")

        if skip_importance_zero:
            conditions.append("importance > 0")

        conditions.append("NOT immutable")

        if skip_expiring_within_days > 0:
            cutoff = datetime.now(timezone.utc) + timedelta(days=skip_expiring_within_days)
            conditions.append("(expires_at IS NULL OR expires_at > %s)")
            params.append(cutoff)

        where = " AND ".join(conditions)
        with self.pool.connection() as conn:
            rows = conn.execute(
                f"SELECT id, content FROM memories WHERE {where}",
                params,
            ).fetchall()
            return [(row["id"], row["content"]) for row in rows]

    def update_memory_compression(
        self,
        memory_id: str,
        compressed: bytes,
        algo: str,
        compressed_at: str,
    ) -> bool:
        """Store compressed bytes for a memory. Returns True if row was found and updated."""
        from datetime import datetime as _dt
        compressed_at_dt = _dt.fromisoformat(compressed_at) if isinstance(compressed_at, str) else compressed_at
        with self.pool.connection() as conn:
            row = conn.execute(
                "WITH updated AS ("
                "  UPDATE memories SET content_compressed=%s, compression_algo=%s, compressed_at=%s"
                "  WHERE id=%s RETURNING id"
                ") SELECT count(*) AS c FROM updated",
                (compressed, algo, compressed_at_dt, memory_id),
            ).fetchone()
            conn.commit()
            return row["c"] > 0

    def get_compression_stats(self, project: str) -> dict:
        """Return compression coverage statistics for a project."""
        with self.pool.connection() as conn:
            row = conn.execute(
                """SELECT
                    COUNT(*) as total,
                    COUNT(content_compressed) as compressed_count,
                    AVG(CASE WHEN content_compressed IS NOT NULL THEN
                        CAST(octet_length(content) AS REAL) / octet_length(content_compressed)
                        ELSE NULL END) as avg_ratio
                FROM memories WHERE project = %s""",
                (project,),
            ).fetchone()
            total = row["total"] or 0
            compressed = row["compressed_count"] or 0
            return {
                "total_memories": total,
                "compressed_count": compressed,
                "coverage_pct": round(100 * compressed / total, 1) if total else 0.0,
                "avg_ratio": round(float(row["avg_ratio"] or 0.0), 3),
            }

    def list_all_projects(self) -> list[str]:
        """Return all distinct project names in this database."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT project FROM memories ORDER BY project"
            ).fetchall()
            return [row["project"] for row in rows]

    def get_all_memory_ids(self, project: str) -> set[str]:
        """Return all memory IDs for a project."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE project=%s", (project,)
            ).fetchall()
            return {row["id"] for row in rows}

    def get_memories_pending_summary(self, project: str, limit: int = 20) -> list[tuple[str, str]]:
        """Return list of (id, content) for memories where summary IS NULL."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT id, content FROM memories WHERE project=%s AND summary IS NULL LIMIT %s",
                (project, limit)
            ).fetchall()
        return [(r["id"], r["content"]) for r in rows]

    def store_summary(self, memory_id: str, summary: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE memories SET summary=%s WHERE id=%s", (summary, memory_id))
            conn.commit()

    def get_pending_summary_count(self, project: str) -> int:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE project=%s AND summary IS NULL",
                (project,)
            ).fetchone()
        return row["count"] if row else 0

    # ── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _row_to_memory(row: dict) -> Memory:
        tags = row["tags"]
        # psycopg auto-decodes JSONB to Python list; handle string fallback
        if isinstance(tags, str):
            tags = json.loads(tags)

        last_accessed = row["last_accessed"]
        created_at = row["created_at"]
        updated_at = row["updated_at"]

        # Handle both datetime objects (psycopg TIMESTAMPTZ) and ISO strings
        if isinstance(last_accessed, str):
            last_accessed = datetime.fromisoformat(last_accessed)
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        expires_at_raw = row.get("expires_at")
        if isinstance(expires_at_raw, str):
            expires_at_raw = datetime.fromisoformat(expires_at_raw)
        # datetime objects pass through as-is; None stays None

        # Compression fields — may not exist before schema v5
        content_compressed = row.get("content_compressed")
        if isinstance(content_compressed, memoryview):
            content_compressed = bytes(content_compressed)
        compression_algo = row.get("compression_algo")
        compressed_at_raw = row.get("compressed_at")
        compressed_at = compressed_at_raw.isoformat() if isinstance(compressed_at_raw, datetime) else compressed_at_raw

        # Summary field — added in schema v6; may be None on older rows
        summary = row.get("summary")

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
            immutable=bool(row.get("immutable", False)),
            expires_at=expires_at_raw,
            content_compressed=content_compressed,
            compression_algo=compression_algo,
            compressed_at=compressed_at,
            summary=summary,
        )

    @staticmethod
    def _row_to_chunk(row: dict) -> Chunk:
        embedding = row["embedding"]
        # psycopg returns bytea as memoryview; convert to bytes
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
