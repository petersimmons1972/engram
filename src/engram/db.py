from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from .types import (
    Chunk,
    Memory,
    MemoryStats,
    MemoryType,
    Relationship,
)

DEFAULT_DB_DIR = Path.home() / ".engram"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL DEFAULT 'context',
    project TEXT NOT NULL DEFAULT 'default',
    tags TEXT NOT NULL DEFAULT '[]',
    importance INTEGER NOT NULL DEFAULT 2,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    tags,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content, tags)
    VALUES ('delete', old.rowid, old.content, old.tags);
    INSERT INTO memory_fts(rowid, content, tags)
    VALUES (new.rowid, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_hash TEXT NOT NULL DEFAULT '',
    embedding BLOB,
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
    created_at TEXT NOT NULL,
    FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rel_pair ON relationships(source_id, target_id, rel_type);

CREATE TABLE IF NOT EXISTS project_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
"""

CURRENT_SCHEMA_VERSION = 2


class MemoryDB:
    def __init__(self, project: str = "default", db_dir: str | Path | None = None):
        import re
        project = re.sub(r'[^a-zA-Z0-9_-]', '', project) or "default"
        self.project = project
        db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{project}.db"
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT value FROM project_meta WHERE key = 'schema_version'").fetchone()
        current = int(row["value"]) if row else 1
        if current < 2:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
            conn.commit()
        conn.execute("INSERT OR REPLACE INTO project_meta (key, value) VALUES ('schema_version', ?)",
                     (str(CURRENT_SCHEMA_VERSION),))
        conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Project Metadata ─────────────────────────────────────────

    def get_meta(self, key: str) -> str | None:
        """Retrieve a project metadata value by key."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT value FROM project_meta WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Store or update a project metadata value."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO project_meta (key, value) VALUES (?, ?)",
                (key, value),
            )
            conn.commit()

    # ── Memory CRUD ──────────────────────────────────────────────

    def store_memory(self, memory: Memory) -> Memory:
        with self._lock:
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            memory.created_at = datetime.fromisoformat(now)
            memory.updated_at = datetime.fromisoformat(now)
            memory.last_accessed = datetime.fromisoformat(now)
            memory.project = self.project

            conn.execute(
                """INSERT INTO memories (id, content, memory_type, project, tags,
                   importance, access_count, last_accessed, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    def update_memory(self, memory_id: str, content: str | None = None,
                      tags: list[str] | None = None,
                      importance: int | None = None) -> Memory | None:
        with self._lock:
            mem = self.get_memory(memory_id)
            if not mem:
                return None

            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            if content is not None:
                mem.content = content
            if tags is not None:
                mem.tags = tags
            if importance is not None:
                mem.importance = importance

            conn.execute(
                """UPDATE memories SET content=?, tags=?, importance=?, updated_at=?
                   WHERE id=?""",
                (mem.content, json.dumps(mem.tags), mem.importance, now, memory_id),
            )
            conn.commit()
            mem.updated_at = datetime.fromisoformat(now)
            return mem

    def delete_memory(self, memory_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_memory_atomic(self, memory_id: str) -> bool:
        """Delete a memory and all its chunks/relationships in a single transaction."""
        with self._lock:
            conn = self._get_conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
                conn.execute(
                    "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                    (memory_id, memory_id),
                )
                cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception:
                conn.rollback()
                raise

    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]:
        with self._lock:
            conn = self._get_conn()
            query = "SELECT * FROM memories WHERE project = ?"
            params: list = [self.project]

            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type.value)
            if min_importance is not None:
                query += " AND importance <= ?"
                params.append(min_importance)
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%"{tag}"%')
                query += " AND (" + " OR ".join(tag_conditions) + ")"

            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            rows = conn.execute(query, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def touch_memory(self, memory_id: str) -> None:
        """Bump access_count and last_accessed for decay scoring."""
        with self._lock:
            conn = self._get_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                (now, memory_id),
            )
            conn.commit()

    # ── Chunk CRUD ───────────────────────────────────────────────

    def store_chunks(self, chunks: list[Chunk]) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.executemany(
                """INSERT OR IGNORE INTO chunks (id, memory_id, chunk_text, chunk_index,
                   chunk_hash, embedding) VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (c.id, c.memory_id, c.chunk_text, c.chunk_index, c.chunk_hash, c.embedding)
                    for c in chunks
                ],
            )
            conn.commit()

    def get_chunks_for_memory(self, memory_id: str) -> list[Chunk]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM chunks WHERE memory_id = ? ORDER BY chunk_index",
                (memory_id,),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_chunks_with_embeddings(self, limit: int = 10_000) -> list[Chunk]:
        """Fetch chunks with embeddings, capped to prevent OOM on large databases."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT c.* FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE c.embedding IS NOT NULL
                   AND m.project = ?
                   ORDER BY m.last_accessed DESC
                   LIMIT ?""",
                (self.project, limit),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_chunk_texts(self, limit: int = 5000) -> list[str]:
        """Fetch chunk texts for dedup comparison. Works regardless of embedding status."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """SELECT c.chunk_text FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE m.project = ?
                   LIMIT ?""",
                (self.project, limit),
            ).fetchall()
            return [r["chunk_text"] for r in rows]

    def chunk_hash_exists(self, chunk_hash: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE chunk_hash = ? LIMIT 1", (chunk_hash,)
            ).fetchone()
            return row is not None

    def delete_chunks_for_memory(self, memory_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM chunks WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete specific chunks by their IDs."""
        if not chunk_ids:
            return 0
        with self._lock:
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in chunk_ids)
            cursor = conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)
            conn.commit()
            return cursor.rowcount

    # ── Relationship CRUD ────────────────────────────────────────

    def store_relationship(self, rel: Relationship) -> Relationship:
        with self._lock:
            conn = self._get_conn()
            src = conn.execute("SELECT 1 FROM memories WHERE id = ?", (rel.source_id,)).fetchone()
            tgt = conn.execute("SELECT 1 FROM memories WHERE id = ?", (rel.target_id,)).fetchone()
            if not src:
                raise ValueError(f"Source memory '{rel.source_id}' does not exist")
            if not tgt:
                raise ValueError(f"Target memory '{rel.target_id}' does not exist")
            try:
                conn.execute(
                    """INSERT INTO relationships (id, source_id, target_id, rel_type,
                       strength, created_at) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        rel.id,
                        rel.source_id,
                        rel.target_id,
                        rel.rel_type.value,
                        rel.strength,
                        rel.created_at.isoformat(),
                    ),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                conn.execute(
                    "UPDATE relationships SET strength = ?"
                    " WHERE source_id = ? AND target_id = ? AND rel_type = ?",
                    (rel.strength, rel.source_id, rel.target_id, rel.rel_type.value),
                )
                conn.commit()
            return rel

    def get_connected(
        self, memory_id: str, max_hops: int = 2,
    ) -> list[tuple[Memory, str, str, float]]:
        """BFS traversal up to max_hops. Returns (memory, rel_type, direction, strength)."""
        with self._lock:
            conn = self._get_conn()
            visited: set[str] = {memory_id}
            results: list[tuple[Memory, str, str, float]] = []
            frontier = [memory_id]

            for _ in range(max_hops):
                if not frontier:
                    break
                next_frontier: list[str] = []
                placeholders = ",".join("?" for _ in frontier)

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
        """Strengthen all edges connected to a memory. Used by feedback loops."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """UPDATE relationships
                   SET strength = MIN(1.0, strength + ?)
                   WHERE source_id = ? OR target_id = ?""",
                (factor, memory_id, memory_id),
            )
            conn.commit()
            return cursor.rowcount

    def decay_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int:
        """Weaken all edges connected to a memory. Used by negative feedback."""
        with self._lock:
            conn = self._get_conn()
            cursor = conn.execute(
                """UPDATE relationships
                   SET strength = MAX(0.0, strength - ?)
                   WHERE source_id = ? OR target_id = ?""",
                (factor, memory_id, memory_id),
            )
            conn.commit()
            return cursor.rowcount

    def get_connection_count(self, memory_id: str) -> int:
        """Count how many edges connect to this memory (graph degree)."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """SELECT COUNT(*) as c FROM relationships
                   WHERE source_id = ? OR target_id = ?""",
                (memory_id, memory_id),
            ).fetchone()
            return row["c"]

    def decay_all_edges(
        self, decay_factor: float = 0.02, min_strength: float = 0.1,
    ) -> tuple[int, int]:
        """Apply decay to all edges. Prune edges below min_strength.
        Returns (decayed_count, pruned_count)."""
        with self._lock:
            conn = self._get_conn()
            decayed = conn.execute(
                "UPDATE relationships SET strength = MAX(0.0, strength - ?)",
                (decay_factor,),
            ).rowcount
            pruned = conn.execute(
                "DELETE FROM relationships WHERE strength < ?",
                (min_strength,),
            ).rowcount
            conn.commit()
            return decayed, pruned

    def prune_stale_memories(self, max_age_hours: float = 720, max_importance: int = 3) -> int:
        """Remove low-importance memories that haven't been accessed in max_age_hours.
        Never prunes memories with importance <= 1 (critical/high)."""
        with self._lock:
            conn = self._get_conn()
            from datetime import datetime, timedelta, timezone
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
            cursor = conn.execute(
                """DELETE FROM memories
                   WHERE project = ? AND importance >= ? AND last_accessed < ? AND access_count = 0""",
                (self.project, max_importance, cutoff),
            )
            conn.commit()
            return cursor.rowcount

    def delete_relationships_for_memory(self, memory_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM relationships WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )
            conn.commit()

    # ── FTS5 Search ──────────────────────────────────────────────

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Strip FTS5 special syntax to prevent malformed MATCH queries."""
        import re
        sanitized = re.sub(r'\w+:', ' ', query)  # strip column filters
        sanitized = re.sub(r'[*^"()]', ' ', sanitized)
        sanitized = re.sub(r'\b(AND|OR|NOT|NEAR)\b', '', sanitized, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', sanitized).strip()

    def fts_search(self, query: str, limit: int = 20) -> list[tuple[Memory, float]]:
        with self._lock:
            conn = self._get_conn()
            safe_query = self._sanitize_fts_query(query)
            if not safe_query:
                return []
            try:
                rows = conn.execute(
                    """SELECT m.*, bm25(memory_fts) AS rank
                       FROM memory_fts f
                       JOIN memories m ON m.rowid = f.rowid
                       WHERE memory_fts MATCH ?
                       AND m.project = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (safe_query, self.project, limit),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                logger.debug("FTS query failed for %r: %s", safe_query, exc)
                return []

            results = []
            for row in rows:
                mem = self._row_to_memory(row)
                score = -row["rank"]  # BM25 returns negative scores; negate for positive
                results.append((mem, score))
            return results

    def rebuild_fts(self) -> None:
        """Rebuild the FTS index from the memories table.

        Uses the FTS5 'rebuild' command for content-sync tables, which
        re-reads all content from the source table. Falls back to manual
        delete+reinsert if 'rebuild' fails (e.g., corrupted index).
        """
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute("INSERT INTO memory_fts(memory_fts) VALUES('rebuild')")
                conn.commit()
            except sqlite3.DatabaseError:
                logger.warning("FTS rebuild command failed, falling back to manual rebuild")
                conn.execute("DELETE FROM memory_fts")
                conn.execute(
                    "INSERT INTO memory_fts(rowid, content, tags) "
                    "SELECT rowid, content, tags FROM memories WHERE project = ?",
                    (self.project,),
                )
                conn.commit()
            logger.info("Rebuilt FTS index for project %s", self.project)

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> MemoryStats:
        with self._lock:
            conn = self._get_conn()

            total = conn.execute(
                "SELECT COUNT(*) as c FROM memories WHERE project = ?", (self.project,)
            ).fetchone()["c"]

            total_chunks = conn.execute(
                """SELECT COUNT(*) as c FROM chunks c
                   JOIN memories m ON m.id = c.memory_id
                   WHERE m.project = ?""",
                (self.project,),
            ).fetchone()["c"]

            total_rels = conn.execute(
                """SELECT COUNT(*) as c FROM relationships r
                   WHERE r.source_id IN (SELECT id FROM memories WHERE project = ?)
                   OR r.target_id IN (SELECT id FROM memories WHERE project = ?)""",
                (self.project, self.project),
            ).fetchone()["c"]

            type_rows = conn.execute(
                "SELECT memory_type, COUNT(*) as c FROM memories"
                " WHERE project = ? GROUP BY memory_type",
                (self.project,),
            ).fetchall()
            by_type = {r["memory_type"]: r["c"] for r in type_rows}

            imp_rows = conn.execute(
                "SELECT importance, COUNT(*) as c FROM memories WHERE project = ? GROUP BY importance",
                (self.project,),
            ).fetchall()
            by_importance = {str(r["importance"]): r["c"] for r in imp_rows}

            oldest_row = conn.execute(
                "SELECT MIN(created_at) as v FROM memories WHERE project = ?", (self.project,)
            ).fetchone()
            newest_row = conn.execute(
                "SELECT MAX(created_at) as v FROM memories WHERE project = ?", (self.project,)
            ).fetchone()

            db_size = os.path.getsize(self.db_path) if self.db_path.exists() else 0

            return MemoryStats(
                total_memories=total,
                total_chunks=total_chunks,
                total_relationships=total_rels,
                by_type=by_type,
                by_importance=by_importance,
                oldest=oldest_row["v"] if oldest_row else None,
                newest=newest_row["v"] if newest_row else None,
                db_size_bytes=db_size,
            )

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            project=row["project"],
            tags=json.loads(row["tags"]),
            importance=row["importance"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            memory_id=row["memory_id"],
            chunk_text=row["chunk_text"],
            chunk_index=row["chunk_index"],
            chunk_hash=row["chunk_hash"],
            embedding=row["embedding"],
        )
