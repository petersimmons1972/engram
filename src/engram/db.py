"""Database abstraction layer for engram.

Provides a ``DatabaseBackend`` protocol that both SQLite and Postgres backends
implement, a ``create_database`` factory driven by the ``DATABASE_URL`` env-var,
and a backwards-compatible ``MemoryDB`` re-export so existing imports keep working.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from .types import Chunk, Memory, MemoryStats, MemoryType, Relationship


# ── Helpers ──────────────────────────────────────────────────────────

def _normalize_project(project: str) -> str:
    """Sanitize a project name to safe filesystem/table characters."""
    return re.sub(r"[^a-zA-Z0-9_-]", "", project) or "default"


# ── Protocol ─────────────────────────────────────────────────────────

@runtime_checkable
class DatabaseBackend(Protocol):
    """Common interface that every engram storage backend must implement."""

    project: str

    # -- lifecycle --
    def close(self) -> None: ...
    def __enter__(self) -> "DatabaseBackend": ...
    def __exit__(self, *exc: object) -> None: ...

    # -- project metadata --
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...

    # -- memory CRUD --
    def store_memory(self, memory: Memory) -> Memory: ...
    def get_memory(self, memory_id: str) -> Memory | None: ...
    def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        tags: list[str] | None = None,
        importance: int | None = None,
    ) -> Memory | None: ...
    def delete_memory(self, memory_id: str) -> bool: ...
    def delete_memory_atomic(self, memory_id: str) -> bool: ...
    def list_memories(
        self,
        memory_type: MemoryType | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Memory]: ...
    def touch_memory(self, memory_id: str) -> None: ...

    # -- chunk CRUD --
    def store_chunks(self, chunks: list[Chunk]) -> None: ...
    def get_chunks_for_memory(self, memory_id: str) -> list[Chunk]: ...
    def get_all_chunks_with_embeddings(self, limit: int = 10_000) -> list[Chunk]: ...
    def get_all_chunk_texts(self, limit: int = 5000) -> list[str]: ...
    def chunk_hash_exists(self, chunk_hash: str) -> bool: ...
    def delete_chunks_for_memory(self, memory_id: str) -> None: ...
    def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int: ...

    # -- relationship CRUD --
    def store_relationship(self, rel: Relationship) -> Relationship: ...
    def get_connected(
        self, memory_id: str, max_hops: int = 2,
    ) -> list[tuple[Memory, str, str, float]]: ...
    def boost_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int: ...
    def decay_edges_for_memory(self, memory_id: str, factor: float = 0.05) -> int: ...
    def get_connection_count(self, memory_id: str) -> int: ...
    def decay_all_edges(
        self, decay_factor: float = 0.02, min_strength: float = 0.1,
    ) -> tuple[int, int]: ...
    def prune_stale_memories(self, max_age_hours: float = 720, max_importance: int = 3) -> int: ...
    def delete_relationships_for_memory(self, memory_id: str) -> None: ...

    # -- FTS search --
    def fts_search(self, query: str, limit: int = 20) -> list[tuple[Memory, float]]: ...
    def rebuild_fts(self) -> None: ...

    # -- stats --
    def get_stats(self) -> MemoryStats: ...


# ── Factory ──────────────────────────────────────────────────────────

def create_database(
    project: str = "default",
    db_dir: str | Path | None = None,
) -> DatabaseBackend:
    """Create the appropriate backend based on the ``DATABASE_URL`` env var.

    * If ``DATABASE_URL`` starts with ``"postgresql"``, returns a
      ``PostgresBackend`` (imported lazily to avoid hard dependency on psycopg).
    * Otherwise returns a ``SqliteBackend``.
    """
    database_url = os.environ.get("DATABASE_URL", "")

    if database_url.startswith("postgresql"):
        from .db_postgres import PostgresBackend  # type: ignore[import-not-found]
        return PostgresBackend(project=_normalize_project(project), dsn=database_url)

    from .db_sqlite import SqliteBackend
    return SqliteBackend(project=_normalize_project(project), db_dir=db_dir)


# ── Backwards-compatible re-export ───────────────────────────────────

from .db_sqlite import SqliteBackend as MemoryDB  # noqa: E402, F811

__all__ = [
    "DatabaseBackend",
    "MemoryDB",
    "SqliteBackend",
    "create_database",
    "_normalize_project",
]
