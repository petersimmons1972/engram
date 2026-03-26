#!/usr/bin/env python3
"""Restore memories from archived SQLite databases to PostgreSQL."""

import sqlite3
import json
from pathlib import Path
import logging
import psycopg

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path.home() / ".engram-archive"
DB_NAMES = ["default", "clearwatch", "3dprint", "engram", "global"]

def restore_from_sqlite():
    """Migrate all archived SQLite databases to PostgreSQL."""

    # Connect to PostgreSQL
    pg_conn = psycopg.connect(
        "postgresql://engram:VYGw4buWza3wkhqRNpdLRRDkps00wAg7@localhost:5432/engram"
    )

    try:
        total_restored = {"memories": 0, "chunks": 0, "relationships": 0}

        for db_name in DB_NAMES:
            db_path = ARCHIVE_DIR / f"{db_name}.db"
            if not db_path.exists():
                logger.warning(f"Skipping {db_name}: {db_path} not found")
                continue

            logger.info(f"Restoring {db_name}...")

            # Open SQLite database
            sqlite_conn = sqlite3.connect(str(db_path))
            sqlite_conn.row_factory = sqlite3.Row

            try:
                # Get all memories from SQLite
                cursor = sqlite_conn.execute("SELECT * FROM memories")
                memories = list(cursor.fetchall())
                logger.info(f"  Memories: {len(memories)}")

                for mem in memories:
                    try:
                        tags = json.loads(mem["tags"]) if isinstance(mem["tags"], str) else (mem["tags"] or [])
                        if not isinstance(tags, list):
                            tags = []

                        pg_conn.execute(
                            """
                            INSERT INTO memories
                            (id, content, memory_type, project, tags, importance,
                             access_count, last_accessed, created_at, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                mem["id"],
                                mem["content"],
                                mem["memory_type"],
                                mem["project"],
                                json.dumps(tags),
                                mem["importance"],
                                mem["access_count"],
                                mem["last_accessed"],
                                mem["created_at"],
                                mem["updated_at"],
                            )
                        )
                        total_restored["memories"] += 1
                    except Exception as e:
                        logger.debug(f"    Skipped memory {mem['id']}: {e}")

                # Get all chunks from SQLite
                cursor = sqlite_conn.execute("SELECT * FROM chunks")
                chunks = list(cursor.fetchall())
                logger.info(f"  Chunks: {len(chunks)}")

                for chunk in chunks:
                    try:
                        pg_conn.execute(
                            """
                            INSERT INTO chunks
                            (id, memory_id, chunk_text, chunk_index, chunk_hash, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                chunk["id"],
                                chunk["memory_id"],
                                chunk["chunk_text"],
                                chunk["chunk_index"],
                                chunk["chunk_hash"],
                                chunk["embedding"],
                            )
                        )
                        total_restored["chunks"] += 1
                    except Exception as e:
                        logger.debug(f"    Skipped chunk {chunk['id']}: {e}")

                # Get all relationships from SQLite
                cursor = sqlite_conn.execute("SELECT * FROM relationships")
                rels = list(cursor.fetchall())
                logger.info(f"  Relationships: {len(rels)}")

                for rel in rels:
                    try:
                        pg_conn.execute(
                            """
                            INSERT INTO relationships
                            (id, source_id, target_id, rel_type, strength, project, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO NOTHING
                            """,
                            (
                                rel["id"],
                                rel["source_id"],
                                rel["target_id"],
                                rel["rel_type"],
                                rel["strength"],
                                rel.get("project", ""),
                                rel["created_at"],
                            )
                        )
                        total_restored["relationships"] += 1
                    except Exception as e:
                        logger.debug(f"    Skipped relationship {rel['id']}: {e}")

                pg_conn.commit()

            finally:
                sqlite_conn.close()

        logger.info(f"\nRestore complete!")
        logger.info(f"  Total memories: {total_restored['memories']}")
        logger.info(f"  Total chunks: {total_restored['chunks']}")
        logger.info(f"  Total relationships: {total_restored['relationships']}")

    finally:
        pg_conn.close()

if __name__ == "__main__":
    restore_from_sqlite()
