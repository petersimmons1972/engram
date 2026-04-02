"""Tests for BackgroundReembedder and memory_migrate_embedder (issue #73)."""
from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)

from engram.db_postgres import PostgresBackend
from engram.embeddings import to_blob
from engram.reembedder import BackgroundReembedder
from engram.types import Memory, Chunk


class TestNullAllEmbeddings:
    def test_nulls_all_embeddings(self, db, embedder):
        """After null_all_embeddings, all chunks for the project have NULL embedding."""
        from engram.search import SearchEngine
        engine = SearchEngine(db=db, embedder=embedder)
        engine.store(Memory(content="memory with embedding one two three"))
        engine.store(Memory(content="another memory four five six"))

        count = db.null_all_embeddings(db.project)
        assert count > 0

        pending = db.get_chunks_pending_embedding(db.project)
        assert len(pending) == count

    def test_does_not_affect_other_projects(self):
        """null_all_embeddings only touches the specified project."""
        dsn = os.environ["TEST_DATABASE_URL"]
        db_a = PostgresBackend(project="proj-a-reembed", dsn=dsn)
        db_b = PostgresBackend(project="proj-b-reembed", dsn=dsn)
        try:
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path(__file__).parent))
            from conftest import FakeEmbedder
            from engram.search import SearchEngine
            emb = FakeEmbedder()
            eng_a = SearchEngine(db=db_a, embedder=emb)
            eng_b = SearchEngine(db=db_b, embedder=emb)
            eng_a.store(Memory(content="project a memory alpha beta"))
            eng_b.store(Memory(content="project b memory gamma delta"))

            db_a.null_all_embeddings("proj-a-reembed")

            assert db_b.get_pending_embedding_count("proj-b-reembed") == 0
        finally:
            with db_a.pool.connection() as conn:
                conn.execute("DELETE FROM memories WHERE project = %s", ("proj-a-reembed",))
                conn.commit()
            with db_b.pool.connection() as conn:
                conn.execute("DELETE FROM memories WHERE project = %s", ("proj-b-reembed",))
                conn.commit()
            db_a.close()
            db_b.close()


class TestBackgroundReembedder:
    def test_reembeds_pending_chunks(self, db, embedder):
        """BackgroundReembedder fills in NULL embeddings."""
        from engram.search import SearchEngine
        engine = SearchEngine(db=db, embedder=embedder)
        engine.store(Memory(content="content to be reembedded alpha beta gamma"))
        db.null_all_embeddings(db.project)
        assert db.get_pending_embedding_count(db.project) > 0

        db.set_meta("embedding_migration_in_progress", "true")
        reembedder = BackgroundReembedder(db=db, embedder=embedder, project=db.project)
        reembedder.start()
        time.sleep(0.5)
        reembedder.stop()

        assert db.get_pending_embedding_count(db.project) == 0

    def test_clears_migration_flag_when_done(self, db, embedder):
        """Once queue is drained, migration flag is set to false."""
        from engram.search import SearchEngine
        engine = SearchEngine(db=db, embedder=embedder)
        engine.store(Memory(content="migration flag test memory one two"))
        db.null_all_embeddings(db.project)
        db.set_meta("embedding_migration_in_progress", "true")

        reembedder = BackgroundReembedder(db=db, embedder=embedder, project=db.project)
        reembedder.start()
        time.sleep(1.0)
        reembedder.stop()

        assert db.get_meta("embedding_migration_in_progress") == "false"

    def test_does_not_start_without_migration_flag(self, db, embedder):
        """BackgroundReembedder.start() is a no-op when no migration is in progress."""
        reembedder = BackgroundReembedder(db=db, embedder=embedder, project=db.project)
        reembedder.start()
        assert not reembedder._thread.is_alive()


class TestCheckEmbedderMetadataRelaxed:
    def test_skips_check_during_migration(self, db, embedder):
        """_check_embedder_metadata does not raise when migration is in progress."""
        from engram.search import SearchEngine
        from engram.errors import EmbeddingConfigMismatchError

        engine = SearchEngine(db=db, embedder=embedder)
        # Store a memory to set embedder metadata
        engine.store(Memory(content="set the embedder metadata here"))

        # Simulate wrong embedder by storing a different name in project_meta
        db.set_meta("embedder_name", "different/embedder")
        db.set_meta("embedding_migration_in_progress", "true")

        # Should not raise
        engine._check_embedder_metadata()
