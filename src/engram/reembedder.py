"""Background re-embedding of chunks after an embedding provider migration.

Runs as a daemon thread per project. Polls for chunks with NULL embedding,
embeds them in batches using the current provider, and stores the results.
When the queue is fully drained, clears the migration flag from project_meta.

Controlled by the ``embedding_migration_in_progress`` key in project_meta.
The thread does NOT start unless that flag is "true" — safe to construct on
every SearchEngine init without side effects.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DatabaseBackend
    from .embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30  # seconds between passes when queue is empty
BATCH_SIZE = 20     # chunks per embedding call


class BackgroundReembedder:
    """Daemon thread that re-embeds chunks with NULL embedding.

    Started when an embedding migration is in progress. Polls for chunks
    with embedding=NULL, embeds them in batches, and stores results.
    When the queue is drained, clears the migration flag from project_meta.
    """

    def __init__(self, db: "DatabaseBackend", embedder: "EmbeddingProvider", project: str):
        self.db = db
        self.embedder = embedder
        self.project = project
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"reembedder-{project}",
            daemon=True,
        )

    def start(self) -> None:
        # Only start if a migration is actually in progress — safe no-op otherwise
        if self.db.get_meta("embedding_migration_in_progress") != "true":
            return
        self._thread.start()
        logger.info("BackgroundReembedder started for project=%s", self.project)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=35.0)
        if self._thread.is_alive():
            logger.warning(
                "BackgroundReembedder thread did not exit within 35 s — possible orphaned "
                "thread (project=%s)",
                self.project,
            )

    def _run(self) -> None:
        from .embeddings import to_blob
        while not self._stop_event.is_set():
            try:
                chunks = self.db.get_chunks_pending_embedding(self.project, limit=BATCH_SIZE)
                if not chunks:
                    # Queue drained — clear migration flag
                    if self.db.get_meta("embedding_migration_in_progress") == "true":
                        self.db.set_meta("embedding_migration_in_progress", "false")
                        logger.info(
                            "Embedding migration complete for project=%s", self.project
                        )
                    # Nothing to do — wait before next poll
                    self._stop_event.wait(timeout=POLL_INTERVAL)
                    continue

                texts = [c.chunk_text for c in chunks]
                embeddings = self.embedder.embed_batch(texts)
                for chunk, emb in zip(chunks, embeddings):
                    rows_updated = self.db.update_chunk_embedding(chunk.id, to_blob(emb))
                    if rows_updated == 0:
                        # Chunk was deleted between fetch and update — this is the
                        # application-level race documented in issue #93: store()
                        # may have deleted the chunk via delete_memory_atomic during
                        # a store rollback. Skip silently but log so it is traceable.
                        logger.warning(
                            "update_chunk_embedding: chunk %s not found "
                            "(may have been deleted during store rollback) — skipping",
                            chunk.id,
                        )
                    else:
                        logger.debug("Re-embedded chunk %s", chunk.id)
                # If we got a full batch there may be more — loop immediately.
                # If we got a partial batch the queue is nearly drained — loop
                # immediately to clear it and set the completion flag.
            except Exception as e:
                logger.warning("Reembedder error: %s", e)
                self._stop_event.wait(timeout=POLL_INTERVAL)
