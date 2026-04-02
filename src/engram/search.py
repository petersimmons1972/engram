from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timezone

from .chunker import chunk_hash, chunk_text
from .db import DatabaseBackend
from .embeddings import EmbeddingProvider, NullEmbedder, cosine_similarity, from_blob, to_blob
from .errors import EmbeddingConfigMismatchError
from .reembedder import BackgroundReembedder
from .summarizer import BackgroundSummarizer
from .types import (
    Chunk,
    ConnectedMemory,
    Memory,
    SearchResult,
)

logger = logging.getLogger(__name__)

WEIGHT_VECTOR = 0.50
WEIGHT_BM25 = 0.35
WEIGHT_RECENCY = 0.15

DECAY_RATE = 0.01  # per hour


class SearchEngine:
    def __init__(self, db: DatabaseBackend, embedder: EmbeddingProvider):
        self.db = db
        self.embedder = embedder
        self._is_null = isinstance(embedder, NullEmbedder)
        self._consolidation_lock = threading.Lock()
        self._summarizer = BackgroundSummarizer(db=self.db, project=self.db.project)
        self._summarizer.start()
        self._reembedder = BackgroundReembedder(db=self.db, embedder=self.embedder, project=self.db.project)
        self._reembedder.start()

    @property
    def has_vectors(self) -> bool:
        """Whether this engine produces vector embeddings."""
        return not self._is_null

    def _check_embedder_metadata(self) -> None:
        """Enforce that the current embedder matches the project's stored config.

        On first embed, stores the embedder name and dimensions.
        On subsequent embeds, raises EmbeddingConfigMismatchError on mismatch.

        When a migration is in progress, the embedder metadata has already been
        updated to the new provider — skip the mismatch check so stores and
        recalls can continue during re-embedding.
        """
        if self._is_null:
            return

        # If a migration is in progress, the embedder has already been updated in
        # project_meta — don't block operations during re-embedding
        if self.db.get_meta("embedding_migration_in_progress") == "true":
            return

        stored_name = self.db.get_meta("embedder_name")
        stored_dims = self.db.get_meta("embedder_dimensions")

        if stored_name is None:
            self.db.set_meta("embedder_name", self.embedder.name)
            self.db.set_meta("embedder_dimensions", str(self.embedder.dimensions))
            self.db.set_meta("embedder_version", getattr(self.embedder, "version", "unknown"))
            return

        if stored_name != self.embedder.name or int(stored_dims or 0) != self.embedder.dimensions:
            raise EmbeddingConfigMismatchError(
                stored_name=stored_name,
                stored_dims=int(stored_dims or 0),
                current_name=self.embedder.name,
                current_dims=self.embedder.dimensions,
            )

    def store(self, memory: Memory) -> Memory:
        """Store a memory: chunk it, embed it (if provider available), index it.

        The entire operation is transactional: if ANY step fails (embedding,
        chunk storage), the memory is rolled back to prevent orphans.
        """
        self._check_embedder_metadata()

        memory = self.db.store_memory(memory)

        try:
            chunks = chunk_text(memory.content)

            texts_to_embed: list[str] = []
            chunk_objects: list[Chunk] = []

            for i, text in enumerate(chunks):
                h = chunk_hash(text)
                if self.db.chunk_hash_exists(h):
                    continue
                chunk_objects.append(
                    Chunk(memory_id=memory.id, chunk_text=text, chunk_index=i, chunk_hash=h)
                )
                texts_to_embed.append(text)

            if texts_to_embed and self.has_vectors:
                embeddings = self.embedder.embed_batch(texts_to_embed)
                for chunk_obj, emb in zip(chunk_objects, embeddings):
                    chunk_obj.embedding = to_blob(emb)

            if chunk_objects:
                self.db.store_chunks(chunk_objects)
        except Exception as e:
            # Any failure after memory creation: roll back to prevent orphans
            self.db.delete_memory_atomic(memory.id)
            raise ValueError(
                f"Failed to store memory {memory.id}. "
                f"Memory deleted to prevent orphaned record. Error: {e}"
            ) from e

        return memory

    def store_batch(self, memories: list[Memory]) -> list[Memory]:
        """Store multiple memories with batched embedding.

        Chunks all memories, embeds all chunks in one batch call, then stores.
        Individual memory failures are skipped without aborting the batch.
        """
        self._check_embedder_metadata()
        stored: list[Memory] = []
        all_chunks: list[Chunk] = []
        all_texts: list[str] = []

        for memory in memories:
            try:
                memory = self.db.store_memory(memory)
                stored.append(memory)
                chunks = chunk_text(memory.content)
                for i, text in enumerate(chunks):
                    h = chunk_hash(text)
                    if self.db.chunk_hash_exists(h):
                        continue
                    chunk_obj = Chunk(
                        memory_id=memory.id, chunk_text=text,
                        chunk_index=i, chunk_hash=h,
                    )
                    all_chunks.append(chunk_obj)
                    all_texts.append(text)
            except Exception as e:
                logger.warning("Failed to process memory %s in batch: %s", memory.id, e)
                continue  # Skip failed individual memories

        if all_texts and self.has_vectors:
            embeddings = self.embedder.embed_batch(all_texts)
            for chunk_obj, emb in zip(all_chunks, embeddings):
                chunk_obj.embedding = to_blob(emb)

        if all_chunks:
            self.db.store_chunks(all_chunks)

        return stored

    def recall(
        self,
        query: str,
        top_k: int = 10,
        memory_type: str | None = None,
        tags: list[str] | None = None,
        min_importance: int | None = None,
        graph_hops: int = 1,
        since: datetime | None = None,
        before: datetime | None = None,
    ) -> list[SearchResult]:
        """Three-signal recall: BM25 + vector + recency. Graph is enrichment only."""

        candidates: dict[str, _Candidate] = {}

        if self._is_null:
            # NullEmbedder: redistribute vector weight to BM25 (recency keeps its weight)
            w_vector = 0.0
            w_bm25 = WEIGHT_BM25 + WEIGHT_VECTOR  # BM25 gets all non-recency, non-vector weight
            w_recency = WEIGHT_RECENCY
        else:
            w_vector = WEIGHT_VECTOR
            w_bm25 = WEIGHT_BM25
            w_recency = WEIGHT_RECENCY

        # Layer 1: FTS5 / BM25
        fts_results = self.db.fts_search(query, limit=top_k * 2, since=since, before=before)
        if fts_results:
            max_bm25 = max(score for _, score in fts_results)
            min_bm25 = min(score for _, score in fts_results)
            score_range = max_bm25 - min_bm25
            for mem, score in fts_results:
                # Single result or all same score: normalize to 1.0 (not 0.0)
                norm_score = (score - min_bm25) / score_range if score_range > 0 else 1.0
                cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                cand.bm25_score = norm_score
                cand.matched_chunk = mem.content[:200]

        # Layer 2: Vector / Semantic (skipped for NullEmbedder)
        if self.has_vectors:
            query_vec = self.embedder.embed(query)

            # FTS-first: load embeddings only for FTS candidate memories
            fts_memory_ids = [mem.id for mem, _ in fts_results] if fts_results else []
            need_fallback = len(fts_memory_ids) < top_k

            if fts_memory_ids:
                fts_chunks = self.db.get_chunks_for_memories(fts_memory_ids)
            else:
                fts_chunks = []

            # If FTS didn't return enough candidates, fall back to full scan
            # for remaining slots to catch semantic-only matches
            if need_fallback:
                all_chunks = self.db.get_all_chunks_with_embeddings()
                fts_chunk_ids = {c.id for c in fts_chunks}
                fallback_chunks = [c for c in all_chunks if c.id not in fts_chunk_ids]
            else:
                fallback_chunks = []

            # Score all candidate chunks (FTS candidates + fallback)
            all_candidate_chunks = fts_chunks + fallback_chunks
            chunk_scores: list[tuple[Chunk, float]] = []
            for chunk in all_candidate_chunks:
                if chunk.embedding is None:
                    continue
                chunk_vec = from_blob(chunk.embedding)
                if len(chunk_vec) == 0:
                    continue
                sim = cosine_similarity(query_vec, chunk_vec)
                chunk_scores.append((chunk, sim))

            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            top_chunks = chunk_scores[: top_k * 2]

            if top_chunks:
                max_vec = top_chunks[0][1] if top_chunks[0][1] > 0 else 1.0
                for chunk, sim in top_chunks:
                    norm_score = sim / max_vec if max_vec > 0 else 0
                    mem = self.db.get_memory(chunk.memory_id)
                    if not mem:
                        continue
                    cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
                    if norm_score > cand.vector_score:
                        cand.vector_score = norm_score
                        cand.matched_chunk = chunk.chunk_text[:200]

        # Score each candidate
        now = datetime.now(timezone.utc)
        scored: list[SearchResult] = []

        for cand in candidates.values():
            mem = cand.memory

            # Apply filters
            if memory_type and mem.memory_type.value != memory_type:
                continue
            if min_importance is not None and mem.importance > min_importance:
                continue
            if tags and not (set(tags) & set(mem.tags)):
                continue

            # Layer 3: Recency decay
            hours = max((now - mem.last_accessed).total_seconds() / 3600, 0.01)
            recency_score = math.exp(-DECAY_RATE * hours)

            composite = (
                w_vector * cand.vector_score
                + w_bm25 * cand.bm25_score
                + w_recency * recency_score
            )

            # Importance multiplier: bounded to <= 2x variance
            # importance 0 => 1.3x, importance 4 => 0.7x (1.86x variance)
            importance_mult = 1.3 - (mem.importance * 0.15)
            final_score = composite * importance_mult

            scored.append(
                SearchResult(
                    memory=mem,
                    score=round(final_score, 4),
                    score_breakdown={
                        "vector": round(cand.vector_score, 4),
                        "bm25": round(cand.bm25_score, 4),
                        "recency": round(recency_score, 4),
                        "importance_mult": round(importance_mult, 2),
                    },
                    matched_chunk=cand.matched_chunk,
                )
            )

        # Post-filter by temporal bounds (catches vector-only candidates that
        # bypassed the FTS temporal filter)
        if since:
            scored = [r for r in scored if r.memory.created_at >= since]
        if before:
            scored = [r for r in scored if r.memory.created_at <= before]

        scored.sort(key=lambda r: r.score, reverse=True)
        top_results = scored[:top_k]

        # Graph expansion: attach connected memories
        for result in top_results:
            self.db.touch_memory(result.memory.id)
            connected_raw = self.db.get_connected(result.memory.id, max_hops=graph_hops)
            result.connected = [
                ConnectedMemory(
                    memory=mem,
                    rel_type=rel_type,
                    direction=direction,
                    strength=strength,
                )
                for mem, rel_type, direction, strength in connected_raw
            ]

        return top_results

    def feedback(self, memory_ids: list[str], helpful: bool) -> dict:
        """Reinforce or weaken graph edges connected to recalled memories.

        Inspired by Cognee's feedback loop: when results are helpful,
        the graph paths that produced them get stronger. When unhelpful,
        they get weaker. Over time, the graph self-optimizes.
        """
        total_affected = 0
        boost = 0.05 if helpful else -0.05

        for mid in memory_ids:
            mem = self.db.get_memory(mid)
            if not mem:
                continue
            if helpful:
                self.db.boost_edges_for_memory(mid, abs(boost))
                self.db.touch_memory(mid)
            else:
                self.db.decay_edges_for_memory(mid, abs(boost))
            total_affected += 1

        return {
            "action": "reinforced" if helpful else "weakened",
            "memories_affected": total_affected,
        }

    def consolidate(self) -> dict:
        """Memory consolidation pass — dedup, decay, and prune.

        Three stages:
        1. Deduplicate chunks (by hash)
        2. Decay all edge strengths and prune weak edges
        3. Prune stale, never-accessed, low-importance memories
        """
        if not self._consolidation_lock.acquire(blocking=False):
            return {
                "status": "already_running",
                "chunks_deduped": 0,
                "edges_decayed": 0,
                "edges_pruned": 0,
                "stale_memories_pruned": 0,
                "message": "Consolidation already in progress for this project",
            }
        try:
            # Stage 1: Dedup chunks
            deduped = self._dedup_chunks()

            # Stage 1b: Rebuild FTS index to remove stale entries from deduped/pruned memories
            if deduped > 0:
                self.db.rebuild_fts()

            # Stage 2: Decay and prune edges
            decayed, pruned_edges = self.db.decay_all_edges(decay_factor=0.02, min_strength=0.1)

            # Stage 3: Prune stale memories (30 days, low importance, never accessed)
            pruned_memories = self.db.prune_stale_memories(max_age_hours=720, max_importance=3)

            # Rebuild FTS if any memories were pruned (triggers may not fire for all deletions)
            if pruned_memories > 0:
                self.db.rebuild_fts()

            return {
                "chunks_deduped": deduped,
                "edges_decayed": decayed,
                "edges_pruned": pruned_edges,
                "stale_memories_pruned": pruned_memories,
            }
        finally:
            self._consolidation_lock.release()

    @property
    def project(self) -> str:
        """Convenience accessor for the underlying DB project name."""
        return self.db.project

    def close(self) -> None:
        """Stop background threads cleanly."""
        self._summarizer.stop()
        self._reembedder.stop()

    def compress_memories(
        self,
        algorithm: str = "zlib",
        min_size_chars: int = 500,
        dry_run: bool = False,
        recompress: bool = False,
    ) -> dict:
        """Compress stored memories to reduce context window cost.

        The original ``content`` field is never modified. Compressed bytes are
        stored in ``content_compressed`` only.
        """
        from .compression import compress, compression_ratio, CompressionAlgoUnavailableError

        rows = self.db.get_uncompressed_memories(
            self.db.project, min_size_chars=min_size_chars, recompress=recompress
        )

        compressed_count = 0
        failed_count = 0
        total_bytes_saved = 0
        ratios: list[float] = []

        for memory_id, content in rows:
            if dry_run:
                try:
                    cbytes, algo = compress(content, algorithm)
                    ratio = compression_ratio(content, cbytes)
                    ratios.append(ratio)
                    compressed_count += 1
                except Exception:
                    failed_count += 1
                continue

            try:
                cbytes, algo = compress(content, algorithm)
                ratio = compression_ratio(content, cbytes)
                compressed_at = datetime.now(timezone.utc).isoformat()
                updated = self.db.update_memory_compression(memory_id, cbytes, algo, compressed_at)
                if updated:
                    compressed_count += 1
                    ratios.append(ratio)
                    total_bytes_saved += max(0, len(content.encode()) - len(cbytes))
                # If updated=False, memory was deleted between select and update — safe to ignore
            except CompressionAlgoUnavailableError:
                raise  # surface this — not a per-memory failure
            except Exception as e:
                failed_count += 1
                logger.warning(f"Failed to compress memory {memory_id}: {e}")

        avg_ratio = round(sum(ratios) / len(ratios), 3) if ratios else 0.0

        return {
            "status": "dry_run" if dry_run else "compressed",
            "compressed": compressed_count,
            "failed": failed_count,
            "bytes_saved": total_bytes_saved,
            "avg_ratio": avg_ratio,
            "algorithm": algorithm,
        }

    def _dedup_chunks(self) -> int:
        all_chunks = self.db.get_all_chunks_with_embeddings()
        seen_hashes: set[str] = set()
        dup_ids: list[str] = []
        for chunk in all_chunks:
            h = chunk.chunk_hash
            if h in seen_hashes:
                dup_ids.append(chunk.id)
            else:
                seen_hashes.add(h)
        if dup_ids:
            self.db.delete_chunks_by_ids(dup_ids)
        return len(dup_ids)


class _Candidate:
    __slots__ = ("memory", "bm25_score", "vector_score", "matched_chunk")

    def __init__(self, memory: Memory):
        self.memory = memory
        self.bm25_score: float = 0.0
        self.vector_score: float = 0.0
        self.matched_chunk: str = ""
