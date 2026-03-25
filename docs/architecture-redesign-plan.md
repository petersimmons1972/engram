# Engram Architecture Redesign Plan
**Date:** 2026-03-25
**Operation:** ENGRAM REDESIGN
**Team:** Montgomery (coord), Rickover (quality), Groves (feasibility), Slim (creative), Rochefort (academic research), Layton (industry analysis), Scoring Specialist
**Status:** Plan approved by review team. Implementation pending.

---

## Background

Adversarial code review found 30 issues (GitHub Issues #45-#75):
- 8 CRITICAL: Race conditions, algorithm bugs, data loss, YAML corruption
- 10 HIGH: Security, performance N+1, missing validation, error handling
- 8 MEDIUM: Index optimization, migration, rate limiting, edge cases
- 4 ARCHITECTURAL: Scoring complexity, query optimization, migration path, schema versioning

Three research streams (academic papers, industry analysis of 6 production systems, scoring algorithm research) investigated solutions. A 4-general review panel debated and reached consensus.

## Research Sources

- **Academic**: Cormack et al. 2009 (RRF), Mem0 paper (arXiv 2504.19413), Zep paper (arXiv 2501.13956), HippoRAG (NeurIPS 2024), Cognee docs
- **Industry**: Mem0, Cognee, Zep/Graphiti, Weaviate, Qdrant, LlamaIndex
- **Scoring**: Elasticsearch function_score, OpenSearch normalization, Vespa phased ranking, Qdrant decay functions

## Key Industry Finding

Engram's weighted linear combination (`vector*0.45 + BM25*0.25 + recency*0.15 + graph*0.15`) is used by **zero production systems**. Every production memory system uses either RRF (Weaviate, Zep, Elasticsearch) or vector-authoritative with graph enrichment (Mem0).

---

## Phase 1: "The Filing Cabinet Fix"

**Scope:** ~200-300 lines of net change, fixes ~20 of 30 issues
**Approach:** Test-first for every change. Write failing test, then implement fix.
**Prerequisite:** All changes on a feature branch with full test suite passing before merge.

### 1.1 FTS-First Retrieval with Vector Re-rank

**Fixes:** #51 (N+1 queries), #72 (query optimization), partially #46, #71

**Current:** `get_all_chunks_with_embeddings()` loads ALL chunks into Python, brute-force cosine similarity on every chunk. O(n) per query.

**New:** FTS finds candidates (fast, indexed), vector re-ranks only those candidates. Fallback to full vector scan only when FTS returns fewer than `top_k` results.

```python
def recall(self, query, top_k=10, ...):
    # Step 1: FTS finds lexical matches (fast, indexed)
    fts_hits = self.db.fts_search(query, limit=top_k * 3)

    # Step 2: If enough candidates, vector re-ranks just those
    if len(fts_hits) >= top_k and self.has_vectors:
        candidate_ids = [m.id for m, _ in fts_hits]
        chunks = self.db.get_chunks_for_memories(candidate_ids)
        # cosine similarity on ~30 chunks, not ~10,000

    # Step 3: If FTS came up short, vector fills the gap
    elif self.has_vectors:
        fts_ids = {m.id for m, _ in fts_hits}
        all_chunks = self.db.get_all_chunks_with_embeddings()
        # Only score chunks NOT already found by FTS
```

**Tests required:**
- `test_lexical_match_found_without_vector_scan`
- `test_semantic_only_match_triggers_vector_fallback`
- `test_mixed_results_merge_fts_and_vector`

**Team consensus:** Approved by Rickover and Slim. pgvector deferred to Phase 2.

### 1.2 Simplified Scoring: BM25 + Vector Re-rank + Recency Tiebreaker

**Fixes:** #46 (weight math), #57 (BM25 normalization), #59 (importance inversion), #71 (scoring complexity)

**Current:** 4-signal weighted combination with multiplicative importance (3.33x variance). Magic weights, fragile normalization, can invert rankings.

**New:** Three signals. BM25 as primary retrieval score, vector similarity as re-rank boost, recency as tiebreaker. Importance becomes a WHERE filter, not a multiplier. Graph removed from scoring (becomes enrichment only).

```python
# Scoring after FTS-first retrieval:
# 1. BM25 rank from FTS (already sorted by relevance)
# 2. Vector similarity boost (additive, bounded)
vector_boost = 0.3 * cosine_similarity(query_vec, chunk_vec) if has_vectors else 0
# 3. Recency tiebreaker (half-life model, 7-day half-life, 24h offset)
RECENCY_OFFSET = 24  # hours — full score within 24h
RECENCY_HALF_LIFE = 168  # hours — 7 days
hours = (now - mem.last_accessed).total_seconds() / 3600
effective_hours = max(0, hours - RECENCY_OFFSET)
recency_bonus = 0.15 * (0.5 ** (effective_hours / RECENCY_HALF_LIFE))
# 4. Importance as filter
# WHERE importance <= min_importance (not a multiplier)

final_score = bm25_normalized + vector_boost + recency_bonus
```

**Tests required:**
- `test_single_result_has_valid_score` (no division-by-zero)
- `test_importance_filters_not_multiplies` (no 3.33x variance)
- `test_bm25_only_mode_produces_valid_ranking`
- `test_recency_24h_offset_gives_full_score`

**Team consensus:** Rickover approved three signals. Slim proposed two (no recency). Rickover won — recency matters for memory systems.

### 1.3 Graph as Enrichment, Not Scoring Signal

**Fixes:** #58 (magic number 5 graph saturation), simplifies scoring

**Current:** `graph_score = min(1.0, conn_count / 5.0)` contributes to composite score with weight 0.15.

**New:** Graph score removed from ranking formula. Connected memories are still attached to results via `get_connected()` after ranking. The `supersedes` detection (server.py:258-264) continues to work because it operates on attached connections, not the score.

**Tests required:**
- `test_recall_returns_connected_memories`
- `test_superseded_warning_still_fires`

**Team consensus:** Unanimous. Both Rickover and Slim agree.

### 1.4 Add Project Column to Relationships Table

**Fixes:** #75 (cross-project edge pollution)

**Current:** Relationships table has no project column. `decay_all_edges()` decays ALL edges globally across all projects.

**New:** Add `project TEXT NOT NULL DEFAULT 'default'` to relationships table. Filter all edge operations by project. Schema migration backfills from source memory's project.

```sql
-- Migration
ALTER TABLE relationships ADD COLUMN project TEXT NOT NULL DEFAULT 'default';
UPDATE relationships r SET project = m.project
    FROM memories m WHERE r.source_id = m.id;
CREATE INDEX idx_rel_project ON relationships(project);
```

**Tests required:**
- `test_decay_edges_only_affects_current_project`
- `test_cross_project_connect_rejected`

**Team consensus:** Unanimous. Rickover proposed, Slim agreed (after JSONB alternative rejected).

### 1.5 Fix Markdown Serializer Bugs

**Fixes:** #49 (YAML dump/load mismatch), #50 (frontmatter delimiter), #52 (ID loss)

Three targeted fixes in `markdown_io.py`:
1. **Line 52:** Change `yaml.dump()` to `yaml.safe_dump()`
2. **Line 68:** Fix `---` splitting — use regex to match only frontmatter delimiters (line-start `---` followed by newline), not content containing `---`
3. **Line 95:** Change `frontmatter.get("id", "")` to explicit None check; raise error if ID missing instead of silently generating new UUID

**Tests required:**
- `test_yaml_round_trip_preserves_all_fields`
- `test_content_with_triple_dashes_preserved`
- `test_missing_id_raises_error`

### 1.6 Transactional Store

**Fixes:** #60 (rollback failure), #62 (chunk embedding loss)

**Current:** Memory stored first, then chunks, then embeddings. If embedding fails, attempts to delete memory but deletion might also fail.

**New:** Wrap memory + chunks + embeddings in a single database transaction. If any step fails, the entire operation rolls back atomically.

**Tests required:**
- `test_embedding_failure_rolls_back_memory_and_chunks`
- `test_partial_chunk_failure_rolls_back_all`

### 1.7 Lazy Chunking

**Fixes:** Reduces overhead for 90%+ of memories

**Current:** All memories are chunked regardless of size. A 50-word session handoff gets the same chunking treatment as a 10,000-word architecture doc.

**New:** Skip chunking for memories under 2000 chars (~500 tokens). Embed the content as a single chunk. Keep existing chunker for long content.

```python
CHUNK_THRESHOLD = 2000  # chars

if len(memory.content) <= CHUNK_THRESHOLD:
    chunks = [Chunk(memory_id=memory.id, chunk_text=memory.content,
                   chunk_index=0, chunk_hash=chunk_hash(memory.content))]
else:
    chunks = chunk_text(memory.content)  # existing chunker
```

**Tests required:**
- `test_short_memory_single_chunk`
- `test_long_memory_chunked_with_overlap`

### 1.8 Rename `memify` to `consolidate`

**Fixes:** Honest documentation

Engram's memify (dedup + decay + prune) is not Cognee's memify (6-stage LLM extraction pipeline). Rename to avoid false claims. The MCP tool `memory_consolidate` already uses the correct name; the internal method `SearchEngine.memify()` should match.

---

## Phase 2: "The Search Engine Upgrade" (when data exceeds ~1,000 memories)

**Trigger:** When any project accumulates >1,000 memories, or recall latency exceeds 500ms.

### 2.1 Reciprocal Rank Fusion (RRF)

Replace simplified scoring with RRF for BM25 + vector fusion:
```python
RRF_score(d) = 1/(k + bm25_rank) + 1/(k + vector_rank), k=60
```
Recency and importance remain as additive post-fusion adjustments.

**Source:** Cormack et al. 2009, used by Weaviate, Elasticsearch, OpenSearch, Azure AI Search.

### 2.2 pgvector for In-Database Vector Search

Add pgvector extension to PostgreSQL. Store embeddings as `vector(768)` type with HNSW index. Replace brute-force scan with `ORDER BY embedding <=> query_vector LIMIT k`.

Transforms recall from O(n) to O(log n).

### 2.3 Embedding Migration Tooling

Add `memory_reembed` CLI command and MCP tool:
1. Store `embedder_name` per chunk (not just globally)
2. Iterate all chunks, re-embed in batches
3. Update project metadata
4. Validate with golden queries

**Source:** Qdrant full re-index strategy. Practical at Engram's scale.

### 2.4 Pluggable Reranker Stage

Separate retrieval from ranking (Zep pattern). Retrieve from BM25 + vector, then apply configurable reranker: RRF, MMR, or cross-encoder.

### 2.5 Async SearchEngine

Make `SearchEngine.recall()` async for the PostgreSQL path. Use `psycopg3 AsyncConnectionPool`.

---

## Phase 3: "The Intelligence Upgrade" (future, research-informed)

### 3.1 Intelligent Store (Mem0-inspired)

At store time, compare new memory against existing memories. Classify as ADD/UPDATE/DELETE/NOOP. Prevents duplicate memories and keeps knowledge current.

**Source:** Mem0 paper (arXiv 2504.19413). 26% improvement over baseline.

### 3.2 Temporal Validity Windows (Zep-inspired)

Add `valid_from` and `valid_until` to graph edges. Superseded facts are time-bounded, not deleted. Historical queries can see what was true at a given time.

**Source:** Zep paper (arXiv 2501.13956). 18.5% accuracy improvement.

### 3.3 Graph Embedding Similarity (Cognee-inspired)

Embed local graph neighborhoods. Graph scoring uses semantic similarity between query and graph structure, not just connection count.

**Source:** Cognee architecture, HippoRAG (NeurIPS 2024).

---

## Issue Resolution Map

### Phase 1 Resolves (~20 issues):

| Issue # | Title | Fixed By |
|---------|-------|----------|
| #46 | Weight redistribution math | 1.2 Simplified scoring |
| #47 | Ollama model hardcoded | Deferred (config, not architecture) |
| #48 | Asymmetric superseded detection | 1.3 Graph enrichment preserves detection |
| #49 | YAML dump/load mismatch | 1.5 Markdown fixes |
| #50 | Frontmatter delimiter bug | 1.5 Markdown fixes |
| #51 | N+1 queries in vector search | 1.1 FTS-first retrieval |
| #52 | Memory ID loss on re-import | 1.5 Markdown fixes |
| #57 | BM25 normalization edge case | 1.2 Simplified scoring |
| #58 | Graph score magic number 5 | 1.3 Graph as enrichment |
| #59 | Importance multiplier 3.33x | 1.2 Importance as filter |
| #60 | Missing transaction rollback | 1.6 Transactional store |
| #62 | Chunk dedup loses embeddings | 1.6 Transactional store |
| #71 | Scoring complexity | 1.2 Simplified scoring |
| #72 | No query optimization | 1.1 FTS-first retrieval |
| #75 | Cross-project edge pollution | 1.4 Project column |

### Phase 2 Resolves:

| Issue # | Title | Fixed By |
|---------|-------|----------|
| #63 | Hardcoded chunking params | 2.3 Configurable at migration time |
| #69 | Unclear embedding mismatch error | 2.3 Migration tooling |
| #70 | Engine cache bottleneck | 2.5 Async SearchEngine |
| #73 | No embedding migration path | 2.3 Re-embed tooling |
| #74 | Weak schema versioning | 2.3 Real migration framework |

### Deferred / Won't Fix:

| Issue # | Title | Reason |
|---------|-------|--------|
| #45 | LRU cache race condition | Acceptable at current scale (64 max, one user) |
| #53 | Race condition duplicate | Same as #45 |
| #54 | SSRF no port check | Low risk for local Ollama |
| #55 | URL validation bypass | Low risk, private attribute |
| #56 | Exception swallowing | Improve logging, not architecture |
| #64 | No rate limiting | One user, not needed |
| #65 | Missing compound index | Add when queries are slow |
| #66 | Cascade delete FTS | Monitor, fix if observed |
| #67 | No corruption detection | Over-engineering for 68 memories |
| #68 | Jaccard threshold | Not user-facing |

---

## Verification Criteria

Phase 1 is complete when:
1. All Phase 1 tests pass (minimum 15 new tests)
2. Existing 180 tests still pass (zero regressions)
3. `ruff check src/ tests/` — zero errors
4. `docker compose build && docker compose up` — all services healthy
5. Store + recall round-trip works for all 6 memory types
6. Superseded memory warning still fires correctly
7. Cross-project isolation verified (project A memories invisible from B)

---

## Implementation Order (Groves' recommended sequence)

**Estimated time: 2-4 days for Phase 1**

| Order | Change | Rationale |
|-------|--------|-----------|
| 1st | 1.5 Markdown fixes | Isolated in markdown_io.py, three targeted line changes |
| 2nd | 1.4 Project column on relationships | Pure schema, no code dependencies |
| 3rd | 1.8 Rename memify → consolidate | Trivial find-and-replace |
| 4th | 1.2 + 1.3 Simplified scoring + graph as enrichment | Core algorithm change, do together |
| 5th | 1.1 FTS-first retrieval | Retrieval rewrite, depends on new scoring |
| 6th | 1.7 Lazy chunking | Store-path change, coordinate with 1.6 |
| 7th | 1.6 Transactional store | Store-path change, do after retrieval rewrite |

Also add to "just do it" bucket: **#47 (hardcoded Ollama model)** — 5-minute env var fix, shouldn't be deferred.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Simplified scoring produces worse results | A/B test: run both old and new scoring on same queries, compare rankings before committing |
| FTS-first misses semantic matches | Vector fallback catches vocabulary mismatch; enrichment at store time improves FTS surface |
| Schema migration breaks existing data | Backup pgdata volume before migration; test on copy first |
| Phase 1 changes interact unexpectedly | Each change is independent and test-first; merge one at a time |
| Phase 2 never happens | Phase 1 is self-sufficient; Phase 2 is optional scaling, not required correctness |

---

## Team Decisions Log

| Decision | For | Against | Verdict |
|----------|-----|---------|---------|
| RRF vs simplified scoring | Research (3 streams) | Rickover, Slim (overkill for 68 memories) | **Simplified now, RRF in Phase 2** |
| pgvector vs FTS-first | Research, Layton | Slim, Groves (premature) | **FTS-first now, pgvector in Phase 2** |
| JSONB graph vs relationships table | Slim (structural safety) | Rickover (breaks supersedes detection) | **Table stays, add project column** |
| Graph as score vs enrichment | Scoring specialist | Rickover, Slim, Mem0 pattern | **Enrichment only** |
| Multiplicative vs additive importance | Current code | All reviewers | **Filter, not multiplier** |
| Lazy chunking | Slim, Rickover | None | **Approved** |
