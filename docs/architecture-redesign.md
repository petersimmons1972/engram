# Engram Architecture Redesign Proposal

**Author:** Montgomery (Multi-team Coordinator, Intel Synthesis)
**Date:** 2026-03-25
**Status:** DRAFT — Pending review by Rickover, Groves, Slim
**Scope:** Addresses 29 issues from adversarial review

---

## Executive Summary

Engram's three-layer search (BM25 + vector + knowledge graph) works but is built on fragile foundations. The scoring algorithm uses a weighted linear combination that nobody ships in production. Vector search loads all embeddings into Python memory. The embedding model is locked at deploy time with no migration path. The graph contributes a naive degree-centrality score. Data integrity has YAML parsing bugs and silent ID regeneration.

This proposal replaces the scoring core with Reciprocal Rank Fusion (RRF), moves vector search to pgvector (Postgres) or keeps SQLite with a hard cap and deprecation warning, adds an embedding migration mechanism, demotes the graph from scoring to enrichment, and hardens data integrity across the stack.

Five phases. Each phase is independently shippable. Total estimated effort: ~2 weeks of focused work.

---

## ADR-1: Score Fusion Algorithm

### Context

Current scoring in `search.py:194-203`:

```python
composite = (
    w_vector * cand.vector_score
    + w_bm25 * cand.bm25_score
    + w_recency * recency_score
    + w_graph * graph_score
)
importance_mult = 2.0 - (mem.importance * 0.35)
final_score = composite * importance_mult
```

This has multiple problems:
- **Weight tuning is unsolvable.** BM25 scores are negative-then-negated floats from SQLite's `bm25()`. Vector scores are cosine similarities in [0,1]. These distributions don't share a scale. Min-max normalization (lines 132-138) is batch-dependent — the same document gets different normalized scores depending on what else matched.
- **Importance multiplier inverts rankings.** A critical memory (importance=0, mult=2.0) can promote an irrelevant result above a relevant one. Multiplicative importance on a composite score is mathematically wrong.
- **NullEmbedder weight redistribution** (lines 118-122) is a hack. Redistributing vector weight proportionally to other signals changes the relative importance of BM25 vs. recency vs. graph depending on whether embeddings are available.
- Four tuneable weights that nobody will ever tune correctly.

### Decision

**Replace weighted linear combination with Reciprocal Rank Fusion (RRF).**

Formula: `score = SUM(1 / (k + rank_i))` where `k = 60` (standard constant from Cormack et al.)

Each retrieval layer (BM25, vector) independently ranks candidates. RRF merges by rank position, not raw score. This eliminates:
- Score normalization bugs (ranks are ordinal, not cardinal)
- Weight tuning (RRF is parameter-free except for k, which is robust across domains)
- NullEmbedder redistribution hack (if only one signal exists, RRF degrades gracefully to that signal's ranking)
- BM25 score distribution mismatches between SQLite and Postgres backends

**Post-fusion adjustments** (additive, applied after RRF):
- **Recency:** Half-life decay model. `recency_bonus = 0.1 * exp(-ln(2) * hours_since_access / (7 * 24))`. 24-hour offset (recent memories get full bonus), 7-day half-life. Additive, max contribution 0.1.
- **Importance:** Additive bonus. `importance_bonus = (4 - importance) * 0.04`. Max bonus 0.16 for critical (importance=0), zero for trivial (importance=4). Never inverts rankings because it's additive and capped.

**Graph is removed from scoring entirely.** See ADR-4.

### Consequences

- Eliminates 4 of 29 adversarial review issues (#46 weight math, #57 BM25 normalization, #59 importance inversion, NullEmbedder redistribution).
- Score breakdowns in `SearchResult.score_breakdown` change shape. Callers that inspect breakdown keys will need updating. The MCP server response format changes (new keys: `rrf_bm25`, `rrf_vector`, `recency_bonus`, `importance_bonus`).
- `WEIGHT_VECTOR`, `WEIGHT_BM25`, `WEIGHT_RECENCY`, `WEIGHT_GRAPH` constants are deleted entirely.

### Implementation

Replace `SearchEngine.recall()` in `src/engram/search.py`:

```python
RRF_K = 60

def recall(self, query, top_k=10, ...):
    # Layer 1: BM25 ranking
    bm25_results = self.db.fts_search(query, limit=top_k * 3)
    bm25_ranks = {mem.id: rank for rank, (mem, _) in enumerate(bm25_results, 1)}

    # Layer 2: Vector ranking (skip if NullEmbedder)
    vec_ranks = {}
    if self.has_vectors:
        vec_results = self._vector_search(query, limit=top_k * 3)
        vec_ranks = {mid: rank for rank, (mid, _) in enumerate(vec_results, 1)}

    # RRF fusion
    all_ids = set(bm25_ranks) | set(vec_ranks)
    candidates = {}
    for mid in all_ids:
        rrf = 0.0
        if mid in bm25_ranks:
            rrf += 1.0 / (RRF_K + bm25_ranks[mid])
        if mid in vec_ranks:
            rrf += 1.0 / (RRF_K + vec_ranks[mid])
        candidates[mid] = rrf

    # Post-fusion adjustments (additive)
    for mid, rrf in candidates.items():
        mem = self.db.get_memory(mid)
        hours = (now - mem.last_accessed).total_seconds() / 3600
        recency_bonus = 0.1 * math.exp(-math.log(2) * max(hours - 24, 0) / 168)
        importance_bonus = (4 - mem.importance) * 0.04
        candidates[mid] = rrf + recency_bonus + importance_bonus
```

---

## ADR-2: Vector Search Backend (pgvector)

### Context

Current vector search in `search.py:142-169`:

```python
all_chunks = self.db.get_all_chunks_with_embeddings()
for chunk in all_chunks:
    sim = cosine_similarity(query_vec, chunk_vec)
```

This is O(n) brute-force scan over every embedding in the project. All embeddings are loaded into Python memory. At 768 dimensions * 4 bytes * 10,000 chunks = 30MB. At 100,000 chunks = 300MB. This does not scale.

The `limit=10_000` cap in `get_all_chunks_with_embeddings()` is a band-aid — it means vector search silently stops working correctly once a project exceeds 10K chunks.

### Decision

**For PostgreSQL backend: use pgvector extension.**

- Store embeddings as `vector(768)` column type instead of `BYTEA`
- Create HNSW index: `CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`
- Vector search becomes: `SELECT ... ORDER BY embedding <=> %s LIMIT %s`
- O(log n) approximate nearest neighbor instead of O(n) brute force
- Eliminates loading all embeddings into Python memory

**For SQLite backend: keep current approach with hard limit and deprecation warning.**

- SQLite is the local/single-user mode. At Engram's typical scale (hundreds to low thousands of memories), brute-force scan is acceptable.
- Add a warning log when chunk count exceeds 5,000: "Vector search is scanning {n} chunks. Consider switching to PostgreSQL with pgvector for better performance."
- sqlite-vec extension exists but adds a native dependency that complicates installation. Not worth it for the SQLite use case.

### Schema Changes

**PostgreSQL (`db_postgres.py`):**

```sql
-- Migration: change embedding column from BYTEA to vector
ALTER TABLE chunks ADD COLUMN embedding_vec vector(768);
-- Backfill from BYTEA (one-time migration)
-- Then drop old column, rename new
CREATE INDEX idx_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding_vec vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**New method on `DatabaseBackend` protocol:**

```python
def vector_search(self, query_embedding: bytes, limit: int = 20) -> list[tuple[Chunk, float]]:
    """Return chunks ranked by vector similarity. Backend-optimized."""
    ...
```

SQLite implementation: brute-force scan (current code, extracted to method).
Postgres implementation: `ORDER BY embedding_vec <=> %s LIMIT %s`.

### Consequences

- `get_all_chunks_with_embeddings()` is no longer called during search (only during consolidation/dedup).
- `embeddings.py` serialization helpers (`to_blob`/`from_blob`) are still needed for SQLite but pgvector uses native vector type.
- Requires `pgvector` extension installed on Postgres server. Add to installation docs.
- Add `pgvector>=0.3.0` to `[project.optional-dependencies] postgres`.

---

## ADR-3: Embedding Migration Mechanism

### Context

Current behavior when switching embedding models (`search.py:39-63`):

```python
if stored_name != self.embedder.name or int(stored_dims) != self.embedder.dimensions:
    raise EmbeddingConfigMismatchError(...)
```

The error message says: "delete the project database and re-store memories, or start a fresh project database." This is unacceptable. Users lose all memories, relationships, access counts, and temporal data.

### Decision

**Add a `memory_reembed` MCP tool that re-embeds all chunks with the current model.**

This is the minimum viable approach. At Engram's scale (hundreds to low thousands of memories), full re-index is practical — a few hundred embeddings takes seconds with Ollama, minutes with OpenAI.

Shadow index / blue-green swap is over-engineered for this scale. The entire re-embed operation fits in a single transaction.

**Implementation:**

1. Store `embedder_name` per chunk (new column), not just in project_meta.
2. `memory_reembed` tool:
   - Reads all chunks for the project
   - Re-embeds each with the current embedder
   - Updates chunk embeddings and embedder_name in a transaction
   - Updates project_meta to reflect new embedder
   - Returns count of chunks re-embedded
3. Remove `EmbeddingConfigMismatchError`. Replace with: if embedder_name in project_meta differs from current, log a warning and suggest running `memory_reembed`. Do not block operations.

**Golden query validation:** After re-embedding, run the 5 most recently accessed memories as queries and verify they appear in their own top-3 results. If any fail, warn but don't rollback (the new model may legitimately rank differently).

### Schema Changes

```sql
-- Both SQLite and Postgres
ALTER TABLE chunks ADD COLUMN embedder_name TEXT NOT NULL DEFAULT '';
```

### New MCP Tool

```python
@mcp.tool()
def memory_reembed(project: str = "") -> dict:
    """Re-embed all chunks with the current embedding model.

    Use this after switching embedding providers (e.g., from Ollama to OpenAI).
    Preserves all memories, relationships, and metadata.
    """
```

### Consequences

- `EmbeddingConfigMismatchError` is removed from `errors.py`.
- `_check_embedder_metadata()` becomes a soft warning, not a hard error.
- Cost consideration: re-embedding with OpenAI costs ~$0.001 per 1000 tokens. A project with 5000 chunks at 500 tokens each = $2.50. Log estimated cost before proceeding.

---

## ADR-4: Graph's Role — Enrichment, Not Scoring

### Context

The graph currently contributes to scoring in two places:

1. **Scoring** (`search.py:191-192`): `graph_score = min(1.0, conn_count / 5.0)` — pure degree centrality. A memory with 5+ connections gets maximum graph score regardless of whether those connections are relevant to the query.

2. **Enrichment** (`search.py:224-235`): Connected memories are attached to search results via BFS traversal.

The scoring contribution is nearly useless. Degree centrality says "this memory is well-connected" but not "this memory is connected to things relevant to YOUR QUERY." A memory about database configuration with 10 connections to deployment decisions would get a high graph score when you search for "auth flow" — if it happened to also match on BM25 or vector.

### Decision

**Graph is enrichment only. It does not affect ranking.**

Position B from the research. Here's why:

The graph answers the question "given that I found this memory, what else should I see?" — not "which memories should rank higher?" These are fundamentally different questions. BM25 and vector search are designed to answer the ranking question. The graph is designed to answer the context question.

Concretely:
- Remove `w_graph * graph_score` from the composite score calculation.
- Remove `WEIGHT_GRAPH` constant.
- Remove `get_connection_count()` call during scoring (saves one DB query per candidate).
- Keep the post-scoring graph expansion (lines 224-235) exactly as-is. This is the valuable part.
- In the `SearchResult.score_breakdown`, replace `"graph"` with `"connected_count"` as informational metadata (not a scoring signal).

**Strengthen the enrichment side:**
- Add edge strength to the BFS traversal output (already present in `ConnectedMemory.strength`).
- Sort connected memories by strength descending so the most relevant connections appear first.
- Add `rel_type` filtering: when recalling, optionally filter connected memories by relationship type (e.g., only show `caused_by` and `resolved_by` connections for error-type queries).

### Why Not Position A (Graph as Scoring)?

The proposed exponential soft saturation formula (`1.0 - exp(-0.5 * total_strength)`) is more sophisticated than degree centrality, but it still doesn't solve the fundamental problem: graph connectivity is query-independent. A memory's graph score is the same regardless of what you searched for. This is a constant bias, not a relevance signal.

You could make graph scoring query-dependent (e.g., do the connected memories also match the query?) but at that point you've reinvented vector search with extra steps. The graph's value is in surfacing context you didn't search for — that's enrichment, not ranking.

### Consequences

- Scoring becomes simpler: RRF(BM25, vector) + recency + importance. Four signals, not five.
- Graph-heavy memories are not penalized or promoted. They get the same ranking treatment as isolated memories but deliver richer context in results.
- The `feedback()` mechanism still works. Edge strengthening/weakening affects which connected memories appear, even though it doesn't affect the parent memory's rank.

---

## ADR-5: Data Integrity Model

### Context

Multiple data integrity issues identified:

1. **YAML frontmatter parsing** (`markdown_io.py:68`): `parts = markdown_content.split("---", 2)` — if content contains `---` (e.g., horizontal rules in markdown), parsing breaks. This corrupts import/export.
2. **YAML serialization**: Uses `yaml.dump()` (line 52), not `yaml.safe_dump()`. Can serialize arbitrary Python objects.
3. **ID handling**: `Memory.id` defaults to `uuid.uuid4().hex` (line 40). During ingest, if frontmatter has `id: ""`, a new ID is silently generated. The ingested memory loses its identity — relationships pointing to the old ID are broken.
4. **Chunk dedup in consolidation** (`search.py:297-309`): Cross-memory dedup. If two different memories happen to have the same chunk text (common for boilerplate), one loses its chunk. This is wrong.
5. **`yaml` dependency**: Not in `pyproject.toml` dependencies. Works because PyYAML is a transitive dependency, but this is fragile.

### Decision

**Fix all five issues:**

1. **YAML frontmatter delimiter**: Use regex match for frontmatter block: `re.match(r'^---\n(.*?)\n---\n', content, re.DOTALL)`. This matches only the first `---..---` block, not content `---`.

2. **Safe serialization**: Replace `yaml.dump()` with `yaml.safe_dump()` everywhere in `markdown_io.py`.

3. **ID validation on import**: During ingest, if frontmatter provides an ID, validate it (non-empty, reasonable format). If the ID already exists in the database, skip or update — never silently generate a new ID. Add a `conflict` parameter to `memory_ingest`: `"skip"` (default), `"update"`, or `"error"`.

4. **Scoped chunk dedup**: Change `_dedup_chunks()` to deduplicate within the same `memory_id` only, not across memories. Two memories can legitimately share chunk text.

5. **Explicit PyYAML dependency**: Add `pyyaml>=6.0` to `[project.dependencies]` in `pyproject.toml`.

### Additional: Rename `memify` to `consolidate`

The method `SearchEngine.memify()` and the MCP tool `memory_consolidate` already disagree on naming. The internal method should match the public interface. Engram's implementation is dedup+decay+prune — not Cognee's 6-stage LLM pipeline. Calling it `memify` is misleading. Rename the internal method to `consolidate()`.

### Consequences

- Breaking change for anyone importing `SearchEngine.memify()` directly. The MCP tool name (`memory_consolidate`) does not change.
- Markdown files exported by older versions may fail to import if they contain `---` in content. Add a migration note to docs.
- `pyproject.toml` gains an explicit PyYAML dependency.

---

## Implementation Phases

### Phase 1: Score Fusion (RRF)

**Changes:**
- Rewrite `SearchEngine.recall()` to use RRF
- Delete `WEIGHT_VECTOR`, `WEIGHT_BM25`, `WEIGHT_RECENCY`, `WEIGHT_GRAPH` constants
- Remove graph scoring from `recall()`, keep graph enrichment
- Update `score_breakdown` keys in response
- Update MCP tool docstring for `memory_recall` (remove weight formula)
- Add `_vector_search()` private method to `SearchEngine` (extracted from recall)

**Files modified:**
- `src/engram/search.py` — major rewrite of `recall()`
- `src/engram/server.py` — update docstring for `memory_recall`

**Issues resolved:** #46 (weight math), #57 (BM25 normalization), #59 (importance inversion), NullEmbedder weight redistribution, graph scoring bias

**Dependencies:** None. Can ship independently.

**Complexity:** Medium. Core algorithm change but well-contained in one method.

**Tests required:**
- RRF produces deterministic rankings for known inputs
- Single-signal degradation (NullEmbedder produces valid results)
- Recency bonus decays correctly over time
- Importance bonus is additive and bounded

### Phase 2: Data Integrity

**Changes:**
- Fix YAML frontmatter parsing with regex
- Switch to `yaml.safe_dump()`
- Add ID validation on ingest
- Scope chunk dedup to within same memory_id
- Add `pyyaml>=6.0` to dependencies
- Rename `memify()` to `consolidate()`

**Files modified:**
- `src/engram/markdown_io.py` — frontmatter parsing fix, safe_dump
- `src/engram/search.py` — rename memify to consolidate, fix _dedup_chunks scope
- `src/engram/server.py` — update internal call from memify() to consolidate()
- `pyproject.toml` — add pyyaml dependency
- `tests/test_consolidate.py` — update method name references

**Issues resolved:** YAML parsing bugs, silent ID regeneration, cross-memory chunk dedup, unsafe serialization

**Dependencies:** None. Can ship independently.

**Complexity:** Low. Targeted fixes, no architectural change.

**Tests required:**
- Frontmatter with `---` in content body parses correctly
- Round-trip: dump then ingest preserves all fields
- Duplicate chunk hashes across different memories are preserved
- Duplicate chunk hashes within same memory are deduped
- Ingest with existing ID: skip/update/error modes

### Phase 3: Vector Search (pgvector)

**Changes:**
- Add `vector_search()` method to `DatabaseBackend` protocol
- Implement pgvector-backed vector search in `PostgresBackend`
- Implement brute-force fallback in `SqliteBackend`
- Add HNSW index to Postgres schema
- Add migration to convert BYTEA embedding column to vector type
- Add `pgvector>=0.3.0` to postgres optional dependency
- Add deprecation warning for large SQLite vector searches

**Files modified:**
- `src/engram/db.py` — add `vector_search()` to protocol
- `src/engram/db_postgres.py` — pgvector schema, vector_search implementation, migration
- `src/engram/db_sqlite.py` — brute-force vector_search implementation
- `src/engram/search.py` — use `db.vector_search()` instead of loading all chunks
- `src/engram/embeddings.py` — add pgvector serialization helper
- `pyproject.toml` — add pgvector dependency

**Issues resolved:** O(n) vector search, memory explosion on large databases, 10K chunk cap

**Dependencies:** Phase 1 (RRF) should land first so vector_search returns ranked results, not scores.

**Complexity:** High. Schema migration, new index type, dual-backend implementation.

**Tests required:**
- pgvector search returns same top-k as brute-force for small datasets
- HNSW index creation succeeds
- BYTEA to vector migration preserves embeddings
- SQLite warning fires at threshold
- vector_search with empty/null embeddings doesn't crash

### Phase 4: Embedding Migration

**Changes:**
- Add `embedder_name` column to chunks table
- Add `memory_reembed` MCP tool
- Change `_check_embedder_metadata()` from hard error to soft warning
- Remove `EmbeddingConfigMismatchError`
- Add golden query validation after re-embed
- Add cost estimation logging for OpenAI re-embed

**Files modified:**
- `src/engram/types.py` — add `embedder_name` to `Chunk` model
- `src/engram/db_sqlite.py` — schema migration for new column
- `src/engram/db_postgres.py` — schema migration for new column
- `src/engram/search.py` — soft warning instead of hard error, store embedder_name per chunk
- `src/engram/server.py` — new `memory_reembed` tool
- `src/engram/errors.py` — remove `EmbeddingConfigMismatchError`

**Issues resolved:** Locked embedding model, destructive migration path, EmbeddingConfigMismatchError UX

**Dependencies:** Phase 3 (pgvector) should land first so re-embed writes to the correct column type.

**Complexity:** Medium. New tool, schema migration, but straightforward logic.

**Tests required:**
- Re-embed changes all chunk embeddings
- Re-embed updates embedder_name per chunk
- Golden query validation passes for same-model re-embed
- Cost estimate is logged
- Soft warning fires on model mismatch (no crash)

### Phase 5: Operational Improvements

**Changes:**
- Add compound index `(project, memory_type)` to memories table (both backends)
- Pre-filter by memory_type/tags in SQL WHERE clause during recall, not in Python post-filter
- Make `SearchEngine.recall()` async for Postgres path (sync wrapper for SQLite)
- Fix LRU cache eviction race: collect engine reference, release lock, then close
- Add `memory_type` and `tags` parameters to `vector_search()` for SQL pre-filtering

**Files modified:**
- `src/engram/db_sqlite.py` — new index, pre-filtered search methods
- `src/engram/db_postgres.py` — new index, pre-filtered search methods, async recall
- `src/engram/search.py` — pass filters to DB layer instead of post-filtering
- `src/engram/server.py` — fix LRU eviction race condition

**Issues resolved:** Post-filter inefficiency, missing compound index, LRU cache race

**Dependencies:** Phases 1-4 should land first. This phase optimizes the new architecture.

**Complexity:** Low-Medium. Incremental optimizations.

---

## Migration Strategy

### Upgrading from Current Architecture

**SQLite users (default):**
1. Phases 1-2 require no data migration. The scoring algorithm and data integrity fixes are code-only changes.
2. Phase 4 adds a column to chunks table. Auto-migration via `_migrate()` method, same pattern as existing schema_version system.
3. No action required from users. Engram detects schema version and migrates on startup.

**PostgreSQL users:**
1. Phase 3 requires pgvector extension. Users must run `CREATE EXTENSION vector;` on their Postgres server.
2. Phase 3 migration converts BYTEA to vector column. This is a one-time operation. Engram's `_migrate()` runs it automatically.
3. If pgvector is not installed, Engram falls back to BYTEA storage with brute-force search (current behavior) and logs a warning.

### Data Preservation Guarantees

- **No memory loss.** All phases preserve existing memories, chunks, and relationships.
- **No score continuity.** RRF scores are on a different scale than weighted linear scores. Any code that thresholds on absolute score values will need updating. Relative rankings should improve.
- **Embedding continuity.** Phase 4 adds per-chunk embedder tracking but does not invalidate existing embeddings. Existing chunks get `embedder_name = ''` (unknown) which is accepted.

### Rollback Plan

Each phase is a separate branch/PR. Rollback = revert the PR.

- **Phase 1 rollback:** Restore old recall() method. No schema changes to undo.
- **Phase 2 rollback:** Restore old markdown_io.py and search.py. Revert pyproject.toml.
- **Phase 3 rollback:** The vector column migration is forward-only (BYTEA -> vector). To rollback, either keep the vector column unused (Engram falls back to brute-force) or dump/reload the database.
- **Phase 4 rollback:** The embedder_name column is additive. Reverting code ignores it. No data loss.
- **Phase 5 rollback:** Index and filter changes are backward-compatible. Revert code only.

---

## Risk Assessment

### What Could Go Wrong

| Risk                                             | Likelihood | Impact | Mitigation                                                                               |
|:-------------------------------------------------|:----------:|:------:|:-----------------------------------------------------------------------------------------|
| RRF ranking worse than weighted linear for some queries | Medium   | Medium | A/B test with golden queries before merging. RRF is well-studied but Engram's data is unique. |
| pgvector not available on user's Postgres         | Medium     | Low    | Graceful fallback to BYTEA + brute-force. Warning log.                                    |
| Re-embed cost surprise with OpenAI               | Low        | Medium | Log estimated cost and require confirmation for >1000 chunks.                             |
| YAML parsing regex misses edge cases             | Low        | Low    | Comprehensive test suite for frontmatter parsing. Existing tests catch regressions.       |
| LRU eviction fix introduces new race             | Low        | Medium | The fix is to copy the reference before releasing the lock. Well-understood pattern.      |

### What We're NOT Changing and Why

1. **Chunking strategy** (`chunker.py`): Sentence-boundary splitting with overlap works fine. No evidence of problems. The adversarial review didn't flag it.

2. **Embedding providers** (`embeddings.py`): OpenAI + Ollama + Null is sufficient. Adding more providers (Cohere, Voyage, etc.) is feature creep. The embedding migration mechanism in Phase 4 makes it easy to switch later.

3. **MCP tool interface**: Tool names, parameter names, and basic response shapes are preserved. Breaking MCP clients would be worse than any internal improvement.

4. **Per-project database isolation**: Each project gets its own SQLite file or Postgres schema. This is correct. Cross-project search is a feature request, not a bug fix.

5. **Memory lifecycle (ADD-only)**: Mem0's ADD/UPDATE/DELETE/NOOP classification at store time (26% improvement per their paper) is compelling but requires an LLM call on every store operation. This adds latency, cost, and a hard dependency on an LLM being available. Engram's `memory_correct` tool already handles the UPDATE case. The NOOP case (duplicate detection) can be added later via the existing `is_duplicate()` in `chunker.py`. Filing this as a future enhancement, not blocking the redesign.

6. **Temporal validity windows** (Zep's valid_from/valid_until): Elegant but adds schema complexity for a feature that `supersedes` relationships already approximate. The current approach (demote old memory, link to new) is good enough. If temporal queries become a real need, it's an additive schema change.

7. **Cognee-style LLM entity extraction in consolidation**: Would make consolidation expensive (LLM call per memory) and add a hard LLM dependency to a maintenance operation. The current dedup+decay+prune is the right scope for a consolidation pass.

---

## Files to Create/Modify — Summary

| File                              | Phase | Action | Description                                                    |
|:----------------------------------|:-----:|:------:|:---------------------------------------------------------------|
| `src/engram/search.py`            |  1-5  | Modify | RRF scoring, vector_search delegation, consolidate rename, pre-filtering |
| `src/engram/server.py`            |  1,4  | Modify | Updated docstrings, new memory_reembed tool, LRU fix          |
| `src/engram/db.py`                |  3    | Modify | Add vector_search to protocol                                 |
| `src/engram/db_postgres.py`       |  3-5  | Modify | pgvector schema, vector_search, migrations, compound index    |
| `src/engram/db_sqlite.py`         |  2-5  | Modify | Brute-force vector_search, migrations, compound index         |
| `src/engram/embeddings.py`        |  3    | Modify | pgvector serialization helper                                 |
| `src/engram/markdown_io.py`       |  2    | Modify | Frontmatter regex fix, safe_dump                              |
| `src/engram/errors.py`            |  4    | Modify | Remove EmbeddingConfigMismatchError                           |
| `src/engram/types.py`             |  4    | Modify | Add embedder_name to Chunk                                    |
| `pyproject.toml`                  |  2-3  | Modify | Add pyyaml, pgvector dependencies                             |
| `tests/test_consolidate.py`       |  2    | Modify | Update memify -> consolidate references                       |
| `tests/test_rrf.py`               |  1    | Create | RRF scoring tests                                             |
| `tests/test_vector_search.py`     |  3    | Create | pgvector and brute-force vector search tests                  |
| `tests/test_reembed.py`           |  4    | Create | Embedding migration tests                                     |
| `tests/test_markdown_integrity.py`|  2    | Create | Frontmatter parsing, round-trip, ID validation tests          |
| `docs/installation.md`            |  3    | Modify | pgvector installation instructions                            |

---

## Appendix: Issue Resolution Map

| Phase | Issues Resolved                                                    |
|:-----:|:-------------------------------------------------------------------|
|   1   | Weight math, BM25 normalization, importance inversion, NullEmbedder redistribution, graph scoring bias |
|   2   | YAML parsing, unsafe serialization, silent ID regen, cross-memory dedup, memify naming |
|   3   | O(n) vector scan, memory explosion, 10K chunk cap                  |
|   4   | Locked embedder, destructive migration, EmbeddingConfigMismatchError UX |
|   5   | Post-filter inefficiency, missing compound index, LRU cache race   |

---

*This document is opinionated by design. Reviewers (Rickover, Groves, Slim) are expected to challenge these positions. The point is to have positions worth challenging.*
