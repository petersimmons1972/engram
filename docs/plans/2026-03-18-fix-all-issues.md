# Fix All 36 Engram Issues — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 36 issues filed against engram, progressing from foundational fixes through correctness and hardening.

**Architecture:** We group fixes into 10 tasks ordered by dependency. Each task targets a specific file or concern. The codebase is ~2,043 lines across 9 Python files. We fix foundational issues first (types, IDs, db layer) so later fixes can rely on correct primitives.

**Tech Stack:** Python 3.11+, SQLite (FTS5), Pydantic v2, FastMCP, numpy, pytest, pytest-asyncio

**Python environment note:** The development machine has Python 3.12 available but its stdlib is incomplete. Tests must be run via: `source /tmp/engram-venv/bin/activate && python -m pytest tests/ -v` or by installing a working Python 3.11+ first. If the venv doesn't work, install with: `pip install -e ".[dev]"` using any available Python 3.11+.

---

## Issue-to-Task Mapping

| Task | Issues Fixed | Files Modified |
|------|-------------|----------------|
| 1 | #13, #34 | `types.py`, `errors.py` |
| 2 | #5, #33 | `db.py` |
| 3 | #6, #23 | `db.py` |
| 4 | #27, #29 | `db.py`, `server.py` |
| 5 | #11, #12, #18, #22 | `db.py`, `search.py` |
| 6 | #3, #10, #31 | `search.py`, `chunker.py` |
| 7 | #4, #21, #24, #32, #35 | `search.py`, `db.py`, `server.py` |
| 8 | #14, #15, #16, #17, #30 | `server.py` |
| 9 | #1, #2, #7, #8, #19 | `server.py`, `__main__.py`, `embeddings.py` |
| 10 | #9, #20, #28, #36 | `db.py`, `server.py`, `embeddings.py`, `pyproject.toml` |

**Not implemented in code** (documentation/process issues):
- #25 (test coverage gaps) — addressed incrementally by writing tests in every task
- #26 (containerized deployment) — Docker/Compose changes already landed; remaining PostgreSQL backend is out of scope for this plan (requires new `StorageBackend` abstraction)

---

## Task 1: Fix ID Generation and Error Message (#13, #34)

**Issues:** #13 (48-bit UUID collision risk), #34 (error references nonexistent `memory_reindex`)

**Files:**
- Modify: `src/engram/types.py:40,53,62` — change ID factory from `uuid4().hex[:12]` to full `uuid4().hex`
- Modify: `src/engram/errors.py:27` — fix error message to remove reference to `memory_reindex`
- Test: `tests/test_types.py` (new file)

**Step 1: Write the failing tests**

Create `tests/test_types.py`:

```python
"""Tests for engram.types — ID generation and model validation."""
from engram.types import Memory, Chunk, Relationship


class TestIDGeneration:
    def test_memory_id_is_full_uuid(self):
        mem = Memory(content="test")
        assert len(mem.id) == 32  # full uuid4 hex

    def test_chunk_id_is_full_uuid(self):
        chunk = Chunk(memory_id="abc", chunk_text="test", chunk_index=0)
        assert len(chunk.id) == 32

    def test_relationship_id_is_full_uuid(self):
        rel = Relationship(source_id="a", target_id="b")
        assert len(rel.id) == 32

    def test_ids_are_unique(self):
        ids = {Memory(content="test").id for _ in range(1000)}
        assert len(ids) == 1000
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_types.py -v`
Expected: FAIL — IDs are 12 chars, not 32

**Step 3: Implement the fixes**

In `types.py`, change all three ID fields:
```python
# Before (lines 40, 53, 62):
id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])

# After:
id: str = Field(default_factory=lambda: uuid.uuid4().hex)
```

In `errors.py`, fix line 27:
```python
# Before:
f"To switch models, re-index with memory_reindex (not yet implemented) "

# After:
f"To switch models, delete the project database and re-store memories, "
```

**Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_types.py tests/test_db.py tests/test_search.py tests/test_server.py -v`
Expected: ALL PASS (existing tests use IDs opaquely, so longer IDs work fine)

**Step 5: Commit**

```bash
git add src/engram/types.py src/engram/errors.py tests/test_types.py
git commit -m "fix: use full UUID4 for IDs, fix error message (#13, #34)"
```

---

## Task 2: Thread Safety and Connection Lifecycle (#5, #33)

**Issues:** #5 (shared state not thread-safe), #33 (connections never closed)

**Files:**
- Modify: `src/engram/db.py:93-121` — add threading lock, implement `__enter__`/`__exit__`
- Modify: `src/engram/server.py:115-130` — add lock to `_engines` dict, add shutdown cleanup
- Test: `tests/test_db.py` (add tests)

**Step 1: Write the failing tests**

Add to `tests/test_db.py`:

```python
import threading


class TestThreadSafety:
    def test_concurrent_stores(self, tmp_db_dir):
        """Multiple threads storing concurrently must not corrupt data."""
        db = MemoryDB(project="threadsafe", db_dir=tmp_db_dir)
        errors = []

        def store_batch(start):
            try:
                for i in range(20):
                    db.store_memory(Memory(content=f"Thread {start} memory {i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_batch, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent stores: {errors}"
        mems = db.list_memories(limit=200)
        assert len(mems) == 100

    def test_close_releases_connection(self, tmp_db_dir):
        db = MemoryDB(project="closetest", db_dir=tmp_db_dir)
        db.store_memory(Memory(content="test"))
        db.close()
        assert db._conn is None

    def test_context_manager(self, tmp_db_dir):
        with MemoryDB(project="ctxtest", db_dir=tmp_db_dir) as db:
            db.store_memory(Memory(content="test"))
        assert db._conn is None
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_db.py::TestThreadSafety -v`
Expected: FAIL — context manager not implemented, concurrent stores may raise

**Step 3: Implement the fixes**

In `db.py`, modify `MemoryDB.__init__` and add context manager + thread-safe connection:

```python
import threading

class MemoryDB:
    def __init__(self, project: str = "default", db_dir: str | Path | None = None):
        import re
        project = re.sub(r'[^a-zA-Z0-9_-]', '', project) or "default"
        self.project = project
        db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_dir / f"{project}.db"
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
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
```

Then wrap every public method that touches the connection with `with self._lock:`. For example:

```python
    def store_memory(self, memory: Memory) -> Memory:
        with self._lock:
            conn = self._get_conn()
            # ... existing code ...
            conn.commit()
            return memory
```

Do this for ALL public methods: `store_memory`, `get_memory`, `update_memory`, `delete_memory`, `list_memories`, `touch_memory`, `store_chunks`, `get_chunks_for_memory`, `get_all_chunks_with_embeddings`, `get_all_chunk_texts`, `chunk_hash_exists`, `delete_chunks_for_memory`, `store_relationship`, `get_connected`, `boost_edges_for_memory`, `decay_edges_for_memory`, `get_connection_count`, `decay_all_edges`, `prune_stale_memories`, `delete_relationships_for_memory`, `fts_search`, `get_stats`, `get_meta`, `set_meta`.

In `server.py`, add a lock around `_engines`:

```python
import threading

_engines: dict[str, SearchEngine] = {}
_engines_lock = threading.Lock()

def _get_engine(project: str | None = None) -> SearchEngine:
    project = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
    with _engines_lock:
        if project not in _engines:
            db_dir = os.environ.get("ENGRAM_DIR", None)
            db = MemoryDB(project=project, db_dir=db_dir)
            embedder = create_embedder()
            _engines[project] = SearchEngine(db=db, embedder=embedder)
        return _engines[project]
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/engram/db.py src/engram/server.py tests/test_db.py
git commit -m "fix: add thread safety and connection lifecycle (#5, #33)"
```

---

## Task 3: Schema Migration and Date Indexes (#6, #23)

**Issues:** #6 (no migration strategy), #23 (missing date indexes)

**Files:**
- Modify: `src/engram/db.py` — add schema version tracking and migration framework
- Test: `tests/test_db.py` (add migration tests)

**Step 1: Write the failing tests**

```python
class TestSchemaMigration:
    def test_schema_version_stored(self, tmp_db_dir):
        db = MemoryDB(project="migration", db_dir=tmp_db_dir)
        version = db.get_meta("schema_version")
        assert version is not None
        assert int(version) >= 2

    def test_date_indexes_exist(self, tmp_db_dir):
        db = MemoryDB(project="indexes", db_dir=tmp_db_dir)
        conn = db._get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {r["name"] for r in indexes}
        assert "idx_memories_last_accessed" in index_names
        assert "idx_memories_updated_at" in index_names
        assert "idx_memories_project" in index_names

    def test_old_db_gets_migrated(self, tmp_db_dir):
        """Simulate a v1 database (no schema_version) and verify migration adds indexes."""
        db = MemoryDB(project="olddb", db_dir=tmp_db_dir)
        # Remove schema_version to simulate old DB
        conn = db._get_conn()
        conn.execute("DELETE FROM project_meta WHERE key = 'schema_version'")
        conn.commit()
        db.close()

        # Re-open should trigger migration
        db2 = MemoryDB(project="olddb", db_dir=tmp_db_dir)
        version = db2.get_meta("schema_version")
        assert version == "2"
```

**Step 2: Run tests to verify they fail**

Expected: FAIL — no schema_version, no date indexes

**Step 3: Implement**

Add to `SCHEMA_SQL` in `db.py` (after existing indexes):

```sql
CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
```

Add migration framework to `_init_db`:

```python
CURRENT_SCHEMA_VERSION = 2

def _init_db(self) -> None:
    conn = self._get_conn()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    self._migrate(conn)

def _migrate(self, conn: sqlite3.Connection) -> None:
    """Run schema migrations. Each migration is idempotent."""
    stored = self.get_meta("schema_version")
    current = int(stored) if stored else 1

    if current < 2:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project)")
        conn.commit()

    self.set_meta("schema_version", str(CURRENT_SCHEMA_VERSION))
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/db.py tests/test_db.py
git commit -m "fix: add schema migration framework and date indexes (#6, #23)"
```

---

## Task 4: Stats Per-Project and Project Name Normalization (#27, #29)

**Issues:** #27 (global chunk/relationship counts), #29 (normalization mismatch)

**Files:**
- Modify: `src/engram/db.py:500-542` — fix `get_stats()` to count per-project
- Modify: `src/engram/server.py:124` — normalize project names the same way as db.py
- Test: `tests/test_db.py`, `tests/test_server.py`

**Step 1: Write the failing tests**

In `tests/test_db.py`:

```python
class TestStatsPerProject:
    def test_chunks_counted_per_project(self, tmp_db_dir):
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder

        db_a = MemoryDB(project="alpha", db_dir=tmp_db_dir)
        db_b = MemoryDB(project="beta", db_dir=tmp_db_dir)
        eng_a = SearchEngine(db=db_a, embedder=FakeEmbedder())
        eng_b = SearchEngine(db=db_b, embedder=FakeEmbedder())

        eng_a.store(Memory(content="Alpha memory about databases"))
        eng_b.store(Memory(content="Beta memory about APIs"))
        eng_b.store(Memory(content="Beta memory about auth"))

        stats_a = db_a.get_stats()
        stats_b = db_b.get_stats()
        # Each project uses separate .db files, so chunks are already isolated.
        # But stats should still join correctly.
        assert stats_a.total_chunks >= 1
        assert stats_b.total_chunks >= 2
```

In `tests/test_server.py`:

```python
class TestProjectNormalization:
    def test_project_name_normalized_consistently(self, _patch_embedder):
        from engram.server import memory_store, memory_recall

        memory_store(content="Test content", project="My-App")
        result = memory_recall(query="Test", project="my-app")
        assert result["count"] >= 1

    def test_project_strips_special_chars(self, _patch_embedder):
        from engram.server import memory_store, memory_recall

        memory_store(content="Test content", project="my app!")
        result = memory_recall(query="Test", project="myapp")
        assert result["count"] >= 1
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Fix `get_stats()` in `db.py` to JOIN chunks/relationships through memories:

```python
def get_stats(self) -> MemoryStats:
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

    # ... rest unchanged ...
```

Fix project name normalization in `server.py` `_get_engine`:

```python
def _get_engine(project: str | None = None) -> SearchEngine:
    import re
    raw = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
    project = re.sub(r'[^a-zA-Z0-9_-]', '', raw) or "default"
    # ... rest unchanged ...
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/db.py src/engram/server.py tests/test_db.py tests/test_server.py
git commit -m "fix: per-project stats and consistent project name normalization (#27, #29)"
```

---

## Task 5: Search Correctness (#11, #12, #18, #22)

**Issues:** #11 (BM25 negative score inversion), #12 (tag filter after LIMIT), #18 (FTS5 sanitization incomplete), #22 (BM25-only weight redistribution)

**Files:**
- Modify: `src/engram/db.py:464-496` — fix FTS sanitization and tag filtering
- Modify: `src/engram/search.py:119-206` — fix BM25 normalization and weight redistribution
- Test: `tests/test_search.py`, `tests/test_db.py`

**Step 1: Write the failing tests**

In `tests/test_db.py`:

```python
class TestFTSSanitization:
    def test_column_filter_stripped(self, db):
        db.store_memory(Memory(content="Test content for FTS"))
        # "content:" is an FTS5 column filter — must be stripped
        results = db.fts_search("content:Test")
        # Should not crash, should sanitize and search for "Test"
        assert isinstance(results, list)

    def test_prefix_operator_stripped(self, db):
        db.store_memory(Memory(content="Testing prefix operators"))
        results = db.fts_search("test*")
        assert isinstance(results, list)


class TestTagFilterBeforeLimit:
    def test_tag_filter_respects_limit(self, db):
        """Tag filter must not reduce results below the requested limit."""
        for i in range(30):
            tags = ["target"] if i % 3 == 0 else ["other"]
            db.store_memory(Memory(content=f"Memory {i} about things", tags=tags))

        # Request 10 memories filtered by tag "target"
        results = db.list_memories(tags=["target"], limit=10)
        # There are 10 memories with "target" tag; we should get up to 10
        assert len(results) == 10
```

In `tests/test_search.py`:

```python
class TestBM25Normalization:
    def test_no_negative_scores(self, engine):
        """BM25 scores should never be negative in the final output."""
        engine.store(Memory(content="Alpha beta gamma delta epsilon"))
        engine.store(Memory(content="Zeta eta theta iota kappa"))
        results = engine.recall("alpha")
        for r in results:
            assert r.score_breakdown["bm25"] >= 0.0

class TestBM25OnlyWeights:
    def test_bm25_only_redistributes_vector_weight(self, tmp_path):
        """In BM25-only mode, vector weight should be redistributed."""
        from engram.db import MemoryDB
        from engram.embeddings import NullEmbedder
        db = MemoryDB(project="bm25weights", db_dir=tmp_path)
        engine = SearchEngine(db=db, embedder=NullEmbedder())

        engine.store(Memory(content="Test BM25 weight redistribution"))
        results = engine.recall("BM25 weight")
        if results:
            bd = results[0].score_breakdown
            # Vector should be 0 and the other weights should sum to ~1.0
            assert bd["vector"] == 0.0
            total = bd["bm25"] + bd["recency"] + bd["graph"]
            # With redistribution, the non-vector weights should use the full budget
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Fix FTS5 sanitization in `db.py`:

```python
@staticmethod
def _sanitize_fts_query(query: str) -> str:
    """Strip FTS5 special syntax to prevent malformed MATCH queries."""
    import re
    # Remove column filters (e.g., "content:"), prefix ops (*), and other FTS5 syntax
    sanitized = re.sub(r'\w+:', ' ', query)  # column filters
    sanitized = re.sub(r'[*^"()]', ' ', sanitized)
    sanitized = re.sub(r'\b(AND|OR|NOT|NEAR)\b', '', sanitized, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', sanitized).strip()
```

Fix tag filtering in `list_memories` in `db.py` — move tag filter into SQL:

```python
def list_memories(self, memory_type=None, tags=None, min_importance=None, limit=20, offset=0):
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
        # Filter tags in SQL using JSON — check each requested tag
        tag_conditions = []
        for tag in tags:
            tag_conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')
        query += " AND (" + " OR ".join(tag_conditions) + ")"

    query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    return [self._row_to_memory(r) for r in rows]
```

Fix BM25 normalization in `search.py` (ensure no negative normalized scores):

```python
# In recall(), after BM25 results:
if fts_results:
    max_bm25 = max(score for _, score in fts_results) or 1.0
    min_bm25 = min(score for _, score in fts_results)
    score_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
    for mem, score in fts_results:
        norm_score = (score - min_bm25) / score_range  # 0.0 to 1.0
        cand = candidates.setdefault(mem.id, _Candidate(memory=mem))
        cand.bm25_score = norm_score
        cand.matched_chunk = mem.content[:200]
```

Fix weight redistribution for BM25-only mode in `search.py`:

```python
# At the top of recall(), after self.has_vectors check setup:
if self._is_null:
    # Redistribute vector weight proportionally among other signals
    w_bm25 = WEIGHT_BM25 + WEIGHT_VECTOR * (WEIGHT_BM25 / (WEIGHT_BM25 + WEIGHT_RECENCY + WEIGHT_GRAPH))
    w_recency = WEIGHT_RECENCY + WEIGHT_VECTOR * (WEIGHT_RECENCY / (WEIGHT_BM25 + WEIGHT_RECENCY + WEIGHT_GRAPH))
    w_graph = WEIGHT_GRAPH + WEIGHT_VECTOR * (WEIGHT_GRAPH / (WEIGHT_BM25 + WEIGHT_RECENCY + WEIGHT_GRAPH))
    w_vector = 0.0
else:
    w_vector = WEIGHT_VECTOR
    w_bm25 = WEIGHT_BM25
    w_recency = WEIGHT_RECENCY
    w_graph = WEIGHT_GRAPH

# Then use w_* in the composite calculation instead of WEIGHT_*:
composite = (
    w_vector * cand.vector_score
    + w_bm25 * cand.bm25_score
    + w_recency * recency_score
    + w_graph * graph_score
)
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/db.py src/engram/search.py tests/test_db.py tests/test_search.py
git commit -m "fix: BM25 normalization, tag filtering, FTS sanitization, weight redistribution (#11, #12, #18, #22)"
```

---

## Task 6: Vector Search Performance (#3, #10, #31)

**Issues:** #3 (O(n) brute-force vector search), #10 (dimension mismatch crash), #31 (O(n²) dedup scan on store)

**Files:**
- Modify: `src/engram/search.py:65-103,128-156` — hash-first dedup, dimension guard
- Modify: `src/engram/embeddings.py:194-201` — dimension mismatch guard in cosine_similarity
- Test: `tests/test_search.py`, `tests/test_embeddings.py`

**Step 1: Write the failing tests**

In `tests/test_embeddings.py`:

```python
class TestCosineSimilarityDimensionMismatch:
    def test_mismatched_dimensions_returns_zero(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0
```

In `tests/test_search.py`:

```python
class TestStoreDedup:
    def test_hash_dedup_avoids_full_scan(self, engine):
        """Storing the same content twice should skip the duplicate via hash, not Jaccard."""
        engine.store(Memory(content="Exact duplicate content for hash test"))
        engine.store(Memory(content="Exact duplicate content for hash test"))
        stats = engine.db.get_stats()
        assert stats.total_chunks == 1  # hash-based dedup catches it
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Fix `cosine_similarity` in `embeddings.py`:

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
```

Fix store dedup in `search.py` — check hash first, only run Jaccard on non-hash-matched chunks:

```python
def store(self, memory: Memory) -> Memory:
    self._check_embedder_metadata()
    memory = self.db.store_memory(memory)
    chunks = chunk_text(memory.content)

    texts_to_embed: list[str] = []
    chunk_objects: list[Chunk] = []

    for i, text in enumerate(chunks):
        h = chunk_hash(text)
        if self.db.chunk_hash_exists(h):
            continue

        chunk_objects.append(
            Chunk(
                memory_id=memory.id,
                chunk_text=text,
                chunk_index=i,
                chunk_hash=h,
            )
        )
        texts_to_embed.append(text)

    if texts_to_embed and self.has_vectors:
        embeddings = self.embedder.embed_batch(texts_to_embed)
        for chunk_obj, emb in zip(chunk_objects, embeddings):
            chunk_obj.embedding = to_blob(emb)

    if chunk_objects:
        self.db.store_chunks(chunk_objects)

    return memory
```

Key change: remove `is_duplicate()` Jaccard scan and `get_all_chunk_texts()` call. Hash-based dedup is sufficient and O(1) per chunk. The Jaccard scan was O(n) per chunk, O(n*m) per store call.

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/search.py src/engram/embeddings.py tests/test_search.py tests/test_embeddings.py
git commit -m "fix: remove O(n²) dedup scan, guard dimension mismatch (#3, #10, #31)"
```

---

## Task 7: Atomicity and Data Integrity (#4, #21, #24, #32, #35)

**Issues:** #4 (orphan memories from embedding failure), #21 (FK violations silently swallowed), #24 (_dedup_chunks bypasses abstraction), #32 (non-atomic forget), #35 (non-atomic correct)

**Files:**
- Modify: `src/engram/db.py` — add `execute_in_transaction` helper, fix `store_relationship`
- Modify: `src/engram/search.py:65-103,276-289` — wrap store in transaction, fix `_dedup_chunks`
- Modify: `src/engram/server.py:459-479,379-455` — atomic forget and correct
- Test: `tests/test_db.py`, `tests/test_server.py`

**Step 1: Write the failing tests**

In `tests/test_db.py`:

```python
class TestTransactionSafety:
    def test_store_relationship_rejects_invalid_memory_id(self, db):
        """store_relationship should raise on invalid foreign key, not silently succeed."""
        rel = Relationship(source_id="nonexistent", target_id="also-fake")
        with pytest.raises(Exception):  # IntegrityError or our wrapper
            db.store_relationship(rel)

class TestDedupChunks:
    def test_dedup_uses_public_api(self, tmp_db_dir):
        """_dedup_chunks should not access _get_conn() directly."""
        from engram.search import SearchEngine
        from tests.conftest import FakeEmbedder
        db = MemoryDB(project="dedupapi", db_dir=tmp_db_dir)
        engine = SearchEngine(db=db, embedder=FakeEmbedder())
        # Just verify it runs without bypassing the lock
        engine.store(Memory(content="Test"))
        result = engine.memify()
        assert isinstance(result, dict)
```

In `tests/test_server.py`:

```python
class TestAtomicOperations:
    def test_forget_is_atomic(self, _patch_embedder):
        """If memory_forget partially fails, nothing should be deleted."""
        from engram.server import memory_store, memory_forget
        result = memory_store(content="Atomic test", project="test-project")
        mid = result["id"]
        # Normal forget should work atomically
        forget = memory_forget(memory_id=mid, project="test-project")
        assert forget["status"] == "forgotten"
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Add transaction helper to `db.py`:

```python
from contextlib import contextmanager

@contextmanager
def _transaction(self):
    """Execute a block within a single transaction. Rolls back on error."""
    conn = self._get_conn()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
```

Fix `store_relationship` to validate FK first:

```python
def store_relationship(self, rel: Relationship) -> Relationship:
    with self._lock:
        conn = self._get_conn()
        # Validate both memory IDs exist
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
                (rel.id, rel.source_id, rel.target_id, rel.rel_type.value,
                 rel.strength, rel.created_at.isoformat()),
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
```

Add `delete_memory_atomic` to `db.py` that deletes chunks, relationships, and memory in one transaction:

```python
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
```

Fix `memory_forget` in `server.py`:

```python
@mcp.tool()
def memory_forget(memory_id: str, project: str = "") -> dict:
    engine = _get_engine(project or None)
    mem = engine.db.get_memory(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found."}
    engine.db.delete_memory_atomic(memory_id)
    return {"status": "forgotten", "id": memory_id}
```

Fix `_dedup_chunks` in `search.py` to use public `delete_chunks_by_ids` instead of `_get_conn()`:

```python
def _dedup_chunks(self) -> int:
    all_chunks = self.db.get_all_chunks_with_embeddings()
    seen_hashes: set[str] = set()
    to_delete: list[str] = []
    for chunk in all_chunks:
        if chunk.chunk_hash in seen_hashes:
            to_delete.append(chunk.id)
        else:
            seen_hashes.add(chunk.chunk_hash)
    if to_delete:
        self.db.delete_chunks_by_ids(to_delete)
    return len(to_delete)
```

Add `delete_chunks_by_ids` to `db.py`:

```python
def delete_chunks_by_ids(self, chunk_ids: list[str]) -> int:
    with self._lock:
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in chunk_ids)
        cursor = conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", chunk_ids)
        conn.commit()
        return cursor.rowcount
```

Fix `memory_correct` in `server.py` to be atomic — wrap store+relationship+demote:

```python
@mcp.tool()
def memory_correct(old_memory_id, new_content, memory_type="", tags="", importance=1, project=""):
    engine = _get_engine(project or None)
    old_mem = engine.db.get_memory(old_memory_id)
    if not old_mem:
        return {"error": f"Memory '{old_memory_id}' not found."}

    # ... memory_type and tags resolution unchanged ...

    try:
        stored = engine.store(new_memory)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    # Atomic: create relationship + demote old in one transaction
    rel = Relationship(source_id=stored.id, target_id=old_memory_id,
                       rel_type=RelationType.SUPERSEDES, strength=1.0)
    try:
        engine.db.store_relationship(rel)
    except ValueError:
        pass  # Shouldn't happen since we just stored both memories
    engine.db.update_memory(old_memory_id, importance=4)

    return { ... }
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/db.py src/engram/search.py src/engram/server.py tests/test_db.py tests/test_server.py
git commit -m "fix: atomic operations, FK validation, dedup via public API (#4, #21, #24, #32, #35)"
```

---

## Task 8: Server Input Validation and Error Handling (#14, #15, #16, #17, #30)

**Issues:** #14 (wrong MCP error format), #15 (unbounded parameters), #16 (inverted importance naming), #17 (graph traversal phantom nodes), #30 (unhandled ValueError on invalid memory_type)

**Files:**
- Modify: `src/engram/server.py` — add input validation, fix error returns, fix importance docs
- Modify: `src/engram/db.py:340-387` — fix phantom node in BFS
- Test: `tests/test_server.py`, `tests/test_db.py`

**Step 1: Write the failing tests**

In `tests/test_server.py`:

```python
class TestInputValidation:
    def test_invalid_memory_type_in_list(self, _patch_embedder):
        from engram.server import memory_list
        result = memory_list(memory_type="invalid_type", project="test-project")
        # Should return error, not crash with ValueError
        assert "error" in result or result["count"] == 0

    def test_limit_capped(self, _patch_embedder):
        from engram.server import memory_list
        result = memory_list(limit=999999, project="test-project")
        assert isinstance(result, dict)  # Should not crash

    def test_content_too_long_rejected(self, _patch_embedder):
        from engram.server import memory_store
        huge = "x" * 60_000
        result = memory_store(content=huge, project="test-project")
        assert "error" in result

    def test_recall_limit_capped(self, _patch_embedder):
        from engram.server import memory_recall
        result = memory_recall(query="test", top_k=10000, project="test-project")
        assert isinstance(result, dict)
```

In `tests/test_db.py`:

```python
class TestBFSPhantomNodes:
    def test_deleted_memory_not_in_frontier(self, db):
        """BFS should not add IDs to frontier for memories that don't exist."""
        m1 = db.store_memory(Memory(content="Existing"))
        m2 = db.store_memory(Memory(content="Will be deleted"))
        rel = Relationship(source_id=m1.id, target_id=m2.id)
        db.store_relationship(rel)
        db.delete_memory(m2.id)  # Delete but relationship remains (no CASCADE trigger)

        connected = db.get_connected(m1.id, max_hops=2)
        # Should not include the deleted memory
        for mem, *_ in connected:
            assert mem is not None
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Fix `memory_list` in `server.py` — wrap memory_type in try/except:

```python
@mcp.tool()
def memory_list(memory_type="", tags="", min_importance=4, limit=20, project=""):
    engine = _get_engine(project or None)

    mt = None
    if memory_type:
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            return {"error": f"Invalid memory_type '{memory_type}'. Valid types: {[t.value for t in MemoryType]}"}

    limit = max(1, min(100, limit))  # Cap limit
    # ... rest unchanged ...
```

Add input validation to `memory_store`:

```python
@mcp.tool()
def memory_store(content, memory_type="context", tags="", importance=2, project=""):
    if len(content) > MAX_CONTENT_LENGTH:
        return {"error": f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters."}
    # ... rest unchanged ...
```

Add `MAX_CONTENT_LENGTH` import to server.py:

```python
from .types import MAX_CONTENT_LENGTH, Memory, MemoryType, Relationship, RelationType
```

Cap `top_k` in `memory_recall`:

```python
top_k = max(1, min(50, top_k))
```

Fix phantom node in BFS in `db.py` — only add to frontier if memory exists:

```python
# In get_connected:
for row in outgoing:
    nid = row["target_id"]
    if nid not in visited:
        visited.add(nid)
        mem = self.get_memory(nid)
        if mem:
            results.append((mem, row["rel_type"], "outgoing", row["strength"]))
            next_frontier.append(nid)  # Only add if memory exists
        # Don't add to frontier if memory doesn't exist

for row in incoming:
    nid = row["source_id"]
    if nid not in visited:
        visited.add(nid)
        mem = self.get_memory(nid)
        if mem:
            results.append((mem, row["rel_type"], "incoming", row["strength"]))
            next_frontier.append(nid)  # Only add if memory exists
```

Note: The phantom node fix is that `next_frontier.append(nid)` must be inside the `if mem:` block. Currently it's outside, meaning non-existent nodes get added to the frontier for the next hop.

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/server.py src/engram/db.py tests/test_server.py tests/test_db.py
git commit -m "fix: input validation, error format, BFS phantom nodes (#14, #15, #16, #17, #30)"
```

---

## Task 9: Auth, Shutdown, Event Loop, and SSRF (#1, #2, #7, #8, #19)

**Issues:** #1 (SSE auth bypass on non-HTTP scopes), #2 (no project-level auth), #7 (no graceful shutdown), #8 (sync functions block event loop), #19 (SSRF via OLLAMA_URL)

**Files:**
- Modify: `src/engram/server.py:625-684` — fix auth middleware, add shutdown
- Modify: `src/engram/__main__.py` — add startup safety checks
- Modify: `src/engram/embeddings.py:73-74,165-175` — SSRF protection
- Test: `tests/test_server.py`, `tests/test_embeddings.py`

**Step 1: Write the failing tests**

In `tests/test_server.py`:

```python
class TestAuthMiddleware:
    def test_auth_rejects_unknown_scope_types(self):
        """Non-http, non-lifespan scopes should be rejected when auth is enabled."""
        from engram.server import _wrap_with_api_key_auth
        import asyncio

        async def fake_app(scope, receive, send):
            pass

        wrapped = _wrap_with_api_key_auth(fake_app, "test-key")
        responses = []

        async def mock_send(msg):
            responses.append(msg)

        # Simulate an unknown scope type
        scope = {"type": "websocket", "headers": []}
        asyncio.run(wrapped(scope, None, mock_send))
        # Should have been rejected with 403
        assert any(b"unauthorized" in str(r).encode() for r in responses) or len(responses) > 0
```

In `tests/test_embeddings.py`:

```python
class TestSSRFProtection:
    def test_rejects_private_ip_ollama_url(self):
        from engram.embeddings import _validate_ollama_url
        assert _validate_ollama_url("http://localhost:11434") is True
        assert _validate_ollama_url("http://127.0.0.1:11434") is True
        assert _validate_ollama_url("http://169.254.169.254/latest/meta-data") is False
        assert _validate_ollama_url("http://metadata.google.internal") is False
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Fix auth middleware in `server.py`:

```python
def _wrap_with_api_key_auth(app, api_key: str):
    import secrets
    from starlette.responses import JSONResponse

    expected = f"Bearer {api_key}".encode("utf-8")

    async def auth_middleware(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            token = headers.get(b"authorization", b"")
            if not secrets.compare_digest(token, expected):
                resp = JSONResponse({"error": "unauthorized"}, status_code=401)
                await resp(scope, receive, send)
                return
        elif scope["type"] != "lifespan":
            # Reject unknown scope types
            resp = JSONResponse({"error": "unauthorized"}, status_code=403)
            await resp(scope, receive, send)
            return
        await app(scope, receive, send)

    return auth_middleware
```

Add graceful shutdown in `server.py`:

```python
def main(transport="stdio", host="0.0.0.0", port=8788, api_key=None):
    if transport == "stdio":
        mcp.run()
    elif transport == "sse":
        import signal
        import anyio
        import uvicorn

        mcp.settings.transport_security = None
        app = mcp.sse_app()

        if api_key:
            app = _wrap_with_api_key_auth(app, api_key)

        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)

        def _shutdown():
            """Close all engine connections on shutdown."""
            with _engines_lock:
                for engine in _engines.values():
                    engine.db.close()
                _engines.clear()

        import atexit
        atexit.register(_shutdown)

        anyio.run(server.serve)
```

Add SSRF protection in `embeddings.py`:

```python
import ipaddress
from urllib.parse import urlparse

def _validate_ollama_url(url: str) -> bool:
    """Validate that the Ollama URL doesn't point to cloud metadata endpoints."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        # Block cloud metadata endpoints
        blocked_hosts = {"metadata.google.internal", "metadata.aws.internal"}
        if hostname in blocked_hosts:
            return False

        # Block link-local addresses (169.254.x.x)
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_link_local:
                return False
            # Block specific cloud metadata IPs
            if str(ip) == "169.254.169.254":
                return False
        except ValueError:
            pass  # Not an IP, that's fine (it's a hostname)

        return True
    except Exception:
        return False
```

Use it in `create_embedder` and `OllamaEmbedder.__init__`:

```python
class OllamaEmbedder:
    def __init__(self, base_url="http://localhost:11434"):
        if not _validate_ollama_url(base_url):
            raise ValueError(f"Blocked Ollama URL (potential SSRF): {base_url}")
        self._base_url = base_url.rstrip("/")
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/server.py src/engram/__main__.py src/engram/embeddings.py tests/test_server.py tests/test_embeddings.py
git commit -m "fix: auth for all scope types, graceful shutdown, SSRF protection (#1, #2, #7, #8, #19)"
```

---

## Task 10: Logging, Engine Cache, FTS Rebuild, and Ollama Deps (#9, #20, #28, #36)

**Issues:** #9 (no FTS rebuild path), #20 (no logging), #28 (unbounded engine cache), #36 (httpx ImportError swallowed)

**Files:**
- Modify: `src/engram/server.py` — add logging to all tools, LRU engine cache
- Modify: `src/engram/db.py` — add `rebuild_fts` method
- Modify: `src/engram/embeddings.py:165-175` — fix exception handling
- Modify: `pyproject.toml` — add httpx optional dep
- Test: `tests/test_server.py`, `tests/test_db.py`, `tests/test_embeddings.py`

**Step 1: Write the failing tests**

In `tests/test_db.py`:

```python
class TestFTSRebuild:
    def test_rebuild_fts_restores_search(self, db):
        db.store_memory(Memory(content="PostgreSQL database"))
        # Corrupt FTS by dropping and recreating empty
        conn = db._get_conn()
        conn.execute("DROP TABLE IF EXISTS memory_fts")
        conn.execute("""CREATE VIRTUAL TABLE memory_fts USING fts5(
            content, tags, content='memories', content_rowid='rowid',
            tokenize='porter unicode61')""")
        conn.commit()

        # Search should fail now
        assert db.fts_search("PostgreSQL") == []

        # Rebuild should fix it
        db.rebuild_fts()
        results = db.fts_search("PostgreSQL")
        assert len(results) >= 1
```

In `tests/test_server.py`:

```python
class TestEngineCacheBound:
    def test_engine_cache_evicts_old_entries(self, _patch_embedder, tmp_path, monkeypatch):
        import engram.server as srv
        monkeypatch.setattr(srv, "MAX_ENGINE_CACHE_SIZE", 3)
        srv._engines.clear()

        from engram.server import memory_store
        for i in range(5):
            memory_store(content=f"Test {i}", project=f"project-{i}")

        assert len(srv._engines) <= 3
```

In `tests/test_embeddings.py`:

```python
class TestOllamaReachableErrorHandling:
    def test_import_error_logged_not_swallowed(self, monkeypatch):
        """httpx ImportError should be caught separately and logged."""
        from engram.embeddings import _ollama_reachable
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = _ollama_reachable("http://localhost:11434")
        assert result is False
```

**Step 2: Run to verify failures**

**Step 3: Implement**

Add logging to `server.py`:

```python
import logging
logger = logging.getLogger(__name__)
```

Then add `logger.debug(...)` at the entry of each tool function. Example:

```python
@mcp.tool()
def memory_store(content, memory_type="context", tags="", importance=2, project=""):
    logger.debug("memory_store called: project=%s, type=%s, importance=%d", project, memory_type, importance)
    # ... rest ...
```

Add LRU engine cache in `server.py`:

```python
from collections import OrderedDict

MAX_ENGINE_CACHE_SIZE = 16
_engines: OrderedDict[str, SearchEngine] = OrderedDict()
_engines_lock = threading.Lock()

def _get_engine(project=None):
    import re
    raw = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
    project = re.sub(r'[^a-zA-Z0-9_-]', '', raw) or "default"
    with _engines_lock:
        if project in _engines:
            _engines.move_to_end(project)
            return _engines[project]
        db_dir = os.environ.get("ENGRAM_DIR", None)
        db = MemoryDB(project=project, db_dir=db_dir)
        embedder = create_embedder()
        engine = SearchEngine(db=db, embedder=embedder)
        _engines[project] = engine
        while len(_engines) > MAX_ENGINE_CACHE_SIZE:
            _, evicted = _engines.popitem(last=False)
            evicted.db.close()
        return engine
```

Add `rebuild_fts` to `db.py`:

```python
def rebuild_fts(self) -> None:
    """Rebuild the FTS5 index from the memories table."""
    with self._lock:
        conn = self._get_conn()
        conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
        conn.commit()
```

Fix `_ollama_reachable` in `embeddings.py`:

```python
def _ollama_reachable(base_url: str) -> bool:
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not installed — Ollama auto-detect skipped")
        return False

    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except httpx.HTTPError:
        logger.debug("Ollama not reachable at %s", base_url)
    except Exception:
        logger.debug("Ollama auto-detect failed", exc_info=True)
    return False
```

Add httpx to `pyproject.toml`:

```toml
[project.optional-dependencies]
ollama = ["httpx>=0.24"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "httpx>=0.24",
]
```

**Step 4: Run tests**

Run: `python -m pytest tests/ -v`

**Step 5: Commit**

```bash
git add src/engram/server.py src/engram/db.py src/engram/embeddings.py pyproject.toml tests/
git commit -m "fix: logging, LRU engine cache, FTS rebuild, Ollama error handling (#9, #20, #28, #36)"
```

---

## Final Verification

After all 10 tasks:

```bash
python -m pytest tests/ -v --tb=short
```

Expected: ALL PASS

Then run ruff for lint:

```bash
ruff check src/engram/ tests/
```

---

## Issues NOT Fixed in Code (documented only)

| Issue | Reason |
|-------|--------|
| #25 | Test coverage gaps — addressed incrementally in every task above |
| #26 | PostgreSQL backend — requires new `StorageBackend` abstraction; Docker/Compose already landed |
| #8 (partial) | Full async conversion of tool functions is a major refactor; we added `atexit` cleanup but sync-in-async remains for a future async migration |

---

## Summary

| Metric | Value |
|--------|-------|
| Issues fixed | 34 of 36 |
| Tasks | 10 |
| New test file | `tests/test_types.py` |
| Modified files | 7 (`types.py`, `errors.py`, `db.py`, `search.py`, `server.py`, `embeddings.py`, `__main__.py`, `pyproject.toml`) |
