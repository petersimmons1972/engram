# Contributing to Engram

**First-time contributors are very welcome here.** If you've never contributed to an open-source project before, this is a good place to start. The codebase is focused, the test suite is comprehensive, and the maintainers will give you real feedback — not form-letter responses.

This document covers everything from "I found a bug" to "I set up a dev environment and want to write code." Read the parts that apply to you.

---

<p align="center">
  <img src="docs/contributing-flow.svg" alt="Five-step contribution flow" width="900">
</p>

---

## Community Standards

This project runs on a low-drama, high-quality collaboration model:

- Be kind. Be direct. Be specific.
- Assume good intent. Ask a clarifying question before concluding someone is being difficult.
- Critique code, not people. "This function does X when it should do Y" is useful feedback. "This is wrong" is not.

Full details in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). By participating, you agree to follow it.

---

## Ways to Contribute

You don't have to write code to contribute. Here's what helps:

**Bug reports** — If something doesn't work, a clear bug report with reproduction steps is genuinely valuable. Writing it carefully is itself a contribution.

**Feature requests** — Have an idea? Open an issue. Include the use case: what problem does this solve, who has it, and why can't the current tools address it?

**Documentation** — Unclear explanations, missing examples, wrong information. These are all valid issues. A docs fix is an excellent first PR.

**Code contributions** — Bug fixes, new features, test coverage improvements. Start from an existing issue when possible so you're not building something the maintainers won't accept.

**Test cases** — If you found a bug, the most useful fix includes a failing test that the fix makes pass. That test stays in the suite and prevents the bug from coming back.

---

## Development Setup

### Clone and Install

```bash
git clone https://github.com/shugav/engram.git
cd engram

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install the package with all dev dependencies
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`, `mypy`, and all optional dependencies (numpy, psycopg, httpx, openai). You need all of these to run the full test suite.

### Environment

```bash
cp .env.example .env
# Edit .env with your configuration
# For most local dev work you can leave embeddings disabled:
# ENGRAM_EMBEDDER=none
```

### Start the Server

```bash
# Local stdio mode (no HTTP server — for testing with an IDE)
python -m engram

# Network SSE mode (HTTP endpoint at localhost:8788)
python -m engram server --transport sse --host 127.0.0.1 --port 8788

# Streamable HTTP mode (recommended for network deployments)
python -m engram server --transport streamable-http --host 127.0.0.1 --port 8788
```

### Start PostgreSQL for Testing

The integration tests require a PostgreSQL instance. Docker makes this easy:

```bash
docker compose --profile test up -d test-postgres
# Starts postgres on port 5433 for tests (separate from the main postgres on 5432)
```

---

## Running Tests

```bash
# All tests (SQLite-backed, fast)
pytest tests/

# Specific test file
pytest tests/test_search.py -v

# PostgreSQL integration tests (requires the test-postgres container)
DATABASE_URL=postgresql://engram:test@localhost:5433/engram_test pytest tests/test_db_postgres.py -v

# With coverage
pytest --cov=src/engram tests/

# Linting
ruff check src/ tests/

# Type checking
mypy src/engram/
```

**The SQLite backend is used as the test fixture** — it runs without any external services and covers the full store → recall → correct → forget lifecycle. PostgreSQL is the supported deployment backend; its tests (`test_db_postgres.py`) need the test container and verify everything specific to the Postgres backend (tsvector search, connection pooling, JSONB tags).

If you're adding a feature, add tests in both `test_search.py` (for logic) and `test_db_postgres.py` (for the PostgreSQL implementation) where applicable.

---

## Project Structure

```
src/engram/
├── server.py       — MCP tool definitions and server entry point (FastMCP)
├── db.py           — DatabaseBackend protocol + factory (picks Postgres or SQLite)
├── db_postgres.py  — PostgreSQL backend (psycopg v3, connection pool, tsvector FTS)
├── db_sqlite.py    — SQLite backend (FTS5, single-file, fallback/testing)
├── search.py       — Three-signal scoring engine (BM25 + vector + recency)
├── embeddings.py   — Embedding providers: OpenAI, Ollama, Null
├── chunker.py      — Text chunking with configurable overlap
├── markdown_io.py  — memory_dump and memory_ingest (markdown ↔ Memory objects)
└── types.py        — Pydantic data models (Memory, Chunk, Relationship, etc.)

tests/
├── conftest.py         — Fixtures, FakeEmbedder, shared test utilities
├── test_server.py      — MCP tool smoke tests (full store/recall/correct lifecycle)
├── test_search.py      — SearchEngine unit tests
├── test_db_postgres.py — PostgreSQL backend integration tests
├── test_db.py          — SQLite backend tests
├── test_embeddings.py  — Embedding provider tests
├── test_markdown_io.py — Dump/ingest round-trip tests
├── test_consolidate.py — Dedup, decay, prune tests
├── test_batch.py       — memory_store_batch efficiency tests
├── test_immutability.py — Immutable memory flag tests
├── test_temporal.py    — expires_at and time-range recall tests
├── test_chunker.py     — Text chunking behavior tests
└── test_types.py       — Pydantic model validation tests
```

Understanding the flow: `server.py` tools call `SearchEngine` methods in `search.py`, which call `DatabaseBackend` methods in `db_postgres.py` or `db_sqlite.py`. The `create_database()` factory in `db.py` picks the right backend based on `DATABASE_URL`.

---

## Pull Request Guidelines

**One logical change per PR.** The question to ask yourself: "If this PR is reverted, what is reverted?" If the answer includes two separate features, split the PR.

**Link the related issue.** "Closes #42" in the PR description automatically closes the issue when merged. If there's no issue, consider opening one first so there's a record of the decision to make the change.

**Update docs when behavior changes.** If your change modifies what a tool does, what a config variable means, or how installation works — update the README. Tests verify behavior; docs explain intent.

**Clear commit messages.** `fix: prevent race condition in engine cache eviction` is useful. `fix stuff` is not. The commit log is how future maintainers understand why the code is the way it is.

**Don't add unrelated changes.** If you notice something else that could be improved while working on your change, open a separate issue for it. Keeping PRs focused makes review faster and reverts cleaner.

---

## Review Process

Maintainers aim for reviews that are:

- **Actionable** — Every comment ends with a concrete suggestion, not just a critique.
- **Respectful** — The goal is good code, not winning arguments.
- **Transparent** — If a change won't be accepted, you'll hear why early rather than after significant work.

If a review comment is unclear, ask a follow-up question in the PR thread. That is actively encouraged — clarity is better than guessing at intent.

First-time contributors: the review process exists to help, not to gatekeep. A revision request means "this is close and worth refining," not "this is rejected."

---

## Commit Style

```
type: short description of what changed (imperative mood, no period)

Optional longer explanation if the change needs context.
```

Common types: `fix`, `feat`, `docs`, `test`, `refactor`, `chore`.

Examples:
```
fix: handle expired memories in recall results
feat: add expires_at support to memory_store_batch
docs: update scoring formula in README to match search.py constants
test: add test for embedder dimension mismatch on project switch
```

---

## Need Help Getting Started?

Open an issue and say "I'm looking for a good first issue." Maintainers will point you at something appropriate for your experience level. No contribution is too small to be worth making.

---

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
