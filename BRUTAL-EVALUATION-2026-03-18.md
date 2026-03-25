# BRUTAL EVALUATION: Engram

**Date**:                 2026-03-18
**Stage**:                BETA (v0.1.0)
**Dimensions Evaluated**: technical (security, data integrity, performance, concurrency)
**Verdict**:              RETHINK (SSE mode) / PROCEED WITH FIXES (stdio mode)

---

## Executive Summary

Engram is a well-structured MCP memory server that works fine as a single-user localhost tool (stdio mode). The moment you enable SSE mode — which is the entire multi-agent value proposition — the codebase has zero defense-in-depth: auth bypass vectors, no TLS, no thread safety, no graceful shutdown, O(n) brute-force search, and embedding failures that silently create ghost records. The ~2,000 lines of source code have more architectural time bombs than features.

---

## Issues Filed: 25 total

### Critical (8 issues)

| #  | Issue | Link |
|----|-------|------|
| 1  | SSE auth bypass: non-HTTP ASGI scopes skip authentication | [#1](https://github.com/petersimmons1972/engram/issues/1) |
| 2  | No project-level authorization — any client can access any project | [#2](https://github.com/petersimmons1972/engram/issues/2) |
| 3  | O(n) brute-force vector search loads all embeddings into RAM | [#3](https://github.com/petersimmons1972/engram/issues/3) |
| 4  | Embedding API failures create orphan memories with no searchable chunks | [#4](https://github.com/petersimmons1972/engram/issues/4) |
| 5  | Thread safety: no locking on shared state, SQLite connections not thread-safe | [#5](https://github.com/petersimmons1972/engram/issues/5) |
| 6  | No schema migration strategy | [#6](https://github.com/petersimmons1972/engram/issues/6) |
| 7  | No graceful shutdown — signal handling and resource cleanup missing | [#7](https://github.com/petersimmons1972/engram/issues/7) |
| 8  | Sync tool functions block SSE event loop during I/O | [#8](https://github.com/petersimmons1972/engram/issues/8) |

### High (9 issues)

| #  | Issue | Link |
|----|-------|------|
| 9  | FTS5 index has no recovery/rebuild path | [#9](https://github.com/petersimmons1972/engram/issues/9) |
| 10 | Cosine similarity crashes on embedding dimension mismatch | [#10](https://github.com/petersimmons1972/engram/issues/10) |
| 11 | BM25 normalization can invert rankings when scores are negative | [#11](https://github.com/petersimmons1972/engram/issues/11) |
| 12 | Tag filtering applied after SQL LIMIT — returns fewer results than expected | [#12](https://github.com/petersimmons1972/engram/issues/12) |
| 13 | 48-bit UUID IDs risk collision at scale — silent data loss | [#13](https://github.com/petersimmons1972/engram/issues/13) |
| 14 | MCP tool errors return wrong format — not flagged as errors to client | [#14](https://github.com/petersimmons1972/engram/issues/14) |
| 15 | Unbounded query parameters enable DoS (limit, content size, rate) | [#15](https://github.com/petersimmons1972/engram/issues/15) |
| 16 | Importance filter parameter is misleadingly named (inverted scale) | [#16](https://github.com/petersimmons1972/engram/issues/16) |
| 17 | Graph traversal adds phantom nodes to BFS frontier + N+1 queries | [#17](https://github.com/petersimmons1972/engram/issues/17) |

### Medium (8 issues)

| #  | Issue | Link |
|----|-------|------|
| 18 | FTS5 query sanitization incomplete | [#18](https://github.com/petersimmons1972/engram/issues/18) |
| 19 | Ollama SSRF via OLLAMA_URL environment variable | [#19](https://github.com/petersimmons1972/engram/issues/19) |
| 20 | No logging in tool functions — zero observability | [#20](https://github.com/petersimmons1972/engram/issues/20) |
| 21 | store_relationship swallows FK violations — silent failure | [#21](https://github.com/petersimmons1972/engram/issues/21) |
| 22 | Score weights not redistributed in BM25-only mode | [#22](https://github.com/petersimmons1972/engram/issues/22) |
| 23 | Missing database indexes on date columns + low busy_timeout | [#23](https://github.com/petersimmons1972/engram/issues/23) |
| 24 | _dedup_chunks bypasses DB abstraction and ignores BM25-only chunks | [#24](https://github.com/petersimmons1972/engram/issues/24) |
| 25 | Test coverage gaps: zero tests for SSE, auth, error paths, concurrency | [#25](https://github.com/petersimmons1972/engram/issues/25) |

---

## Architectural Assessment

### Two Fundamental Weaknesses

1. **No failure isolation between embedding and storage.** When the embedding provider hiccups, you lose data (ghost memories with no chunks). No retry, no deferred embedding, no graceful degradation.

2. **O(n) brute-force vector search** capped at 10k chunks. Works for a toy project. The moment you cross 10k chunks, search quality silently degrades with no warning.

### SSE Mode Is Not Production-Ready

The SSE transport bolts network exposure onto a codebase with zero defense-in-depth:
- Auth middleware has a scope-type bypass
- No TLS
- No project-level authorization
- Default config binds unauthenticated to all interfaces
- Sync tools block the event loop
- No thread safety on shared state
- No graceful shutdown
- No logging

If exposed to any network beyond a trusted VPN, it's an open door to every piece of context your AI agents have ever seen.

### stdio Mode Is Usable Today

For single-developer, single-machine, single-agent usage via stdio transport, most critical issues don't apply. The search scaling issues (#3) and data integrity issues (#4, #6, #9) still apply but at lower impact. Viable for personal use with awareness of limitations.

---

## What's Actually Good

- Clean layered architecture: chunker -> embedder -> search -> server
- Hybrid search concept (BM25 + vector + recency + graph) is the right approach
- Proper use of Pydantic models for type safety
- Good test coverage for happy paths (~1,141 lines of tests)
- SQLite + WAL mode is a pragmatic choice for the target use case
- FTS5 with porter tokenizer is well-chosen
- MIT license, comprehensive community docs (CONTRIBUTING, SECURITY, CoC)
- The consolidation/decay concept is genuinely novel for agent memory

---

## Review Methodology

4 adversarial critic agents reviewed the codebase in parallel:
- **Security critic**: Auth, injection, input validation, SSRF, secrets
- **SQLite/data integrity critic**: Schema, concurrency, transactions, FTS5
- **Search/embeddings critic**: Ranking math, embedding pipeline, chunking
- **Server/MCP protocol critic**: Error handling, transport, async, deployment

Findings were deduplicated (many issues flagged by 2-4 agents independently) and consolidated from ~60 raw findings into 25 scoped GitHub issues.
