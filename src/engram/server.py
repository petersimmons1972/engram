"""Engram MCP Server -- persistent three-layer memory for AI agents."""

from __future__ import annotations

import base64 as _base64
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

from .db import create_database
from .embeddings import create_embedder
from .errors import EmbeddingConfigMismatchError
from .search import SearchEngine
from .summarizer import OLLAMA_URL, SUMMARIZE_ENABLED, SUMMARIZE_MODEL
from .types import (
    MAX_CONTENT_LENGTH,
    Memory,
    MemoryType,
    Relationship,
    RelationType,
)

ENGRAM_INSTRUCTIONS = """\
You have access to Engram, a persistent memory system shared across all of the \
user's machines. Memories survive across sessions, workspaces, and devices. \
Other AI agents working for this user also read and write to this same system.

## CRITICAL -- Project Scoping

Every tool accepts a `project` parameter. You MUST set this to the current \
workspace/project name (lowercase, hyphenated). Derive it from the workspace \
folder name -- for example, if you are working in `/home/user/my-cool-app`, \
set project="my-cool-app".

Each project gets its own isolated database. Memories stored in project "my-app" \
are invisible to project "web-dashboard" and vice versa. This prevents cross-project \
pollution.

For user-wide preferences that should apply everywhere (e.g. "user prefers dark mode", \
"always use Tailscale hostnames"), store them in project="global" so any project \
can find them.

## CRITICAL -- Session Start (Recall Before Working)

At the START of every task, before writing any code:

1. Call memory_recall with query "session handoff" AND the current project name \
   to find where the last agent left off.
2. Call memory_recall with a task-relevant query AND the current project name.
3. Also call memory_recall with the same query against project="global" to pick \
   up user-wide preferences.

If a session handoff note is found, present it to the user first and ask if they \
want to continue from there.

## CRITICAL -- Session End (Handoff Before Finishing)

Before your FINAL response in any significant task, you MUST store a session \
handoff memory:

memory_store(content="SESSION HANDOFF: [what was done] | NEXT: [what should \
happen next] | BLOCKED: [blockers] | FILES CHANGED: [files modified]", \
memory_type="context", tags="session-handoff", importance=1, project="<project>")

This is how the next agent picks up exactly where you left off. Think of it like \
a nurse handing off to the next shift. Every significant task ends with a handoff.

## When to Store Memories

Store a memory whenever you encounter something a future agent would benefit \
from knowing:

- **Decisions** (type: decision): "Chose PostgreSQL over MySQL because ..."
- **Patterns** (type: pattern): "This codebase uses repository pattern for DB access"
- **Errors** (type: error): "Port 3000 is already in use on my-server"
- **Architecture** (type: architecture): "Auth flow: JWT -> middleware -> httpOnly cookie"
- **Preferences** (type: preference): "User prefers tabs over spaces" \
  (store in project="global" if user-wide)
- **Context** (type: context): General project/environment details

## Importance Levels

- 0 = Critical identity (core user preferences, system-wide decisions)
- 1 = Key facts (important project decisions, recurring patterns)
- 2 = General context (default -- most memories go here)
- 3 = Low priority (minor notes, temporary context)
- 4 = Trivial (auto-pruned after 30 days if never accessed)

## Tags

Always add relevant tags. Use short, lowercase, hyphenated tags: \
"auth", "docker", "tailscale", "python", "frontend".

## Knowledge Graph

After storing related memories, use memory_connect to link them. Connected \
memories surface automatically during recall.

## Feedback Loop

After using recall results, call memory_feedback to mark them helpful or not. \
This trains the graph to surface better results over time.

## Maintenance

Run memory_consolidate periodically to deduplicate, decay unused edges, and \
prune stale memories.

## Onboarding New Projects

When you first connect to a project with zero memories, store foundational \
context: what the project is, its tech stack, key architecture decisions, \
and the user's conventions for this codebase. This bootstraps future agents.
"""

mcp = FastMCP("engram", instructions=ENGRAM_INSTRUCTIONS)


def _normalize_project(project: str) -> str:
    """Sanitize a project name for use as a DB namespace."""
    return re.sub(r'[^a-z0-9_-]', '', (project or "default").strip().lower()) or "default"


def _compression_ratio_from_fields(content: str, compressed: bytes) -> float:
    """Compute compression ratio from plaintext content and compressed bytes."""
    if not compressed:
        return 0.0
    return round(len(content.encode("utf-8")) / len(compressed), 3)


def _build_content_envelope(
    memory_id: str,
    content: str,
    content_compressed: bytes | None,
    compression_algo: str | None,
    compressed_at: "datetime | None",
    content_format: str,
) -> dict:
    """Build content fields for a memory response dict based on content_format.

    INVARIANT: 'content' in the returned dict is ALWAYS the plaintext string.
    Binary bytes never appear in response dicts — always base64-encoded.
    """
    from .compression import CompressionAlgoUnavailableError, decompress

    base = {"content": content}  # always present, always plaintext

    if content_format == "text":
        return base

    # Build compressed envelope
    compressed_envelope = None
    warning = None

    if content_compressed is not None:
        try:
            # Verify decompressable (round-trip check)
            decompress(content_compressed, compression_algo or "zlib")
            compressed_envelope = {
                "data": _base64.b64encode(content_compressed).decode("ascii"),
                "algo": compression_algo,
                "ratio": _compression_ratio_from_fields(content, content_compressed),
                "compressed_at": compressed_at.isoformat() if compressed_at else None,
                "warning": None,
            }
        except CompressionAlgoUnavailableError:
            warning = f"algo_unavailable:{compression_algo}"
            compressed_envelope = None
        except Exception:
            warning = "decompression_failed"
            compressed_envelope = None
    else:
        if content_format == "compressed_only":
            return {"error": f"Memory {memory_id} not yet compressed"}
        warning = "not_yet_compressed"

    if content_format == "compressed_only":
        if compressed_envelope:
            return {"content": content, "content_compressed": compressed_envelope}
        # Compressed bytes present but undecompressable (corrupt or algo unavailable)
        return {"error": f"Memory {memory_id} compressed data unavailable: {warning}"}

    if content_format in ("compressed", "both"):
        result = dict(base)
        if compressed_envelope:
            result["content_compressed"] = compressed_envelope
        elif warning:
            result["content_compressed"] = {
                "data": None, "algo": None, "ratio": None,
                "compressed_at": None, "warning": warning,
            }
        return result

    return base  # fallback for unknown format values


MAX_ENGINE_CACHE_SIZE = 64
_engines: dict[str, SearchEngine] = {}
_engines_lock = threading.Lock()            # guards dict reads/writes only
_creation_locks: dict[str, threading.Lock] = {}
_creation_locks_lock = threading.Lock()     # guards _creation_locks dict

# Rate limiting for memory_store — sliding window per project
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = int(os.environ.get("ENGRAM_RATE_LIMIT", "100"))  # calls per window
_store_calls: dict[str, deque] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def _get_engine(project: str | None = None) -> SearchEngine:
    """Return (or create) a SearchEngine for the given project.

    Uses double-checked locking: a short global lock guards the cache dict,
    and per-project locks guard expensive engine creation. Two threads
    requesting different projects never block each other.
    """
    raw = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
    proj = re.sub(r"[^a-z0-9_-]", "", raw) or "default"

    # Fast path — engine already cached
    with _engines_lock:
        if proj in _engines:
            return _engines[proj]

    # Slow path — get or create a per-project creation lock
    with _creation_locks_lock:
        if proj not in _creation_locks:
            _creation_locks[proj] = threading.Lock()
        proj_lock = _creation_locks[proj]

    with proj_lock:
        # Double-check after acquiring project lock
        with _engines_lock:
            if proj in _engines:
                return _engines[proj]

        db = create_database(project=proj)
        embedder = create_embedder()
        engine = SearchEngine(db=db, embedder=embedder)

        with _engines_lock:
            _engines[proj] = engine
            if len(_engines) > MAX_ENGINE_CACHE_SIZE:
                oldest = next(iter(_engines))
                evicted = _engines.pop(oldest)
                logger.info("Evicted engine for project=%s", oldest)
                evicted.close()

        return engine


@mcp.tool()
def memory_store(
    content: str,
    memory_type: str = "context",
    tags: str = "",
    importance: int = 2,
    immutable: bool = False,
    expires_at: str = "",
    project: str = "",
) -> dict:
    """Store a new memory. Auto-chunks, embeds, and indexes for three-layer search.

    Args:
        content: The memory content to store. Be specific and detailed.
        memory_type: One of: decision, pattern, error, context, architecture, preference.
        tags: Comma-separated tags for filtering (e.g. "auth,security,jwt").
        importance: Priority 0-4. 0=critical identity, 1=key facts, 2=general, 3=low, 4=trivial.
        immutable: If true, memory cannot be corrected or deleted. Use for critical preferences.
        expires_at: ISO datetime after which memory is auto-pruned (e.g. "2026-04-30T00:00:00+00:00"). Empty = never expires.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        The stored memory's ID and metadata.
    """
    logger.debug("memory_store called: project=%s, type=%s, importance=%d", project, memory_type, importance)
    if len(content) > MAX_CONTENT_LENGTH:
        return {"error": f"Content exceeds maximum length of {MAX_CONTENT_LENGTH} characters."}

    # Rate limiting: sliding window per project
    proj_key = (project or "default").strip().lower() or "default"
    now = time.monotonic()
    with _rate_limit_lock:
        window = _store_calls[proj_key]
        # Evict calls outside the window
        while window and now - window[0] > _RATE_LIMIT_WINDOW:
            window.popleft()
        if len(window) >= _RATE_LIMIT_MAX:
            retry_after = int(_RATE_LIMIT_WINDOW - (now - window[0])) + 1
            return {
                "error": f"Rate limit exceeded: max {_RATE_LIMIT_MAX} memory_store calls per {_RATE_LIMIT_WINDOW}s per project.",
                "retry_after_seconds": retry_after,
            }
        window.append(now)

    engine = _get_engine(project or None)

    try:
        mt = MemoryType(memory_type)
    except ValueError:
        mt = MemoryType.CONTEXT

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    importance = max(0, min(4, importance))

    expires_at_dt = None
    if expires_at:
        try:
            from datetime import datetime
            expires_at_dt = datetime.fromisoformat(expires_at)
        except ValueError:
            return {"error": f"Invalid expires_at format: '{expires_at}'. Use ISO 8601 (e.g. '2026-04-30T00:00:00+00:00')."}

    memory = Memory(
        content=content,
        memory_type=mt,
        tags=tag_list,
        importance=importance,
        immutable=immutable,
        expires_at=expires_at_dt,
    )

    try:
        stored = engine.store(memory)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    return {
        "status": "stored",
        "id": stored.id,
        "memory_type": stored.memory_type.value,
        "tags": stored.tags,
        "importance": stored.importance,
    }


@mcp.tool()
def memory_store_batch(
    memories: str,
    project: str = "",
) -> dict:
    """Store multiple memories in one call with batched embedding.

    More efficient than calling memory_store in a loop — chunks all memories
    first, then embeds all chunks in a single batch call.

    Args:
        memories: JSON array of memory objects. Each object may contain:
            content (required), memory_type, tags (comma-separated string or list),
            importance, immutable, expires_at.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Count of stored/failed memories and list of stored IDs.
    """
    import json as _json

    logger.debug("memory_store_batch called: project=%s", project)

    try:
        items = _json.loads(memories)
    except (ValueError, TypeError):
        return {"error": "Invalid JSON. 'memories' must be a JSON array of objects."}

    if not isinstance(items, list):
        return {"error": "'memories' must be a JSON array of objects."}

    # Rate limiting: count each memory against sliding window
    proj_key = (project or "default").strip().lower() or "default"
    now = time.monotonic()
    with _rate_limit_lock:
        window = _store_calls[proj_key]
        while window and now - window[0] > _RATE_LIMIT_WINDOW:
            window.popleft()
        remaining = _RATE_LIMIT_MAX - len(window)
        if remaining <= 0:
            retry_after = int(_RATE_LIMIT_WINDOW - (now - window[0])) + 1
            return {
                "error": f"Rate limit exceeded: max {_RATE_LIMIT_MAX} stores per {_RATE_LIMIT_WINDOW}s per project.",
                "retry_after_seconds": retry_after,
            }
        # Reserve slots for the batch (cap at remaining capacity)
        batch_size = min(len(items), remaining)
        for _ in range(batch_size):
            window.append(now)
        items = items[:batch_size]

    engine = _get_engine(project or None)

    memory_objects: list[Memory] = []
    failed = 0
    for item in items:
        if not isinstance(item, dict):
            failed += 1
            continue
        content = item.get("content", "")
        if not content or len(content) > MAX_CONTENT_LENGTH:
            failed += 1
            continue

        try:
            mt = MemoryType(item.get("memory_type", "context"))
        except ValueError:
            mt = MemoryType.CONTEXT

        raw_tags = item.get("tags", "")
        if isinstance(raw_tags, list):
            tag_list = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tag_list = [t.strip() for t in str(raw_tags).split(",") if t.strip()]

        importance = max(0, min(4, int(item.get("importance", 2))))
        immutable = bool(item.get("immutable", False))

        expires_at_dt = None
        ea = item.get("expires_at", "")
        if ea:
            try:
                from datetime import datetime as _dt
                expires_at_dt = _dt.fromisoformat(ea)
            except ValueError:
                pass  # Ignore invalid expires_at

        memory_objects.append(Memory(
            content=content,
            memory_type=mt,
            tags=tag_list,
            importance=importance,
            immutable=immutable,
            expires_at=expires_at_dt,
        ))

    try:
        stored = engine.store_batch(memory_objects)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    return {
        "stored": len(stored),
        "failed": failed + (len(memory_objects) - len(stored)),
        "ids": [m.id for m in stored],
    }


@mcp.tool()
def memory_recall(
    query: str,
    top_k: int = 5,
    memory_type: str = "",
    tags: str = "",
    min_importance: int = 4,
    graph_hops: int = 1,
    since: str = "",
    before: str = "",
    project: str = "",
    content_format: str = "text",
    include_summary: bool = False,
) -> dict:
    """Search memories using all three layers: keyword (BM25), semantic (vector), and graph.

    Results are ranked by a composite score:
      Final = (vector * 0.50 + BM25 * 0.35 + recency * 0.15) * importance_multiplier

    Connected memories from the knowledge graph are attached automatically.

    Args:
        query: What to search for. Can be a keyword, question, or concept.
        top_k: Number of results to return (default 5).
        memory_type: Filter by type (decision/pattern/error/context/architecture/
            preference). Empty = all.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this value (0=only critical, 4=all).
        graph_hops: How many relationship hops to traverse (1 or 2).
        since: Only return memories created at or after this ISO datetime (e.g. "2026-03-01T00:00:00+00:00"). Empty = no lower bound.
        before: Only return memories created at or before this ISO datetime. Empty = no upper bound.
        project: Project namespace (e.g. "my-app"). Empty = "default".
        content_format: "text" (default), "compressed" (include base64 envelope if available), or "compressed_only".
        include_summary: If True and a summary exists, include "summary" in each result dict.

    Returns:
        Ranked list of memories with scores, matched chunks, and connected context.
    """
    logger.debug("memory_recall called: project=%s, query=%r, top_k=%d", project, query[:50], top_k)
    engine = _get_engine(project or None)
    top_k = max(1, min(50, top_k))

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mt = memory_type if memory_type else None
    mi = min_importance if min_importance < 4 else None

    from datetime import datetime as _dt
    since_dt = _dt.fromisoformat(since) if since else None
    before_dt = _dt.fromisoformat(before) if before else None

    results = engine.recall(
        query=query,
        top_k=top_k,
        memory_type=mt,
        tags=tag_list,
        min_importance=mi,
        graph_hops=max(1, min(2, graph_hops)),
        since=since_dt,
        before=before_dt,
    )

    output = []
    for r in results:
        # Determine supersede status from both edge directions.
        # Convention (set by memory_correct): source=new supersedes target=old.
        #   incoming edge (other → this): this memory IS superseded by other.
        #   outgoing edge (this → other): this memory supersedes other (this is the new version).
        superseded_by = None
        supersedes_list = []
        for c in r.connected:
            if c.rel_type != "supersedes":
                continue
            if c.direction == "incoming":
                # Another memory declared itself as superseding this one.
                superseded_by = {"id": c.memory.id, "content": c.memory.content[:300]}
            elif c.direction == "outgoing":
                # This memory supersedes another — surface it so callers know
                # the old version still exists.
                supersedes_list.append({"id": c.memory.id, "content": c.memory.content[:300]})

        content_fields = _build_content_envelope(
            memory_id=r.memory.id,
            content=r.memory.content,
            content_compressed=r.memory.content_compressed,
            compression_algo=r.memory.compression_algo,
            compressed_at=r.memory.compressed_at,
            content_format=content_format,
        )

        entry = {
            "id": r.memory.id,
            **content_fields,
            "type": r.memory.memory_type.value,
            "tags": r.memory.tags,
            "importance": r.memory.importance,
            "score": r.score,
            "score_breakdown": r.score_breakdown,
            "matched_chunk": r.matched_chunk,
            "connected": [
                {
                    "id": c.memory.id,
                    "content": c.memory.content[:300],
                    "rel_type": c.rel_type,
                    "direction": c.direction,
                    "strength": c.strength,
                }
                for c in r.connected
            ],
        }

        if superseded_by:
            entry["WARNING"] = "THIS MEMORY HAS BEEN SUPERSEDED. Use the newer version instead."
            entry["superseded_by"] = superseded_by
        if supersedes_list:
            entry["supersedes"] = supersedes_list
        if include_summary and r.memory.summary:
            entry["summary"] = r.memory.summary

        output.append(entry)

    return {"results": output, "count": len(output)}


@mcp.tool()
def memory_connect(
    source_id: str,
    target_id: str,
    rel_type: str = "relates_to",
    strength: float = 1.0,
    project: str = "",
) -> dict:
    """Create a typed relationship between two memories in the knowledge graph.

    This is how memories become interconnected. When one memory is recalled,
    its connected memories are pulled in automatically.

    Args:
        source_id: ID of the source memory.
        target_id: ID of the target memory.
        rel_type: Type: caused_by, relates_to, depends_on,
            supersedes, used_in, resolved_by.
        strength: Connection strength from 0.0 to 1.0 (default 1.0).
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        The created relationship.
    """
    logger.debug("memory_connect called: project=%s, %s -> %s (%s)", project, source_id, target_id, rel_type)
    engine = _get_engine(project or None)

    source = engine.db.get_memory(source_id)
    target = engine.db.get_memory(target_id)
    if not source:
        return {"error": f"Source memory '{source_id}' not found."}
    if not target:
        return {"error": f"Target memory '{target_id}' not found."}

    try:
        rt = RelationType(rel_type)
    except ValueError:
        rt = RelationType.RELATES_TO

    rel = Relationship(
        source_id=source_id,
        target_id=target_id,
        rel_type=rt,
        strength=max(0.0, min(1.0, strength)),
    )
    try:
        engine.db.store_relationship(rel)
    except ValueError as exc:
        return {"error": str(exc)}

    return {
        "status": "connected",
        "id": rel.id,
        "source_id": source_id,
        "target_id": target_id,
        "rel_type": rt.value,
        "strength": rel.strength,
    }


@mcp.tool()
def memory_list(
    memory_type: str = "",
    tags: str = "",
    min_importance: int = 4,
    limit: int = 20,
    project: str = "",
    content_format: str = "text",
    include_summary: bool = False,
) -> dict:
    """List recent memories with optional filters.

    Args:
        memory_type: Filter by type. Empty = all types.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this (0=only critical, 4=all).
        limit: Max number of memories to return.
        project: Project namespace (e.g. "my-app"). Empty = "default".
        content_format: "text" (default) or "compressed" to include compression envelope.

    Returns:
        List of memories sorted by most recently updated.
    """
    logger.debug("memory_list called: project=%s, type=%s, limit=%d", project, memory_type, limit)
    engine = _get_engine(project or None)
    limit = max(1, min(100, limit))

    mt = None
    if memory_type:
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            return {"error": f"Invalid memory_type '{memory_type}'. Valid: {[t.value for t in MemoryType]}"}

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mi = min_importance if min_importance < 4 else None

    memories = engine.db.list_memories(
        memory_type=mt,
        tags=tag_list,
        min_importance=mi,
        limit=limit,
    )

    result_items = []
    for m in memories:
        content_fields = _build_content_envelope(
            memory_id=m.id,
            content=m.content[:300],
            content_compressed=m.content_compressed,
            compression_algo=m.compression_algo,
            compressed_at=m.compressed_at,
            content_format=content_format,
        )
        item = {
            "id": m.id,
            **content_fields,
            "type": m.memory_type.value,
            "tags": m.tags,
            "importance": m.importance,
            "access_count": m.access_count,
            "created_at": m.created_at.isoformat(),
            "updated_at": m.updated_at.isoformat(),
        }
        if include_summary and m.summary:
            item["summary"] = m.summary
        result_items.append(item)

    return {
        "memories": result_items,
        "count": len(memories),
    }


@mcp.tool()
def memory_correct(
    old_memory_id: str,
    new_content: str,
    memory_type: str = "",
    tags: str = "",
    importance: int = 1,
    project: str = "",
) -> dict:
    """Correct or supersede a wrong/outdated memory.

    Use this when a recalled memory contains wrong information, an outdated
    decision, a bug fix that turned out to be incorrect, or anything that
    should no longer be trusted. The old memory is demoted and linked to the
    new one via a 'supersedes' relationship. Future recalls will prefer the
    new memory and deprioritize the old one.

    Args:
        old_memory_id: ID of the memory that is wrong or outdated.
        new_content: The corrected/updated information.
        memory_type: Type for the new memory. Empty = inherit from old memory.
        tags: Comma-separated tags. Empty = inherit from old memory.
        importance: Importance for the new memory (default 1 = high).
        project: Project namespace. Derive from workspace folder name. Empty = "default".

    Returns:
        The new memory ID and confirmation that the old one was superseded.
    """
    logger.debug("memory_correct called: project=%s, old_id=%s", project, old_memory_id)
    engine = _get_engine(project or None)

    old_mem = engine.db.get_memory(old_memory_id)
    if not old_mem:
        return {"error": f"Memory '{old_memory_id}' not found."}

    if old_mem.immutable:
        return {"error": f"Memory '{old_memory_id}' is immutable and cannot be corrected."}

    if not memory_type:
        mt = old_mem.memory_type
    else:
        try:
            mt = MemoryType(memory_type)
        except ValueError:
            mt = old_mem.memory_type

    tag_list = (
        [t.strip() for t in tags.split(",") if t.strip()] if tags else old_mem.tags
    )

    new_memory = Memory(
        content=new_content,
        memory_type=mt,
        tags=tag_list,
        importance=max(0, min(4, importance)),
    )
    try:
        stored = engine.store(new_memory)
    except EmbeddingConfigMismatchError as e:
        return {"error": str(e)}

    # Link: new supersedes old
    rel = Relationship(
        source_id=stored.id,
        target_id=old_memory_id,
        rel_type=RelationType.SUPERSEDES,
        strength=1.0,
    )
    engine.db.store_relationship(rel)

    # Demote old memory to trivial so it gets pruned over time
    engine.db.update_memory(old_memory_id, importance=4)

    return {
        "status": "corrected",
        "old_id": old_memory_id,
        "old_content_preview": old_mem.content[:200],
        "new_id": stored.id,
        "new_content_preview": new_content[:200],
        "relationship": "new supersedes old",
        "old_demoted_to": "trivial (will be pruned if unused)",
    }


@mcp.tool()
def memory_forget(memory_id: str, project: str = "") -> dict:
    """Remove a memory and all its relationships from the knowledge graph.

    Args:
        memory_id: The ID of the memory to remove.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Confirmation of deletion.
    """
    logger.debug("memory_forget called: project=%s, id=%s", project, memory_id)
    engine = _get_engine(project or None)

    mem = engine.db.get_memory(memory_id)
    if not mem:
        return {"error": f"Memory '{memory_id}' not found."}

    if mem.immutable:
        return {"error": f"Memory '{memory_id}' is immutable and cannot be deleted."}

    engine.db.delete_memory_atomic(memory_id)

    return {"status": "forgotten", "id": memory_id}


@mcp.tool()
def memory_summarize(
    project: str = "",
    limit: int = 50,
    model: str = "",
) -> dict:
    """Manually trigger summarization backfill for memories without summaries.

    Runs synchronously (blocking) on the calling thread. For large backlogs,
    the background summarizer (always running) handles this automatically.

    Args:
        project: Project namespace (e.g. "my-app"). Empty = "default".
        limit: Maximum number of memories to summarize in this call (default 50).
        model: Ollama model name. Empty = use ENGRAM_SUMMARIZE_MODEL env var.

    Returns:
        Count of summarized/failed memories and remaining backlog size.
    """
    from .summarizer import summarize_content as _summarize_content

    proj = _normalize_project(project or "")
    engine = _get_engine(proj)
    effective_model = model or SUMMARIZE_MODEL

    pending = engine.db.get_memories_pending_summary(proj, limit=limit)
    summarized = 0
    failed = 0

    for memory_id, content in pending:
        summary = _summarize_content(content, model=effective_model)
        if summary:
            engine.db.store_summary(memory_id, summary)
            summarized += 1
        else:
            failed += 1

    return {
        "summarized": summarized,
        "failed": failed,
        "remaining": engine.db.get_pending_summary_count(proj),
        "model": effective_model,
    }


@mcp.tool()
def memory_status(project: str = "") -> dict:
    """Get statistics about the memory system.

    Args:
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Total memories, chunks, relationships, breakdown by type and importance,
        database size, and age range.
    """
    logger.debug("memory_status called: project=%s", project)
    engine = _get_engine(project or None)
    proj = engine.db.project
    stats = engine.db.get_stats()
    result = stats.model_dump()
    result["summarization"] = {
        "pending": engine.db.get_pending_summary_count(proj),
        "model": SUMMARIZE_MODEL,
        "enabled": SUMMARIZE_ENABLED,
        "ollama_url": OLLAMA_URL,
    }
    integrity_stats = engine.db.get_integrity_stats(proj)
    result["integrity"] = {
        "total": integrity_stats["total"],
        "hashed": integrity_stats["hashed"],
        "coverage_pct": round(integrity_stats["hashed"] / integrity_stats["total"] * 100, 1) if integrity_stats["total"] > 0 else 0.0,
    }
    result["embedding_migration"] = {
        "in_progress": engine.db.get_meta("embedding_migration_in_progress") == "true",
        "pending_chunks": engine.db.get_pending_embedding_count(proj),
        "embedder": engine.db.get_meta("embedder_name") or "none",
    }
    return result


@mcp.tool()
def memory_feedback(
    memory_ids: str,
    helpful: bool = True,
    project: str = "",
) -> dict:
    """Provide feedback on recall results to strengthen or weaken graph connections.

    When recall results are helpful, their graph edges get reinforced -- making
    those connections more likely to surface in future recalls. When unhelpful,
    edges weaken. Over time the knowledge graph self-optimizes based on what
    actually helps you.

    Call this after memory_recall when you know whether the results were useful.

    Args:
        memory_ids: Comma-separated IDs of memories from the recall results.
        helpful: True if the results were useful, False if they were not.
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Number of memories whose graph edges were adjusted.
    """
    logger.debug("memory_feedback called: project=%s, helpful=%s", project, helpful)
    engine = _get_engine(project or None)
    ids = [mid.strip() for mid in memory_ids.split(",") if mid.strip()]
    if not ids:
        return {"error": "No memory IDs provided."}
    result = engine.feedback(ids, helpful)
    return result


@mcp.tool()
def memory_consolidate(project: str = "", compress: bool = False) -> dict:
    """Run a memory consolidation pass — dedup, decay, and prune.

    Three stages:
    1. Deduplicates chunks by hash to remove exact duplicates.
    2. Applies temporal decay to all graph edges and prunes weak connections
       (strength < 0.1) -- edges that are never reinforced by feedback fade away.
    3. Prunes stale, never-accessed, low-importance memories older than 30 days.

    Frequently-used connections survive and strengthen. Unused ones decay.
    Run this periodically to keep the memory system healthy and focused.

    Args:
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Breakdown of chunks deduped, edges decayed/pruned, and stale memories removed.
    """
    logger.debug("memory_consolidate called: project=%s, compress=%s", project, compress)
    engine = _get_engine(project or None)
    result = engine.consolidate()
    if compress:
        compress_result = engine.compress_memories()
        result["compression"] = compress_result
    return {"status": "consolidated", **result}


@mcp.tool()
def memory_compress(
    project: str = "",
    algorithm: str = "zlib",
    min_size_chars: int = 500,
    dry_run: bool = False,
    recompress: bool = False,
) -> dict:
    """Compress stored memories to reduce context window cost.

    Compression is deferred/additive — the original `content` field is ALWAYS preserved.
    Use `memory_recall(content_format="compressed")` to retrieve compressed form.

    Args:
        project: Project namespace.
        algorithm: Compression algorithm — "zlib" (default, stdlib) or "zstd" (requires engram[zstd]).
        min_size_chars: Skip memories shorter than this (default 500 chars).
        dry_run: Report what would be compressed without writing.
        recompress: Re-compress already-compressed memories (useful for algorithm changes).
    """
    project = _normalize_project(project)
    from .compression import SUPPORTED_ALGOS
    if algorithm not in SUPPORTED_ALGOS:
        return {"error": f"Unsupported algorithm {algorithm!r}. Available on this installation: {sorted(SUPPORTED_ALGOS)}"}

    engine = _get_engine(project)
    try:
        return engine.compress_memories(
            algorithm=algorithm,
            min_size_chars=min_size_chars,
            dry_run=dry_run,
            recompress=recompress,
        )
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def memory_verify(
    project: str = "",
    fix: bool = False,
) -> dict:
    """Verify content integrity of all memories using SHA-256 hashes.

    Scans all memories and checks their content_hash. Reports:
    - ok: memories with a valid hash
    - missing_hash: pre-v7 memories not yet hashed
    - corrupt: hash present but doesn't match content (indicates external modification)

    With fix=True, backfills missing hashes and recomputes corrupt ones.
    """
    from .db_postgres import _content_hash

    engine = _get_engine(project or None)
    proj = engine.db.project

    stats = engine.db.get_integrity_stats(proj)
    fixed = 0

    if fix:
        # Backfill missing hashes
        pending = engine.db.get_memories_missing_hash(proj, limit=10_000)
        for memory_id, content in pending:
            engine.db.update_memory_hash(memory_id, _content_hash(content))
            fixed += 1

        # Recompute corrupt hashes — re-fetch stats to find them
        if stats["corrupt"] > 0:
            with engine.db.pool.connection() as conn:
                rows = conn.execute(
                    "SELECT id, content FROM memories "
                    "WHERE project = %s AND content_hash IS NOT NULL "
                    "AND content_hash != encode(sha256(content::bytea), 'hex')",
                    (proj,),
                ).fetchall()
            for row in rows:
                engine.db.update_memory_hash(row["id"], _content_hash(row["content"]))
                fixed += 1

        stats = engine.db.get_integrity_stats(proj)

    return {
        "ok": stats["hashed"] - stats["corrupt"],
        "missing_hash": stats["total"] - stats["hashed"],
        "corrupt": stats["corrupt"],
        "fixed": fixed,
        "project": proj,
    }


@mcp.tool()
def memory_migrate_embedder(
    project: str = "",
    new_embedder: str = "",
    dry_run: bool = False,
) -> dict:
    """Switch the embedding provider for a project.

    Nulls all existing chunk embeddings and updates the project's embedder
    metadata. Vector search degrades to BM25+recency during re-embedding.
    Re-embedding runs in the background automatically.

    Args:
        project: Project namespace. Empty = "default".
        new_embedder: Embedder in "provider/model" format.
            Examples: "ollama/nomic-embed-text", "openai/text-embedding-3-small"
        dry_run: If true, report what would happen without making changes.

    Returns:
        Migration status, chunk count queued, old and new embedder names,
        and estimated completion time in minutes.
    """
    from .embeddings import create_embedder as _create_embedder
    import os as _os

    proj = _normalize_project(project)
    engine = _get_engine(proj)

    old_embedder = engine.db.get_meta("embedder_name") or "unknown"
    total_chunks = engine.db.get_pending_embedding_count(proj)

    if dry_run:
        # Count all chunks (not just pending) for the dry-run estimate
        with engine.db.pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks c "
                "JOIN memories m ON m.id = c.memory_id "
                "WHERE m.project = %s",
                (proj,),
            ).fetchone()
        all_chunks = row["c"] if row else 0
        return {
            "status": "dry_run",
            "old_embedder": old_embedder,
            "new_embedder": new_embedder or "(current)",
            "chunks_would_be_queued": all_chunks,
        }

    if not new_embedder:
        return {"error": "new_embedder is required"}

    # Null all embeddings — triggers BM25-only fallback during migration
    nulled = engine.db.null_all_embeddings(proj)

    # Parse "provider/model" format
    parts = new_embedder.split("/", 1)
    provider = parts[0] if len(parts) > 1 else "ollama"
    model = parts[1] if len(parts) > 1 else parts[0]

    # Create the new embedder to read its .name and .dimensions, then restore env
    old_provider_env = _os.environ.get("ENGRAM_EMBEDDER")
    old_model_env = _os.environ.get("ENGRAM_OLLAMA_MODEL") or _os.environ.get("ENGRAM_OPENAI_MODEL")
    try:
        _os.environ["ENGRAM_EMBEDDER"] = provider
        if provider == "ollama":
            _os.environ["ENGRAM_OLLAMA_MODEL"] = model
        elif provider == "openai":
            _os.environ["ENGRAM_OPENAI_MODEL"] = model
        new_emb = _create_embedder()
        engine.db.set_meta("embedder_name", new_emb.name)
        engine.db.set_meta("embedder_dimensions", str(new_emb.dimensions))
        engine.db.set_meta("embedder_version", getattr(new_emb, "version", "unknown"))
    finally:
        # Restore env — engine will use updated project_meta from now on
        if old_provider_env is not None:
            _os.environ["ENGRAM_EMBEDDER"] = old_provider_env
        elif "ENGRAM_EMBEDDER" in _os.environ:
            del _os.environ["ENGRAM_EMBEDDER"]

    # Set migration flag and timestamp
    from datetime import datetime, timezone
    engine.db.set_meta("embedding_migration_in_progress", "true")
    engine.db.set_meta("embedding_migration_started_at", datetime.now(timezone.utc).isoformat())

    # Restart the reembedder with the new embedder instance
    engine._reembedder.stop()
    from .reembedder import BackgroundReembedder
    engine._reembedder = BackgroundReembedder(db=engine.db, embedder=new_emb, project=proj)
    engine._reembedder.start()

    # Rough estimate: ~0.5s per batch of 20
    estimated_minutes = round((nulled / 20) * 0.5 / 60, 1)

    return {
        "status": "migration_started",
        "chunks_queued": nulled,
        "old_embedder": old_embedder,
        "new_embedder": new_emb.name,
        "estimated_minutes": estimated_minutes,
    }


@mcp.tool()
def memory_export_all(
    output_path: str = "./engram-export",
    include_compressed: bool = False,
) -> dict:
    """Export ALL memories from ALL projects to a portable ZIP archive.

    Use this as the uninstall story: walk away with everything in portable markdown.

    Structure:
        engram-export-<timestamp>/
            <project>/
                001-type-id.md
                ...
            README.md
            manifest.json
        engram-export-<timestamp>.zip

    Args:
        output_path: Directory to write export into (default: ./engram-export).
        include_compressed: Include compression metadata in YAML frontmatter.
    """
    from datetime import datetime, timezone
    import json
    import zipfile
    from pathlib import Path

    from .markdown_io import dump_all_projects, create_export_readme

    # Use the global engine to get a DB handle, then access all projects
    engine = _get_engine("global")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-") + "Z"
    export_dir_name = f"engram-export-{timestamp}"
    base_path = Path(output_path)
    export_dir = base_path / export_dir_name

    try:
        manifest = dump_all_projects(engine.db, export_dir, include_compressed=include_compressed)
        readme_path = create_export_readme(manifest, export_dir)

        # Write manifest.json
        manifest_path = export_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Create ZIP
        zip_path = base_path / f"{export_dir_name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in export_dir.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, arcname=file_path.relative_to(base_path))

        return {
            "status": "exported",
            "projects": list(manifest["projects"].keys()),
            "total_memories": manifest["total_memories"],
            "export_dir": str(export_dir),
            "zip_path": str(zip_path),
        }
    except Exception as e:
        logger.exception("Export failed")
        return {"error": str(e)}


@mcp.tool()
def memory_import_claudemd(
    file_path: str,
    project: str = "global",
    dry_run: bool = False,
    backup_dir: str = "",
) -> dict:
    """Import non-operational memories from a CLAUDE.md-style file.

    Extracts lessons, patterns, and anti-patterns from structured headings.
    Does NOT import behavioral rules, workflow steps, or CLI commands.
    Creates a pre-import backup ZIP before writing anything.

    Args:
        file_path: Path to CLAUDE.md or similar markdown file.
        project: Target project namespace (default: "global").
        dry_run: Preview extracted memories without writing.
        backup_dir: Where to write pre-import backup ZIP (default: same dir as file_path).
    """
    from pathlib import Path
    project = _normalize_project(project)

    if not Path(file_path).exists():
        return {"error": f"File not found: {file_path}"}

    from .markdown_io import (
        create_snapshot_zip,
        dump_memories_to_directory,
        parse_claudemd_memories,
    )

    extracted = parse_claudemd_memories(file_path, project=project)

    if dry_run:
        return {
            "status": "dry_run",
            "extracted": len(extracted),
            "preview": [
                {"content": m.content[:100], "type": m.memory_type.value,
                 "tags": m.tags, "importance": m.importance}
                for m in extracted[:10]
            ],
        }

    if not extracted:
        return {"status": "no_memories_found", "extracted": 0}

    engine = _get_engine(project)

    # Pre-import backup
    import tempfile
    backup_path = None
    try:
        current_memories = engine.db.list_memories(
            memory_type=None, tags=[], min_importance=4, limit=10000
        )
        out_dir = backup_dir or str(Path(file_path).parent)
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_memories_to_directory(current_memories, tmp_dir)
            backup_zip = create_snapshot_zip(tmp_dir, current_memories, out_dir)
            backup_path = str(backup_zip)
    except Exception as e:
        logger.warning(f"Pre-import backup failed (continuing): {e}")

    # Store extracted memories
    stored = 0
    failed = 0
    for memory in extracted:
        try:
            engine.store(memory)
            stored += 1
        except Exception as e:
            failed += 1
            logger.warning(f"Failed to store memory: {e}")

    return {
        "status": "imported",
        "extracted": len(extracted),
        "stored": stored,
        "failed": failed,
        "pre_import_backup": backup_path,
    }


@mcp.tool()
def memory_dump(project: str = "", output_path: str = "./memory-dump") -> dict:
    """Export all memories from a project as markdown files.

    Creates a directory with all memories serialized as .md files with YAML frontmatter.
    Use this for backups or when uninstalling Engram.

    Args:
        project: Project namespace to dump (e.g. "my-app"). Empty = "default".
        output_path: Directory to write markdown files. Default: ./memory-dump

    Returns:
        Count of memories dumped and output directory path.
    """
    from .markdown_io import dump_memories_to_directory

    logger.debug("memory_dump called: project=%s, output=%s", project, output_path)
    engine = _get_engine(project or None)

    # List all memories in the project
    memories = engine.db.list_memories(limit=10_000)

    if not memories:
        return {
            "status": "no_memories",
            "project": engine.db.project,
            "count": 0,
            "message": f"No memories found in project '{engine.db.project}'",
        }

    # Dump to markdown
    count = dump_memories_to_directory(memories, output_path)

    return {
        "status": "dumped",
        "project": engine.db.project,
        "count": count,
        "output_path": str(output_path),
    }


@mcp.tool()
def memory_ingest(
    project: str = "",
    directory: str = "./memory-ingest",
    memory_type: str = "",
    importance: int = 2,
    backup_dir: str = "",
    skip_existing_ids: bool = True,
) -> dict:
    """Import markdown files from a directory as memories.

    Reads all .md files with YAML frontmatter from a directory and stores them as memories.
    Creates a pre-import backup ZIP of existing memories BEFORE writing anything, so
    existing data is always preserved.

    Args:
        project: Project namespace to ingest into (e.g. "my-app"). Empty = "default".
        directory: Directory containing markdown files to import.
        memory_type: Optional type filter - only ingest memories of this type.
        importance: Optional importance override for all ingested memories.
        backup_dir: Where to write the pre-import backup ZIP. Defaults to the source directory.
        skip_existing_ids: If True (default), skip memories whose ID already exists in the DB.
            This makes ingest non-destructive — existing memories are never overwritten.

    Returns:
        Count of memories ingested, path to pre-import backup zip, skipped count, and any failed files.
    """
    import tempfile
    from pathlib import Path

    from .markdown_io import (
        create_snapshot_zip,
        dump_memories_to_directory,
        ingest_memories_from_directory,
    )

    logger.debug("memory_ingest called: project=%s, directory=%s", project, directory)
    engine = _get_engine(project or None)

    # --- Pre-import backup: snapshot existing DB state before writing anything ---
    pre_import_backup: str | None = None
    backup_warning: str | None = None
    try:
        current_memories = engine.db.list_memories(
            memory_type=None, tags=[], min_importance=4, limit=100_000
        )
        out_dir = backup_dir or directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            dump_memories_to_directory(current_memories, tmp_dir)
            backup_zip = create_snapshot_zip(tmp_dir, current_memories, out_dir)
            pre_import_backup = str(backup_zip)
            logger.info(f"Pre-import backup created: {pre_import_backup}")
    except Exception as e:
        backup_warning = f"Pre-import backup failed: {e}"
        logger.warning(backup_warning)

    # Parse markdown files
    memories, failed = ingest_memories_from_directory(directory, project=engine.db.project)

    if not memories:
        return {
            "status": "no_memories",
            "project": engine.db.project,
            "count": 0,
            "failed": len(failed),
            "message": f"No memories to ingest from {directory}",
            "pre_import_backup": pre_import_backup,
            "backup_warning": backup_warning,
        }

    # Apply type filter if specified
    if memory_type:
        try:
            mt = MemoryType(memory_type)
            memories = [m for m in memories if m.memory_type == mt]
        except ValueError:
            pass

    # Apply importance override if specified
    if importance is not None and importance != 2:
        for m in memories:
            m.importance = max(0, min(4, importance))

    # Skip memories whose ID already exists in the DB (non-destructive ingest)
    skipped_existing = 0
    if skip_existing_ids:
        existing_ids = engine.db.get_all_memory_ids(engine.db.project)
        to_import = []
        for m in memories:
            if m.id in existing_ids:
                skipped_existing += 1
                logger.debug(f"Skipping existing memory {m.id}")
            else:
                to_import.append(m)
        memories = to_import

    # Store all memories
    stored_count = 0
    for memory in memories:
        try:
            engine.store(memory)
            stored_count += 1
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    # Create source-files snapshot zip (existing behavior — documents what was imported)
    snapshot_zip = create_snapshot_zip(directory, memories, ".")
    logger.info(f"Created snapshot zip: {snapshot_zip}")

    return {
        "status": "ingested",
        "project": engine.db.project,
        "count": stored_count,
        "skipped_existing": skipped_existing,
        "failed": len(failed),
        "failed_files": failed,
        "pre_import_backup": pre_import_backup,
        "backup_warning": backup_warning,
        "snapshot_zip": str(snapshot_zip),
    }


@mcp.prompt()
def onboarding(project: str = "") -> str:
    """Get a quick-start guide for using engram effectively. Call this if you're
    unsure how to use the memory system or want a refresher on best practices.

    Args:
        project: Project namespace to show stats for. Empty = "default".
    """
    proj = (project or "default").strip().lower()
    engine = _get_engine(proj)
    stats = engine.db.get_stats()
    s = stats.model_dump()

    mem_count = s.get("total_memories", 0)
    is_new = mem_count == 0

    header = (
        f"# Engram Quick-Start -- project: `{proj}`\n\n"
        f"**Memory DB status:** {mem_count} memories, "
        f"{s.get('total_chunks', 0)} chunks, "
        f"{s.get('total_relationships', 0)} graph edges.\n\n"
    )

    bootstrap = ""
    if is_new:
        bootstrap = (
            "## NEW PROJECT -- Bootstrap Required\n\n"
            "This project has zero memories. You should store foundational context:\n\n"
            "1. **What is this project?** Purpose, goals, current status.\n"
            "2. **Tech stack:** Languages, frameworks, databases, infra.\n"
            "3. **Architecture:** Key patterns, data flow, directory structure.\n"
            "4. **Conventions:** Coding style, naming, testing approach.\n"
            "5. **Known issues:** Current bugs, tech debt, gotchas.\n\n"
            "Use type `architecture` for #2-3, type `context` for #1, "
            "type `preference` for #4, type `error` for #5.\n\n"
            "Also recall from project=`global` for user-wide preferences.\n\n"
        )

    workflow = (
        "## Your Workflow\n\n"
        f"1. **Recall first:** `memory_recall('topic', project='{proj}')`\n"
        f"2. **Also check global:** `memory_recall('topic', project='global')`\n"
        "3. **Work:** Use recalled context to inform your decisions.\n"
        f"4. **Store:** `memory_store('...', project='{proj}')`\n"
        f"5. **Connect:** `memory_connect(src, tgt, project='{proj}')`\n"
        "6. **Feedback:** Mark recall results helpful/unhelpful.\n\n"
    )

    types_and_tips = (
        "## Memory Types\n\n"
        "| Type | Use for |\n"
        "|------|--------|\n"
        "| decision | Choices made and their reasoning |\n"
        "| pattern | Recurring code/architecture patterns |\n"
        "| error | Bugs, gotchas, and their fixes |\n"
        "| architecture | System design, data flow, integrations |\n"
        "| preference | User preferences and conventions |\n"
        "| context | General project/environment context |\n\n"
        "## Project Scoping\n\n"
        f"- **This project:** `project='{proj}'` -- for project-specific memories.\n"
        "- **User-wide:** `project='global'` -- for preferences that apply everywhere.\n"
        "- Never mix: don't store project-specific decisions in global, or vice versa.\n\n"
        "## Tips\n\n"
        "- Be specific. 'Auth uses JWT' < 'Auth uses RS256 JWT issued by /api/login "
        "with 24h expiry in httpOnly cookie.'\n"
        "- Always add tags. Future recall depends on them.\n"
        "- Use importance 0-1 sparingly -- only for things that should never be pruned.\n"
    )

    return header + bootstrap + workflow + types_and_tips


def _wrap_with_api_key_auth(app, api_key: str):
    """ASGI middleware that rejects requests missing a valid Bearer token.

    Uses constant-time comparison to prevent timing side-channel attacks.
    Validates all scope types: http, websocket, and lifespan.
    """
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
            resp = JSONResponse({"error": "unauthorized"}, status_code=403)
            await resp(scope, receive, send)
            return
        await app(scope, receive, send)

    return auth_middleware


def main(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8788,
    api_key: str | None = None,
) -> None:
    """Start the engram MCP server.

    Args:
        transport: "stdio" for local subprocess, "sse" for legacy SSE, or
                   "streamable-http" for stateless HTTP (recommended for Docker).
        host: Bind address for network transports.
        port: Port for network transports.
        api_key: Optional Bearer token for auth.
    """
    if transport == "stdio":
        mcp.run()
    elif transport in ("sse", "streamable-http"):
        import atexit

        import anyio
        import uvicorn

        # Disable DNS rebinding protection for network access
        mcp.settings.transport_security = None

        if transport == "streamable-http":
            # Stateless per-request transport: no session state, survives server restarts.
            # Recommended over SSE for Docker deployments — eliminates stale session issues.
            app = mcp.streamable_http_app()
            startup_msg = f"Starting engram streamable-HTTP server on {host}:{port}/mcp"
        else:
            app = mcp.sse_app()
            startup_msg = f"Starting engram SSE server on {host}:{port}"

        if api_key:
            app = _wrap_with_api_key_auth(app, api_key)

        def _shutdown():
            for engine in _engines.values():
                engine.db.close()
            _engines.clear()

        atexit.register(_shutdown)

        print(startup_msg)
        config = uvicorn.Config(app, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        anyio.run(server.serve)


if __name__ == "__main__":
    main()
