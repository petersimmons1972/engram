"""Engram MCP Server -- persistent three-layer memory for AI agents."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict, defaultdict, deque

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

from .db import create_database
from .embeddings import create_embedder
from .errors import EmbeddingConfigMismatchError
from .search import SearchEngine
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

MAX_ENGINE_CACHE_SIZE = 64
_engines: OrderedDict[str, SearchEngine] = OrderedDict()
_engines_lock = threading.Lock()

# Rate limiting for memory_store — sliding window per project
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = int(os.environ.get("ENGRAM_RATE_LIMIT", "100"))  # calls per window
_store_calls: dict[str, deque] = defaultdict(deque)
_rate_limit_lock = threading.Lock()


def _get_engine(project: str | None = None) -> SearchEngine:
    """Return (or create) a SearchEngine for the given project.

    Each project gets its own SQLite database file (~/.engram/{project}.db),
    so memories are fully isolated between projects. Uses LRU eviction to
    bound cache size to _MAX_ENGINES entries.
    """
    with _engines_lock:
        raw = (project or os.environ.get("ENGRAM_PROJECT", "default")).strip().lower()
        project = re.sub(r'[^a-z0-9_-]', '', raw) or "default"
        if project in _engines:
            _engines.move_to_end(project)
            return _engines[project]
        db_dir = os.environ.get("ENGRAM_DIR", None)
        db = create_database(project=project, db_dir=db_dir)
        embedder = create_embedder()
        engine = SearchEngine(db=db, embedder=embedder)
        _engines[project] = engine
        if len(_engines) > MAX_ENGINE_CACHE_SIZE:
            _, evicted = _engines.popitem(last=False)
            evicted.db.close()
            logger.info("Evicted least-recently-used engine from cache")
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
def memory_recall(
    query: str,
    top_k: int = 5,
    memory_type: str = "",
    tags: str = "",
    min_importance: int = 4,
    graph_hops: int = 1,
    project: str = "",
) -> dict:
    """Search memories using all three layers: keyword (BM25), semantic (vector), and graph.

    Results are ranked by a composite score:
      Final = (vector * 0.45 + BM25 * 0.25 + recency * 0.15 + graph * 0.15) * importance_multiplier

    Connected memories from the knowledge graph are attached automatically.

    Args:
        query: What to search for. Can be a keyword, question, or concept.
        top_k: Number of results to return (default 5).
        memory_type: Filter by type (decision/pattern/error/context/architecture/
            preference). Empty = all.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this value (0=only critical, 4=all).
        graph_hops: How many relationship hops to traverse (1 or 2).
        project: Project namespace (e.g. "my-app"). Empty = "default".

    Returns:
        Ranked list of memories with scores, matched chunks, and connected context.
    """
    logger.debug("memory_recall called: project=%s, query=%r, top_k=%d", project, query[:50], top_k)
    engine = _get_engine(project or None)
    top_k = max(1, min(50, top_k))

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    mt = memory_type if memory_type else None
    mi = min_importance if min_importance < 4 else None

    results = engine.recall(
        query=query,
        top_k=top_k,
        memory_type=mt,
        tags=tag_list,
        min_importance=mi,
        graph_hops=max(1, min(2, graph_hops)),
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

        entry = {
            "id": r.memory.id,
            "content": r.memory.content,
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
) -> dict:
    """List recent memories with optional filters.

    Args:
        memory_type: Filter by type. Empty = all types.
        tags: Comma-separated tags to filter by. Empty = all.
        min_importance: Only return memories with importance <= this (0=only critical, 4=all).
        limit: Max number of memories to return.
        project: Project namespace (e.g. "my-app"). Empty = "default".

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

    return {
        "memories": [
            {
                "id": m.id,
                "content": m.content[:300],
                "type": m.memory_type.value,
                "tags": m.tags,
                "importance": m.importance,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat(),
                "updated_at": m.updated_at.isoformat(),
            }
            for m in memories
        ],
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
    stats = engine.db.get_stats()
    return stats.model_dump()


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
def memory_consolidate(project: str = "") -> dict:
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
    logger.debug("memory_consolidate called: project=%s", project)
    engine = _get_engine(project or None)
    result = engine.consolidate()
    return {"status": "consolidated", **result}


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
) -> dict:
    """Import markdown files from a directory as memories.

    Reads all .md files with YAML frontmatter from a directory and stores them as memories.
    Also creates an install-time snapshot zip containing original files and memory snapshot.

    Args:
        project: Project namespace to ingest into (e.g. "my-app"). Empty = "default".
        directory: Directory containing markdown files to import.
        memory_type: Optional type filter - only ingest memories of this type.
        importance: Optional importance override for all ingested memories.

    Returns:
        Count of memories ingested, path to snapshot zip, and any failed files.
    """
    from pathlib import Path

    from .markdown_io import create_snapshot_zip, ingest_memories_from_directory

    logger.debug("memory_ingest called: project=%s, directory=%s", project, directory)
    engine = _get_engine(project or None)

    # Parse markdown files
    memories, failed = ingest_memories_from_directory(directory, project=engine.db.project)

    if not memories:
        return {
            "status": "no_memories",
            "project": engine.db.project,
            "count": 0,
            "failed": len(failed),
            "message": f"No memories to ingest from {directory}",
        }

    # Apply type filter if specified
    if memory_type:
        try:
            mt = MemoryType(memory_type)
            memories = [m for m in memories if m.memory_type == mt]
        except ValueError:
            pass

    # Apply importance override if specified
    if importance and importance != 2:
        for m in memories:
            m.importance = max(0, min(4, importance))

    # Store all memories
    stored_count = 0
    for memory in memories:
        try:
            engine.store(memory)
            stored_count += 1
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")

    # Create snapshot zip
    snapshot_zip = create_snapshot_zip(directory, memories, ".")
    logger.info(f"Created snapshot zip: {snapshot_zip}")

    return {
        "status": "ingested",
        "project": engine.db.project,
        "count": stored_count,
        "failed": len(failed),
        "failed_files": failed,
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
