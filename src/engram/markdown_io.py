"""Markdown serialization and deserialization for memory dump/ingest.

Handles bidirectional conversion between Memory objects and markdown files with YAML frontmatter.
Also creates install-time snapshot zips to prove data preservation at ingest time.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .types import Memory, MemoryType

logger = logging.getLogger(__name__)


def memory_to_markdown(memory: Memory) -> str:
    """Convert a Memory object to markdown with YAML frontmatter.

    Format:
        ---
        id: abc123
        type: decision
        tags: [auth, security]
        importance: 1
        created: 2026-03-25T10:00:00Z
        last_accessed: 2026-03-25T12:00:00Z
        ---

        # Memory content (title is first line if available)

        Rest of the content...
    """
    frontmatter = {
        "id": memory.id,
        "type": memory.memory_type.value,
        "tags": memory.tags,
        "importance": memory.importance,
        "created": memory.created_at.isoformat(),
        "last_accessed": memory.last_accessed.isoformat(),
        "project": memory.project,
    }

    # Use safe_dump to ensure consistent round-trip serialization (issue #49)
    yaml_str = yaml.safe_dump(frontmatter, default_flow_style=False, sort_keys=False)

    return f"---\n{yaml_str}---\n\n{memory.content}\n"


def _parse_timestamp(value: str | datetime | date) -> datetime:
    """Parse a timestamp from YAML frontmatter, handling str, datetime, and date types.

    yaml.safe_load may return:
    - str: ISO format like '2026-03-25T10:00:00+00:00'
    - datetime: when the value looks like a datetime without quotes
    - date: when the value looks like a bare date (e.g., 2026-03-25)
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    # str
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def markdown_to_memory(markdown_content: str, project: str = "default") -> Memory | None:
    """Parse markdown with YAML frontmatter back into a Memory object.

    Returns None if parsing fails.
    """
    try:
        # Split frontmatter from content using regex to match --- at line boundaries
        # This avoids breaking on --- appearing inside YAML values or content
        fm_match = re.match(r"\A---\n(.*?\n)---\n(.*)\Z", markdown_content, re.DOTALL)
        if not fm_match:
            logger.warning("Markdown does not contain valid YAML frontmatter block")
            return None

        yaml_str = fm_match.group(1)
        content = fm_match.group(2).strip()

        # Parse YAML frontmatter
        try:
            frontmatter = yaml.safe_load(yaml_str)
        except yaml.YAMLError as e:
            logger.warning("Failed to parse YAML frontmatter: %s", e)
            return None

        if not isinstance(frontmatter, dict):
            logger.warning("Frontmatter did not parse to a dict")
            return None

        # Extract fields with defaults
        memory_type_str = frontmatter.get("type", "context")
        try:
            memory_type = MemoryType(memory_type_str)
        except ValueError:
            memory_type = MemoryType.CONTEXT

        # Issue #52: Generate a new ID if none is present in frontmatter
        memory_id = frontmatter.get("id", "") or uuid.uuid4().hex

        memory = Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            project=frontmatter.get("project", project),
            tags=frontmatter.get("tags", []),
            importance=int(frontmatter.get("importance", 2)),
        )

        # Update timestamps if present in frontmatter
        # Issue #49: yaml.safe_load may return str, datetime, or date objects
        if "created" in frontmatter:
            memory.created_at = _parse_timestamp(frontmatter["created"])
        if "last_accessed" in frontmatter:
            memory.last_accessed = _parse_timestamp(frontmatter["last_accessed"])

        return memory

    except Exception as e:
        logger.error("Error parsing markdown: %s", e)
        return None


def dump_memories_to_directory(memories: list[Memory], output_dir: str | Path) -> int:
    """Dump all memories as markdown files to a directory.

    Returns the count of memories written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, memory in enumerate(memories, 1):
        # Sanitize filename: use project-type-id
        filename = f"{i:03d}-{memory.memory_type.value}-{memory.id[:8]}.md"
        filepath = output_path / filename

        try:
            markdown = memory_to_markdown(memory)
            filepath.write_text(markdown, encoding="utf-8")
            count += 1
            logger.debug(f"Wrote memory {memory.id} to {filepath}")
        except Exception as e:
            logger.error(f"Failed to write memory {memory.id}: {e}")

    logger.info(f"Dumped {count} memories to {output_path}")
    return count


def ingest_memories_from_directory(
    source_dir: str | Path, project: str = "default"
) -> tuple[list[Memory], list[str]]:
    """Ingest markdown files from a directory into Memory objects.

    Returns (memories_list, failed_files).
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        logger.error(f"Source directory does not exist: {source_path}")
        return [], []

    memories = []
    failed_files = []

    # Find all .md files
    md_files = sorted(source_path.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {source_path}")

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            memory = markdown_to_memory(content, project=project)

            if memory:
                memories.append(memory)
                logger.debug(f"Ingested {md_file.name}")
            else:
                failed_files.append(md_file.name)
                logger.warning(f"Failed to parse {md_file.name}")

        except Exception as e:
            failed_files.append(md_file.name)
            logger.error(f"Error reading {md_file.name}: {e}")

    logger.info(f"Ingested {len(memories)} memories, {len(failed_files)} failed")
    return memories, failed_files


def create_snapshot_zip(
    source_dir: str | Path,
    memories: list[Memory],
    output_dir: str | Path,
) -> Path:
    """Create an install-time snapshot zip containing source files, manifest, and memory snapshot.

    Returns path to the created zip file.

    Zip structure:
        source-files/
            <original files from source_dir>
        manifest.json
        memory-snapshot.json
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename: memory-backup-YYYY-MM-DDTHH-MM-SSZ.zip
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(":", "-") + "Z"
    zip_filename = f"memory-backup-{timestamp}.zip"
    zip_path = output_path / zip_filename

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all source files to source-files/ prefix
        if source_path.exists():
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    arcname = f"source-files/{file_path.relative_to(source_path)}"
                    zf.write(file_path, arcname=arcname)
                    logger.debug(f"Added {arcname} to zip")

        # Create and add manifest.json
        manifest: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_directory": str(source_path),
            "file_count": sum(1 for _ in source_path.rglob("*") if _.is_file()),
            "memory_count": len(memories),
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Create and add memory snapshot
        memory_snapshot = [
            {
                "id": m.id,
                "type": m.memory_type.value,
                "tags": m.tags,
                "importance": m.importance,
                "created": m.created_at.isoformat(),
                "project": m.project,
                "content_length": len(m.content),
            }
            for m in memories
        ]
        zf.writestr("memory-snapshot.json", json.dumps(memory_snapshot, indent=2))

    logger.info(f"Created snapshot zip: {zip_path}")
    return zip_path


def parse_claudemd_memories(
    file_path: "str | Path",
    project: str = "global",
) -> "list[Memory]":
    """Extract non-operational memories from a CLAUDE.md-style file.

    Extracts lessons/patterns from:
    - Bullet lists under headings containing: Lessons, Learned, Anti-Pattern, Key, Pattern
    - Lines matching "(Nx) <lesson text>" format (usage-counted lessons)

    Skips operational content:
    - Behavioral rules (NEVER/ALWAYS sections)
    - Workflow steps, CLI commands
    - Topic file references (lines with " -> " or "→" pointers)
    - Code blocks
    """
    import re as _re
    from pathlib import Path as _Path

    content = _Path(file_path).read_text(encoding="utf-8")
    memories: list[Memory] = []

    # Section headings that contain extractable lessons
    LESSON_HEADING_KEYWORDS = _re.compile(
        r"lesson|learned|anti.?pattern|key lesson|patterns?|quick ref",
        _re.IGNORECASE,
    )
    # Headings to skip (operational)
    SKIP_HEADING_KEYWORDS = _re.compile(
        r"never|always|workflow|rules|critical|decisions|pre.?flight|behavior",
        _re.IGNORECASE,
    )
    # Usage-counted lesson format: "(4x) Lesson text"
    COUNTED_LESSON = _re.compile(r"^\s*\((\d+)x\)\s+(.+)$")
    # Topic file reference (skip these)
    TOPIC_REF = _re.compile(r"[→]|->")
    # Code block detection
    in_code_block = False

    current_tags: list[str] = []
    in_lesson_section = False

    for line in content.splitlines():
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # Detect section headings
        heading_match = _re.match(r"^#{1,3}\s+(.+)$", line)
        if heading_match:
            current_heading = heading_match.group(1).strip()
            if LESSON_HEADING_KEYWORDS.search(current_heading):
                in_lesson_section = True
                # Derive tags from heading words
                words = _re.sub(r"[^a-zA-Z0-9 ]", "", current_heading).lower().split()
                current_tags = [w for w in words if len(w) > 3][:3]
            elif SKIP_HEADING_KEYWORDS.search(current_heading):
                in_lesson_section = False
            else:
                in_lesson_section = False
            continue

        if not in_lesson_section:
            continue

        # Skip topic file references
        if TOPIC_REF.search(line):
            continue

        # Usage-counted format: "(4x) lesson text"
        counted = COUNTED_LESSON.match(line)
        if counted:
            count = int(counted.group(1))
            lesson_text = counted.group(2).strip()
            # Higher count = more important = lower importance number
            importance = max(1, 3 - min(count // 2, 2))
            memories.append(Memory(
                content=lesson_text,
                memory_type=MemoryType.PATTERN,
                project=project,
                tags=current_tags + ["lessons-learned"],
                importance=importance,
            ))
            continue

        # Regular bullet point
        bullet_match = _re.match(r"^\s*[-*]\s+(.+)$", line)
        if bullet_match:
            lesson_text = bullet_match.group(1).strip()
            if len(lesson_text) < 20:  # skip trivially short bullets
                continue
            memories.append(Memory(
                content=lesson_text,
                memory_type=MemoryType.PATTERN,
                project=project,
                tags=current_tags + ["lessons-learned"],
                importance=2,
            ))

    return memories


def dump_all_projects(
    db: Any,
    output_dir: "str | Path",
    include_compressed: bool = False,
) -> dict:
    """Dump all memories from all projects to per-project subdirectories.

    Returns manifest dict with project names and memory counts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    projects = db.list_all_projects()
    manifest: dict[str, Any] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "projects": {},
        "total_memories": 0,
    }

    for project_name in projects:
        memories = db.list_memories(
            memory_type=None, tags=[], min_importance=4, limit=100000,
        )
        project_dir = output_path / project_name
        count = dump_memories_to_directory(memories, project_dir)
        manifest["projects"][project_name] = count
        manifest["total_memories"] += count

    return manifest


def create_export_readme(manifest: dict, output_dir: "str | Path") -> Path:
    """Write a README.md to the export directory with re-import instructions."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    projects = manifest.get("projects", {})
    project_lines = "\n".join(
        f"    engram ingest --project {p} --directory ./{p}/"
        for p in sorted(projects.keys())
    )
    content = f"""# Engram Memory Export

Exported: {manifest['exported_at']} | Projects: {len(projects)} | Memories: {manifest['total_memories']}

## Re-import instructions

To restore to a fresh Engram instance:

{project_lines}

## Format

Each .md file has YAML frontmatter (id, type, tags, importance, created, project)
followed by the memory content. Compatible with `memory_ingest` / `engram ingest`.

## Projects in this export

{chr(10).join(f"- **{p}**: {c} memories" for p, c in sorted(projects.items()))}
"""
    readme_path = output_path / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    return readme_path
