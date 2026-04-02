"""Tests for markdown_io round-trip serialization (issues #49, #50, #52)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from engram.markdown_io import (
    dump_memories_to_directory,
    ingest_memories_from_directory,
    markdown_to_memory,
    memory_to_markdown,
)
from engram.types import Memory, MemoryType


def _make_memory(**overrides) -> Memory:
    """Helper to create a Memory with sensible defaults."""
    defaults = {
        "id": uuid.uuid4().hex,
        "content": "Test memory content",
        "memory_type": MemoryType.DECISION,
        "project": "test-project",
        "tags": ["tag1", "tag2"],
        "importance": 1,
        "created_at": datetime(2026, 3, 25, 10, 0, 0, tzinfo=timezone.utc),
        "last_accessed": datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return Memory(**defaults)


class TestMarkdownRoundTrip:
    """Round-trip tests for memory_to_markdown / markdown_to_memory."""

    def test_content_with_triple_dash_survives(self, tmp_path):
        """Issue #50: Content containing --- as HR must survive dump/parse round-trip."""
        content = "Some text above\n\n---\n\nSome text below the HR"
        mem = _make_memory(content=content)

        md = memory_to_markdown(mem)
        result = markdown_to_memory(md, project="test-project")

        assert result is not None, "Failed to parse markdown with --- in content"
        assert result.content == content

    def test_id_preserved_on_reimport(self, tmp_path):
        """Issue #52: Memory ID must survive dump/parse round-trip."""
        original_id = uuid.uuid4().hex
        mem = _make_memory(id=original_id)

        md = memory_to_markdown(mem)
        result = markdown_to_memory(md, project="test-project")

        assert result is not None
        assert result.id == original_id, (
            f"ID changed from {original_id} to {result.id}"
        )

    def test_yaml_datetime_round_trip(self, tmp_path):
        """Issue #49: created_at and last_accessed must survive round-trip within 1s tolerance."""
        created = datetime(2026, 3, 25, 10, 30, 45, tzinfo=timezone.utc)
        accessed = datetime(2026, 3, 25, 14, 15, 20, tzinfo=timezone.utc)
        mem = _make_memory(created_at=created, last_accessed=accessed)

        md = memory_to_markdown(mem)
        result = markdown_to_memory(md, project="test-project")

        assert result is not None
        assert abs((result.created_at - created).total_seconds()) < 1, (
            f"created_at drifted: {result.created_at} vs {created}"
        )
        assert abs((result.last_accessed - accessed).total_seconds()) < 1, (
            f"last_accessed drifted: {result.last_accessed} vs {accessed}"
        )

    def test_yaml_date_only_in_frontmatter(self):
        """Issue #49: yaml.safe_load may return datetime.date for unquoted dates.

        Hand-edited frontmatter with bare dates (e.g., 2026-03-25 without quotes)
        must still parse without losing the timestamp.
        """
        md = (
            "---\n"
            "id: abc123\n"
            "type: context\n"
            "tags: []\n"
            "importance: 2\n"
            "created: 2026-03-25\n"
            "last_accessed: 2026-03-25\n"
            "project: test\n"
            "---\n\n"
            "Content here"
        )
        result = markdown_to_memory(md, project="test")
        assert result is not None
        # The date should be parsed, not silently dropped
        assert result.created_at.year == 2026
        assert result.created_at.month == 3
        assert result.created_at.day == 25

    def test_tags_round_trip(self, tmp_path):
        """Tags list must survive round-trip identically."""
        tags = ["auth", "security", "high-priority"]
        mem = _make_memory(tags=tags)

        md = memory_to_markdown(mem)
        result = markdown_to_memory(md, project="test-project")

        assert result is not None
        assert result.tags == tags

    def test_full_dump_ingest_cycle(self, tmp_path):
        """10 memories round-trip through dump/ingest with matching IDs and content."""
        originals = [
            _make_memory(
                id=uuid.uuid4().hex,
                content=f"Memory number {i} with unique content",
            )
            for i in range(10)
        ]

        count = dump_memories_to_directory(originals, tmp_path)
        assert count == 10

        ingested, failed = ingest_memories_from_directory(
            tmp_path, project="test-project"
        )
        assert len(failed) == 0, f"Failed files: {failed}"
        assert len(ingested) == 10

        original_ids = {m.id for m in originals}
        ingested_ids = {m.id for m in ingested}
        assert original_ids == ingested_ids, (
            f"ID mismatch: missing={original_ids - ingested_ids}, "
            f"extra={ingested_ids - original_ids}"
        )

        original_contents = {m.id: m.content for m in originals}
        for m in ingested:
            assert m.content == original_contents[m.id], (
                f"Content mismatch for {m.id}"
            )

    def test_frontmatter_delimiter_in_content(self, tmp_path):
        """Issue #50: Content with multiple --- delimiters must survive."""
        content = "Section 1\n\n---\n\nSection 2\n\n---\n\nSection 3"
        mem = _make_memory(content=content)

        md = memory_to_markdown(mem)
        result = markdown_to_memory(md, project="test-project")

        assert result is not None, "Failed to parse markdown with multiple --- in content"
        assert result.content == content

    def test_empty_id_gets_generated(self):
        """Issue #52: markdown_to_memory must generate an ID when none is present."""
        md = (
            "---\n"
            "type: context\n"
            "tags: []\n"
            "importance: 2\n"
            "created: '2026-03-25T10:00:00+00:00'\n"
            "last_accessed: '2026-03-25T12:00:00+00:00'\n"
            "project: test\n"
            "---\n\n"
            "Some content"
        )
        result = markdown_to_memory(md, project="test")
        assert result is not None
        assert result.id != "", "Empty ID must not survive import"
        assert len(result.id) > 0

    def test_frontmatter_with_dashes_in_yaml_value(self):
        """Issue #50: --- inside a YAML value must not break frontmatter parsing.

        This is the actual bug: split('---', 2) splits on '---' anywhere in the
        string, not just at line boundaries. A YAML value containing '---' as a
        substring breaks the parser.
        """
        # Manually craft markdown where a YAML value contains ---
        # (This simulates what happens if someone hand-edits frontmatter)
        md = (
            "---\n"
            "id: abc123\n"
            "type: context\n"
            "tags: []\n"
            "importance: 2\n"
            "created: '2026-03-25T10:00:00+00:00'\n"
            "last_accessed: '2026-03-25T12:00:00+00:00'\n"
            "project: my---project\n"
            "---\n\n"
            "Content here"
        )
        result = markdown_to_memory(md, project="test")
        assert result is not None, "Failed to parse frontmatter with --- in YAML value"
        assert result.project == "my---project"
        assert result.content == "Content here"


def test_pre_import_backup_created(tmp_path, engine):
    """create_snapshot_zip produces a dated zip file."""
    from engram.markdown_io import create_snapshot_zip, ingest_memories_from_directory
    from engram.types import Memory

    # Store a memory and write a markdown file to import
    m = Memory(content="existing memory " * 50, project="test")
    engine.store(m)

    md_file = tmp_path / "001-context-abc12345.md"
    md_file.write_text(
        "---\nid: abc12345abc12345abc12345abc12345\ntype: context\ntags: []\n"
        "importance: 2\nproject: test\ncreated: 2026-01-01T00:00:00+00:00\n"
        "last_accessed: 2026-01-01T00:00:00+00:00\n---\n\nThis is a new memory to import.\n"
    )

    memories, failed = ingest_memories_from_directory(tmp_path, project="test")
    assert len(memories) == 1

    zip_path = create_snapshot_zip(tmp_path, memories, tmp_path)
    assert zip_path.exists()
    assert zip_path.suffix == ".zip"


def test_claudemd_parser_extracts_lessons():
    """parse_claudemd_memories extracts counted lessons and skips topic refs."""
    import os
    import tempfile

    from engram.markdown_io import parse_claudemd_memories

    content = """# Claude Instructions

## Behavioral Rules
Never do X. Always do Y.

## Key Lessons

- (4x) Backup before modifying critical files
- (2x) Validate what the customer sees, not intermediate formats
- cert-manager patterns -> memory/homelab-cert-manager.md

## Workflow
1. Step one
2. Step two
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        fname = f.name

    try:
        memories = parse_claudemd_memories(fname, project="test")
        # Should extract the two counted lessons, skip the topic ref and behavioral rules
        assert len(memories) == 2
        contents = [m.content for m in memories]
        assert any("Backup before modifying" in c for c in contents)
        assert any("Validate what the customer" in c for c in contents)
        # Topic reference should be skipped
        assert not any("cert-manager" in c for c in contents)
    finally:
        os.unlink(fname)


def test_claudemd_parser_skips_operational_sections():
    """NEVER/ALWAYS/Workflow sections are not extracted."""
    import os
    import tempfile

    from engram.markdown_io import parse_claudemd_memories

    content = """# Instructions

## Critical Rules
NEVER commit secrets.
ALWAYS run tests.

## Key Lessons
- (1x) One real lesson here

## Workflow
- Do this step
- Do that step
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(content)
        fname = f.name

    try:
        memories = parse_claudemd_memories(fname, project="test")
        assert len(memories) == 1
        assert "One real lesson" in memories[0].content
    finally:
        os.unlink(fname)


def test_export_all_creates_readme(tmp_path, engine):
    """dump_all_projects + create_export_readme produce correct outputs."""
    from engram.markdown_io import create_export_readme, dump_all_projects
    from engram.types import Memory

    # Store a memory to ensure there's at least one project
    m = Memory(content="A test memory for export", project="test")
    engine.store(m)

    export_dir = tmp_path / "export"
    manifest = dump_all_projects(engine.db, export_dir)
    assert "projects" in manifest
    assert "total_memories" in manifest

    readme = create_export_readme(manifest, export_dir)
    assert readme.exists()
    content = readme.read_text()
    assert "Re-import instructions" in content
