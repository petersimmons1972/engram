"""Tests for engram type models — ID generation and field defaults."""

from engram.types import Chunk, Memory, Relationship


class TestIDGeneration:
    """IDs must be full 32-char UUID4 hex strings, not truncated."""

    def test_memory_id_length(self):
        m = Memory(content="test")
        assert len(m.id) == 32, f"Memory ID should be 32 chars, got {len(m.id)}"

    def test_chunk_id_length(self):
        c = Chunk(memory_id="abc", chunk_text="text", chunk_index=0)
        assert len(c.id) == 32, f"Chunk ID should be 32 chars, got {len(c.id)}"

    def test_relationship_id_length(self):
        r = Relationship(source_id="a", target_id="b")
        assert len(r.id) == 32, f"Relationship ID should be 32 chars, got {len(r.id)}"

    def test_ids_are_hex(self):
        m = Memory(content="test")
        int(m.id, 16)  # raises ValueError if not valid hex

    def test_unique_ids_across_1000(self):
        ids = {Memory(content="test").id for _ in range(1000)}
        assert len(ids) == 1000, f"Expected 1000 unique IDs, got {len(ids)}"
