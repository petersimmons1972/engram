"""Tests for engram type models — ID generation and field defaults."""

from engram.types import Chunk, Memory, Relationship, SearchResult


class TestSearchResultFields:
    """Phase 1 — Store Big: SearchResult gains chunk_score and matched_chunk_index."""

    def test_search_result_default_chunk_score(self):
        """SearchResult without chunk_score defaults to 0.0."""
        mem = Memory(content="test content")
        result = SearchResult(memory=mem, score=0.5)
        assert result.chunk_score == 0.0

    def test_search_result_default_matched_chunk_index(self):
        """SearchResult without matched_chunk_index defaults to -1."""
        mem = Memory(content="test content")
        result = SearchResult(memory=mem, score=0.5)
        assert result.matched_chunk_index == -1

    def test_search_result_accepts_chunk_score(self):
        """SearchResult accepts an explicit chunk_score."""
        mem = Memory(content="test content")
        result = SearchResult(memory=mem, score=0.5, chunk_score=0.87)
        assert result.chunk_score == 0.87

    def test_search_result_accepts_matched_chunk_index(self):
        """SearchResult accepts an explicit matched_chunk_index."""
        mem = Memory(content="test content")
        result = SearchResult(memory=mem, score=0.5, matched_chunk_index=2)
        assert result.matched_chunk_index == 2

    def test_old_construction_without_new_fields_still_works(self):
        """Existing callers that omit chunk_score and matched_chunk_index are unaffected."""
        mem = Memory(content="backward compatible content")
        result = SearchResult(
            memory=mem,
            score=0.75,
            matched_chunk="some chunk text",
        )
        assert result.chunk_score == 0.0
        assert result.matched_chunk_index == -1
        assert result.matched_chunk == "some chunk text"


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
