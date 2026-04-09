"""Tests for engram.chunker -- text chunking, hashing, and deduplication."""

from __future__ import annotations

from engram.chunker import chunk_hash, chunk_text, is_duplicate, jaccard_similarity


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.")
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_long_text_splits_into_multiple(self):
        # Need enough content to exceed the LAZY_CHUNK_THRESHOLD (8000 chars)
        sentences = [f"Sentence number {i} is here for testing purposes." for i in range(300)]
        text = " ".join(sentences)
        assert len(text) > 8000  # must exceed lazy chunking threshold
        chunks = chunk_text(text, max_tokens=100)
        assert len(chunks) > 1

    def test_overlap_preserves_context(self):
        sentences = [f"Important fact {i} for context." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=80, overlap_tokens=20)

        if len(chunks) >= 2:
            words_first = set(chunks[0].split())
            words_second = set(chunks[1].split())
            overlap = words_first & words_second
            assert len(overlap) > 0, "Chunks should have overlapping words"

    def test_single_long_sentence(self):
        text = "word " * 1000
        chunks = chunk_text(text.strip(), max_tokens=100)
        assert len(chunks) >= 1


class TestChunkHash:
    def test_deterministic(self):
        h1 = chunk_hash("Hello world")
        h2 = chunk_hash("Hello world")
        assert h1 == h2

    def test_whitespace_normalized(self):
        h1 = chunk_hash("hello  world")
        h2 = chunk_hash("hello world")
        assert h1 == h2

    def test_case_normalized(self):
        h1 = chunk_hash("Hello World")
        h2 = chunk_hash("hello world")
        assert h1 == h2

    def test_different_text_different_hash(self):
        h1 = chunk_hash("alpha")
        h2 = chunk_hash("beta")
        assert h1 != h2


class TestJaccardSimilarity:
    def test_identical_strings(self):
        assert jaccard_similarity("hello world", "hello world") == 1.0

    def test_completely_different(self):
        assert jaccard_similarity("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        sim = jaccard_similarity("the quick brown fox", "the slow brown dog")
        assert 0.0 < sim < 1.0

    def test_empty_string(self):
        assert jaccard_similarity("", "hello") == 0.0
        assert jaccard_similarity("", "") == 0.0


class TestIsDuplicate:
    def test_exact_duplicate(self):
        existing = ["Hello world this is a test"]
        assert is_duplicate("Hello world this is a test", existing) is True

    def test_near_duplicate(self):
        # 13/15 unique words overlap -> Jaccard ~0.867, above the 0.85 threshold
        existing = ["alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"]
        candidate = (
            "alpha beta gamma delta epsilon zeta eta theta"
            " iota kappa lambda mu nu omicron"
        )
        assert is_duplicate(candidate, existing) is True

    def test_not_duplicate(self):
        existing = ["Quantum computing fundamentals"]
        assert is_duplicate("Web development with Python", existing) is False

    def test_empty_existing(self):
        assert is_duplicate("Anything", []) is False


class TestLazyChunking:
    """B5: Content under 8000 chars should skip chunking and be stored as a single chunk."""

    def test_short_content_produces_single_chunk(self):
        """500-char content with sentences -> exactly 1 chunk even with small max_tokens."""
        sentences = [f"Fact number {i}." for i in range(30)]
        text = " ".join(sentences)
        assert len(text) < 8000
        # With a small max_tokens, the chunker would normally split this.
        # Lazy chunking should skip splitting entirely for content under 8000 chars.
        chunks = chunk_text(text, max_tokens=20)
        assert len(chunks) == 1, f"Short content should produce 1 chunk, got {len(chunks)}"
        assert chunks[0] == text

    def test_long_content_still_chunks(self):
        """10000-char content -> more than 1 chunk."""
        sentences = [
            f"Sentence number {i} is a fairly long sentence for testing purposes."
            for i in range(200)
        ]
        text = " ".join(sentences)
        assert len(text) > 10000
        chunks = chunk_text(text)
        assert len(chunks) > 1

    def test_threshold_boundary(self):
        """8000 chars -> 1 chunk. 8001+ chars -> chunking algorithm runs."""
        # Build text of exactly ~8000 chars with sentence boundaries
        sentences = []
        total = 0
        i = 0
        while total + len(f"Sentence number {i}.") + 1 <= 8000:
            s = f"Sentence number {i}."
            sentences.append(s)
            total += len(s) + (1 if sentences else 0)
            i += 1
        text_at = " ".join(sentences)
        assert len(text_at) <= 8000
        # Even with tiny max_tokens, should produce 1 chunk (lazy skip)
        chunks_at = chunk_text(text_at, max_tokens=20)
        assert len(chunks_at) == 1, f"8000 chars should produce 1 chunk, got {len(chunks_at)}"

        # 8001+ chars with sentence boundaries: chunking algorithm runs
        sentences_over = list(sentences)
        while len(" ".join(sentences_over)) <= 8000:
            sentences_over.append(f"Sentence number {i}.")
            i += 1
        text_over = " ".join(sentences_over)
        assert len(text_over) > 8000
        chunks_over = chunk_text(text_over, max_tokens=20)
        # With max_tokens=20 (80 chars), content >8000 chars should produce multiple chunks
        assert len(chunks_over) > 1, f"Content over 8000 chars should chunk, got {len(chunks_over)}"


class TestLazyChunkThreshold:
    """Phase 1 — Store Big: threshold raised from 2000 to 8000 chars."""

    def test_content_just_under_8000_is_single_chunk(self):
        """7999-char content must NOT be chunked (returned as-is)."""
        # Build exactly 7999 chars of valid sentence content.
        # Use a fixed sentence, fill to length.
        base = "Sentence for threshold boundary test. "
        text = (base * 300)[:7999]
        # Strip trailing partial word to avoid odd results -- just use what we have.
        chunks = chunk_text(text, max_tokens=20)
        assert len(chunks) == 1, (
            f"7999-char content should produce 1 chunk, got {len(chunks)}"
        )

    def test_content_just_over_8000_is_chunked(self):
        """8001-char content MUST be chunked (threshold is now 8000)."""
        base = "Sentence for threshold boundary test. "
        text = (base * 300)[:8001]
        # With very small max_tokens, content over threshold should split.
        chunks = chunk_text(text, max_tokens=20)
        assert len(chunks) > 1, (
            f"8001-char content should produce multiple chunks, got {len(chunks)}"
        )


class TestChunkLengthAccuracy:
    """Regression tests for #39: Space separators not counted in chunk length."""

    def test_chunks_do_not_exceed_max_chars(self):
        """Every chunk's actual length must be <= max_tokens * 4 chars.

        Content must exceed LAZY_CHUNK_THRESHOLD (8000 chars) so the chunking
        algorithm actually runs rather than the lazy single-chunk path.
        """
        # Use 400 sentences to ensure we exceed the 8000-char lazy threshold
        sentences = [f"Sentence number {i} is here for testing purposes." for i in range(400)]
        text = " ".join(sentences)
        assert len(text) > 8000, "Test setup: text must exceed lazy chunk threshold"
        max_tokens = 50  # 200 chars
        chunks = chunk_text(text, max_tokens=max_tokens, overlap_tokens=10)

        max_chars = max_tokens * 4
        for i, chunk in enumerate(chunks):
            assert len(chunk) <= max_chars, (
                f"Chunk {i} is {len(chunk)} chars, exceeds max of {max_chars}: {chunk[:80]}..."
            )
