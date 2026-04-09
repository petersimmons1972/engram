from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

LAZY_CHUNK_THRESHOLD = 8000  # characters

# Regex matching level-1 and level-2 markdown headings at the start of a line.
_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+)$", re.MULTILINE)


@dataclass
class ChunkCandidate:
    """A candidate chunk produced by chunk_document().

    Attributes:
        text: The chunk text (may include overlap from the previous chunk).
        section_heading: The nearest level-1 or level-2 heading ancestor, or None
            when the document has no headings.
        chunk_type: One of "section", "paragraph", or "sentence_window".
            "sentence_window" is used when a single paragraph exceeds target_chunk_chars
            and is split by the existing sentence-window algorithm.
    """

    text: str
    section_heading: str | None
    chunk_type: str  # "section" | "paragraph" | "sentence_window"


def chunk_document(
    text: str,
    target_chunk_chars: int = 1200,
) -> list[ChunkCandidate]:
    """Semantic chunking: headings -> paragraphs -> sentence-window fallback.

    Algorithm:
    1. Split on level-1 and level-2 markdown headings.  Each heading + body = a
       candidate section.
    2. Within each section, split on blank-line paragraph breaks.
    3. Merge adjacent paragraphs up to target_chunk_chars.
    4. If a single paragraph exceeds target_chunk_chars, fall back to sentence-
       window splitting.
    5. Carry the last sentence of the previous chunk as the first sentence of
       the next chunk (overlap).
    6. If no headings are found, fall back to paragraph splitting (section_heading=None),
       then sentence-window as needed.
    """
    if not text.strip():
        return []

    # ── Step 1: split on headings ──────────────────────────────────────────
    heading_positions: list[tuple[int, str]] = []
    for m in _HEADING_RE.finditer(text):
        heading_positions.append((m.start(), m.group(2).strip()))

    if not heading_positions:
        # No headings — treat the whole document as a single anonymous section
        return _chunk_section(text.strip(), heading=None, target=target_chunk_chars)

    # Build (heading, body) pairs
    sections: list[tuple[str, str]] = []
    for i, (pos, heading) in enumerate(heading_positions):
        # Find where this section's body begins: after the heading line
        line_end = text.index("\n", pos) if "\n" in text[pos:] else len(text)
        body_start = line_end + 1

        # Find where this section ends: start of the next heading, or EOF
        if i + 1 < len(heading_positions):
            body_end = heading_positions[i + 1][0]
        else:
            body_end = len(text)

        body = text[body_start:body_end].strip()
        sections.append((heading, body))

    # ── Step 2–5: process each section ────────────────────────────────────
    results: list[ChunkCandidate] = []
    prev_last_sentence: str | None = None

    for heading, body in sections:
        section_chunks = _chunk_section(body, heading=heading, target=target_chunk_chars)

        # Carry overlap from previous section into the first chunk of this section
        if prev_last_sentence and section_chunks:
            first = section_chunks[0]
            if prev_last_sentence not in first.text:
                section_chunks[0] = ChunkCandidate(
                    text=prev_last_sentence + " " + first.text,
                    section_heading=first.section_heading,
                    chunk_type=first.chunk_type,
                )

        results.extend(section_chunks)

        # Record last sentence for next section's overlap
        if results:
            prev_last_sentence = _last_sentence(results[-1].text)

    return results


# ── Internal helpers ───────────────────────────────────────────────────────────

def _chunk_section(
    text: str,
    heading: str | None,
    target: int,
) -> list[ChunkCandidate]:
    """Convert a section body into ChunkCandidates.

    Splits on blank-line paragraphs, merges up to target chars,
    and falls back to sentence-window splitting for oversized paragraphs.
    """
    if not text.strip():
        return []

    # Split into paragraphs on one or more blank lines
    raw_paras = re.split(r"\n{2,}", text.strip())
    paras = [p.strip() for p in raw_paras if p.strip()]

    if not paras:
        return []

    chunks: list[ChunkCandidate] = []
    current_texts: list[str] = []
    current_len = 0
    prev_last: str | None = None

    def _flush(ctype: str = "paragraph") -> None:
        nonlocal current_texts, current_len
        if current_texts:
            merged = "\n\n".join(current_texts)
            if prev_last and prev_last not in merged:
                merged = prev_last + " " + merged
            chunks.append(ChunkCandidate(text=merged, section_heading=heading, chunk_type=ctype))
        current_texts = []
        current_len = 0

    for para in paras:
        if len(para) > target:
            # Oversized paragraph: flush current buffer, then sentence-window split
            _flush()
            sw_chunks = _sentence_window(para, target=target)
            for i, sw_text in enumerate(sw_chunks):
                # Apply overlap: carry prev_last into first sw chunk
                if i == 0 and prev_last and prev_last not in sw_text:
                    sw_text = prev_last + " " + sw_text
                chunks.append(
                    ChunkCandidate(
                        text=sw_text,
                        section_heading=heading,
                        chunk_type="sentence_window",
                    )
                )
                if chunks:
                    _nonlocal_last = _last_sentence(chunks[-1].text)
            continue

        sep = 2 if current_texts else 0  # "\n\n" separator length
        if current_len + len(para) + sep > target and current_texts:
            _flush()

        current_texts.append(para)
        current_len += len(para) + (2 if len(current_texts) > 1 else 0)
        if chunks:
            prev_last_ref = _last_sentence(
                chunks[-1].text if chunks else ""
            )
            # Keep prev_last updated as we accumulate chunks
        _ = prev_last  # suppress unused warning

    _flush()

    return chunks


def _sentence_window(text: str, target: int) -> list[str]:
    """Split a single large paragraph into overlapping sentence-window chunks.

    Uses the same sentence-boundary logic as chunk_text() but tuned for
    target_chunk_chars rather than max_tokens.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)
        added_len = slen + (1 if current else 0)

        if current_len + added_len > target and current:
            chunks.append(" ".join(current))
            # Overlap: carry last sentence
            overlap = [current[-1]] if current else []
            overlap_len = len(current[-1]) if current else 0
            current = overlap
            current_len = overlap_len
            added_len = slen + (1 if current else 0)

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _last_sentence(text: str) -> str:
    """Return the last sentence from text (for overlap carry-over)."""
    sentences = _split_sentences(text)
    return sentences[-1] if sentences else ""


def chunk_text(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Uses a rough 1 token ~ 4 chars approximation to avoid a tokenizer dependency.
    Content under LAZY_CHUNK_THRESHOLD characters is returned as a single chunk
    to avoid unnecessary splitting of short memories.
    """
    if not text.strip():
        return []

    # Lazy chunking: short content is a single chunk
    if len(text) <= LAZY_CHUNK_THRESHOLD:
        return [text]

    chars_per_token = 4
    max_chars = max_tokens * chars_per_token
    overlap_chars = overlap_tokens * chars_per_token

    sentences = _split_sentences(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        slen = len(sentence)
        # Account for the space separator when appending to a non-empty chunk
        added_len = slen + (1 if current else 0)

        if current_len + added_len > max_chars and current:
            chunks.append(" ".join(current))

            # Build overlap from the tail of the current chunk
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                sep = 1 if overlap else 0
                if overlap_len + len(s) + sep > overlap_chars:
                    break
                overlap.insert(0, s)
                overlap_len += len(s) + sep
            current = overlap
            current_len = overlap_len
            added_len = slen + (1 if current else 0)

        current.append(sentence)
        current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def chunk_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalized.encode()).hexdigest()[:32]


def jaccard_similarity(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(re.sub(r"\s+", " ", a.strip().lower()).split())
    words_b = set(re.sub(r"\s+", " ", b.strip().lower()).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def is_duplicate(new_text: str, existing_texts: list[str], threshold: float | None = None) -> bool:
    """Return True if new_text is too similar to any existing text.

    Threshold lowered from 0.85 to 0.75 — the old value was too aggressive
    and deduplicated legitimately distinct memories that share common phrasing.
    Override via ENGRAM_DEDUP_THRESHOLD env var if needed.

    When `threshold` is None (the default), the value is read from the
    ENGRAM_DEDUP_THRESHOLD environment variable, falling back to 0.75.
    An explicit numeric argument always takes precedence over the env var.
    """
    if threshold is None:
        threshold = float(os.environ.get("ENGRAM_DEDUP_THRESHOLD", "0.75"))
    for existing in existing_texts:
        if jaccard_similarity(new_text, existing) >= threshold:
            return True
    return False


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the delimiter attached."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
