"""Shared fixtures for engram tests.

Provides a FakeEmbedder (deterministic, no API calls), temporary database
directories, and pre-wired SearchEngine instances for fast, isolated testing.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from engram.db import MemoryDB
from engram.search import SearchEngine


class FakeEmbedder:
    """Deterministic embedder that hashes words into a fixed-size vector.

    Produces similar vectors for texts with overlapping vocabulary, enabling
    realistic semantic search tests without any external API calls.
    """

    DIMS = 64

    def embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vec = np.zeros(self.DIMS, dtype=np.float32)
        for w in words:
            idx = hash(w) % self.DIMS
            vec[idx] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def tmp_db_dir(tmp_path: Path) -> Path:
    """Provide a fresh temporary directory for database files."""
    return tmp_path


@pytest.fixture
def db(tmp_db_dir: Path) -> MemoryDB:
    """Provide a MemoryDB instance in a temp directory."""
    return MemoryDB(project="test", db_dir=tmp_db_dir)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def engine(db: MemoryDB, embedder: FakeEmbedder) -> SearchEngine:
    """Provide a SearchEngine wired to a temp DB and fake embedder."""
    return SearchEngine(db=db, embedder=embedder)
