"""Shared fixtures for engram tests."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pytest

from engram.db_postgres import PostgresBackend
from engram.search import SearchEngine


pytestmark = pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="No TEST_DATABASE_URL set",
)


class FakeEmbedder:
    """Deterministic embedder that hashes words into a fixed-size vector.

    Produces similar vectors for texts with overlapping vocabulary, enabling
    realistic semantic search tests without any external API calls.
    Implements the EmbeddingProvider protocol.
    """

    name = "fake/test-embedder"
    dimensions = 64
    version = "v1-test"

    def embed(self, text: str) -> np.ndarray:
        words = set(text.lower().split())
        vec = np.zeros(self.dimensions, dtype=np.float32)
        for w in words:
            idx = hash(w) % self.dimensions
            vec[idx] = 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


@pytest.fixture
def db():
    """PostgresBackend in a clean test project. Cleans up after each test."""
    dsn = os.environ.get("TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("No TEST_DATABASE_URL set")

    backend = PostgresBackend(project="test", dsn=dsn)
    yield backend
    with backend.pool.connection() as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM relationships")
        conn.execute("DELETE FROM memories")
        conn.execute("DELETE FROM project_meta")
        conn.commit()
    backend.close()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def engine(db, embedder: FakeEmbedder) -> SearchEngine:
    """Provide a SearchEngine wired to a Postgres DB and fake embedder."""
    return SearchEngine(db=db, embedder=embedder)
