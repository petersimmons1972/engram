"""Embedding providers for engram.

Supports three backends:
  - OpenAI (premium, highest quality): text-embedding-3-small, 1536 dims
  - Ollama (local, free, good quality): nomic-embed-text via REST API, 768 dims
  - Null (no vector search): BM25-only mode, zero dependencies

Selected via ENGRAM_EMBEDDER env var (openai|ollama|none). Default: auto-detect.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import struct
from typing import Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding backends must implement."""

    name: str
    dimensions: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536 dimensions)."""

    name = "openai/text-embedding-3-small"
    dimensions = 1536
    version = "v1"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(input=[text], model="text-embedding-3-small")
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._client.embeddings.create(input=batch, model="text-embedding-3-small")
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(
                np.array(d.embedding, dtype=np.float32) for d in sorted_data
            )
        return all_embeddings


def _validate_ollama_url(url: str) -> bool:
    """Validate that an Ollama URL is not targeting internal/metadata services (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        blocked_hosts = {"metadata.google.internal", "metadata.aws.internal"}
        if hostname in blocked_hosts:
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_link_local:
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


class OllamaEmbedder:
    """Ollama nomic-embed-text via local REST API (768 dimensions).

    Calls Ollama's /api/embed endpoint directly with httpx -- no ollama
    Python package needed.
    """

    name = "ollama/nomic-embed-text"
    dimensions = 768
    version = "v1.5"

    def __init__(self, base_url: str = "http://localhost:11434"):
        if not _validate_ollama_url(base_url):
            raise ValueError(f"Blocked Ollama URL (potential SSRF): {base_url}")
        self._base_url = base_url.rstrip("/")

    def embed(self, text: str) -> np.ndarray:
        import httpx
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embeddings"][0], dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        import httpx
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = httpx.post(
                f"{self._base_url}/api/embed",
                json={"model": "nomic-embed-text", "input": batch},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend(
                np.array(emb, dtype=np.float32) for emb in data["embeddings"]
            )
        return all_embeddings


class NullEmbedder:
    """No-op embedder for BM25-only mode. Zero external dependencies."""

    name = "none"
    dimensions = 0
    version = "n/a"

    def embed(self, text: str) -> np.ndarray:
        return np.array([], dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        return [np.array([], dtype=np.float32) for _ in texts]


def create_embedder(
    provider: str | None = None,
    api_key: str | None = None,
    ollama_url: str = "http://localhost:11434",
) -> EmbeddingProvider:
    """Factory that creates the appropriate embedder based on config.

    Args:
        provider: "openai", "ollama", "none", or None for auto-detect.
        api_key: OpenAI API key (only needed for openai provider).
        ollama_url: Ollama base URL (only needed for ollama provider).

    Auto-detect order: Ollama (if reachable) -> OpenAI (if key set) -> None.
    """
    if provider is None:
        provider = os.environ.get("ENGRAM_EMBEDDER", "").strip().lower()

    if provider == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            logger.warning("OPENAI_API_KEY not set, falling back to BM25-only mode")
            return NullEmbedder()
        return OpenAIEmbedder(api_key=key)

    if provider == "ollama":
        url = os.environ.get("OLLAMA_URL", ollama_url)
        return OllamaEmbedder(base_url=url)

    if provider == "none":
        return NullEmbedder()

    # Auto-detect
    auto_url = os.environ.get("OLLAMA_URL", ollama_url)
    if _ollama_reachable(auto_url):
        logger.info("Auto-detected Ollama at %s, using local embeddings", auto_url)
        return OllamaEmbedder(base_url=auto_url)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbedder(api_key=key)

    logger.info("No embedding provider available, using BM25-only mode")
    return NullEmbedder()


def _ollama_reachable(base_url: str) -> bool:
    """Quick check if Ollama is running and has nomic-embed-text."""
    try:
        import httpx
        resp = httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except Exception:
        pass
    return False


# ── Serialization helpers (unchanged) ────────────────────────────


def to_blob(vec: np.ndarray) -> bytes:
    if len(vec) == 0:
        return b""
    return struct.pack(f"{len(vec)}f", *vec.tolist())


def from_blob(blob: bytes) -> np.ndarray:
    if not blob:
        return np.array([], dtype=np.float32)
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
