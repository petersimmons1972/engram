"""Embedding providers for engram.

Supports three backends:
  - OpenAI (premium, highest quality): text-embedding-3-small, 1536 dims
  - Ollama (local, free, good quality): nomic-embed-text via REST API, 768 dims
  - Null (no vector search): BM25-only mode, zero dependencies

Selected via ENGRAM_EMBEDDER env var (openai|ollama|none). Default: auto-detect.

numpy and httpx are optional dependencies. Install with:
  pip install engram[embeddings]   # for numpy (vector math)
  pip install engram[ollama]       # for httpx (Ollama REST client)
  pip install engram[all]          # everything
"""

from __future__ import annotations

import ipaddress
import logging
import os
import struct
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable
from urllib.parse import urlparse

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np

    _has_numpy = True
except ImportError:
    np = None  # type: ignore[assignment]
    _has_numpy = False

try:
    import httpx as _httpx
except ImportError:
    _httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _require_numpy(caller: str = "") -> None:
    """Raise a helpful ImportError if numpy is not installed."""
    if not _has_numpy:
        raise ImportError(
            f"numpy is required for {caller or 'embedding operations'}: "
            "pip install engram[embeddings]"
        )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol that all embedding backends must implement."""

    name: str
    dimensions: int

    def embed(self, text: str) -> Any: ...

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[Any]: ...


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536 dimensions)."""

    name = "openai/text-embedding-3-small"
    dimensions = 1536
    version = "v1"

    def __init__(self, api_key: str | None = None):
        _require_numpy("OpenAI embeddings")
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
        if _httpx is None:
            raise ImportError(
                "httpx is required for Ollama embeddings: pip install engram[ollama]"
            )
        _require_numpy("Ollama embeddings")
        if not _validate_ollama_url(base_url):
            raise ValueError(f"Blocked Ollama URL (potential SSRF): {base_url}")
        self._base_url = base_url.rstrip("/")

    def embed(self, text: str) -> np.ndarray:
        resp = _httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": "nomic-embed-text", "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["embeddings"][0], dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = _httpx.post(
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

    def embed(self, text: str) -> Any:
        if _has_numpy:
            return np.array([], dtype=np.float32)
        return []

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[Any]:
        if _has_numpy:
            return [np.array([], dtype=np.float32) for _ in texts]
        return [[] for _ in texts]


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
    if _httpx is None:
        logger.debug("httpx not installed — Ollama auto-detect skipped")
        return False
    try:
        resp = _httpx.get(f"{base_url}/api/tags", timeout=2.0)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except _httpx.HTTPError:
        logger.debug("Ollama not reachable at %s", base_url)
    except Exception:
        logger.debug("Ollama auto-detect failed", exc_info=True)
    return False


# ── Serialization helpers (unchanged) ────────────────────────────


def to_blob(vec: Any) -> bytes:
    if _has_numpy:
        if len(vec) == 0:
            return b""
        return struct.pack(f"{len(vec)}f", *vec.tolist())
    return b""


def from_blob(blob: bytes) -> Any:
    _require_numpy("from_blob")
    if not blob:
        return np.array([], dtype=np.float32)
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def cosine_similarity(a: Any, b: Any) -> float:
    if not _has_numpy:
        return 0.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        return 0.0
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)
