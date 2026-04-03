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

    _MAX_RETRIES = 3
    _RETRY_BASE_DELAY = 1.0  # seconds; doubles on each retry (exponential backoff)

    def _embed_with_retry(self, **kwargs) -> object:
        """Call the OpenAI embeddings API with exponential backoff on 429."""
        import time as _time
        try:
            from openai import RateLimitError
        except ImportError:
            RateLimitError = Exception  # type: ignore[misc,assignment]

        delay = self._RETRY_BASE_DELAY
        for attempt in range(self._MAX_RETRIES):
            try:
                return self._client.embeddings.create(**kwargs)
            except RateLimitError:
                if attempt == self._MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "OpenAI rate limit (429) hit — retrying in %.1fs (attempt %d/%d)",
                    delay, attempt + 1, self._MAX_RETRIES,
                )
                _time.sleep(delay)
                delay *= 2
        raise RuntimeError("Unreachable")  # pragma: no cover

    def embed(self, text: str) -> np.ndarray:
        resp = self._embed_with_retry(input=[text], model="text-embedding-3-small")
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str], batch_size: int = 64) -> list[np.ndarray]:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = self._embed_with_retry(input=batch, model="text-embedding-3-small")
            sorted_data = sorted(resp.data, key=lambda d: d.index)
            all_embeddings.extend(
                np.array(d.embedding, dtype=np.float32) for d in sorted_data
            )
        return all_embeddings


_ALLOWED_PORTS = {80, 443, 11434, 8080, 3000, 8788}

# RFC-1918 private address ranges to block for SSRF protection.
# We block non-loopback private ranges and link-local (169.254.x.x / metadata endpoints).
# Loopback (127.0.0.1) is explicitly allowed — local Ollama is a primary use case.
_SSRF_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),       # RFC-1918
    ipaddress.ip_network("172.16.0.0/12"),     # RFC-1918
    ipaddress.ip_network("192.168.0.0/16"),    # RFC-1918
    ipaddress.ip_network("169.254.0.0/16"),    # link-local / AWS/GCP metadata
    ipaddress.ip_network("fc00::/7"),          # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


def _is_ssrf_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP address should be blocked for SSRF protection.

    Blocks RFC-1918 private ranges and link-local addresses.
    Loopback (127.x.x.x / ::1) is explicitly allowed for local Ollama.
    """
    if ip.is_loopback:
        return False
    return any(ip in net for net in _SSRF_BLOCKED_NETWORKS)


def _validate_ollama_url(url: str) -> bool:
    """Validate that an Ollama URL is not targeting internal/metadata services (SSRF protection)."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        port = parsed.port
        blocked_hosts = {"metadata.google.internal", "metadata.aws.internal"}
        if hostname in blocked_hosts:
            return False
        if port and port not in _ALLOWED_PORTS:
            logger.warning("Blocked Ollama URL: port %d not in allowlist %s", port, _ALLOWED_PORTS)
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if _is_ssrf_blocked(ip):
                return False
        except ValueError:
            # hostname is not a bare IP literal — attempt DNS resolution to check
            import socket
            try:
                resolved_ip = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(resolved_ip)
                if _is_ssrf_blocked(ip):
                    return False
            except (socket.gaierror, ValueError):
                pass
        return True
    except Exception:
        return False


class OllamaEmbedder:
    """Ollama embedding via local REST API.

    Calls Ollama's /api/embed endpoint directly with httpx -- no ollama
    Python package needed. Supports optional Bearer auth for protected Ollama endpoints.

    Model is configurable via ENGRAM_OLLAMA_MODEL env var (default: nomic-embed-text).
    """

    dimensions = 768
    version = "v1.5"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        model: str | None = None,
    ):
        if _httpx is None:
            raise ImportError(
                "httpx is required for Ollama embeddings: pip install engram[ollama]"
            )
        _require_numpy("Ollama embeddings")
        if not _validate_ollama_url(base_url):
            raise ValueError(f"Blocked Ollama URL (potential SSRF): {base_url}")
        self._base_url = base_url.rstrip("/")
        self._model = model or os.environ.get("ENGRAM_OLLAMA_MODEL", "nomic-embed-text")
        # Canonicalize name: strip :latest suffix so nomic-embed-text and
        # nomic-embed-text:latest are treated as the same model by the
        # consistency guard. The full tag is still sent to the Ollama API.
        _canonical = self._model.removesuffix(":latest")
        self.name = f"ollama/{_canonical}"
        self._headers: dict[str, str] = {}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def embed(self, text: str) -> np.ndarray:
        resp = _httpx.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": text},
            headers=self._headers,
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
                json={"model": self._model, "input": batch},
                headers=self._headers,
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
        ollama_key = os.environ.get("OLLAMA_API_KEY")
        return OllamaEmbedder(base_url=url, api_key=ollama_key)

    if provider == "none":
        return NullEmbedder()

    # Auto-detect
    auto_url = os.environ.get("OLLAMA_URL", ollama_url)
    ollama_key = os.environ.get("OLLAMA_API_KEY")
    if _ollama_reachable(auto_url, api_key=ollama_key, model=os.environ.get("ENGRAM_OLLAMA_MODEL")):
        logger.info("Auto-detected Ollama at %s, using local embeddings", auto_url)
        return OllamaEmbedder(base_url=auto_url, api_key=ollama_key)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if key:
        logger.info("Using OpenAI embeddings")
        return OpenAIEmbedder(api_key=key)

    logger.info("No embedding provider available, using BM25-only mode")
    return NullEmbedder()


def _ollama_reachable(base_url: str, api_key: str | None = None, model: str | None = None) -> bool:
    """Quick check if Ollama is running and has the required embedding model."""
    if _httpx is None:
        logger.debug("httpx not installed — Ollama auto-detect skipped")
        return False
    target_model = model or os.environ.get("ENGRAM_OLLAMA_MODEL", "nomic-embed-text")
    try:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        resp = _httpx.get(f"{base_url}/api/tags", headers=headers, timeout=2.0)
        if resp.status_code != 200:
            logger.warning("Ollama at %s returned HTTP %d — check if service is healthy", base_url, resp.status_code)
            return False
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        if not any(target_model in m for m in models):
            logger.warning("Ollama at %s is running but model '%s' not found. Available: %s", base_url, target_model, ", ".join(models) or "(none)")
            return False
        return True
    except _httpx.ConnectError:
        logger.debug("Ollama not reachable at %s (connection refused)", base_url)
    except _httpx.HTTPError as e:
        logger.warning("Ollama HTTP error at %s: %s", base_url, e)
    except Exception:
        logger.warning("Ollama auto-detect failed at %s", base_url, exc_info=True)
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
