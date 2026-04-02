"""Compression codec for Engram memory content.

Follows the embeddings.py optional-dependency pattern:
- zlib is always available (stdlib)
- zstd is optional (zstandard package)
- Unknown/unavailable algos raise CompressionAlgoUnavailableError — never silently return garbage
"""
from __future__ import annotations
import zlib

try:
    import zstandard as _zstd
    _has_zstd = True
except ImportError:
    _has_zstd = False

SUPPORTED_ALGOS: frozenset[str] = frozenset({"zlib"} | ({"zstd"} if _has_zstd else set()))
DEFAULT_ALGO = "zlib"
MIN_COMPRESS_LENGTH = 500  # chars — skip tiny memories, overhead not worth it


class CompressionAlgoUnavailableError(Exception):
    """Raised when a stored compression_algo is not available on this machine."""


def compress(content: str, algo: str = DEFAULT_ALGO) -> tuple[bytes, str]:
    """Compress content string. Returns (compressed_bytes, algo_name).

    Raises ValueError if algo is unsupported.
    Raises CompressionAlgoUnavailableError if algo requires missing package.
    """
    if algo == "zlib":
        return zlib.compress(content.encode("utf-8"), level=6), "zlib"
    elif algo == "zstd":
        if not _has_zstd:
            raise CompressionAlgoUnavailableError(
                "zstd not available — install with: pip install engram[zstd]"
            )
        cctx = _zstd.ZstdCompressor(level=3)
        return cctx.compress(content.encode("utf-8")), "zstd"
    else:
        raise ValueError(f"Unsupported compression algorithm: {algo!r}. Supported: {SUPPORTED_ALGOS}")


def decompress(compressed: bytes, algo: str) -> str:
    """Decompress bytes to string.

    Raises CompressionAlgoUnavailableError if algo requires missing package.
    Raises zlib.error / zstd errors if bytes are corrupt.
    """
    if algo == "zlib":
        return zlib.decompress(compressed).decode("utf-8")
    elif algo == "zstd":
        if not _has_zstd:
            raise CompressionAlgoUnavailableError(
                "zstd not available — install with: pip install engram[zstd]"
            )
        dctx = _zstd.ZstdDecompressor()
        return dctx.decompress(compressed).decode("utf-8")
    else:
        raise CompressionAlgoUnavailableError(
            f"Unknown compression algorithm {algo!r} — cannot decompress stored memory"
        )


def should_compress(content: str) -> bool:
    """True if content is long enough to be worth compressing."""
    return len(content) >= MIN_COMPRESS_LENGTH


def compression_ratio(original: str, compressed: bytes) -> float:
    """Ratio of original bytes to compressed bytes. >1.0 = compression saved space."""
    original_bytes = len(original.encode("utf-8"))
    if not compressed:
        return 0.0
    return round(original_bytes / len(compressed), 3)
