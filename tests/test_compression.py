"""Tests for compression.py codec."""
import zlib
import pytest
from engram.compression import (
    compress, decompress, should_compress, compression_ratio,
    CompressionAlgoUnavailableError, DEFAULT_ALGO, MIN_COMPRESS_LENGTH,
)

SHORT_TEXT = "hello"
LONG_TEXT = "x" * 1000

def test_compress_zlib_returns_bytes():
    result, algo = compress(LONG_TEXT, "zlib")
    assert isinstance(result, bytes)
    assert algo == "zlib"

def test_compress_zlib_roundtrip():
    content = "The quick brown fox jumps over the lazy dog " * 50
    compressed, algo = compress(content, "zlib")
    assert decompress(compressed, algo) == content

def test_compress_default_algo_is_zlib():
    _, algo = compress(LONG_TEXT)
    assert algo == DEFAULT_ALGO == "zlib"

def test_decompress_corrupt_data_raises():
    with pytest.raises(zlib.error):
        decompress(b"not valid zlib data", "zlib")

def test_decompress_unknown_algo_raises():
    with pytest.raises(CompressionAlgoUnavailableError):
        decompress(b"anything", "nonexistent-algo")

def test_compress_unknown_algo_raises():
    with pytest.raises(ValueError):
        compress(LONG_TEXT, "nonexistent-algo")

def test_should_compress_long():
    assert should_compress("a" * MIN_COMPRESS_LENGTH) is True

def test_should_compress_short():
    assert should_compress("a" * (MIN_COMPRESS_LENGTH - 1)) is False

def test_compression_ratio_good():
    content = "The quick brown fox " * 100
    compressed, _ = compress(content, "zlib")
    ratio = compression_ratio(content, compressed)
    assert ratio > 1.0  # should compress well

def test_compression_ratio_incompressible():
    # Random-ish content compresses poorly — ratio may be < 1.0
    import hashlib
    content = hashlib.sha256(b"seed").hexdigest() * 20
    compressed, _ = compress(content, "zlib")
    ratio = compression_ratio(content, compressed)
    assert isinstance(ratio, float)

def test_compression_ratio_empty_compressed():
    assert compression_ratio("hello", b"") == 0.0
