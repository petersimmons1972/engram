"""Custom exceptions for engram.

Keeps error types in one place so the engine layer can raise clean exceptions
and the server layer can catch and surface them to MCP clients.
"""


class EngramError(Exception):
    """Base exception for all engram errors."""


class EmbeddingConfigMismatchError(EngramError):
    """Raised when the current embedder doesn't match the project's stored metadata.

    This prevents silently mixing vectors from different embedding models,
    which would corrupt semantic search results.
    """

    def __init__(self, stored_name: str, stored_dims: int, current_name: str, current_dims: int):
        self.stored_name = stored_name
        self.stored_dims = stored_dims
        self.current_name = current_name
        self.current_dims = current_dims
        super().__init__(
            f"Embedding model mismatch: project was created with '{stored_name}' ({stored_dims}d) "
            f"but the current embedder is '{current_name}' ({current_dims}d). "
            f"Mixing embedding models corrupts vector search results.\n\n"
            f"To fix this, either:\n"
            f"  1. Switch back to '{stored_name}' (set ENGRAM_OLLAMA_MODEL or ENGRAM_EMBEDDER)\n"
            f"  2. Export memories with 'engram dump', delete the project, and re-import with 'engram ingest'\n"
            f"  3. Start a fresh project with the new model"
        )
