"""Background summarization of memories using Ollama.

Runs as a daemon thread per project, continuously summarizing memories that
have no summary yet. Controlled by ENGRAM_SUMMARIZE_ENABLED and
ENGRAM_SUMMARIZE_MODEL env vars.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DatabaseBackend

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
SUMMARIZE_MODEL = os.getenv("ENGRAM_SUMMARIZE_MODEL", "llama3.2")
SUMMARIZE_ENABLED = os.getenv("ENGRAM_SUMMARIZE_ENABLED", "true").lower() == "true"
POLL_INTERVAL = 30  # seconds between background passes
BATCH_SIZE = 10     # memories per pass


def summarize_content(content: str, model: str = SUMMARIZE_MODEL) -> str | None:
    """Call Ollama synchronously to summarize a memory. Returns None on failure."""
    try:
        import httpx
        prompt = (
            "Summarize the following memory in 1-2 concise sentences. "
            "Focus on the key fact or decision. No preamble.\n\n"
            f"{content[:2000]}"
        )
        resp = httpx.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or None
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        return None


def _check_model_available(model: str) -> bool:
    """Return True if the model is available in Ollama."""
    try:
        import httpx
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5.0)
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(m.startswith(model) for m in models)
    except Exception:
        return False


class BackgroundSummarizer:
    """Daemon thread that continuously summarizes memories with no summary."""

    def __init__(self, db: DatabaseBackend, project: str):
        self.db = db
        self.project = project
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"summarizer-{project}",
            daemon=True,
        )

    def start(self) -> None:
        if not SUMMARIZE_ENABLED:
            logger.info("Background summarization disabled (ENGRAM_SUMMARIZE_ENABLED=false)")
            return
        if not _check_model_available(SUMMARIZE_MODEL):
            logger.warning(
                "Ollama model %s not available — background summarization disabled", SUMMARIZE_MODEL
            )
            return
        self._thread.start()
        logger.info(
            "Background summarizer started for project=%s model=%s", self.project, SUMMARIZE_MODEL
        )

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=35.0)
        if self._thread.is_alive():
            logger.warning(
                "Summarizer thread did not exit within 35 s — possible orphaned thread "
                "(project=%s)",
                self.project,
            )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                pending = self.db.get_memories_pending_summary(self.project, limit=BATCH_SIZE)
                for memory_id, content in pending:
                    if self._stop_event.is_set():
                        break
                    summary = summarize_content(content)
                    if summary:
                        self.db.store_summary(memory_id, summary)
                        logger.debug("Summarized memory %s", memory_id)
            except Exception as e:
                logger.warning("Summarizer loop error: %s", e)
            self._stop_event.wait(timeout=POLL_INTERVAL)
