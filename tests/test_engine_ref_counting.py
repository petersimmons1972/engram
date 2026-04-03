"""Unit tests for SearchEngine deferred-close with reference counting.

Issue #102: _get_engine() LRU eviction calls close() on an engine that may
still have live references from concurrent threads. Fix: track active refs,
defer teardown until the last ref is released.

These tests use only mocks — no database, no external services, no API calls.
"""

from __future__ import annotations

import threading
import time
import unittest.mock as mock

import pytest

from engram.search import SearchEngine


def _make_engine() -> SearchEngine:
    """Build a SearchEngine with fully mocked internals.

    Bypasses __init__ start() calls via spec-only mocks so no threads
    are actually launched.
    """
    db = mock.MagicMock()
    db.project = "test-project"
    embedder = mock.MagicMock()
    embedder.name = "fake/test"
    embedder.dimensions = 64

    # Patch out the background workers so no real threads start.
    with (
        mock.patch("engram.search.BackgroundSummarizer") as mock_summarizer_cls,
        mock.patch("engram.search.BackgroundReembedder") as mock_reembedder_cls,
    ):
        mock_summarizer_cls.return_value = mock.MagicMock()
        mock_reembedder_cls.return_value = mock.MagicMock()
        engine = SearchEngine(db=db, embedder=embedder)

    return engine


class TestContextManagerRefCount:
    def test_enter_increments_ref_count(self):
        engine = _make_engine()
        assert engine._ref_count == 0
        with engine:
            assert engine._ref_count == 1
        assert engine._ref_count == 0

    def test_nested_context_managers_stack(self):
        engine = _make_engine()
        with engine:
            with engine:
                assert engine._ref_count == 2
            assert engine._ref_count == 1
        assert engine._ref_count == 0

    def test_enter_returns_engine_itself(self):
        engine = _make_engine()
        with engine as ctx:
            assert ctx is engine


class TestDeferredClose:
    def test_close_deferred_while_reference_held(self):
        """close() while inside a with-block must not call _do_close immediately."""
        engine = _make_engine()
        with mock.patch.object(engine, "_do_close") as mock_do_close:
            with engine:
                # Simulate eviction calling close() while the ref is still held.
                engine.close()
                assert engine._closing is True
                # _do_close must NOT have been called yet — ref_count is 1.
                mock_do_close.assert_not_called()
            # __exit__ drops ref_count to 0; now _do_close must fire.
            mock_do_close.assert_called_once()

    def test_close_immediate_when_no_references(self):
        """close() with no active refs must call _do_close immediately."""
        engine = _make_engine()
        with mock.patch.object(engine, "_do_close") as mock_do_close:
            assert engine._ref_count == 0
            engine.close()
            mock_do_close.assert_called_once()

    def test_close_called_exactly_once_not_twice(self):
        """Deferred close must not double-call _do_close after ref drops."""
        engine = _make_engine()
        do_close_calls: list[int] = []

        original_do_close = engine._do_close

        def tracking_do_close():
            do_close_calls.append(1)
            # Don't call original to avoid hitting real db.close().

        engine._do_close = tracking_do_close

        with engine:
            engine.close()
        # Exactly one call, not two.
        assert len(do_close_calls) == 1


class TestEnterAfterClose:
    def test_enter_raises_when_engine_is_closing(self):
        """__enter__ must raise RuntimeError if _closing is True.

        This covers the race where eviction fires between _get_engine() returning
        the engine and the caller entering the context manager.  The caller can
        catch RuntimeError and retry _get_engine().
        """
        engine = _make_engine()
        # Drive the engine into _closing state without going through _do_close
        # (simulate: close() was called, no refs, _do_close ran, but another
        # thread still holds a stale reference to the engine object).
        engine._closing = True

        with pytest.raises(RuntimeError, match="being evicted"):
            engine.__enter__()

    def test_enter_raises_message_mentions_retry(self):
        """The error message should indicate the caller can retry."""
        engine = _make_engine()
        engine._closing = True

        with pytest.raises(RuntimeError) as exc_info:
            engine.__enter__()

        assert "retry" in str(exc_info.value).lower() or "evict" in str(exc_info.value).lower()


class TestDoCloseActualTeardown:
    def test_do_close_stops_threads_and_closes_db(self):
        """_do_close() must stop both background workers and close the DB pool."""
        engine = _make_engine()

        engine._do_close()

        engine._summarizer.stop.assert_called_once()
        engine._reembedder.stop.assert_called_once()
        engine.db.close.assert_called_once()

    def test_legacy_close_with_no_refs_calls_do_close(self):
        """close() is the public API. With no refs it must reach _do_close."""
        engine = _make_engine()
        engine.close()

        engine._summarizer.stop.assert_called_once()
        engine._reembedder.stop.assert_called_once()
        engine.db.close.assert_called_once()


class TestConcurrentRefCounting:
    def test_close_deferred_until_last_thread_exits(self):
        """Thread A holds a ref. Main calls close(). Close defers until A exits."""
        engine = _make_engine()
        close_happened = threading.Event()
        thread_can_exit = threading.Event()
        thread_entered = threading.Event()

        do_close_calls: list[float] = []

        def tracking_do_close():
            do_close_calls.append(time.monotonic())
            close_happened.set()

        engine._do_close = tracking_do_close

        def thread_a():
            with engine:
                thread_entered.set()
                # Hold the reference until the main thread says so.
                thread_can_exit.wait(timeout=5.0)

        t = threading.Thread(target=thread_a)
        t.start()

        thread_entered.wait(timeout=5.0)
        assert engine._ref_count == 1

        # Eviction fires while Thread A still holds the engine.
        close_start = time.monotonic()
        engine.close()
        assert engine._closing is True
        # Must not have closed yet.
        assert not close_happened.is_set(), "_do_close fired before ref dropped"

        # Release Thread A.
        thread_can_exit.set()
        t.join(timeout=5.0)

        # Now _do_close must have fired.
        close_happened.wait(timeout=5.0)
        assert close_happened.is_set(), "_do_close never fired after last ref released"
        assert engine._ref_count == 0

        # The close must have happened AFTER we released the thread.
        assert do_close_calls[0] >= close_start

    def test_multiple_threads_all_increment_ref_count(self):
        """N threads inside context managers: ref_count peaks at N.

        Two barriers ensure:
        1. All threads have entered __enter__ before anyone records the count.
        2. All threads have recorded before anyone exits __exit__.
        This removes the race between recording and decrementing.
        """
        engine = _make_engine()
        # Override _do_close to be a no-op so close() won't tear down anything.
        engine._do_close = mock.MagicMock()

        n = 5
        entry_barrier = threading.Barrier(n)   # all in before anyone reads
        exit_barrier = threading.Barrier(n)    # all read before anyone exits
        peak_counts: list[int] = []
        counts_lock = threading.Lock()

        def worker():
            with engine:
                # Phase 1: wait until all n threads have entered the with block.
                entry_barrier.wait()
                # Phase 2: read ref_count — all n refs are held, no one has exited.
                with engine._ref_lock:
                    count = engine._ref_count
                with counts_lock:
                    peak_counts.append(count)
                # Phase 3: wait until all threads have recorded before exiting.
                exit_barrier.wait()
                # Now everyone exits their with block (decrements).

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert all(c == n for c in peak_counts), f"Expected all {n}, got {peak_counts}"
        assert engine._ref_count == 0
