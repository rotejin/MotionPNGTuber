"""Subprocess runner for mouth_track_gui.

Phase 3 of the mouth_track_gui refactoring plan.
Provides a synchronous subprocess API (``run_stream``) that streams
stdout/stderr to a caller-supplied callback, with soft-stop / force-stop
support.

Thread contract:
- ``run_stream()``:        call from worker thread only (blocking).
- ``request_soft_stop()``: call from any thread (main thread typical).
- ``force_stop()``:        call from any thread (main thread typical).
- ``reset()``:             call from main thread before a new workflow.

Stop semantics:
- ``request_soft_stop()`` sets a *reservation flag* only.  It does NOT
  immediately signal the child process.  The signal is sent inside
  ``_drain_output()`` only when the current ``run_stream()`` was called
  with ``allow_soft_stop=True``.  This matches the GUI's "stop after
  current step" button: a soft stop during an ``allow_soft_stop=False``
  command (e.g. ffmpeg) lets it finish, and the flag is observed by the
  next ``allow_soft_stop=True`` command or by the App-level step loop.
- ``force_stop()`` immediately kills the child process tree.
- ``reset()`` clears both flags.  Call it once at the start of each
  workflow (before the first ``run_stream``).
"""
from __future__ import annotations

import os
import queue
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Callable


@dataclass
class RunResult:
    """Outcome of a single ``run_stream`` invocation."""
    returncode: int
    was_stopped: bool  # True only when this child was actually interrupted
    stop_requested: bool = False  # True if any stop request was observed


@dataclass
class DrainResult:
    """Internal outcome of ``_drain_output``."""
    was_stopped: bool
    stop_requested: bool


class CommandRunner:
    """Synchronous subprocess runner with stop support.

    Parameters
    ----------
    on_log : callable(str) -> None
        Called on the *worker thread* for every line of output.
    """

    def __init__(self, on_log: Callable[[str], None]) -> None:
        self._on_log = on_log
        self._active_proc: subprocess.Popen | None = None
        self._lock = threading.Lock()  # guards _active_proc
        self._soft_requested = threading.Event()
        self._force_requested = threading.Event()

    # ------------------------------------------------------------------
    # Public API – main thread
    # ------------------------------------------------------------------

    def request_soft_stop(self) -> None:
        """Reserve a soft stop.  Does NOT send a signal immediately.

        The signal is only delivered to the child during ``_drain_output``
        when ``allow_soft_stop=True``.
        """
        self._soft_requested.set()

    def force_stop(self) -> None:
        """Kill the running process tree immediately."""
        self._force_requested.set()
        with self._lock:
            p = self._active_proc
        if p and p.poll() is None:
            self._terminate_proc_tree(p)

    def reset(self) -> None:
        """Clear all stop flags.  Call before a new workflow."""
        self._soft_requested.clear()
        self._force_requested.clear()

    @property
    def soft_requested(self) -> bool:
        """True if a soft stop has been requested (reservation)."""
        return self._soft_requested.is_set()

    @property
    def is_running(self) -> bool:
        with self._lock:
            p = self._active_proc
        return p is not None and p.poll() is None

    # ------------------------------------------------------------------
    # Public API – worker thread
    # ------------------------------------------------------------------

    def run_stream(
        self,
        cmd: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        allow_soft_stop: bool = False,
    ) -> RunResult:
        """Run *cmd* synchronously, streaming output via ``on_log``.

        Parameters
        ----------
        cmd : list[str]
            Command and arguments.
        cwd : str | None
            Working directory for the subprocess.
        env : dict | None
            Extra environment variables (merged into ``os.environ``).
        allow_soft_stop : bool
            If True, a pending ``request_soft_stop`` will send SIGINT /
            CTRL_BREAK to the child.  If False, the process runs to
            completion even if soft stop is pending (force stop still
            works).

        Returns
        -------
        RunResult
        """
        # Do NOT clear flags here — a stop requested before this run
        # must be observable.  Flags are cleared by reset().

        run_env = os.environ.copy()
        run_env.setdefault("PYTHONUTF8", "1")
        run_env.setdefault("PYTHONIOENCODING", "utf-8")
        if env:
            run_env.update(env)

        popen_kw: dict = {}
        if sys.platform.startswith("win"):
            popen_kw["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            # ``process_group`` was added in Python 3.11.  This project pins
            # Python 3.10, so use ``start_new_session`` to create a fresh
            # process group that can still be targeted via ``os.killpg``.
            popen_kw["start_new_session"] = True

        self._on_log("[cmd] " + " ".join(cmd))

        try:
            p = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=run_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kw,
            )
        except Exception as e:
            self._on_log(f"[error] failed to start: {e}")
            return RunResult(
                returncode=999, was_stopped=False, stop_requested=False,
            )

        with self._lock:
            self._active_proc = p

        drain = DrainResult(was_stopped=False, stop_requested=False)
        try:
            drain = self._drain_output(p, allow_soft_stop)
        finally:
            rc = p.wait()
            try:
                p.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass
            with self._lock:
                if self._active_proc is p:
                    self._active_proc = None

        return RunResult(
            returncode=rc,
            was_stopped=drain.was_stopped,
            stop_requested=drain.stop_requested,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_output(
        self, p: subprocess.Popen, allow_soft_stop: bool
    ) -> DrainResult:
        """Read stdout in a background thread, relay to on_log.

        A late stop request after child exit must set ``stop_requested``,
        not ``was_stopped``.
        """
        assert p.stdout is not None
        out_q: queue.Queue = queue.Queue()
        sentinel = object()

        def _reader() -> None:
            try:
                for line in p.stdout:  # type: ignore[union-attr]
                    out_q.put(line)
            finally:
                out_q.put(sentinel)

        threading.Thread(target=_reader, daemon=True).start()

        force_observed = False
        soft_observed = False
        was_stopped = False
        stop_requested = False

        while True:
            # Force stop
            if self._force_requested.is_set() and not force_observed:
                force_observed = True
                stop_requested = True
                if p.poll() is None:
                    was_stopped = True
                    self._terminate_proc_tree(p)

            # Soft stop — send signal only if this run allows it
            if (
                allow_soft_stop
                and not soft_observed
                and self._soft_requested.is_set()
            ):
                soft_observed = True
                stop_requested = True
                if p.poll() is None:
                    was_stopped = True
                    self._send_soft_signal(p)

            try:
                item = out_q.get(timeout=0.1)
            except queue.Empty:
                item = None

            if item is sentinel:
                break
            if isinstance(item, str):
                self._on_log(item.rstrip("\n"))

        return DrainResult(
            was_stopped=was_stopped, stop_requested=stop_requested,
        )

    @staticmethod
    def _send_soft_signal(p: subprocess.Popen) -> None:
        """Send a graceful interrupt signal to the child."""
        try:
            if sys.platform.startswith("win"):
                os.kill(p.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
        except Exception:
            pass

    @staticmethod
    def _terminate_proc_tree(p: subprocess.Popen) -> None:
        """Kill the child process (and its tree if possible)."""
        try:
            if sys.platform.startswith("win"):
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    p.kill()
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
