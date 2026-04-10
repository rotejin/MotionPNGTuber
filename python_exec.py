"""Helpers for selecting the right Python executable for child processes."""

from __future__ import annotations

import os
import sys


def resolve_python_subprocess_executable(executable: str | None = None) -> str:
    """Return a Python executable suitable for spawning CLI child processes.

    On Windows, GUI apps are often launched via ``pythonw.exe``. That is fine
    for the top-level GUI, but nested CLI scripts may fail when they expect
    usable stdio streams. In that case, switch child launches to the sibling
    ``python.exe`` when available.
    """

    exe = executable or sys.executable or "python"
    if os.name != "nt":
        return exe

    if os.path.basename(exe).lower() != "pythonw.exe":
        return exe

    candidate = os.path.join(os.path.dirname(exe), "python.exe")
    if os.path.isfile(candidate):
        return candidate
    return exe
