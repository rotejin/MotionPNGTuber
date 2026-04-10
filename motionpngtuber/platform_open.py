"""Cross-platform helpers for opening files/directories with the OS default app."""
from __future__ import annotations

import os
import subprocess
import sys


def open_path_with_default_app(path: str) -> None:
    """Open a file or directory using the platform default application."""
    raw = str(path or "").strip()
    if not raw:
        raise ValueError("path is empty")
    target = os.path.abspath(raw)

    if sys.platform.startswith("win"):
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])


def prefer_native_video_preview() -> bool:
    """Return True on platforms where native preview is safer than OpenCV GUI."""
    return sys.platform == "darwin"
