"""Session/state management for mouth_track_gui.

Phase 1 of the mouth_track_gui refactoring plan.
Centralises load/save, type-safe helpers, and session key definitions.

Thread safety:
- load_session / save_session are protected by a shared module-level
  lock and use atomic file replacement (tempfile + os.replace) so that
  concurrent reads never see a partially-written file.
- All other helpers are pure functions with no shared state.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

# ---------------------------------------------------------------------------
# Path constants (delegated to _paths.py for package-aware resolution)
# ---------------------------------------------------------------------------
from ._paths import PROJECT_ROOT as HERE, SESSION_FILE as LAST_SESSION_FILE

# ---------------------------------------------------------------------------
# Known session keys (documentation + future validation)
# ---------------------------------------------------------------------------
SESSION_KEYS: frozenset[str] = frozenset({
    # GUI display / input
    "video",            # runtime background (may point to mouthless)
    "source_video",     # original video shown in GUI
    "mouth_dir",        # mouth sprite folder
    "character",        # character sub-folder name
    "emotion_preset",   # emotion preset label
    "emotion_hud",      # bool – show emotion HUD
    "coverage",         # float 0.40–0.90
    "pad",              # float 1.00–3.00
    "mouth_brightness",     # float -32..32
    "mouth_saturation",     # float 0.70..1.50
    "mouth_warmth",         # float -24..24
    "mouth_color_strength", # float 0.00..1.00
    "mouth_edge_priority",  # float 0.00..1.00
    "mouth_edge_width_ratio",  # float 0.02..0.20
    "mouth_inspect_boost",  # float 1.00..4.00
    "audio_device",     # int – sounddevice index
    "audio_device_spec", # str – sd:<index> / pa:<source>
    "erase_shading",    # "plane" | "none"
    "smoothing",        # smoothing preset label
    # Track file paths (written by actions)
    "track",
    "track_path",               # alias of track (calib_only)
    "track_calibrated",
    "track_calibrated_path",    # alias (calib_only)
    "calib",                    # alias (calib_only)
    # Legacy (read-only, for backward compat)
    "shading",
})

# ---------------------------------------------------------------------------
# Type-safe value conversion helpers
# ---------------------------------------------------------------------------

def safe_bool(v: object, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def safe_int(
    v: object,
    default: int,
    min_v: int | None = None,
    max_v: int | None = None,
) -> int:
    try:
        if isinstance(v, str):
            v = v.split(":", 1)[0].strip()
        iv = int(float(v)) if isinstance(v, str) else int(v)  # type: ignore[arg-type]
    except Exception:
        return default
    if min_v is not None:
        iv = max(min_v, iv)
    if max_v is not None:
        iv = min(max_v, iv)
    return iv


def safe_float(
    v: object,
    default: float,
    min_v: float | None = None,
    max_v: float | None = None,
) -> float:
    try:
        fv = float(v)  # type: ignore[arg-type]
    except Exception:
        return default
    if min_v is not None:
        fv = max(min_v, fv)
    if max_v is not None:
        fv = min(max_v, fv)
    return fv


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------
_session_lock = threading.Lock()


def _warn_session_issue(message: str) -> None:
    """Emit a lightweight warning for session persistence failures."""
    try:
        print(f"[mouth_track_gui.state] {message}", file=sys.stderr)
    except Exception:
        pass


def _read_session_file() -> dict:
    """Read session JSON from disk without locking (internal helper)."""
    try:
        with open(LAST_SESSION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_session() -> dict:
    """Load the last session from disk.  Returns ``{}`` on any error.

    Thread-safe: acquires the module lock to avoid reading a file that
    is being replaced by a concurrent ``save_session`` call.
    """
    with _session_lock:
        return _read_session_file()


def save_session(d: dict) -> bool:
    """Persist session data by *merging* with the existing file.

    Thread-safe via a module-level lock.  Uses tempfile + os.replace()
    for atomic writes so that a crash mid-write never corrupts the file.
    Returns ``True`` on success, ``False`` on failure.
    """
    with _session_lock:
        try:
            cur = _read_session_file()
            cur.update(d)
            dir_name = os.path.dirname(LAST_SESSION_FILE)
            fd, tmp = tempfile.mkstemp(
                suffix=".tmp", dir=dir_name or "."
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(cur, f, ensure_ascii=False, indent=2)
                os.replace(tmp, LAST_SESSION_FILE)
                return True
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            _warn_session_issue(
                f"save_session failed for {LAST_SESSION_FILE}: {e}",
            )
            return False
