"""Path definitions for the mouth_track_gui package.

All modules should reference PROJECT_ROOT / SESSION_FILE / SCRIPT_ROOT
instead of using __file__ directly, so that both
``python mouth_track_gui.py`` (compat wrapper) and
``python -m mouth_track_gui`` resolve to the same paths.

Live runtime IPC paths are intentionally *not* defined here. They are created
per run in ``mouth_track_gui.live_ipc`` so concurrent sessions stay isolated.
"""
from __future__ import annotations

import os

# Package parent directory = project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_FILE = os.path.join(PROJECT_ROOT, ".mouth_track_last_session.json")
SCRIPT_ROOT = PROJECT_ROOT  # CLI scripts (auto_mouth_track_v2.py etc.)
