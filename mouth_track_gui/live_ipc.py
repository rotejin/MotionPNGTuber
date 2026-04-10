"""Session-scoped IPC helpers for live GUI/runtime coordination."""
from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import tempfile
import uuid


@dataclass(frozen=True)
class LiveIpcSession:
    session_dir: str
    session_token: str
    live_color_control_path: str
    auto_color_request_path: str
    auto_color_result_path: str


def create_live_ipc_session(base_dir: str | None = None) -> LiveIpcSession:
    token = uuid.uuid4().hex
    mkdtemp_kwargs: dict[str, str] = {"prefix": f"motionpngtuber_live_{token[:8]}_"}
    if base_dir:
        mkdtemp_kwargs["dir"] = os.path.abspath(base_dir)
    session_dir = tempfile.mkdtemp(**mkdtemp_kwargs)
    return LiveIpcSession(
        session_dir=session_dir,
        session_token=token,
        live_color_control_path=os.path.join(session_dir, "mouth_color_live_control.json"),
        auto_color_request_path=os.path.join(session_dir, "mouth_color_auto_request.json"),
        auto_color_result_path=os.path.join(session_dir, "mouth_color_auto_result.json"),
    )


def cleanup_live_ipc_session(session: LiveIpcSession | None) -> None:
    if session is None:
        return
    shutil.rmtree(session.session_dir, ignore_errors=True)
