"""Tests for session-scoped live IPC helpers."""
from __future__ import annotations

import os
import tempfile
import unittest

from mouth_track_gui.live_ipc import (
    cleanup_live_ipc_session,
    create_live_ipc_session,
)


class LiveIpcSessionTests(unittest.TestCase):
    def test_create_and_cleanup_session(self):
        with tempfile.TemporaryDirectory() as td:
            session = create_live_ipc_session(td)
            self.assertTrue(os.path.isdir(session.session_dir))
            self.assertTrue(session.live_color_control_path.startswith(session.session_dir))
            self.assertTrue(session.auto_color_request_path.startswith(session.session_dir))
            self.assertTrue(session.auto_color_result_path.startswith(session.session_dir))
            self.assertTrue(session.session_token)

            cleanup_live_ipc_session(session)
            self.assertFalse(os.path.exists(session.session_dir))


if __name__ == "__main__":
    unittest.main()
