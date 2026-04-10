from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from loop_lipsync_runtime_patched_emotion_auto import (
    AsyncMouthColorRebuilder,
    MouthColorAdjust,
    _load_auto_color_request,
    _load_live_color_control,
)


class IpcTokenValidationTests(unittest.TestCase):
    def test_load_live_color_control_rejects_token_mismatch(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "control.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_token": "wrong",
                        "updated_at": 1.0,
                        "mouth_brightness": 1.0,
                    },
                    f,
                )
            self.assertIsNone(_load_live_color_control(path, "expected"))

    def test_load_auto_color_request_accepts_matching_token(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "request.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_token": "expected",
                        "request_id": "req-1",
                        "requested_at": 12.5,
                    },
                    f,
                )
            self.assertEqual(_load_auto_color_request(path, "expected"), ("req-1", 12.5))


class AsyncMouthColorRebuilderTests(unittest.TestCase):
    def test_latest_request_wins(self):
        original = {
            "Neutral": {
                "open": np.zeros((1, 1, 4), dtype=np.uint8),
            },
        }

        def fake_rebuild(_mouth_sets_original, cfg):
            time.sleep(0.05 if cfg.brightness == 1.0 else 0.0)
            return {
                "Neutral": {
                    "open": np.full((1, 1, 4), int(cfg.brightness), dtype=np.uint8),
                },
            }

        with mock.patch(
            "loop_lipsync_runtime_patched_emotion_auto._rebuild_adjusted_mouth_sets",
            side_effect=fake_rebuild,
        ):
            rebuilder = AsyncMouthColorRebuilder(original, (1.0, 2.0, 3.0, 4.0))
            try:
                rebuilder.submit(
                    updated_at=1.0,
                    cfg=MouthColorAdjust(brightness=1.0),
                    reason="live",
                )
                rebuilder.submit(
                    updated_at=2.0,
                    cfg=MouthColorAdjust(brightness=2.0),
                    reason="live",
                )

                ready = None
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    ready = rebuilder.pop_ready()
                    if ready is not None:
                        break
                    time.sleep(0.02)

                self.assertIsNotNone(ready)
                assert ready is not None
                self.assertEqual(ready.updated_at, 2.0)
                self.assertEqual(int(ready.mouth_sets["Neutral"]["open"][0, 0, 0]), 2)
            finally:
                rebuilder.close()


if __name__ == "__main__":
    unittest.main()
