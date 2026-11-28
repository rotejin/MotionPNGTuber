"""erase_mouth_offline.py with --debug, with and without --keep-audio.

Before, the debug images were saved through the same variable as the output video
(out_path), so --out was never written and the last frame_*.png was overwritten with
the MP4. With --keep-audio (what the GUI uses through auto_erase_mouth.py) the
low-quality warning and metrics.json were skipped, because they sat in the else branch.
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import erase_mouth_offline

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _head(path: str, size: int = 12) -> bytes:
    with open(path, "rb") as f:
        return f.read(size)


class EraseDebugOutputTests(unittest.TestCase):
    W, H, N = 160, 120, 130  # 130 frames: debug frames 0 and 120

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.video = os.path.join(self.dir, "clip.mp4")
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (self.W, self.H))
        self.assertTrue(writer.isOpened())
        for _ in range(self.N):
            frame = np.full((self.H, self.W, 3), (180, 200, 230), np.uint8)
            cv2.ellipse(frame, (80, 70), (18, 8), 0, 0, 360, (40, 40, 160), -1)
            writer.write(frame)
        writer.release()
        quad = np.array([[60, 60], [100, 60], [100, 80], [60, 80]], np.float32)
        valid = np.ones(self.N, np.uint8)
        valid[: int(self.N * 0.3)] = 0  # valid rate 0.70: below the 0.80 warning line
        self.track = os.path.join(self.dir, "mouth_track.npz")
        np.savez(self.track, quad=np.tile(quad, (self.N, 1, 1)), valid=valid,
                 w=np.int32(self.W), h=np.int32(self.H))

    def erase(self, *extra: str) -> tuple[int, str, str, str]:
        case = tempfile.mkdtemp(dir=self.dir)
        out = os.path.join(case, "out.mp4")
        debug = os.path.join(case, "debug")
        argv = ["erase_mouth_offline.py", "--video", self.video, "--track", self.track, "--out", out,
                "--debug", debug, "--coverage", "0.6", "--shading", "none", *extra]
        log = io.StringIO()
        with (
            mock.patch.object(erase_mouth_offline.sys, "argv", argv),
            # The test clip has no audio track; muxing fails and the silent video is kept.
            mock.patch.object(erase_mouth_offline, "mux_audio_ffmpeg", return_value=False),
            contextlib.redirect_stdout(log),
        ):
            rc = erase_mouth_offline.main()
        return rc, out, debug, log.getvalue()

    def test_debug_writes_the_video_and_keeps_the_debug_images(self):
        for extra in ((), ("--keep-audio",)):
            with self.subTest(extra=extra):
                rc, out, debug, log = self.erase(*extra)
                self.assertEqual(rc, 0, log)
                self.assertTrue(os.path.isfile(out), log)
                self.assertEqual(_head(out)[4:8], b"ftyp")
                for name in ("frame_000000.png", "frame_000120.png", "mask_ring.png", "ref_patch.png"):
                    self.assertEqual(_head(os.path.join(debug, name), 8), PNG_SIGNATURE, name)
                self.assertFalse([f for f in os.listdir(debug) if ".bak." in f])

    def test_warning_and_metrics_also_with_keep_audio(self):
        for extra in ((), ("--keep-audio",)):
            with self.subTest(extra=extra):
                rc, _out, debug, log = self.erase(*extra)
                self.assertEqual(rc, 0, log)
                self.assertIn("valid_rate is low (0.70)", log)
                self.assertTrue(os.path.isfile(os.path.join(debug, "metrics.json")))


if __name__ == "__main__":
    unittest.main()
