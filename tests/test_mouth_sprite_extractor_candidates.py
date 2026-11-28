"""Mouth sprite extractor: reading the candidate frames, and the face detector's errors.

- Thumbnails, auto assign, the preview and auto crop each seeked to every candidate
  frame again. A seek decodes from the key frame before it, and some clips have a
  single key frame: on a 2160x3900 clip, 20 candidates took 13 s and the window froze.
  The patches are now read in frame order (reading forward between nearby frames) and
  shared.
- The face detector prints its messages (how to get the weights again, missing
  packages, ...) to stdout, but only stderr was shown, so the error was empty.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

import mouth_sprite_extractor_gui as gui
from motionpngtuber import mouth_sprite_extractor as extractor
from motionpngtuber.mouth_sprite_extractor import MouthFrameInfo


def _frame_info(idx: int, x: float = 10.0) -> MouthFrameInfo:
    quad = np.array([[x, 10], [x + 30, 10], [x + 30, 30], [x, 30]], np.float32)
    return MouthFrameInfo(
        frame_idx=idx, quad=quad, center=quad.mean(axis=0), width=30.0, height=20.0,
        confidence=0.9, valid=True,
    )


class _CountingCapture:
    def __init__(self, cap) -> None:
        self._cap = cap
        self.seeks = 0
        self.decoded = 0

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seeks += 1
        return self._cap.set(prop, value)

    def grab(self):
        self.decoded += 1
        return self._cap.grab()

    def read(self):
        self.decoded += 1
        return self._cap.read()

    def __getattr__(self, name):
        return getattr(self._cap, name)


class CandidateFrameTests(unittest.TestCase):
    N = 60

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # MJPG: every frame is a key frame, so OpenCV seeks exactly (with mp4v it can be
        # one frame off, which would test OpenCV rather than the code here).
        self.video = os.path.join(tmp.name, "clip.avi")
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(self.N):
            writer.write(np.full((48, 64, 3), (4 * i, 100, 50), np.uint8))  # blue = 4 * frame
        writer.release()

    def open_counting(self):
        return _CountingCapture(cv2.VideoCapture(self.video))

    def test_read_frames_in_order_returns_each_frame_once_in_order(self):
        cap = self.open_counting()
        try:
            got = list(gui.read_frames_in_order(cap, [25, 3, 7, 3, 29, 59]))
        finally:
            cap.release()
        self.assertEqual([idx for idx, _frame in got], [3, 7, 25, 29, 59])
        for idx, frame in got:
            self.assertAlmostEqual(float(frame[..., 0].mean()), 4 * idx, delta=3.0, msg=idx)
        # One seek to the first frame; the rest is read forward (59 - 29 < 48):
        # frames 3 to 59, each decoded once.
        self.assertEqual(cap.seeks, 1)
        self.assertEqual(cap.decoded, 57)

    def test_read_frames_in_order_seeks_across_large_gaps(self):
        cap = self.open_counting()
        try:
            got = list(gui.read_frames_in_order(cap, [2, 55], max_forward=10))
        finally:
            cap.release()
        self.assertEqual([idx for idx, _frame in got], [2, 55])
        self.assertAlmostEqual(float(got[1][1][..., 0].mean()), 4 * 55, delta=3.0)
        self.assertEqual(cap.seeks, 2)

    def make_app(self):
        return SimpleNamespace(video_path=self.video, unified_size=(32, 20), _patch_cache={})

    def test_candidate_patches_are_shared(self):
        app = self.make_app()
        frames = [_frame_info(i) for i in (40, 5, 12)]
        opened = []
        real_capture = cv2.VideoCapture

        def counting_capture(path):
            opened.append(path)
            return real_capture(path)

        with mock.patch.object(gui.cv2, "VideoCapture", side_effect=counting_capture):
            first = gui.MouthSpriteExtractorApp._candidate_patches(app, frames)
            again = gui.MouthSpriteExtractorApp._candidate_patches(app, list(reversed(frames)))
            # A different quad for the same frame is a different patch.
            moved = [_frame_info(12, x=20.0)]
            gui.MouthSpriteExtractorApp._candidate_patches(app, moved)

        self.assertEqual(len(opened), 2)  # the first call and the moved quad; not the second call
        for patch in first:
            self.assertIsNotNone(patch)
            self.assertEqual(patch.shape, (20, 32, 3))
        self.assertTrue(all(a is b for a, b in zip(first, reversed(again))))
        expected = extractor.warp_frame_to_norm(
            np.full((48, 64, 3), (4 * 5, 100, 50), np.uint8), frames[1].quad, 32, 20,
        )
        self.assertLess(float(np.abs(first[1].astype(int) - expected.astype(int)).mean()), 3.0)

    def test_unreadable_video_gives_none(self):
        app = self.make_app()
        app.video_path = os.path.join(os.path.dirname(self.video), "missing.avi")
        self.assertEqual(gui.MouthSpriteExtractorApp._candidate_patches(app, [_frame_info(1)]), [None])

    def test_auto_assign_reports_an_unreadable_video(self):
        app = SimpleNamespace(
            candidate_frames=[_frame_info(1), _frame_info(2)],
            unified_size=(32, 20),
            _candidate_patches=lambda frames: [None] * len(frames),
            log=mock.Mock(),
            after=mock.Mock(),
        )
        self.assertFalse(gui.MouthSpriteExtractorApp._run_auto_assign_internal(app))
        app.after.assert_not_called()


class FaceDetectorErrorTests(unittest.TestCase):
    def test_the_detectors_message_is_in_the_error(self):
        with tempfile.TemporaryDirectory() as td:
            script = os.path.join(td, "fake_detector.py")
            with open(script, "w", encoding="utf-8") as f:
                f.write(
                    "import sys\n"
                    "print('[error] 重みファイルの SHA-256 が一致しません。重みを消して取り直してください')\n"
                    "sys.exit(1)\n"
                )
            app = SimpleNamespace(video_path=os.path.join(td, "clip.mp4"))
            with (
                mock.patch.object(extractor, "resolve_adjacent_or_repo_script", return_value=script),
                mock.patch.object(extractor, "resolve_python_subprocess_executable", return_value=sys.executable),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    extractor.MouthSpriteExtractor._run_face_detector(app)
        message = str(ctx.exception)
        self.assertIn("重みファイルの SHA-256 が一致しません", message)
        self.assertIn("exit code 1", message)


if __name__ == "__main__":
    unittest.main()
