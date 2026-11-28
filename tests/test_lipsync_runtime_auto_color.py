"""The live "auto mouth colour" button: stable result and no long freeze.

- The mouth colour is sampled from the original sprites. The estimate is absolute and
  is applied to the original sprites again, so sampling the adjusted sprites made the
  value change every time the button was pressed.
- Without the virtual camera, the full-resolution frame was played with
  BgVideo.get_frame(): pressing the button again after a few minutes decoded every
  frame in between (tens of seconds) while the live window froze. It now seeks to the
  frame shown in the preview.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import cv2
import numpy as np

from loop_lipsync_runtime_patched_emotion_auto import (
    BgVideo,
    MouthColorAdjust,
    MouthTrack,
    _estimate_auto_color_for_request,
)
from motionpngtuber.mouth_color_adjust import apply_mouth_color_adjust_4ch

SKIN = (236, 205, 190)
SHAPES = ("closed", "half", "open", "e", "u")


def _sprite() -> np.ndarray:
    spr = np.zeros((30, 40, 4), np.uint8)
    cv2.ellipse(spr, (20, 15), (16, 10), 0, 0, 360, (150, 40, 60, 255), -1)
    return spr


def _track(path: str, w: int, h: int) -> MouthTrack:
    quad = np.array([[60, 50], [100, 50], [100, 80], [60, 80]], np.float32)
    np.savez(path, quad=np.tile(quad, (10, 1, 1)), valid=np.ones(10, np.uint8), w=160, h=120)
    track = MouthTrack.load(path, w, h)
    assert track is not None
    return track


class AutoColorEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        npz = os.path.join(tmp.name, "track.npz")
        self.preview_track = _track(npz, 160, 120)
        self.full_track = _track(npz, 320, 240)
        self.preview_frame = np.full((120, 160, 3), SKIN, np.uint8)
        self.full_frame = np.full((240, 320, 3), SKIN, np.uint8)
        self.original = {"Default": {shape: _sprite() for shape in SHAPES}}

    def estimate(self, mouth_sets, cfg, full_frame_source=None):
        return _estimate_auto_color_for_request(
            mouth_sets_original=mouth_sets,
            current_emotion="Default",
            mouth_shape="open",
            current_cfg=cfg,
            fixed_x=80,
            fixed_y=60,
            preview_frame=self.preview_frame,
            preview_frame_idx=3,
            preview_track=self.preview_track,
            preview_scale=0.5,
            full_frame_source=full_frame_source,
        )

    def assertSameAdjust(self, a: MouthColorAdjust, b: MouthColorAdjust) -> None:
        for name in ("brightness", "saturation", "warmth", "color_strength"):
            self.assertAlmostEqual(getattr(a, name), getattr(b, name), places=4, msg=name)

    def test_pressing_again_gives_the_same_adjustment(self):
        first, _ = self.estimate(self.original, MouthColorAdjust())
        self.assertIsNotNone(first)
        # The runtime applies the result to the original sprites and keeps sampling them.
        second, _ = self.estimate(self.original, first["cfg"])
        third, _ = self.estimate(self.original, second["cfg"])
        self.assertSameAdjust(first["cfg"], second["cfg"])
        self.assertSameAdjust(first["cfg"], third["cfg"])

    def test_sampling_the_adjusted_sprites_would_change_the_result(self):
        # Shows that the scene above can tell the two apart (what the runtime used to do).
        first, _ = self.estimate(self.original, MouthColorAdjust())
        cfg = first["cfg"]
        adjusted = {
            "Default": {
                shape: apply_mouth_color_adjust_4ch(spr, cfg, color_order="RGBA")
                for shape, spr in self.original["Default"].items()
            }
        }
        second, _ = self.estimate(adjusted, cfg)
        self.assertGreater(
            abs(second["cfg"].brightness - cfg.brightness) + abs(second["cfg"].warmth - cfg.warmth),
            1.0,
        )

    def test_full_resolution_uses_the_frame_shown_in_the_preview(self):
        asked = []

        def full_frame_source(frame_idx):
            asked.append(frame_idx)
            return self.full_frame, frame_idx, self.full_track

        full, source = self.estimate(self.original, MouthColorAdjust(), full_frame_source)
        preview, _ = self.estimate(self.original, MouthColorAdjust())
        self.assertEqual(source, "full")
        self.assertEqual(asked, [3])
        self.assertAlmostEqual(full["cfg"].brightness, preview["cfg"].brightness, delta=2.0)
        self.assertAlmostEqual(full["cfg"].warmth, preview["cfg"].warmth, delta=2.0)

    def test_falls_back_to_the_preview(self):
        def broken(_frame_idx):
            raise RuntimeError("cannot open the video")

        for full_frame_source in (broken, lambda _idx: None):
            with self.subTest(source=full_frame_source):
                estimated, source = self.estimate(self.original, MouthColorAdjust(), full_frame_source)
                self.assertEqual(source, "preview")
                self.assertIsNotNone(estimated)


class _CountingCapture:
    def __init__(self, cap) -> None:
        self._cap = cap
        self.reads = 0

    def read(self):
        self.reads += 1
        return self._cap.read()

    def __getattr__(self, name):
        return getattr(self._cap, name)


class BgVideoReadFrameAtTests(unittest.TestCase):
    """read_frame_at() seeks to one frame instead of playing every frame in between."""

    N = 30

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # MJPG: every frame is a key frame, so OpenCV seeks exactly (with mp4v its seek
        # can be one frame off, which would test OpenCV rather than read_frame_at).
        self.path = os.path.join(tmp.name, "loop.avi")
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(self.N):
            writer.write(np.full((48, 64, 3), (8 * i, 100, 50), np.uint8))  # BGR: blue = 8 * frame
        writer.release()
        self.video = BgVideo(self.path, 32, 24)
        self.addCleanup(self.video.close)

    def blue_of(self, frame_rgb: np.ndarray) -> float:
        return float(frame_rgb[..., 2].mean())

    def test_reads_the_requested_frame_with_one_read(self):
        counting = _CountingCapture(self.video.cap)
        self.video.cap = counting
        for idx in (25, 7, 25 + self.N):
            with self.subTest(idx=idx):
                counting.reads = 0
                frame = self.video.read_frame_at(idx)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (24, 32, 3))
                self.assertEqual(counting.reads, 1)
                expected = idx % self.N
                self.assertEqual(self.video.frame_idx, expected)
                self.assertAlmostEqual(self.blue_of(frame), 8 * expected, delta=3.0)

    def test_returns_none_when_the_frame_cannot_be_read(self):
        self.video.cap.release()
        self.assertIsNone(self.video.read_frame_at(5))


if __name__ == "__main__":
    unittest.main()
