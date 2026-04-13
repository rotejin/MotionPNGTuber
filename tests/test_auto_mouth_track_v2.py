"""Focused tests for auto_mouth_track_v2 regression paths."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from auto_mouth_track_v2 import Metrics, _segment_repair, build_arg_parser


class SegmentRepairTests(unittest.TestCase):
    def _write_track_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            quad=np.zeros((4, 4, 2), dtype=np.float32),
            valid=np.array([1, 0, 1, 1], dtype=np.uint8),
            confidence=np.array([0.9, 0.1, 0.9, 0.9], dtype=np.float32),
            w=np.int32(64),
            h=np.int32(64),
        )

    def test_segment_repair_returns_persistent_file(self):
        with tempfile.TemporaryDirectory() as td:
            best_npz = os.path.join(td, "best.npz")
            self._write_track_npz(best_npz)

            def fake_extract(_video: str, _s: int, _e: int, sub_mp4: str) -> bool:
                with open(sub_mp4, "wb") as f:
                    f.write(b"video")
                return True

            def fake_run(_detector: str, seg_args: dict[str, str | None]) -> int:
                seg_out = str(seg_args["--out"])
                shutil.copy2(best_npz, seg_out)
                return 0

            def fake_stitch(_base_npz: str, seg_npz: str, dst_npz: str, offset: int) -> None:
                self.assertEqual(offset, 0)
                shutil.copy2(seg_npz, dst_npz)

            before = Metrics(0.50, 0.40, 0.30, 0.10, 4, 2)
            after = Metrics(1.00, 0.95, 0.90, 0.01, 4, 4)

            with (
                mock.patch("auto_mouth_track_v2._find_bad_segments", return_value=[(0, 1)]),
                mock.patch("auto_mouth_track_v2._extract_subvideo", side_effect=fake_extract),
                mock.patch("auto_mouth_track_v2.run_detector", side_effect=fake_run),
                mock.patch("auto_mouth_track_v2._stitch_segment", side_effect=fake_stitch),
                mock.patch("auto_mouth_track_v2.compute_metrics_npz", side_effect=[before, after]),
            ):
                repaired = _segment_repair(
                    video="dummy.mp4",
                    detector_py="detector.py",
                    best_npz=best_npz,
                    base_args={},
                    bad_conf_thr=0.3,
                    bad_max_len=5,
                    pad_frames=2,
                    max_segments=1,
                )

            self.assertIsNotNone(repaired)
            assert repaired is not None
            self.assertTrue(os.path.isfile(repaired))
            with np.load(repaired, allow_pickle=False) as npz:
                self.assertIn("quad", npz.files)
                self.assertEqual(npz["quad"].shape, (4, 4, 2))
            os.unlink(repaired)


class ArgParserTests(unittest.TestCase):
    def test_device_defaults_to_auto(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--video", "loop.mp4", "--out", "out.npz"])
        self.assertEqual(args.device, "auto")


if __name__ == "__main__":
    unittest.main()
