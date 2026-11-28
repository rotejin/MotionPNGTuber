"""Focused tests for auto_mouth_track_v2 regression paths."""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

import auto_mouth_track_v2
from auto_mouth_track_v2 import Metrics, _segment_repair, build_arg_parser


class TempFolderCleanupTests(unittest.TestCase):
    """main() removes its temp folder, also on Windows (an open .npz cannot be removed there).

    The GUI runs this script for every analysis, with segment repair on by default, and
    the best candidate's track used to stay open until main() returned, so
    %TEMP%\\auto_mouth_track_*\\cand_*.npz was left behind every time.
    """

    def test_main_removes_its_temp_folder(self):
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "clip.mp4")
            detector = os.path.join(td, "detector.py")
            out = os.path.join(td, "mouth_track.npz")
            for path in (video, detector):
                with open(path, "wb"):
                    pass
            created = []
            real_mkdtemp = tempfile.mkdtemp

            def recording_mkdtemp(*args, **kwargs):
                kwargs["dir"] = td
                path = real_mkdtemp(*args, **kwargs)
                created.append(path)
                return path

            def fake_detector(_detector_py, args_map):
                n = 40
                quad = np.tile(np.array([[10, 10], [30, 10], [30, 20], [10, 20]], np.float32), (n, 1, 1))
                valid = np.ones(n, np.uint8)
                valid[5] = 0  # a short bad run, so that segment repair looks at the track
                np.savez_compressed(
                    args_map["--out"], quad=quad, valid=valid,
                    confidence=np.full(n, 0.9, np.float32), w=np.int32(64), h=np.int32(64),
                )
                return 0

            argv = ["auto_mouth_track_v2.py", "--video", video, "--out", out, "--detector", detector,
                    "--early-stop", "--max-tries", "4"]
            with (
                mock.patch.object(auto_mouth_track_v2.sys, "argv", argv),
                mock.patch.object(auto_mouth_track_v2.tempfile, "mkdtemp", side_effect=recording_mkdtemp),
                mock.patch.object(auto_mouth_track_v2, "get_video_info", return_value=(25.0, 40)),
                mock.patch.object(auto_mouth_track_v2, "run_detector", side_effect=fake_detector),
                mock.patch.object(auto_mouth_track_v2, "_segment_repair", return_value=None),
                mock.patch("builtins.print"),
            ):
                rc = auto_mouth_track_v2.main()

            self.assertEqual(rc, 0)
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(created)
            for path in created:
                if os.path.exists(path):
                    self.fail(f"temp folder left behind: {os.listdir(path)}")


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
