import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from auto_erase_mouth import (
    Track,
    build_erase_candidates,
    parse_coverages,
    select_ref_frame,
    should_enable_probe,
    _slice_track_file,
)


class ParseCoveragesTests(unittest.TestCase):
    def test_parses_csv(self):
        vals = parse_coverages("0.60, 0.70,0.80")
        self.assertEqual(vals, [0.60, 0.70, 0.80])

    def test_empty_uses_default(self):
        vals = parse_coverages("")
        self.assertEqual(vals, [0.60, 0.70, 0.80])


class BuildEraseCandidatesTests(unittest.TestCase):
    def test_builds_cross_product(self):
        cands = build_erase_candidates(["hold", "strict"], [0.6, 0.7])
        self.assertEqual(
            [(c.valid_policy, c.coverage) for c in cands],
            [("hold", 0.6), ("hold", 0.7), ("strict", 0.6), ("strict", 0.7)],
        )

    def test_deduplicates(self):
        cands = build_erase_candidates(["hold", "hold"], [0.6, 0.6])
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0].valid_policy, "hold")
        self.assertAlmostEqual(cands[0].coverage, 0.6)


class ShouldEnableProbeTests(unittest.TestCase):
    def test_requires_enough_candidates(self):
        self.assertFalse(should_enable_probe(total_frames=1000, fps=30.0, candidate_count=3))

    def test_requires_long_video(self):
        self.assertFalse(should_enable_probe(total_frames=120, fps=30.0, candidate_count=4))

    def test_enables_for_long_video_and_many_candidates(self):
        self.assertTrue(should_enable_probe(total_frames=600, fps=30.0, candidate_count=4))


class SliceTrackFileTests(unittest.TestCase):
    def test_slices_frame_aligned_arrays_and_keeps_metadata(self):
        quad = np.arange(10 * 4 * 2, dtype=np.float32).reshape(10, 4, 2)
        valid = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=np.uint8)
        conf = np.linspace(0.1, 1.0, 10, dtype=np.float32)
        calib_offset = np.array([3.0, -2.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src.npz")
            dst = os.path.join(td, "dst.npz")
            np.savez_compressed(
                src,
                quad=quad,
                valid=valid,
                confidence=conf,
                calib_offset=calib_offset,
                w=np.int32(1280),
                h=np.int32(720),
            )

            count = _slice_track_file(src, dst, start_f=2, count_f=4)
            self.assertEqual(count, 4)

            with np.load(dst, allow_pickle=False) as npz:
                self.assertEqual(npz["quad"].shape[0], 4)
                self.assertEqual(npz["valid"].shape[0], 4)
                self.assertEqual(npz["confidence"].shape[0], 4)
                np.testing.assert_array_equal(npz["quad"], quad[2:6])
                np.testing.assert_array_equal(npz["valid"], valid[2:6])
                np.testing.assert_array_equal(npz["confidence"], conf[2:6])
                np.testing.assert_array_equal(npz["calib_offset"], calib_offset)
                self.assertEqual(int(npz["w"]), 1280)
                self.assertEqual(int(npz["h"]), 720)


class SelectRefFrameTests(unittest.TestCase):
    def test_smart_mode_passes_ref_topk(self):
        track = Track(
            quad=np.zeros((3, 4, 2), dtype=np.float32),
            valid=np.array([1, 1, 1], dtype=bool),
            filled=np.zeros((3, 4, 2), dtype=np.float32),
            confidence=np.array([0.1, 0.9, 0.4], dtype=np.float32),
            total=3,
        )
        with mock.patch("auto_erase_mouth.choose_ref_frame_smart", return_value=1) as mocked:
            ref_idx = select_ref_frame(
                "dummy.mp4",
                track,
                n_out=3,
                norm_w=64,
                norm_h=32,
                ref_mode="smart",
                coverage_for_mask=0.7,
                ref_topk=7,
            )

        self.assertEqual(ref_idx, 1)
        mocked.assert_called_once_with(
            "dummy.mp4",
            track,
            n_out=3,
            norm_w=64,
            norm_h=32,
            coverage_for_mask=0.7,
            top_k=7,
        )

    def test_confidence_mode_falls_back_to_first_valid_without_confidence(self):
        track = Track(
            quad=np.zeros((4, 4, 2), dtype=np.float32),
            valid=np.array([0, 1, 1, 0], dtype=bool),
            filled=np.zeros((4, 4, 2), dtype=np.float32),
            confidence=None,
            total=4,
        )
        ref_idx = select_ref_frame(
            "dummy.mp4",
            track,
            n_out=4,
            norm_w=64,
            norm_h=32,
            ref_mode="confidence",
            coverage_for_mask=0.6,
            ref_topk=5,
        )
        self.assertEqual(ref_idx, 1)


if __name__ == "__main__":
    unittest.main()
