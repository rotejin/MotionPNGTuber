"""Tests for convert_npz_to_json Unicode-path handling."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from convert_npz_to_json import convert_npz_to_json


class ConvertNpzToJsonTests(unittest.TestCase):
    def test_writes_json_under_unicode_directory(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "日本語フォルダ"
            out_dir.mkdir()
            npz_path = out_dir / "mouth_track_calibrated.npz"

            np.savez_compressed(
                npz_path,
                fps=np.float32(30.0),
                w=np.int32(640),
                h=np.int32(360),
                ref_sprite_w=np.int32(128),
                ref_sprite_h=np.int32(96),
                calib_offset=np.array([1.5, -2.0], dtype=np.float32),
                calib_scale=np.float32(1.0),
                calib_rotation=np.float32(0.0),
                quad=np.zeros((2, 4, 2), dtype=np.float32),
                valid=np.ones((2,), dtype=np.uint8),
            )

            output = convert_npz_to_json(npz_path, out_dir)

            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["width"], 640)
            self.assertEqual(len(payload["frames"]), 2)

    def test_rejects_missing_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            npz_path = out_dir / "mouth_track_calibrated.npz"

            np.savez_compressed(
                npz_path,
                fps=np.float32(30.0),
                w=np.int32(640),
                h=np.int32(360),
                ref_sprite_w=np.int32(128),
                ref_sprite_h=np.int32(96),
                calib_offset=np.array([1.5, -2.0], dtype=np.float32),
                calib_scale=np.float32(1.0),
                calib_rotation=np.float32(0.0),
                quad=np.zeros((2, 4, 2), dtype=np.float32),
            )

            with self.assertRaisesRegex(ValueError, "missing keys"):
                convert_npz_to_json(npz_path, out_dir)


if __name__ == "__main__":
    unittest.main()
