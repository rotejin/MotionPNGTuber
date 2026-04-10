"""Tests for Unicode-safe sprite loading in calibrate_mouth_track."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from calibrate_mouth_track import load_bgra
from motionpngtuber.mouth_sprite_extractor import write_image_file


class LoadBgraTests(unittest.TestCase):
    def test_loads_unicode_path_png(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "日本語フォルダ"
            out_dir.mkdir()
            out_path = out_dir / "open.png"

            img = np.zeros((8, 8, 4), dtype=np.uint8)
            img[:, :, 0] = 10
            img[:, :, 1] = 20
            img[:, :, 2] = 30
            img[:, :, 3] = 255
            self.assertTrue(write_image_file(str(out_path), img))

            loaded = load_bgra(str(out_path))

            self.assertEqual(loaded.shape, (8, 8, 4))
            self.assertEqual(loaded[0, 0].tolist(), [10, 20, 30, 255])
            self.assertEqual(int(loaded[0, 0, 3]), 255)


if __name__ == "__main__":
    unittest.main()
