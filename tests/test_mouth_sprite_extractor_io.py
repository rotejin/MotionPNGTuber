"""Tests for mouth_sprite_extractor image output helpers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from motionpngtuber.mouth_sprite_extractor import write_image_file


class WriteImageFileTests(unittest.TestCase):
    def test_writes_png_to_unicode_path(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "日本語フォルダ"
            out_dir.mkdir()
            out_path = out_dir / "open.png"

            img = np.zeros((8, 8, 4), dtype=np.uint8)
            img[:, :, 3] = 255

            ok = write_image_file(str(out_path), img)

            self.assertTrue(ok)
            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
