"""Round-trip tests for BGRA/RGBA image loading helpers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from motionpngtuber.image_io import read_image_bgra, write_image_file
from motionpngtuber.lipsync_core import load_rgba


class ImageIoColorOrderTests(unittest.TestCase):
    def test_write_then_read_bgra_and_rgba_are_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "sprite.png"

            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[:, :] = [255, 0, 0, 255]
            bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            self.assertTrue(write_image_file(str(out_path), bgra))

            loaded_bgra = read_image_bgra(str(out_path))
            loaded_rgba = load_rgba(str(out_path))

            self.assertIsNotNone(loaded_bgra)
            self.assertEqual(loaded_bgra[0, 0].tolist(), [0, 0, 255, 255])
            self.assertEqual(loaded_rgba[0, 0].tolist(), [255, 0, 0, 255])


if __name__ == "__main__":
    unittest.main()
