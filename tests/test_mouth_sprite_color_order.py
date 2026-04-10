"""Color-order regression tests for mouth sprite extraction."""
from __future__ import annotations

import unittest

import numpy as np

from mouth_sprite_extractor import extract_mouth_sprite
from mouth_sprite_extractor_gui import extract_sprite_with_crop


class MouthSpriteColorOrderTests(unittest.TestCase):
    def setUp(self):
        self.frame_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        self.frame_bgr[:, :] = [0, 0, 255]  # red in BGR
        self.quad = np.array(
            [[0, 0], [9, 0], [9, 9], [0, 9]],
            dtype=np.float32,
        )

    def test_extract_mouth_sprite_returns_true_rgba_channel_order(self):
        rgba = extract_mouth_sprite(
            self.frame_bgr,
            self.quad,
            10,
            10,
            feather_px=0,
            mask_scale=1.0,
        )
        self.assertEqual(rgba[5, 5].tolist(), [255, 0, 0, 255])

    def test_extract_sprite_with_crop_returns_true_rgba_channel_order(self):
        rgba = extract_sprite_with_crop(
            self.frame_bgr,
            self.quad,
            10,
            10,
            feather_px=0,
        )
        self.assertEqual(rgba[5, 5].tolist(), [255, 0, 0, 255])


if __name__ == "__main__":
    unittest.main()
