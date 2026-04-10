from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from motionpngtuber.lipsync_core import load_mouth_sprites


class LoadMouthSpritesRegressionTests(unittest.TestCase):
    def test_open_only_sprite_set_can_generate_wider_e_variant_without_crash(self):
        with tempfile.TemporaryDirectory() as td:
            mouth_dir = Path(td)
            open_rgba = np.zeros((50, 100, 4), dtype=np.uint8)
            open_rgba[10:40, 20:80] = [255, 40, 40, 255]
            Image.fromarray(open_rgba, "RGBA").save(mouth_dir / "open.png")

            sprites = load_mouth_sprites(str(mouth_dir), full_w=200, full_h=200)

        self.assertEqual(set(sprites.keys()), {"closed", "half", "open", "u", "e"})
        self.assertEqual(sprites["open"].shape, (50, 100, 4))
        self.assertEqual(sprites["e"].shape, sprites["open"].shape)
        self.assertGreater(int(np.count_nonzero(sprites["e"][..., 3])), 0)


if __name__ == "__main__":
    unittest.main()
