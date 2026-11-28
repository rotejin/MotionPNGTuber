from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from motionpngtuber.lipsync_core import load_mouth_sprites, load_rgba


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


class LoadRgbaPngOnlyTests(unittest.TestCase):
    """Mouth sprites are read with the PNG decoder only (see .github/SECURITY.md)."""

    def test_other_formats_named_png_are_refused(self):
        rgba = np.zeros((8, 8, 4), dtype=np.uint8)
        rgba[..., 3] = 255
        for fmt in ("TIFF", "BMP", "GIF"):
            with self.subTest(fmt=fmt), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "open.png"
                Image.fromarray(rgba, "RGBA").save(path, format=fmt)

                with self.assertRaises(ValueError):
                    load_rgba(str(path))

    def test_png_is_read_and_the_file_is_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "open.png"
            rgba = np.zeros((6, 4, 4), dtype=np.uint8)
            rgba[1, 2] = [10, 20, 30, 255]
            Image.fromarray(rgba, "RGBA").save(path)

            loaded = load_rgba(str(path))
            # On Windows an open file cannot be removed.
            path.unlink()

        self.assertEqual(loaded.shape, (6, 4, 4))
        self.assertEqual(loaded[1, 2].tolist(), [10, 20, 30, 255])


if __name__ == "__main__":
    unittest.main()
