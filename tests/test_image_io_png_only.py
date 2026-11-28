"""Mouth sprites are read with the PNG decoder only, by every tool (see .github/SECURITY.md).

The live runtime (lipsync_core.load_rgba), the calibration (calibrate_mouth_track.load_bgra)
and the lightweight preview (image_io.read_image_bgra) must refuse other formats named
``*.png`` and must read every kind of PNG with the same colours.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrate_mouth_track import load_bgra
from motionpngtuber.image_io import read_image_bgra, read_png_rgba
from motionpngtuber.lipsync_core import load_rgba

BGR = (10, 120, 240)


def _encode(ext: str, image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise AssertionError(f"OpenCV could not write {ext}")
    return buf.tobytes()


class OtherFormatsNamedPngTests(unittest.TestCase):
    def test_every_reader_refuses_them(self):
        # OpenJPEG cannot write very small images, so use a normal sprite size.
        bgr = np.full((32, 48, 3), BGR, dtype=np.uint8)
        bgra = np.dstack([bgr, np.full((32, 48), 200, np.uint8)])
        cases = (
            ("TIFF", ".tiff", bgra),
            ("JPEG", ".jpg", bgr),
            ("BMP", ".bmp", bgr),
            ("WebP", ".webp", bgra),
            ("JPEG 2000", ".jp2", bgr),
            ("PPM", ".ppm", bgr),
            ("Radiance HDR", ".hdr", bgr.astype(np.float32) / 255.0),
        )
        for name, ext, image in cases:
            with self.subTest(fmt=name), tempfile.TemporaryDirectory() as td:
                path = Path(td) / "open.png"
                path.write_bytes(_encode(ext, image))

                self.assertIsNone(read_image_bgra(str(path)))
                with self.assertRaises(ValueError):
                    read_png_rgba(str(path))
                with self.assertRaises(ValueError):
                    load_bgra(str(path))
                with self.assertRaises(ValueError):
                    load_rgba(str(path))

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td) / "open.png")
            self.assertIsNone(read_image_bgra(path))
            with self.assertRaises(FileNotFoundError):
                load_bgra(path)


class PngVariantsTests(unittest.TestCase):
    """Every reader returns the same 8-bit pixels for every kind of PNG."""

    def read_all(self, data: bytes) -> list[list[int]]:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "open.png"
            path.write_bytes(data)
            preview = read_image_bgra(str(path))
            calib = load_bgra(str(path))
            live = load_rgba(str(path))
            # The files are closed afterwards (on Windows an open file cannot be removed).
            path.unlink()
        self.assertIsNotNone(preview)
        for image in (preview, calib, live):
            self.assertEqual(image.dtype, np.uint8)
            self.assertEqual(image.shape, (6, 8, 4))
        preview_rgba = cv2.cvtColor(preview, cv2.COLOR_BGRA2RGBA)
        calib_rgba = cv2.cvtColor(calib, cv2.COLOR_BGRA2RGBA)
        self.assertEqual(preview_rgba[0, 0].tolist(), live[0, 0].tolist())
        self.assertEqual(calib_rgba[0, 0].tolist(), live[0, 0].tolist())
        return live[0, 0].tolist()

    def save_with_pillow(self, image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def test_rgba_8bit(self):
        bgra = np.zeros((6, 8, 4), np.uint8)
        bgra[...] = (*BGR, 200)
        self.assertEqual(self.read_all(_encode(".png", bgra)), [240, 120, 10, 200])

    def test_grayscale(self):
        # Used to be refused by the calibration and the preview (2 channels in OpenCV).
        gray = np.full((6, 8), 120, np.uint8)
        self.assertEqual(self.read_all(_encode(".png", gray)), [120, 120, 120, 255])

    def test_grayscale_with_alpha(self):
        la = np.dstack([np.full((6, 8), 120, np.uint8), np.full((6, 8), 200, np.uint8)])
        data = self.save_with_pillow(Image.fromarray(la, mode="LA"))
        self.assertEqual(self.read_all(data), [120, 120, 120, 200])

    def test_palette_with_transparency(self):
        rgba = np.zeros((6, 8, 4), np.uint8)
        rgba[...] = (240, 120, 10, 200)
        rgba[0, 1] = (0, 0, 0, 0)
        data = self.save_with_pillow(Image.fromarray(rgba, mode="RGBA").quantize(colors=4))
        self.assertEqual(self.read_all(data), [240, 120, 10, 200])

    def test_rgba_16bit_keeps_the_upper_8_bits(self):
        # Low bytes differ from the high bytes, as in real 16-bit images.
        bgra16 = np.zeros((6, 8, 4), np.uint16)
        bgra16[...] = (10 * 256 + 0x80, 120 * 256 + 0x33, 240 * 256 + 0xF0, 200 * 256 + 0x7F)
        self.assertEqual(self.read_all(_encode(".png", bgra16)), [240, 120, 10, 200])

    def test_grayscale_16bit_is_not_white(self):
        gray16 = np.full((6, 8), 120 * 256 + 77, np.uint16)
        self.assertEqual(self.read_all(_encode(".png", gray16)), [120, 120, 120, 255])


if __name__ == "__main__":
    unittest.main()
