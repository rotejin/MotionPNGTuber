"""Small image I/O helpers with Unicode-path support."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


def read_png_rgba(path: str) -> np.ndarray:
    """Read a PNG file as an 8-bit RGBA array (H, W, 4).

    Only Pillow's PNG decoder is used. Pillow and OpenCV otherwise pick the
    decoder from the file contents, so a file named ``*.png`` from someone else
    could reach the TIFF, JPEG or other decoders (see .github/SECURITY.md).
    Grayscale, palette and 16-bit PNGs become 8-bit RGBA, keeping their
    transparency, so every tool sees the same colours.

    Raises ``ValueError`` if the file is not a PNG, and ``OSError`` (for example
    ``FileNotFoundError``) if it cannot be opened or is broken.
    """
    try:
        with Image.open(path, formats=["PNG"]) as im:
            return _to_rgba8(im)
    except UnidentifiedImageError as e:
        raise ValueError(f"not a PNG file: {path}") from e


def _to_rgba8(im: Image.Image) -> np.ndarray:
    if im.mode.startswith("I"):
        # Pillow keeps 16-bit grayscale as "I;16" (or "I"), and convert("RGBA")
        # would clip every value above 255 to white. Keep the upper 8 bits.
        raw = np.asarray(im).astype(np.uint32)
        gray = (raw >> 8).astype(np.uint8)
        alpha = np.full(gray.shape, 255, dtype=np.uint8)
        transparent = im.info.get("transparency")
        if isinstance(transparent, int):
            alpha[raw == transparent] = 0
        return np.dstack([gray, gray, gray, alpha])
    return np.array(im.convert("RGBA"), dtype=np.uint8)


def read_image_bgra(path: str) -> np.ndarray | None:
    """Read a PNG in OpenCV channel order (8-bit BGRA), or None if it cannot be read.

    Uses the PNG decoder only, like :func:`read_png_rgba`.
    """
    try:
        rgba = read_png_rgba(path)
    except (OSError, ValueError):
        return None
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def write_image_file(path: str, image: np.ndarray) -> bool:
    """Write an image using a Unicode-safe path strategy."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    encoded.tofile(path)
    return True
