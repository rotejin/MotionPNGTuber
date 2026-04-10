"""Small image I/O helpers with Unicode-path support."""
from __future__ import annotations

import os

import cv2
import numpy as np


def read_image_file(path: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """Read an image using a Unicode-safe path strategy."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def read_image_bgra(path: str) -> np.ndarray | None:
    """Read an image in OpenCV-native channel order (BGR/BGRA)."""
    img = read_image_file(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        return None
    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return img


def write_image_file(path: str, image: np.ndarray) -> bool:
    """Write an image using a Unicode-safe path strategy."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    encoded.tofile(path)
    return True
