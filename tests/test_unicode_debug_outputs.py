"""Unicode-path tests for debug image outputs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from auto_mouth_track_v2 import save_best_debug_outputs
from erase_mouth_offline import save_debug_metrics
from face_track_anime_detector import save_metrics_png


class UnicodeDebugOutputTests(unittest.TestCase):
    def test_auto_mouth_track_debug_png_under_unicode_dir(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "日本語フォルダ"
            out_dir.mkdir()
            npz_path = out_dir / "best.npz"
            np.savez_compressed(
                npz_path,
                confidence=np.array([0.1, 0.8, 0.6], dtype=np.float32),
                valid=np.array([1, 1, 0], dtype=np.uint8),
                quad=np.zeros((3, 4, 2), dtype=np.float32),
            )

            save_best_debug_outputs(str(out_dir), str(npz_path), {"best": "ok"})

            self.assertTrue((out_dir / "auto_best.png").is_file())
            self.assertTrue((out_dir / "auto_metrics.json").is_file())

    def test_face_track_metrics_png_under_unicode_path(self):
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "日本語フォルダ" / "metrics.png"
            save_metrics_png(
                str(out_path),
                {
                    "confidence": np.array([0.1, 0.5, 0.9], dtype=np.float32),
                    "valid": np.array([1.0, 1.0, 0.0], dtype=np.float32),
                },
                title="unicode",
            )
            self.assertTrue(out_path.is_file())

    def test_erase_debug_metrics_under_unicode_dir(self):
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "日本語フォルダ"
            payload = {"fps": 30.0, "frames": 3}
            valid = np.array([1, 0, 1], dtype=np.uint8)

            save_debug_metrics(str(out_dir), payload, valid)

            self.assertTrue((out_dir / "metrics.json").is_file())
            self.assertTrue((out_dir / "metrics.png").is_file())


if __name__ == "__main__":
    unittest.main()
