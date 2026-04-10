import unittest

import numpy as np

from mouth_color_adjust import (
    MouthColorAdjust,
    alpha_bbox_from_mask,
    apply_basic_color_adjust_3ch,
    apply_inspect_boost_3ch,
    apply_mouth_color_adjust_4ch,
    build_edge_weight,
    clamp_mouth_color_adjust,
    estimate_auto_mouth_color_adjust,
    sample_background_ring_mean_3ch,
    sample_colored_edge_mean_4ch,
)


class AlphaBboxFromMaskTests(unittest.TestCase):
    def test_returns_none_for_empty_mask(self):
        alpha = np.zeros((5, 6), dtype=np.uint8)
        self.assertIsNone(alpha_bbox_from_mask(alpha))

    def test_returns_bbox_for_nonzero_region(self):
        alpha = np.zeros((6, 7), dtype=np.uint8)
        alpha[2:5, 1:4] = 255
        self.assertEqual(alpha_bbox_from_mask(alpha), (1, 2, 4, 5))


class ClampMouthColorAdjustTests(unittest.TestCase):
    def test_returns_new_instance(self):
        cfg = MouthColorAdjust(brightness=100.0)
        clamped = clamp_mouth_color_adjust(cfg)
        self.assertIsNot(cfg, clamped)
        self.assertEqual(cfg.brightness, 100.0)
        self.assertEqual(clamped.brightness, 32.0)


class BuildEdgeWeightTests(unittest.TestCase):
    def test_empty_alpha_returns_zero_weight(self):
        alpha = np.zeros((10, 10), dtype=np.uint8)
        weight = build_edge_weight(alpha, 0.1)
        self.assertEqual(weight.dtype, np.float32)
        self.assertEqual(float(weight.max()), 0.0)

    def test_edge_has_higher_weight_than_center(self):
        alpha = np.zeros((20, 20), dtype=np.uint8)
        alpha[3:17, 3:17] = 255
        weight = build_edge_weight(alpha, 0.15)
        self.assertGreater(float(weight[3, 10]), float(weight[10, 10]))


class ApplyBasicColorAdjust3chTests(unittest.TestCase):
    def test_bgr_warmth_increases_red_and_decreases_blue(self):
        img = np.full((2, 2, 3), 100, dtype=np.uint8)
        out = apply_basic_color_adjust_3ch(
            img,
            MouthColorAdjust(warmth=10.0),
            color_order="BGR",
        )
        self.assertGreater(int(out[..., 2].mean()), 100)
        self.assertLess(int(out[..., 0].mean()), 100)

    def test_rgb_warmth_increases_red_and_decreases_blue(self):
        img = np.full((2, 2, 3), 100, dtype=np.uint8)
        out = apply_basic_color_adjust_3ch(
            img,
            MouthColorAdjust(warmth=10.0),
            color_order="RGB",
        )
        self.assertGreater(int(out[..., 0].mean()), 100)
        self.assertLess(int(out[..., 2].mean()), 100)


class ApplyMouthColorAdjust4chTests(unittest.TestCase):
    def _make_rgba(self, *, color_order: str) -> np.ndarray:
        img = np.zeros((12, 12, 4), dtype=np.uint8)
        img[..., 3] = 0
        img[2:10, 2:10, 3] = 255
        if color_order == "BGRA":
            img[2:10, 2:10, 0] = 80
            img[2:10, 2:10, 1] = 100
            img[2:10, 2:10, 2] = 120
        else:
            img[2:10, 2:10, 0] = 120
            img[2:10, 2:10, 1] = 100
            img[2:10, 2:10, 2] = 80
        return img

    def test_preserves_alpha(self):
        img = self._make_rgba(color_order="BGRA")
        out = apply_mouth_color_adjust_4ch(
            img,
            MouthColorAdjust(warmth=10.0, color_strength=1.0),
            color_order="BGRA",
        )
        np.testing.assert_array_equal(out[..., 3], img[..., 3])

    def test_color_strength_zero_leaves_rgb_unchanged(self):
        img = self._make_rgba(color_order="BGRA")
        out = apply_mouth_color_adjust_4ch(
            img,
            MouthColorAdjust(warmth=20.0, color_strength=0.0),
            color_order="BGRA",
        )
        np.testing.assert_array_equal(out, img)

    def test_edge_priority_biases_edge_more_than_center(self):
        img = self._make_rgba(color_order="BGRA")
        out = apply_mouth_color_adjust_4ch(
            img,
            MouthColorAdjust(warmth=20.0, color_strength=1.0, edge_priority=1.0, edge_width_ratio=0.18),
            color_order="BGRA",
        )
        edge_delta = abs(int(out[2, 6, 2]) - int(img[2, 6, 2]))
        center_delta = abs(int(out[6, 6, 2]) - int(img[6, 6, 2]))
        self.assertGreater(edge_delta, center_delta)

    def test_rgba_uses_rgb_channel_order(self):
        img = self._make_rgba(color_order="RGBA")
        out = apply_mouth_color_adjust_4ch(
            img,
            MouthColorAdjust(warmth=10.0, color_strength=1.0, edge_priority=0.0),
            color_order="RGBA",
        )
        self.assertGreater(int(out[..., 0][img[..., 3] > 0].mean()), int(img[..., 0][img[..., 3] > 0].mean()))
        self.assertLess(int(out[..., 2][img[..., 3] > 0].mean()), int(img[..., 2][img[..., 3] > 0].mean()))


class ApplyInspectBoost3chTests(unittest.TestCase):
    def test_boost_one_returns_copy(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = apply_inspect_boost_3ch(img, 1.0, color_order="BGR")
        np.testing.assert_array_equal(out, img)
        self.assertIsNot(out, img)

    def test_bgr_and_rgb_orders_both_supported(self):
        img_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        img_bgr[..., 2] = 180
        img_bgr[..., 1] = 120
        out_bgr = apply_inspect_boost_3ch(img_bgr, 2.0, color_order="BGR")
        self.assertEqual(out_bgr.shape, img_bgr.shape)

        img_rgb = img_bgr[..., ::-1].copy()
        out_rgb = apply_inspect_boost_3ch(img_rgb, 2.0, color_order="RGB")
        self.assertEqual(out_rgb.shape, img_rgb.shape)


class AutoColorSampleTests(unittest.TestCase):
    def test_sample_colored_edge_mean_ignores_transparent_pixels(self):
        rgba = np.zeros((12, 12, 4), dtype=np.uint8)
        rgba[2:10, 2:10, 3] = 255
        rgba[2:10, 2:10, 0] = 120
        rgba[2:10, 2:10, 1] = 90
        rgba[2:10, 2:10, 2] = 60
        rgba[0:2, 0:2, 0] = 255
        rgba[0:2, 0:2, 1] = 255
        rgba[0:2, 0:2, 2] = 255
        sampled = sample_colored_edge_mean_4ch(
            rgba,
            edge_width_ratio=0.15,
            color_order="RGBA",
            alpha_threshold=24,
        )
        self.assertIsNotNone(sampled)
        mean, count = sampled  # type: ignore[misc]
        self.assertGreater(count, 0)
        self.assertLess(float(mean[0]), 200.0)

    def test_sample_background_ring_mean_reads_outside_alpha_region(self):
        frame = np.full((20, 20, 3), 30, dtype=np.uint8)
        frame[4:16, 4:16] = [100, 110, 120]
        alpha = np.zeros((8, 8), dtype=np.uint8)
        alpha[1:7, 1:7] = 255
        sampled = sample_background_ring_mean_3ch(
            frame,
            alpha,
            6,
            6,
            edge_width_ratio=0.15,
            color_order="RGB",
            alpha_threshold=24,
        )
        self.assertIsNotNone(sampled)
        mean, count = sampled  # type: ignore[misc]
        self.assertGreater(count, 0)
        self.assertGreater(float(mean[0]), 80.0)


class EstimateAutoMouthColorAdjustTests(unittest.TestCase):
    def test_moves_brightness_and_warmth_toward_background(self):
        cfg, debug = estimate_auto_mouth_color_adjust(
            MouthColorAdjust(brightness=0.0, saturation=1.0, warmth=0.0, color_strength=0.5),
            bg_mean=np.array([150.0, 120.0, 90.0], dtype=np.float32),
            mouth_mean=np.array([90.0, 100.0, 130.0], dtype=np.float32),
            color_order="RGB",
        )
        self.assertGreater(cfg.brightness, 0.0)
        self.assertGreater(cfg.warmth, 0.0)
        self.assertGreaterEqual(cfg.color_strength, 0.5)
        self.assertIn("delta_e", debug)

    def test_preserves_non_target_params(self):
        current = MouthColorAdjust(edge_priority=0.91, edge_width_ratio=0.14, inspect_boost=3.0)
        cfg, _debug = estimate_auto_mouth_color_adjust(
            current,
            bg_mean=np.array([120.0, 110.0, 100.0], dtype=np.float32),
            mouth_mean=np.array([118.0, 109.0, 101.0], dtype=np.float32),
            color_order="RGB",
        )
        self.assertAlmostEqual(cfg.edge_priority, current.edge_priority)
        self.assertAlmostEqual(cfg.edge_width_ratio, current.edge_width_ratio)
        self.assertAlmostEqual(cfg.inspect_boost, current.inspect_boost)

    def test_repeated_estimation_does_not_accumulate_brightness_or_warmth(self):
        current = MouthColorAdjust(brightness=12.0, saturation=1.3, warmth=8.0, color_strength=0.8)
        first, _ = estimate_auto_mouth_color_adjust(
            current,
            bg_mean=np.array([150.0, 120.0, 90.0], dtype=np.float32),
            mouth_mean=np.array([90.0, 100.0, 130.0], dtype=np.float32),
            color_order="RGB",
        )
        second, _ = estimate_auto_mouth_color_adjust(
            first,
            bg_mean=np.array([150.0, 120.0, 90.0], dtype=np.float32),
            mouth_mean=np.array([90.0, 100.0, 130.0], dtype=np.float32),
            color_order="RGB",
        )
        self.assertAlmostEqual(first.brightness, second.brightness)
        self.assertAlmostEqual(first.warmth, second.warmth)
        self.assertAlmostEqual(first.saturation, second.saturation)


if __name__ == "__main__":
    unittest.main()
