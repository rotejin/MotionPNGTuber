"""Tests for mouth_track_gui.preview module (Phase 4)."""
import unittest

import numpy as np

from erase_mouth_offline import make_mouth_mask as production_make_mouth_mask
from mouth_track_gui.preview import (
    TrackData,
    MaskParams,
    PreviewSelection,
    load_and_scale_quads,
    fill_invalid_quads,
    scale_quad_about_center,
    build_pad_preview_values,
    compute_norm_patch_size,
    compute_mask_params,
    make_mouth_mask,
    feather_mask,
    build_preview_masks,
    build_pad_button_rects,
    rect_contains,
    warp_rgba_to_quad,
    alpha_blend_rgba_over_bgr,
    resize_for_preview,
    _read_preview_frame,
)


class FillInvalidQuadsTests(unittest.TestCase):
    def _make_quads(self, n: int) -> np.ndarray:
        """Create simple (n,4,2) quads with each frame having a distinct value."""
        quads = np.zeros((n, 4, 2), dtype=np.float32)
        for i in range(n):
            quads[i] = float(i + 1)
        return quads

    def test_all_valid(self):
        quads = self._make_quads(5)
        valid = np.ones(5, dtype=bool)
        filled = fill_invalid_quads(quads, valid)
        self.assertIsNotNone(filled)
        np.testing.assert_array_equal(filled, quads)

    def test_all_invalid_returns_none(self):
        quads = self._make_quads(5)
        valid = np.zeros(5, dtype=bool)
        result = fill_invalid_quads(quads, valid)
        self.assertIsNone(result)

    def test_middle_invalid_filled(self):
        quads = self._make_quads(5)
        valid = np.array([True, True, False, False, True], dtype=bool)
        filled = fill_invalid_quads(quads, valid)
        self.assertIsNotNone(filled)
        # Frame 2 and 3 (invalid) should hold from frame 1
        np.testing.assert_array_equal(filled[2], quads[1])
        np.testing.assert_array_equal(filled[3], quads[1])

    def test_leading_invalid_filled_from_first_valid(self):
        quads = self._make_quads(5)
        valid = np.array([False, False, True, True, True], dtype=bool)
        filled = fill_invalid_quads(quads, valid)
        self.assertIsNotNone(filled)
        # Frame 0 and 1 should be filled from frame 2 (first valid)
        np.testing.assert_array_equal(filled[0], quads[2])
        np.testing.assert_array_equal(filled[1], quads[2])

    def test_trailing_invalid_filled(self):
        quads = self._make_quads(5)
        valid = np.array([True, True, True, False, False], dtype=bool)
        filled = fill_invalid_quads(quads, valid)
        self.assertIsNotNone(filled)
        # Frame 3 and 4 should hold from frame 2
        np.testing.assert_array_equal(filled[3], quads[2])
        np.testing.assert_array_equal(filled[4], quads[2])


class ComputeNormPatchSizeTests(unittest.TestCase):
    def test_returns_even_values(self):
        # Create quads that form 100x50 rectangles
        n = 10
        quads = np.zeros((n, 4, 2), dtype=np.float32)
        for i in range(n):
            quads[i] = np.array([
                [0, 0], [100, 0], [100, 50], [0, 50],
            ], dtype=np.float32)
        norm_w, norm_h = compute_norm_patch_size(quads, n)
        self.assertEqual(norm_w % 2, 0)
        self.assertEqual(norm_h % 2, 0)

    def test_minimum_sizes(self):
        # Tiny quads should still produce at least 96x64
        n = 10
        quads = np.zeros((n, 4, 2), dtype=np.float32)
        for i in range(n):
            quads[i] = np.array([
                [0, 0], [5, 0], [5, 5], [0, 5],
            ], dtype=np.float32)
        norm_w, norm_h = compute_norm_patch_size(quads, n)
        self.assertGreaterEqual(norm_w, 96)
        self.assertGreaterEqual(norm_h, 64)


class ScaleQuadAboutCenterTests(unittest.TestCase):
    def test_center_is_preserved(self):
        quad = np.array([[10, 20], [30, 20], [30, 40], [10, 40]], dtype=np.float32)
        scaled = scale_quad_about_center(quad, 1.5)
        np.testing.assert_allclose(scaled.mean(axis=0), quad.mean(axis=0), atol=1e-5)

    def test_width_and_height_scale(self):
        quad = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=np.float32)
        scaled = scale_quad_about_center(quad, 2.0)
        orig_w = np.linalg.norm(quad[1] - quad[0])
        orig_h = np.linalg.norm(quad[3] - quad[0])
        new_w = np.linalg.norm(scaled[1] - scaled[0])
        new_h = np.linalg.norm(scaled[3] - scaled[0])
        self.assertAlmostEqual(new_w, orig_w * 2.0, places=5)
        self.assertAlmostEqual(new_h, orig_h * 2.0, places=5)


class BuildPadPreviewValuesTests(unittest.TestCase):
    def test_default_pad_returns_expected_triplet(self):
        self.assertEqual(build_pad_preview_values(2.1), (1.9, 2.1, 2.3))

    def test_near_min_still_returns_three_unique_values(self):
        vals = build_pad_preview_values(1.2)
        self.assertEqual(len(vals), 3)
        self.assertEqual(len(set(vals)), 3)
        self.assertTrue(all(1.2 <= v <= 3.2 for v in vals))


class PreviewButtonLayoutTests(unittest.TestCase):
    def test_builds_one_rect_per_panel(self):
        rects = build_pad_button_rects(320, 3)
        self.assertEqual(len(rects), 3)
        self.assertTrue(all(r[2] > r[0] and r[3] > r[1] for r in rects))

    def test_rects_stay_within_each_panel(self):
        panel_w = 300
        rects = build_pad_button_rects(panel_w, 3)
        for i, rect in enumerate(rects):
            panel_left = i * panel_w
            panel_right = (i + 1) * panel_w
            self.assertGreaterEqual(rect[0], panel_left)
            self.assertLessEqual(rect[2], panel_right)

    def test_rect_contains(self):
        rect = (10, 20, 50, 60)
        self.assertTrue(rect_contains(rect, 20, 30))
        self.assertFalse(rect_contains(rect, 9, 30))


class ComputeMaskParamsTests(unittest.TestCase):
    def test_low_coverage(self):
        params = compute_mask_params(0.0, 100)
        self.assertAlmostEqual(params.mask_scale_x, 0.50)
        self.assertAlmostEqual(params.mask_scale_y, 0.44)
        self.assertEqual(params.ring_px, 16)
        self.assertEqual(params.dilate_px, 8)
        self.assertEqual(params.feather_px, 18)
        self.assertAlmostEqual(params.top_clip_frac, 0.84)

    def test_high_coverage(self):
        params = compute_mask_params(1.0, 100)
        self.assertAlmostEqual(params.mask_scale_x, 0.68)
        self.assertAlmostEqual(params.mask_scale_y, 0.58)
        self.assertEqual(params.ring_px, 26)
        self.assertEqual(params.dilate_px, 16)
        self.assertEqual(params.feather_px, 28)
        self.assertAlmostEqual(params.top_clip_frac, 0.78)

    def test_mid_coverage(self):
        params = compute_mask_params(0.65, 200)
        self.assertTrue(0.50 < params.mask_scale_x < 0.68)
        self.assertTrue(0.44 < params.mask_scale_y < 0.58)

    def test_clamps_coverage(self):
        params_neg = compute_mask_params(-0.5, 100)
        params_zero = compute_mask_params(0.0, 100)
        self.assertAlmostEqual(params_neg.mask_scale_x, params_zero.mask_scale_x)

        params_high = compute_mask_params(2.0, 100)
        params_one = compute_mask_params(1.0, 100)
        self.assertAlmostEqual(params_high.mask_scale_x, params_one.mask_scale_x)

    def test_frozen(self):
        params = compute_mask_params(0.5, 100)
        with self.assertRaises(AttributeError):
            params.ring_px = 99  # type: ignore[misc]


class MakeMouthMaskTests(unittest.TestCase):
    def test_output_shape(self):
        mask = make_mouth_mask(100, 80, rx=30, ry=20)
        self.assertEqual(mask.shape, (80, 100))
        self.assertEqual(mask.dtype, np.uint8)

    def test_has_nonzero_pixels(self):
        mask = make_mouth_mask(100, 80, rx=30, ry=20)
        self.assertGreater(np.count_nonzero(mask), 0)

    def test_top_clip_matches_production_geometry(self):
        mask = make_mouth_mask(100, 100, rx=40, ry=40, top_clip_frac=0.6)
        prod = production_make_mouth_mask(100, 100, rx=40, ry=40, top_clip_frac=0.6)
        np.testing.assert_array_equal(mask, prod)

    def test_top_clip_frac_is_clamped_like_production(self):
        mask = make_mouth_mask(100, 100, rx=40, ry=40, top_clip_frac=0.5)
        prod = production_make_mouth_mask(100, 100, rx=40, ry=40, top_clip_frac=0.5)
        np.testing.assert_array_equal(mask, prod)

    def test_center_y_offset(self):
        mask_no_offset = make_mouth_mask(100, 100, rx=30, ry=20, center_y_offset_px=0)
        mask_with_offset = make_mouth_mask(100, 100, rx=30, ry=20, center_y_offset_px=20)
        # With offset, center of mass should be lower
        ys_no = np.where(mask_no_offset > 0)[0]
        ys_with = np.where(mask_with_offset > 0)[0]
        if len(ys_no) > 0 and len(ys_with) > 0:
            self.assertGreater(np.mean(ys_with), np.mean(ys_no))


class FeatherMaskTests(unittest.TestCase):
    def test_output_range(self):
        mask_u8 = np.zeros((50, 50), dtype=np.uint8)
        mask_u8[20:30, 20:30] = 255
        result = feather_mask(mask_u8, dilate_px=3, feather_px=5)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.all(result >= 0.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_no_dilate_no_feather(self):
        mask_u8 = np.zeros((50, 50), dtype=np.uint8)
        mask_u8[20:30, 20:30] = 255
        result = feather_mask(mask_u8, dilate_px=0, feather_px=0)
        expected = mask_u8.astype(np.float32) / 255.0
        np.testing.assert_array_almost_equal(result, expected)

    def test_dilate_expands(self):
        mask_u8 = np.zeros((50, 50), dtype=np.uint8)
        mask_u8[25, 25] = 255
        original_count = np.count_nonzero(mask_u8)
        result = feather_mask(mask_u8, dilate_px=3, feather_px=0)
        dilated_count = np.count_nonzero(result)
        self.assertGreater(dilated_count, original_count)


class BuildPreviewMasksTests(unittest.TestCase):
    def test_output_shapes(self):
        inner_f, ring_f = build_preview_masks(100, 80, 0.65)
        self.assertEqual(inner_f.shape, (80, 100))
        self.assertEqual(ring_f.shape, (80, 100))

    def test_output_dtype(self):
        inner_f, ring_f = build_preview_masks(100, 80, 0.65)
        self.assertEqual(inner_f.dtype, np.float32)
        self.assertEqual(ring_f.dtype, np.float32)

    def test_output_range(self):
        inner_f, ring_f = build_preview_masks(100, 80, 0.65)
        self.assertTrue(np.all(inner_f >= 0.0))
        self.assertTrue(np.all(inner_f <= 1.0))
        self.assertTrue(np.all(ring_f >= 0.0))
        self.assertTrue(np.all(ring_f <= 1.0))

    def test_inner_and_ring_not_identical(self):
        inner_f, ring_f = build_preview_masks(100, 80, 0.65)
        self.assertFalse(np.array_equal(inner_f, ring_f))

    def test_higher_coverage_larger_inner(self):
        inner_low, _ = build_preview_masks(100, 80, 0.3)
        inner_high, _ = build_preview_masks(100, 80, 0.9)
        # Higher coverage should produce a mask with more nonzero area
        self.assertGreater(np.sum(inner_high > 0), np.sum(inner_low > 0))

    def test_mask_matches_production_make_mouth_mask(self):
        params = compute_mask_params(0.65, 80)
        rx = int((100 * params.mask_scale_x) * 0.5)
        ry = int((80 * params.mask_scale_y) * 0.5)
        preview_inner = make_mouth_mask(
            100,
            80,
            rx=rx,
            ry=ry,
            center_y_offset_px=params.center_y_off,
            top_clip_frac=params.top_clip_frac,
        )
        prod_inner = production_make_mouth_mask(
            100,
            80,
            rx=rx,
            ry=ry,
            center_y_offset_px=params.center_y_off,
            top_clip_frac=params.top_clip_frac,
        )
        np.testing.assert_array_equal(preview_inner, prod_inner)


class SpriteOverlayHelpersTests(unittest.TestCase):
    def test_warp_rgba_to_quad_output_shape(self):
        sprite = np.zeros((8, 8, 4), dtype=np.uint8)
        sprite[..., 1] = 255
        sprite[..., 3] = 255
        quad = np.array([[2, 2], [9, 2], [9, 9], [2, 9]], dtype=np.float32)
        warped = warp_rgba_to_quad(sprite, quad, 16, 12)
        self.assertEqual(warped.shape, (12, 16, 4))
        self.assertEqual(warped.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(warped[..., 3]), 0)

    def test_alpha_blend_rgba_over_bgr_changes_pixels(self):
        base = np.zeros((4, 4, 3), dtype=np.uint8)
        over = np.zeros((4, 4, 4), dtype=np.uint8)
        over[..., 2] = 255
        over[..., 3] = 255
        out = alpha_blend_rgba_over_bgr(base, over, opacity=0.5)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertGreater(int(out[..., 2].mean()), 0)

    def test_resize_for_preview_shrinks_large_image(self):
        img = np.zeros((1000, 3000, 3), dtype=np.uint8)
        resized = resize_for_preview(img, max_w=1500, max_h=800)
        self.assertLessEqual(resized.shape[1], 1500)
        self.assertLessEqual(resized.shape[0], 800)


class LoadAndScaleQuadsTests(unittest.TestCase):
    def test_missing_quad_key(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, data=np.zeros(10))
            path = f.name
        import os
        try:
            with self.assertRaises(ValueError):
                load_and_scale_quads(path, 640, 480)
        finally:
            os.unlink(path)

    def test_valid_track(self):
        import tempfile, os
        quads = np.random.rand(10, 4, 2).astype(np.float32) * 100
        valid = np.ones(10, dtype=bool)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, quad=quads, valid=valid, w=100, h=100)
            path = f.name
        try:
            td = load_and_scale_quads(path, 200, 200)
            self.assertEqual(td.n_frames, 10)
            self.assertEqual(td.quads.shape, (10, 4, 2))
            # Scaled 2x (200/100)
            np.testing.assert_allclose(td.quads[..., 0], quads[..., 0] * 2.0, atol=1e-5)
            np.testing.assert_allclose(td.quads[..., 1], quads[..., 1] * 2.0, atol=1e-5)
        finally:
            os.unlink(path)

    def test_no_wh_keys_no_scaling(self):
        import tempfile, os
        quads = np.random.rand(5, 4, 2).astype(np.float32) * 50
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, quad=quads)
            path = f.name
        try:
            td = load_and_scale_quads(path, 50, 50)
            # No w/h in npz => src_w=vid_w, src_h=vid_h => scale=1.0
            np.testing.assert_allclose(td.quads, quads, atol=1e-5)
        finally:
            os.unlink(path)


class TrackDataTests(unittest.TestCase):
    def test_frozen(self):
        td = TrackData(
            quads=np.zeros((3, 4, 2), dtype=np.float32),
            valid=np.ones(3, dtype=bool),
            n_frames=3,
        )
        with self.assertRaises(AttributeError):
            td.n_frames = 5  # type: ignore[misc]


class MaskParamsTests(unittest.TestCase):
    def test_frozen(self):
        mp = MaskParams(
            mask_scale_x=0.5, mask_scale_y=0.44,
            ring_px=16, dilate_px=8, feather_px=18,
            top_clip_frac=0.84, center_y_off=5,
        )
        with self.assertRaises(AttributeError):
            mp.ring_px = 99  # type: ignore[misc]


class PreviewSelectionTests(unittest.TestCase):
    def test_frozen(self):
        sel = PreviewSelection(applied=True, pad=2.1, coverage=0.6)
        with self.assertRaises(AttributeError):
            sel.pad = 2.3  # type: ignore[misc]


class ReadPreviewFrameTests(unittest.TestCase):
    class _FakeCap:
        def __init__(self, frames):
            self.frames = [np.array(f, copy=True) for f in frames]
            self.pos = 0
            self.set_calls = []
            self.read_calls = 0

        def set(self, prop, value):
            self.set_calls.append((prop, value))
            self.pos = int(value)
            return True

        def read(self):
            self.read_calls += 1
            if self.pos < 0 or self.pos >= len(self.frames):
                return False, None
            frame = self.frames[self.pos].copy()
            self.pos += 1
            return True, frame

    def test_reuses_cached_frame_without_read(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(3)]
        cap = self._FakeCap(frames)
        cached = frames[1].copy()

        ok, frame, cached_idx, cached_frame, next_read_idx = _read_preview_frame(
            cap,
            1,
            cached_idx=1,
            cached_frame=cached,
            next_read_idx=2,
        )

        self.assertTrue(ok)
        self.assertEqual(cap.read_calls, 0)
        self.assertEqual(cap.set_calls, [])
        np.testing.assert_array_equal(frame, cached)
        self.assertEqual(cached_idx, 1)
        self.assertEqual(next_read_idx, 2)
        self.assertIs(cached_frame, cached)

    def test_sequential_read_skips_seek(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(5)]
        cap = self._FakeCap(frames)
        cap.pos = 3

        ok, frame, cached_idx, cached_frame, next_read_idx = _read_preview_frame(
            cap,
            3,
            cached_idx=None,
            cached_frame=None,
            next_read_idx=3,
        )

        self.assertTrue(ok)
        self.assertEqual(cap.set_calls, [])
        self.assertEqual(cap.read_calls, 1)
        self.assertEqual(cached_idx, 3)
        self.assertEqual(next_read_idx, 4)
        np.testing.assert_array_equal(frame, frames[3])
        self.assertIsNotNone(cached_frame)

    def test_non_sequential_read_seeks(self):
        frames = [np.full((2, 2, 3), i, dtype=np.uint8) for i in range(5)]
        cap = self._FakeCap(frames)
        cap.pos = 1

        ok, frame, cached_idx, _cached_frame, next_read_idx = _read_preview_frame(
            cap,
            4,
            cached_idx=None,
            cached_frame=None,
            next_read_idx=2,
        )

        self.assertTrue(ok)
        self.assertEqual(cap.read_calls, 1)
        self.assertEqual(len(cap.set_calls), 1)
        self.assertEqual(int(cap.set_calls[0][1]), 4)
        self.assertEqual(cached_idx, 4)
        self.assertEqual(next_read_idx, 5)
        np.testing.assert_array_equal(frame, frames[4])


if __name__ == "__main__":
    unittest.main()
