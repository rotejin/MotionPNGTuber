import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import face_track_anime_detector
from face_track_anime_detector import (
    build_device_try_order,
    create_detector_with_fallback,
    is_cuda_runtime_error,
    mouth_quad_auto,
    mouth_quad_from_landmarks,
    run_detector_with_runtime_fallback,
    select_target_prediction,
    verify_detector_checkpoints,
)


def _pred(bbox):
    return {
        "bbox": np.asarray(bbox, dtype=np.float32),
        "keypoints": np.zeros((28, 3), dtype=np.float32),
    }


class SelectTargetPredictionTests(unittest.TestCase):
    def test_without_previous_prefers_large_face(self):
        preds = [
            _pred([10, 10, 40, 40, 0.9]),
            _pred([100, 100, 180, 180, 0.8]),
        ]
        best = select_target_prediction(preds, prev_bbox=None, min_conf=0.5)
        self.assertIsNotNone(best)
        assert best is not None
        np.testing.assert_array_equal(best["bbox"], preds[1]["bbox"])


class MouthQuadSizingTests(unittest.TestCase):
    def _keypoints(self, mouth_points):
        keypoints = np.zeros((28, 3), dtype=np.float32)
        # 左目 11-16 / 右目 17-22 を水平配置して回転0にする
        for i in range(11, 17):
            keypoints[i] = [30.0 + (i - 11), 30.0, 1.0]
        for i in range(17, 23):
            keypoints[i] = [70.0 + (i - 17), 30.0, 1.0]
        for idx, pt in zip((24, 25, 26, 27), mouth_points):
            keypoints[idx] = [pt[0], pt[1], 1.0]
        return keypoints

    def _quad_size(self, quad):
        xs = quad[:, 0]
        ys = quad[:, 1]
        return float(xs.max() - xs.min()), float(ys.max() - ys.min())

    def test_closed_mouth_respects_square_sprite_floor(self):
        keypoints = self._keypoints([
            (42.0, 60.0),
            (58.0, 60.0),
            (54.0, 62.0),
            (46.0, 62.0),
        ])
        bbox = np.asarray([20.0, 15.0, 80.0, 95.0, 0.95], dtype=np.float32)
        quad, conf = mouth_quad_from_landmarks(
            keypoints,
            bbox=bbox,
            sprite_aspect=1.0,
            pad=1.0,
            min_mouth_w_ratio=0.12,
            min_mouth_w_px=16.0,
        )
        w, h = self._quad_size(quad)
        self.assertGreater(conf, 0.9)
        self.assertGreaterEqual(w, 16.0)
        self.assertGreaterEqual(h / w, 0.95)
        self.assertLessEqual(h / w, 1.05)

    def test_manual_pad_still_expands_quad_when_needed(self):
        keypoints = self._keypoints([
            (42.0, 60.0),
            (58.0, 60.0),
            (54.0, 66.0),
            (46.0, 66.0),
        ])
        bbox = np.asarray([20.0, 15.0, 80.0, 95.0, 0.95], dtype=np.float32)
        quad1, _ = mouth_quad_from_landmarks(
            keypoints, bbox=bbox, sprite_aspect=1.0, pad=1.0,
            min_mouth_w_ratio=0.12, min_mouth_w_px=16.0,
        )
        quad2, _ = mouth_quad_from_landmarks(
            keypoints, bbox=bbox, sprite_aspect=1.0, pad=1.4,
            min_mouth_w_ratio=0.12, min_mouth_w_px=16.0,
        )
        w1, h1 = self._quad_size(quad1)
        w2, h2 = self._quad_size(quad2)
        self.assertGreater(w2, w1)
        self.assertGreater(h2, h1)

    def test_auto_mode_uses_landmarks_before_bbox_fallback(self):
        keypoints = self._keypoints([
            (42.0, 60.0),
            (58.0, 60.0),
            (54.0, 63.0),
            (46.0, 63.0),
        ])
        bbox = np.asarray([20.0, 15.0, 80.0, 95.0, 0.95], dtype=np.float32)
        quad, _ = mouth_quad_auto(
            bbox=bbox,
            keypoints=keypoints,
            sprite_aspect=1.0,
            pad=1.0,
            auto_min_mouth_ratio=0.12,
            auto_min_mouth_px=16.0,
        )
        w, h = self._quad_size(quad)
        self.assertLess(w, 30.0)
        self.assertGreater(h, 10.0)
        self.assertLess(h, 20.0)

    def test_with_previous_prefers_continuity(self):
        prev_bbox = np.asarray([20, 20, 80, 80, 0.9], dtype=np.float32)
        preds = [
            _pred([22, 22, 82, 82, 0.88]),   # close to previous
            _pred([140, 140, 260, 260, 0.95]),  # larger but far
        ]
        best = select_target_prediction(preds, prev_bbox=prev_bbox, min_conf=0.5)
        self.assertIsNotNone(best)
        assert best is not None
        np.testing.assert_array_equal(best["bbox"], preds[0]["bbox"])

    def test_min_conf_prefers_strong_candidates_when_available(self):
        prev_bbox = np.asarray([20, 20, 80, 80, 0.9], dtype=np.float32)
        preds = [
            _pred([22, 22, 82, 82, 0.20]),   # close but weak
            _pred([30, 30, 90, 90, 0.75]),   # slightly farther but valid
        ]
        best = select_target_prediction(preds, prev_bbox=prev_bbox, min_conf=0.5)
        self.assertIsNotNone(best)
        assert best is not None
        np.testing.assert_array_equal(best["bbox"], preds[1]["bbox"])


class DeviceFallbackTests(unittest.TestCase):
    def test_build_device_try_order_auto_prefers_cuda_then_cpu(self):
        self.assertEqual(build_device_try_order("auto"), ["cuda:0", "cpu"])

    def test_build_device_try_order_cuda_adds_cpu_fallback(self):
        self.assertEqual(build_device_try_order("cuda:1"), ["cuda:1", "cpu"])

    def test_is_cuda_runtime_error_detects_known_message(self):
        self.assertTrue(is_cuda_runtime_error(RuntimeError("no kernel image is available for execution on the device")))
        self.assertFalse(is_cuda_runtime_error(RuntimeError("some unrelated failure")))

    def test_create_detector_with_fallback_uses_cpu_after_cuda_init_failure(self):
        calls = []

        def fake_create(model, device):
            calls.append((model, device))
            if device.startswith("cuda"):
                raise RuntimeError("sm_120 is not compatible")
            return {"device": device}

        # create=True: anime-face-detector is not installed in the light CI environment.
        with mock.patch("face_track_anime_detector.create_detector", side_effect=fake_create, create=True), \
                mock.patch("face_track_anime_detector.verify_detector_checkpoints") as verify:
            detector, used_device = create_detector_with_fallback("yolov3", "auto")

        self.assertEqual(detector["device"], "cpu")
        self.assertEqual(used_device, "cpu")
        self.assertEqual(calls, [("yolov3", "cuda:0"), ("yolov3", "cpu")])
        verify.assert_called_once_with("yolov3")

    def test_run_detector_with_runtime_fallback_retries_current_frame_on_cpu(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)

        class FailingCudaDetector:
            def __call__(self, _frame):
                raise RuntimeError("CUDA error: no kernel image is available")

        class CpuDetector:
            def __call__(self, _frame):
                return [{"bbox": np.asarray([0, 0, 1, 1, 0.9], dtype=np.float32), "keypoints": np.zeros((28, 3), dtype=np.float32)}]

        with unittest.mock.patch(
            "face_track_anime_detector.create_detector_with_fallback",
            return_value=(CpuDetector(), "cpu"),
        ):
            preds, detector, used_device = run_detector_with_runtime_fallback(
                FailingCudaDetector(),
                frame,
                model="yolov3",
                current_device="cuda:0",
            )

        self.assertEqual(used_device, "cpu")
        self.assertEqual(len(preds), 1)
        self.assertIsInstance(detector, CpuDetector)


class CheckpointVerificationTests(unittest.TestCase):
    """The weights are read with torch.load, so only the published files may be used."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.files = {}
        hashes = {}
        for name in ("yolov3", "hrnetv2"):
            path = Path(self.tmp.name) / f"{name}.pth"
            path.write_bytes(f"weights of {name}".encode())
            self.files[name] = path
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.get_path = mock.Mock(side_effect=lambda name: self.files[name])
        for patcher in (
            mock.patch.object(face_track_anime_detector, "DETECTOR_CHECKPOINT_SHA256", hashes),
            mock.patch.object(face_track_anime_detector, "_verified_checkpoints", set()),
            mock.patch("face_track_anime_detector.get_checkpoint_path", self.get_path, create=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_published_files_pass_and_are_checked_once(self):
        verify_detector_checkpoints("yolov3")
        verify_detector_checkpoints("yolov3")

        self.assertEqual([c.args[0] for c in self.get_path.call_args_list], ["yolov3", "hrnetv2"])

    def test_changed_file_is_refused(self):
        self.files["hrnetv2"].write_bytes(b"something else")

        with self.assertRaises(RuntimeError) as ctx:
            verify_detector_checkpoints("yolov3")

        self.assertIn(str(self.files["hrnetv2"]), str(ctx.exception))
        self.assertIn("Delete this file", str(ctx.exception))

    def test_model_without_a_known_hash_is_refused(self):
        with self.assertRaises(RuntimeError):
            verify_detector_checkpoints("faster-rcnn")

    def test_detector_is_not_created_from_a_changed_file(self):
        self.files["yolov3"].write_bytes(b"something else")
        create = mock.Mock()

        with mock.patch("face_track_anime_detector.create_detector", create, create=True):
            with self.assertRaises(RuntimeError):
                create_detector_with_fallback("yolov3", "auto")

        create.assert_not_called()


class PinnedCheckpointHashTests(unittest.TestCase):
    def test_both_models_the_tool_uses_have_a_sha256(self):
        pinned = face_track_anime_detector.DETECTOR_CHECKPOINT_SHA256
        self.assertEqual(set(pinned), {"yolov3", face_track_anime_detector.LANDMARK_MODEL})
        for name, value in pinned.items():
            self.assertRegex(value, r"^[0-9a-f]{64}$", name)


class _FrameCountOverride:
    """A real VideoCapture whose CAP_PROP_FRAME_COUNT says something else."""

    def __init__(self, cap, frame_count, *, readable=True) -> None:
        self._cap = cap
        self._frame_count = frame_count
        self._readable = readable

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return self._cap.get(prop)

    def read(self):
        return self._cap.read() if self._readable else (False, None)

    def __getattr__(self, name):
        return getattr(self._cap, name)


class FrameCountTests(unittest.TestCase):
    """The track has one entry per frame actually read, whatever the video header says.

    Before, the header's count decided the length: 0 saved an empty track and exited 0,
    a larger number left zero quads at the end, a smaller one dropped frames, and the
    garbage value of a raw stream crashed np.zeros.
    """

    N = 23

    def setUp(self) -> None:
        import cv2

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.video = str(self.dir / "clip.mp4")
        writer = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(self.N):
            writer.write(np.full((48, 64, 3), 128, np.uint8))
        writer.release()

    def run_detector(self, frame_count, *, readable=True):
        import cv2

        real_capture = cv2.VideoCapture
        out = str(self.dir / "mouth_track.npz")
        argv = ["face_track_anime_detector.py", "--video", self.video, "--out", out, "--device", "cpu"]
        with (
            mock.patch.object(face_track_anime_detector, "HAS_ANIME_DETECTOR", True),
            mock.patch.object(face_track_anime_detector, "create_detector_with_fallback",
                              return_value=(object(), "cpu")),
            mock.patch.object(face_track_anime_detector, "run_detector_with_runtime_fallback",
                              side_effect=lambda detector, frame, model, current_device: ([], detector, current_device)),
            mock.patch.object(face_track_anime_detector.cv2, "VideoCapture",
                              side_effect=lambda path: _FrameCountOverride(real_capture(path), frame_count,
                                                                           readable=readable)),
            mock.patch.object(face_track_anime_detector.sys, "argv", argv),
            mock.patch("builtins.print"),
        ):
            rc = face_track_anime_detector.main()
        return rc, out

    def test_track_length_follows_the_frames_read(self):
        for header in (self.N, 0, self.N + 40, self.N - 9, -230584300921369):
            with self.subTest(header=header):
                rc, out = self.run_detector(header)
                self.assertEqual(rc, 0)
                with np.load(out, allow_pickle=False) as npz:
                    self.assertEqual(npz["quad"].shape, (self.N, 4, 2))
                    self.assertEqual(npz["valid"].shape, (self.N,))
                    self.assertEqual(npz["confidence"].shape, (self.N,))
                Path(out).unlink()

    def test_no_readable_frame_is_an_error(self):
        rc, out = self.run_detector(self.N, readable=False)
        self.assertEqual(rc, 1)
        self.assertFalse(Path(out).exists())


if __name__ == "__main__":
    unittest.main()
