import json
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

from convert_npz_to_json import convert_npz_to_json
from motionpngtuber.workflow_validation import build_workflow_paths

try:
    VideoSetManager = import_module("multi_video_live_gui").VideoSetManager
    HAS_MULTI_VIDEO_GUI = True
except ModuleNotFoundError:
    VideoSetManager = None
    HAS_MULTI_VIDEO_GUI = False


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ASSET_DIR = REPO_ROOT / "assets" / "t0309"
SMOKE_VIDEO = SMOKE_ASSET_DIR / "t0309.mp4"
SMOKE_TRACK = SMOKE_ASSET_DIR / "mouth_track.npz"
SMOKE_TRACK_CALIBRATED = SMOKE_ASSET_DIR / "mouth_track_calibrated.npz"
SMOKE_MOUTH_DIR = SMOKE_ASSET_DIR / "mouth"


@unittest.skipUnless(SMOKE_ASSET_DIR.is_dir(), "smoke asset directory is missing")
class EndToEndSmokeTests(unittest.TestCase):
    def test_build_workflow_paths_on_sample_asset(self):
        paths, err = build_workflow_paths(
            str(SMOKE_VIDEO),
            require_track=True,
            require_calibrated=True,
            prefer_calibrated=True,
        )
        self.assertEqual(err, "")
        self.assertIsNotNone(paths)
        assert paths is not None
        self.assertEqual(Path(paths.track_npz), SMOKE_TRACK)
        self.assertEqual(Path(paths.calib_npz), SMOKE_TRACK_CALIBRATED)
        self.assertEqual(Path(paths.preferred_track), SMOKE_TRACK_CALIBRATED)

    def test_convert_npz_to_json_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            output = convert_npz_to_json(SMOKE_TRACK_CALIBRATED, Path(td))
            payload = json.loads(Path(output).read_text(encoding="utf-8"))

        self.assertGreater(len(payload["frames"]), 0)
        self.assertGreater(payload["width"], 0)
        self.assertGreater(payload["height"], 0)
        self.assertIn("calibration", payload)
        self.assertIn("refSpriteSize", payload)

    @unittest.skipUnless(HAS_MULTI_VIDEO_GUI, "multi_video_live_gui.py is archived in this layout")
    def test_multivideo_set_manager_add_set_smoke(self):
        manager = VideoSetManager()
        vs, warnings = manager.add_set(str(SMOKE_ASSET_DIR))

        self.assertIsNotNone(vs)
        assert vs is not None
        self.assertTrue(Path(vs.video_path).is_file())
        self.assertTrue(Path(vs.track_path).is_file())
        self.assertTrue(Path(vs.mouth_dir).is_dir())
        self.assertEqual(Path(vs.track_path), SMOKE_TRACK_CALIBRATED)
        self.assertEqual(Path(vs.mouth_dir), SMOKE_MOUTH_DIR)
        self.assertIsInstance(warnings, list)


if __name__ == "__main__":
    unittest.main()
