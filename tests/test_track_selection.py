import os
import tempfile
import unittest

from lipsync_core import resolve_preferred_track_path


class ResolvePreferredTrackPathTests(unittest.TestCase):
    def test_prefers_calibrated_track_when_present(self):
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "mouth_track.npz")
            calibrated = os.path.join(td, "mouth_track_calibrated.npz")
            open(raw, "wb").close()
            open(calibrated, "wb").close()

            got = resolve_preferred_track_path(raw, calibrated, prefer_calibrated=True)
            self.assertEqual(got, calibrated)

    def test_falls_back_to_raw_track_when_calibrated_missing(self):
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "mouth_track.npz")
            calibrated = os.path.join(td, "mouth_track_calibrated.npz")
            open(raw, "wb").close()

            got = resolve_preferred_track_path(raw, calibrated, prefer_calibrated=True)
            self.assertEqual(got, raw)

    def test_can_disable_calibrated_preference(self):
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "mouth_track.npz")
            calibrated = os.path.join(td, "mouth_track_calibrated.npz")
            open(raw, "wb").close()
            open(calibrated, "wb").close()

            got = resolve_preferred_track_path(raw, calibrated, prefer_calibrated=False)
            self.assertEqual(got, raw)


if __name__ == "__main__":
    unittest.main()
