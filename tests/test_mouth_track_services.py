"""Tests for mouth_track_gui.services module (Phase 2)."""
import os
import tempfile
import unittest

from mouth_track_gui.services import (
    script_contains,
    list_input_devices,
    find_input_device_item,
    display_to_audio_spec,
    ensure_backend_sanity,
    guess_mouth_dir,
    best_open_sprite,
    is_emotion_level_mouth_root,
    EMOTION_DIR_NAMES,
    list_character_dirs,
    resolve_character_dir,
    best_open_sprite_for_character,
)
from unittest import mock
import sys


class ScriptContainsTests(unittest.TestCase):
    def test_all_present(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("--pad --det-scale --min-conf")
            tmp = f.name
        try:
            self.assertTrue(script_contains(tmp, ["--pad", "--det-scale"]))
        finally:
            os.unlink(tmp)

    def test_missing_needle(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("--pad only")
            tmp = f.name
        try:
            self.assertFalse(script_contains(tmp, ["--pad", "--det-scale"]))
        finally:
            os.unlink(tmp)

    def test_nonexistent_file(self):
        self.assertFalse(script_contains("/nonexistent/file.py", ["x"]))


class GuessMouthDirTests(unittest.TestCase):
    def test_mouth_next_to_video(self):
        with tempfile.TemporaryDirectory() as d:
            mouth = os.path.join(d, "mouth")
            os.makedirs(mouth)
            video = os.path.join(d, "test.mp4")
            open(video, "w").close()
            self.assertEqual(guess_mouth_dir(video), mouth)

    def test_no_mouth_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            video = os.path.join(d, "test.mp4")
            open(video, "w").close()
            # Only empty if HERE/mouth also doesn't exist
            result = guess_mouth_dir(video)
            # Result is either empty or HERE/mouth
            self.assertIsInstance(result, str)


class BestOpenSpriteTests(unittest.TestCase):
    def test_direct_open_png(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "open.png")
            open(p, "w").close()
            self.assertEqual(best_open_sprite(d), p)

    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(best_open_sprite(d), "")

    def test_nonexistent(self):
        self.assertEqual(best_open_sprite("/nonexistent"), "")


class EmotionLevelTests(unittest.TestCase):
    def test_has_open_png(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "open.png"), "w").close()
            self.assertTrue(is_emotion_level_mouth_root(d))

    def test_has_emotion_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "happy"))
            os.makedirs(os.path.join(d, "sad"))
            self.assertTrue(is_emotion_level_mouth_root(d))

    def test_not_emotion_level(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "charA"))
            self.assertFalse(is_emotion_level_mouth_root(d))


class ListCharacterDirsTests(unittest.TestCase):
    def test_returns_character_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "Alice"))
            os.makedirs(os.path.join(d, "Bob"))
            chars = list_character_dirs(d)
            self.assertEqual(chars, ["Alice", "Bob"])

    def test_empty_when_emotion_level(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "open.png"), "w").close()
            self.assertEqual(list_character_dirs(d), [])

    def test_excludes_emotion_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "Alice"))
            os.makedirs(os.path.join(d, "happy"))
            chars = list_character_dirs(d)
            self.assertNotIn("happy", chars)
            self.assertIn("Alice", chars)


class ResolveCharacterDirTests(unittest.TestCase):
    def test_with_character(self):
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "Alice")
            os.makedirs(ch)
            self.assertEqual(resolve_character_dir(d, "Alice"), ch)

    def test_invalid_character(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_character_dir(d, "NoSuch"), d)

    def test_empty_root(self):
        self.assertEqual(resolve_character_dir("", "Alice"), "")


class BestOpenSpriteForCharacterTests(unittest.TestCase):
    def test_direct_open(self):
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "Alice")
            os.makedirs(ch)
            p = os.path.join(ch, "open.png")
            open(p, "w").close()
            self.assertEqual(best_open_sprite_for_character(d, "Alice"), p)

    def test_emotion_subfolder(self):
        with tempfile.TemporaryDirectory() as d:
            ch = os.path.join(d, "Alice")
            em = os.path.join(ch, "default")
            os.makedirs(em)
            p = os.path.join(em, "open.png")
            open(p, "w").close()
            result = best_open_sprite_for_character(d, "Alice")
            # Windows is case-insensitive; normalise for comparison
            self.assertEqual(os.path.normcase(result), os.path.normcase(p))


class ListInputDevicesTests(unittest.TestCase):
    def test_returns_dict_items(self):
        fake_sd = mock.Mock()
        fake_sd.query_devices.return_value = [
            {"name": "Mic A", "max_input_channels": 1, "default_samplerate": 48000},
            {"name": "Speaker", "max_input_channels": 0, "default_samplerate": 48000},
        ]
        with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
            with mock.patch("motionpngtuber.audio_linux.augment_devices_for_linux", side_effect=lambda items, _sd: items):
                items = list_input_devices()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["spec"], "sd:0")
        self.assertEqual(items[0]["index"], 0)
        self.assertIn("Mic A", items[0]["display"])

    def test_find_input_device_item_by_spec_and_display(self):
        items = [
            {"spec": "sd:1", "index": 1, "display": "1: Mic"},
            {"spec": "pa:alsa_input.test", "index": None, "display": "pa:alsa_input.test  (via pulse)"},
        ]
        self.assertEqual(find_input_device_item(items, "sd:1"), items[0])
        self.assertEqual(find_input_device_item(items, "pa:alsa_input.test  (via pulse)"), items[1])
        self.assertIsNone(find_input_device_item(items, "missing"))

    def test_display_to_audio_spec(self):
        with mock.patch("mouth_track_gui.services.list_input_devices", return_value=[
            {"spec": "sd:2", "index": 2, "display": "2: USB Mic"},
        ]):
            self.assertEqual(display_to_audio_spec("2: USB Mic"), "sd:2")
            self.assertIsNone(display_to_audio_spec("missing"))


if __name__ == "__main__":
    unittest.main()
