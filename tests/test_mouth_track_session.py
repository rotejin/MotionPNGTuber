"""Tests for mouth_track_gui.state module (Phase 1)."""
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from mouth_track_gui.state import (
    safe_bool,
    safe_float,
    safe_int,
    load_session,
    save_session,
    LAST_SESSION_FILE,
    SESSION_KEYS,
)


class SafeBoolTests(unittest.TestCase):
    def test_true_values(self):
        for v in [True, "1", "true", "True", "yes", "on"]:
            self.assertTrue(safe_bool(v), f"Expected True for {v!r}")

    def test_false_values(self):
        for v in [False, "0", "false", "False", "no", "off"]:
            self.assertFalse(safe_bool(v), f"Expected False for {v!r}")

    def test_none_returns_default(self):
        self.assertFalse(safe_bool(None, default=False))
        self.assertTrue(safe_bool(None, default=True))

    def test_unknown_returns_default(self):
        self.assertFalse(safe_bool("maybe", default=False))
        self.assertTrue(safe_bool("maybe", default=True))


class SafeIntTests(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(safe_int(42, 0), 42)
        self.assertEqual(safe_int("42", 0), 42)

    def test_clamp(self):
        self.assertEqual(safe_int(5, 0, min_v=10), 10)
        self.assertEqual(safe_int(100, 0, max_v=50), 50)

    def test_device_string(self):
        self.assertEqual(safe_int("31: Built-in Mic", 0), 31)

    def test_invalid_returns_default(self):
        self.assertEqual(safe_int("abc", 99), 99)
        self.assertEqual(safe_int(None, 99), 99)


class SafeFloatTests(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(safe_float(0.6, 0.0), 0.6)
        self.assertAlmostEqual(safe_float("2.1", 0.0), 2.1)

    def test_clamp(self):
        self.assertAlmostEqual(safe_float(0.1, 0.5, min_v=0.4), 0.4)
        self.assertAlmostEqual(safe_float(1.0, 0.5, max_v=0.9), 0.9)

    def test_invalid_returns_default(self):
        self.assertAlmostEqual(safe_float("abc", 0.6), 0.6)


class SessionPersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write("{}")
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            save_session({"video": "test.mp4", "coverage": 0.65})
            data = load_session()
            self.assertEqual(data["video"], "test.mp4")
            self.assertAlmostEqual(data["coverage"], 0.65)
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_save_merges_with_existing(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump({"video": "a.mp4", "pad": 2.0}, f)
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            save_session({"coverage": 0.7})
            data = load_session()
            self.assertEqual(data["video"], "a.mp4")
            self.assertAlmostEqual(data["pad"], 2.0)
            self.assertAlmostEqual(data["coverage"], 0.7)
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_load_returns_empty_on_missing_file(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            mod.LAST_SESSION_FILE = "/nonexistent/path/session.json"
            self.assertEqual(load_session(), {})
        finally:
            mod.LAST_SESSION_FILE = original

    def test_load_returns_empty_on_corrupted_json(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write("{broken json")
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            self.assertEqual(load_session(), {})
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_load_normalises_non_dict_root(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                json.dump([1, 2, 3], f)
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            self.assertEqual(load_session(), {})
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_save_is_atomic(self):
        """After save_session, the file should contain valid JSON."""
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write("{}")
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            save_session({"key": "value"})
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["key"], "value")
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def test_save_reports_failure_and_returns_false(self):
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write("{}")
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp
            with mock.patch("mouth_track_gui.state.tempfile.mkstemp", side_effect=OSError("disk full")):
                with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    ok = save_session({"key": "value"})
            self.assertFalse(ok)
            self.assertIn("save_session failed", stderr.getvalue())
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass


class ConcurrentSessionTests(unittest.TestCase):
    def test_concurrent_read_write(self):
        """Multiple writers + readers must not corrupt the file."""
        import threading
        import mouth_track_gui.state as mod
        original = mod.LAST_SESSION_FILE
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, encoding="utf-8"
            ) as f:
                f.write("{}")
                tmp = f.name
            mod.LAST_SESSION_FILE = tmp

            errors: list = []

            def writer(idx: int) -> None:
                for n in range(50):
                    save_session({f"w{idx}": n})

            def reader() -> None:
                for _ in range(200):
                    d = load_session()
                    if not isinstance(d, dict):
                        errors.append(("non_dict", type(d).__name__))

            threads = (
                [threading.Thread(target=writer, args=(i,)) for i in range(3)]
                + [threading.Thread(target=reader) for _ in range(3)]
            )
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            with open(tmp, "r", encoding="utf-8") as f:
                final = json.load(f)
            self.assertIsInstance(final, dict)
        finally:
            mod.LAST_SESSION_FILE = original
            try:
                os.unlink(tmp)
            except Exception:
                pass


class SessionKeysTests(unittest.TestCase):
    def test_known_keys_is_frozenset(self):
        self.assertIsInstance(SESSION_KEYS, frozenset)

    def test_essential_keys_present(self):
        for key in ["video", "source_video", "mouth_dir", "coverage", "pad"]:
            self.assertIn(key, SESSION_KEYS)

    def test_mouth_color_keys_present(self):
        for key in [
            "mouth_brightness",
            "mouth_saturation",
            "mouth_warmth",
            "mouth_color_strength",
            "mouth_edge_priority",
            "mouth_edge_width_ratio",
            "mouth_inspect_boost",
        ]:
            self.assertIn(key, SESSION_KEYS)


if __name__ == "__main__":
    unittest.main()
