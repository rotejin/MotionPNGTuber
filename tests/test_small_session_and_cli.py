"""Small fixes: the coverage slider is saved, and the runtime's command line wins.

- The "口消し範囲" (coverage) slider only updated its label; the value was saved only
  when a video or mouth folder was chosen or ① / ② ran, so it was lost otherwise.
- The runtime only noticed a path given as two tokens (``--loop-video clip.mp4``):
  with ``--loop-video=clip.mp4`` it loaded the last GUI session and replaced the video.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import loop_lipsync_runtime_patched_emotion_auto as runtime
from mouth_track_gui.ui import bind_coverage_slider


class _Var:
    def __init__(self, value: float) -> None:
        self.value = value
        self.callbacks = []

    def get(self) -> float:
        return self.value

    def set(self, value: float) -> None:
        self.value = value
        for callback in self.callbacks:
            callback("name", "", "write")

    def trace_add(self, _mode: str, callback) -> None:
        self.callbacks.append(callback)


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def config(self, *, text: str) -> None:
        self.text = text


class CoverageSliderTests(unittest.TestCase):
    def test_moving_the_slider_saves_the_value(self):
        var, label, save = _Var(0.60), _Label(), mock.Mock()
        bind_coverage_slider(var, label, save)
        var.set(0.73)
        save.assert_called_once_with({"coverage": 0.73})
        self.assertEqual(label.text, "0.73")


class RuntimePathArgumentTests(unittest.TestCase):
    def args(self, **values):
        defaults = {"loop_video": "", "assets_dir": "", "mouth_dir": "", "track": "", "track_calibrated": ""}
        defaults.update(values)
        return SimpleNamespace(**defaults)

    def test_paths_on_the_command_line_are_noticed(self):
        cases = [
            (["--loop-video", "clip.mp4"], self.args(loop_video="clip.mp4")),
            (["--loop-video=clip.mp4"], self.args(loop_video="clip.mp4")),
            (["--loop-vid", "clip.mp4"], self.args(loop_video="clip.mp4")),  # argparse abbreviation
            (["--mouth-dir=mouth"], self.args(mouth_dir="mouth")),
            (["--tuber-num=11"], self.args()),
        ]
        for argv, args in cases:
            with self.subTest(argv=argv):
                self.assertTrue(runtime._user_specified_paths(argv, args))

    def test_other_options_are_not_paths(self):
        self.assertFalse(runtime._user_specified_paths([], self.args()))
        self.assertFalse(runtime._user_specified_paths(["--device", "3", "--emotion-auto"], self.args()))

    def test_the_last_session_does_not_replace_a_video_given_with_equals(self):
        with tempfile.TemporaryDirectory() as td:
            session_video = os.path.join(td, "session.mp4")
            with open(session_video, "wb"):
                pass
            mine = os.path.join(td, "mine.mp4")
            with (
                mock.patch.object(runtime.sys, "argv", ["runtime", f"--loop-video={mine}"]),
                mock.patch.object(runtime, "load_last_session", return_value={"video": session_video}),
                mock.patch("builtins.print"),
            ):
                args = runtime.parse_args()
        self.assertEqual(args.loop_video, mine)


class RuntimeJsonWriteTests(unittest.TestCase):
    def test_writes_the_file_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "result.json")
            runtime._write_json_atomic(path, {"request_id": "r1", "value": 1.5})
            runtime._write_json_atomic(path, {"request_id": "r2", "value": 2.5})
            with open(path, encoding="utf-8") as f:
                import json

                self.assertEqual(json.load(f), {"request_id": "r2", "value": 2.5})
            self.assertEqual(os.listdir(td), ["result.json"])

    def test_a_failed_write_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "result.json")
            with self.assertRaises(TypeError):
                runtime._write_json_atomic(path, {"bad": object()})
            self.assertEqual(os.listdir(td), [])


if __name__ == "__main__":
    unittest.main()
