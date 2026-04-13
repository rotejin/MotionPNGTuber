"""Focused regression tests for mouth_track_gui.app helper methods."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from mouth_track_gui.app import App
from mouth_track_gui.actions import ActionPlan, ActionStep
from mouth_track_gui.runner import RunResult


class DummyVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


class DummyCombo:
    def __init__(self) -> None:
        self.values = ()
        self.bound = None

    def __setitem__(self, key: str, value) -> None:
        if key == "values":
            self.values = tuple(value)

    def bind(self, _event: str, callback) -> None:
        self.bound = callback


class AppHelperRegressionTests(unittest.TestCase):
    def test_resolve_mouth_root_autofill_uses_post_setter(self):
        fake = SimpleNamespace(
            mouth_dir_var=DummyVar(""),
            _guess_mouth_dir=lambda: "C:/tmp/mouth",
            _post_set_mouth_dir=mock.Mock(),
            _show_error=mock.Mock(),
        )

        with mock.patch("mouth_track_gui.app.validate_existing_dir", return_value=("C:/tmp/mouth", "")):
            path = App._resolve_mouth_root(fake, auto_fill=True)

        self.assertEqual(path, "C:/tmp/mouth")
        fake._post_set_mouth_dir.assert_called_once_with("C:/tmp/mouth")
        fake._show_error.assert_not_called()

    def test_resolve_character_for_action_posts_single_choice(self):
        fake = SimpleNamespace(
            mouth_dir_var=DummyVar("C:/tmp/mouth"),
            character_var=DummyVar(""),
            _post_set_character=mock.Mock(),
            _show_error=mock.Mock(),
        )

        with mock.patch("mouth_track_gui.app.is_emotion_level_mouth_root", return_value=False):
            with mock.patch("mouth_track_gui.app.list_character_dirs", return_value=["tomari"]):
                character = App._resolve_character_for_action(fake)

        self.assertEqual(character, "tomari")
        fake._post_set_character.assert_called_once_with("tomari", persist=True)
        fake._show_error.assert_not_called()

    def test_refresh_audio_devices_restores_audio_device_spec(self):
        fake = SimpleNamespace(
            audio_device_var=DummyVar(31),
            audio_device_spec_var=DummyVar("pa:alsa_input.test"),
            audio_device_menu_var=DummyVar(""),
            cmb_audio=DummyCombo(),
        )
        devices = [
            {"spec": "sd:3", "index": 3, "display": "3: USB Mic"},
            {"spec": "pa:alsa_input.test", "index": None, "display": "pa:alsa_input.test  (via pulse)"},
        ]
        with mock.patch("mouth_track_gui.app.list_input_devices", return_value=devices):
            App._refresh_audio_devices(fake, init=False)

        self.assertEqual(fake.audio_device_menu_var.get(), "pa:alsa_input.test  (via pulse)")
        self.assertEqual(fake.audio_device_spec_var.get(), "pa:alsa_input.test")

    def test_refresh_audio_devices_saves_audio_device_spec_on_select(self):
        fake = SimpleNamespace(
            audio_device_var=DummyVar(0),
            audio_device_spec_var=DummyVar(""),
            audio_device_menu_var=DummyVar(""),
            cmb_audio=DummyCombo(),
            _save_session=mock.Mock(),
        )
        devices = [
            {"spec": "sd:3", "index": 3, "display": "3: USB Mic"},
            {"spec": "pa:alsa_input.test", "index": None, "display": "pa:alsa_input.test  (via pulse)"},
        ]
        with mock.patch("mouth_track_gui.app.list_input_devices", return_value=devices):
            App._refresh_audio_devices(fake, init=False)
            fake.audio_device_menu_var.set("pa:alsa_input.test  (via pulse)")
            assert fake.cmb_audio.bound is not None
            fake.cmb_audio.bound()

        self.assertEqual(fake.audio_device_spec_var.get(), "pa:alsa_input.test")
        fake._save_session.assert_called_with({"audio_device_spec": "pa:alsa_input.test"})

    def test_save_session_logs_warning_when_persistence_fails(self):
        fake = SimpleNamespace(log=mock.Mock())

        with mock.patch("mouth_track_gui.app.save_session", return_value=False):
            ok = App._save_session(fake, {"video": "loop.mp4"})

        self.assertFalse(ok)
        fake.log.assert_called_once()
        self.assertIn("セッション保存に失敗", fake.log.call_args.args[0])

    def test_execute_plan_treats_stopped_result_as_non_error(self):
        fake = SimpleNamespace(
            _save_session=mock.Mock(return_value=True),
            _progress_begin=mock.Mock(),
            _progress_step=mock.Mock(),
            log=mock.Mock(),
            runner=SimpleNamespace(
                soft_requested=False,
                run_stream=mock.Mock(return_value=RunResult(returncode=130, was_stopped=True)),
            ),
            _show_error=mock.Mock(),
            _run_post_action=mock.Mock(),
            stop_mode="soft",
        )
        plan = ActionPlan(
            name="解析/キャリブ",
            steps=(ActionStep(cmd=["python"], label="解析", progress_label="解析"),),
            post_actions=(mock.Mock(),),
            completion_msg="完了",
        )

        rc = App._execute_plan(fake, plan)

        self.assertEqual(rc, 130)
        fake._show_error.assert_not_called()
        fake._run_post_action.assert_not_called()
        self.assertTrue(any("停止しました" in c.args[0] for c in fake.log.call_args_list))

    def test_execute_plan_late_stop_after_success_keeps_current_step_complete(self):
        fake = SimpleNamespace(
            _save_session=mock.Mock(return_value=True),
            _progress_begin=mock.Mock(),
            _progress_step=mock.Mock(),
            log=mock.Mock(),
            runner=SimpleNamespace(
                soft_requested=True,
                run_stream=mock.Mock(
                    return_value=RunResult(
                        returncode=0, was_stopped=False, stop_requested=True,
                    ),
                ),
            ),
            _show_error=mock.Mock(),
            _run_post_action=mock.Mock(),
            stop_mode="soft",
        )
        plan = ActionPlan(
            name="解析/キャリブ",
            steps=(
                ActionStep(cmd=["python"], label="解析", progress_label="解析"),
                ActionStep(
                    cmd=["python"], label="キャリブ", progress_label="キャリブ",
                    skip_on_stop=True,
                ),
            ),
            post_actions=(mock.Mock(),),
            completion_msg="完了",
        )

        rc = App._execute_plan(fake, plan)

        self.assertEqual(rc, 0)
        fake.runner.run_stream.assert_called_once()
        fake._show_error.assert_not_called()
        fake._run_post_action.assert_not_called()
        self.assertTrue(
            any(
                call.args == (1, "解析完了 (1/2)")
                for call in fake._progress_step.call_args_list
            ),
        )
        self.assertTrue(
            any("停止予約のため、キャリブ以降をスキップします。" in c.args[0] for c in fake.log.call_args_list),
        )
        self.assertFalse(
            any("停止しました。" in c.args[0] for c in fake.log.call_args_list),
        )

    def test_finalize_live_run_result_shows_error_on_nonzero_exit(self):
        fake = SimpleNamespace(
            log=mock.Mock(),
            _progress_step=mock.Mock(),
            _show_error=mock.Mock(),
            stop_mode="none",
        )

        App._finalize_live_run_result(fake, RunResult(returncode=42, was_stopped=False))

        fake._progress_step.assert_called_with(1, "ライブ異常終了")
        fake._show_error.assert_called_once()
        self.assertIn("rc=42", fake._show_error.call_args.args[1])

    def test_finalize_live_run_result_late_stop_after_success_keeps_live_finished(self):
        fake = SimpleNamespace(
            log=mock.Mock(),
            _progress_step=mock.Mock(),
            _show_error=mock.Mock(),
            stop_mode="soft",
        )

        App._finalize_live_run_result(
            fake,
            RunResult(returncode=0, was_stopped=False, stop_requested=True),
        )

        fake._progress_step.assert_called_with(1, "ライブ終了")
        fake._show_error.assert_not_called()

    def test_finalize_live_run_result_late_stop_after_failure_still_reports_error(self):
        fake = SimpleNamespace(
            log=mock.Mock(),
            _progress_step=mock.Mock(),
            _show_error=mock.Mock(),
            stop_mode="soft",
        )

        App._finalize_live_run_result(
            fake,
            RunResult(returncode=42, was_stopped=False, stop_requested=True),
        )

        fake._progress_step.assert_called_with(1, "ライブ異常終了")
        fake._show_error.assert_called_once()
        self.assertIn("rc=42", fake._show_error.call_args.args[1])

    def test_export_browser_assets_logs_json_only_when_ffmpeg_missing(self):
        with tempfile.TemporaryDirectory() as td:
            mouthless = os.path.join(td, "loop_mouthless.mp4")
            calib = os.path.join(td, "mouth_track_calibrated.npz")
            with open(mouthless, "wb") as f:
                f.write(b"video")
            import numpy as np
            np.savez_compressed(calib, fps=np.float32(30.0))

            fake = SimpleNamespace(log=mock.Mock(), _run_cmd_stream=mock.Mock(return_value=0))

            def fake_convert(_npz_path, out_dir):
                out_path = os.path.join(str(out_dir), "mouth_track.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("{}")
                return out_path

            with (
                mock.patch("mouth_track_gui.app.shutil.which", return_value=None),
                mock.patch("convert_npz_to_json.convert_npz_to_json", side_effect=fake_convert),
            ):
                App._export_browser_assets(fake, mouthless, calib)

        joined = "\n".join(call.args[0] for call in fake.log.call_args_list)
        self.assertIn("mouth_track.json", joined)
        self.assertIn("ffmpegが見つからないためH.264変換をスキップします", joined)
        self.assertIn("利用可能ファイル", joined)


if __name__ == "__main__":
    unittest.main()
