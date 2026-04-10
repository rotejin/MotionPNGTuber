"""Tests for mouth_track_gui.actions module (Phase 5)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

from mouth_track_gui.actions import (
    ActionPlan,
    ActionStep,
    PostAction,
    build_calib_cmd,
    build_erase_cmd,
    build_erase_coverage_arg,
    build_live_cmd,
    build_track_cmd,
    plan_calib_only,
    plan_erase,
    plan_live,
    plan_track_and_calib,
    resolve_runtime_script,
)


class ActionStepTests(unittest.TestCase):
    def test_fields(self):
        step = ActionStep(cmd=["echo"], label="test")
        self.assertEqual(step.cmd, ["echo"])
        self.assertEqual(step.label, "test")
        self.assertTrue(step.allow_soft_stop)
        self.assertFalse(step.skip_on_stop)

    def test_frozen(self):
        step = ActionStep(cmd=["echo"], label="test")
        with self.assertRaises(AttributeError):
            step.label = "changed"  # type: ignore[misc]


class ActionPlanTests(unittest.TestCase):
    def test_total_steps(self):
        plan = ActionPlan(
            name="test",
            steps=(
                ActionStep(cmd=["a"], label="step1"),
                ActionStep(cmd=["b"], label="step2"),
            ),
        )
        self.assertEqual(plan.total_steps, 2)

    def test_defaults(self):
        plan = ActionPlan(name="test", steps=())
        self.assertEqual(plan.session_init, {})
        self.assertEqual(plan.session_final, {})
        self.assertEqual(plan.post_actions, ())
        self.assertEqual(plan.completion_msg, "完了")


class BuildTrackCmdTests(unittest.TestCase):
    def test_basic(self):
        cmd = build_track_cmd("/base", "vid.mp4", "out.npz", 2.0)
        self.assertIn("--video", cmd)
        self.assertIn("vid.mp4", cmd)
        self.assertIn("--out", cmd)
        self.assertIn("out.npz", cmd)
        self.assertIn("--pad", cmd)
        self.assertIn("2.00", cmd)
        self.assertIn("--early-stop", cmd)
        self.assertNotIn("--smooth-cutoff", cmd)

    def test_with_smoothing(self):
        cmd = build_track_cmd("/base", "v.mp4", "o.npz", 1.0, smoothing_cutoff=1.5)
        idx = cmd.index("--smooth-cutoff")
        self.assertEqual(cmd[idx + 1], "1.5")

    def test_script_path(self):
        cmd = build_track_cmd("/my/base", "v.mp4", "o.npz", 1.0)
        script = os.path.join("/my/base", "auto_mouth_track_v2.py")
        self.assertIn(script, cmd)

    def test_gui_default_pad_uses_legacy_safe_baseline(self):
        cmd = build_track_cmd("/base", "vid.mp4", "out.npz", 2.1)
        idx = cmd.index("--pad")
        self.assertEqual(cmd[idx + 1], "2.10")

    def test_prefers_python_exe_over_pythonw_for_worker_cli(self):
        with mock.patch(
            "mouth_track_gui.actions.resolve_python_subprocess_executable",
            return_value="C:\\venv\\Scripts\\python.exe",
        ):
            cmd = build_track_cmd("/base", "vid.mp4", "out.npz", 2.0)
        self.assertEqual(cmd[0], "C:\\venv\\Scripts\\python.exe")


class BuildCalibCmdTests(unittest.TestCase):
    def test_basic(self):
        cmd = build_calib_cmd(
            "/b", "v.mp4", "track.npz", "open.png", "calib.npz",
            mouth_brightness=0.0,
            mouth_saturation=1.0,
            mouth_warmth=0.0,
            mouth_color_strength=0.75,
            mouth_edge_priority=0.85,
            mouth_edge_width_ratio=0.10,
            mouth_inspect_boost=1.0,
        )
        self.assertIn("--video", cmd)
        self.assertIn("v.mp4", cmd)
        self.assertIn("--track", cmd)
        self.assertIn("track.npz", cmd)
        self.assertIn("--sprite", cmd)
        self.assertIn("open.png", cmd)
        self.assertIn("--out", cmd)
        self.assertIn("calib.npz", cmd)
        self.assertIn("--mouth-brightness", cmd)
        self.assertIn("--mouth-inspect-boost", cmd)


class BuildEraseCoverageArgTests(unittest.TestCase):
    def test_normal_coverage(self):
        result = build_erase_coverage_arg(0.65)
        values = [float(x) for x in result.split(",")]
        self.assertTrue(all(0.40 <= v <= 0.90 for v in values))
        self.assertIn(0.65, values)

    def test_low_coverage_clamps(self):
        result = build_erase_coverage_arg(0.30)
        values = [float(x) for x in result.split(",")]
        self.assertTrue(all(v >= 0.40 for v in values))

    def test_high_coverage_clamps(self):
        result = build_erase_coverage_arg(0.85)
        values = [float(x) for x in result.split(",")]
        self.assertTrue(all(v <= 0.90 for v in values))

    def test_sorted_unique(self):
        result = build_erase_coverage_arg(0.60)
        values = [float(x) for x in result.split(",")]
        self.assertEqual(values, sorted(set(values)))


class BuildEraseCmdTests(unittest.TestCase):
    def test_basic(self):
        cmd = build_erase_cmd("/b", "v.mp4", "t.npz", "out.mp4", 0.65, True)
        self.assertIn("--video", cmd)
        self.assertIn("--track", cmd)
        self.assertIn("--out", cmd)
        self.assertIn("--coverage", cmd)
        self.assertIn("--try-strict", cmd)
        self.assertIn("--keep-audio", cmd)

    def test_shading_plane(self):
        cmd = build_erase_cmd("/b", "v.mp4", "t.npz", "out.mp4", 0.65, True)
        idx = cmd.index("--shading")
        self.assertEqual(cmd[idx + 1], "plane")

    def test_shading_none(self):
        cmd = build_erase_cmd("/b", "v.mp4", "t.npz", "out.mp4", 0.65, False)
        idx = cmd.index("--shading")
        self.assertEqual(cmd[idx + 1], "none")

    def test_uses_resolved_python_executable(self):
        with mock.patch(
            "mouth_track_gui.actions.resolve_python_subprocess_executable",
            return_value="C:\\venv\\Scripts\\python.exe",
        ):
            cmd = build_erase_cmd("/b", "v.mp4", "t.npz", "out.mp4", 0.65, False)
        self.assertEqual(cmd[0], "C:\\venv\\Scripts\\python.exe")


class ResolveRuntimeScriptTests(unittest.TestCase):
    def test_prefers_emotion_auto(self):
        with tempfile.TemporaryDirectory() as td:
            ea = os.path.join(td, "loop_lipsync_runtime_patched_emotion_auto.py")
            plain = os.path.join(td, "loop_lipsync_runtime_patched.py")
            open(ea, "w").close()
            open(plain, "w").close()
            self.assertEqual(resolve_runtime_script(td), ea)

    def test_fallback_to_plain(self):
        with tempfile.TemporaryDirectory() as td:
            plain = os.path.join(td, "loop_lipsync_runtime_patched.py")
            open(plain, "w").close()
            result = resolve_runtime_script(td)
            self.assertEqual(result, plain)


class BuildLiveCmdTests(unittest.TestCase):
    def _make_runtime(self, content: str = "") -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        )
        f.write(content)
        f.close()
        return f.name

    def test_basic_without_emotion(self):
        rt = self._make_runtime("# no flags")
        try:
            cmd = build_live_cmd(
                "/b", rt, "loop.mp4", "/mouth", "t.npz", "c.npz", 0,
            )
            self.assertIn("--no-auto-last-session", cmd)
            self.assertIn("--loop-video", cmd)
            self.assertNotIn("--emotion-auto", cmd)
        finally:
            os.unlink(rt)

    def test_with_emotion_auto(self):
        rt = self._make_runtime(
            "# --emotion-auto --emotion-preset --emotion-hud "
            "--no-emotion-hud --no-emotion-gui "
            "--emotion-silence-db --emotion-min-conf "
            "--emotion-hud-font --emotion-hud-alpha "
            "--mouth-brightness --mouth-saturation --mouth-warmth "
            "--mouth-color-strength --mouth-edge-priority "
            "--mouth-edge-width-ratio --mouth-inspect-boost "
            "--mouth-ipc-token --mouth-live-control --mouth-auto-request --mouth-auto-result"
        )
        try:
            cmd = build_live_cmd(
                "/b", rt, "loop.mp4", "/mouth", "t.npz", "c.npz", 0,
                emotion_preset_key="excited",
                emotion_hud=True,
                mouth_brightness=1.0,
                mouth_saturation=1.1,
                mouth_warmth=2.0,
                mouth_color_strength=0.7,
                mouth_edge_priority=0.8,
                mouth_edge_width_ratio=0.12,
                mouth_inspect_boost=3.0,
                mouth_ipc_token="token-123",
                live_color_control_path="control.json",
                auto_color_request_path="auto_req.json",
                auto_color_result_path="auto_res.json",
            )
            self.assertIn("--emotion-auto", cmd)
            self.assertIn("--emotion-preset", cmd)
            idx = cmd.index("--emotion-preset")
            self.assertEqual(cmd[idx + 1], "excited")
            self.assertIn("--emotion-hud", cmd)
            self.assertIn("--no-emotion-gui", cmd)
            self.assertIn("--emotion-min-conf", cmd)
            min_conf_idx = cmd.index("--emotion-min-conf")
            self.assertEqual(cmd[min_conf_idx + 1], "0.12")
            self.assertIn("--mouth-brightness", cmd)
            self.assertIn("--mouth-inspect-boost", cmd)
            self.assertIn("--mouth-ipc-token", cmd)
            self.assertIn("--mouth-live-control", cmd)
            self.assertIn("--mouth-auto-request", cmd)
            self.assertIn("--mouth-auto-result", cmd)
        finally:
            os.unlink(rt)

    def test_with_live_control_flag(self):
        rt = self._make_runtime("# --mouth-live-control --mouth-auto-request --mouth-auto-result")
        try:
            cmd = build_live_cmd(
                "/b", rt, "loop.mp4", "/mouth", "t.npz", "c.npz", 0,
                live_color_control_path="control.json",
                auto_color_request_path="auto_req.json",
                auto_color_result_path="auto_res.json",
            )
            self.assertIn("--mouth-live-control", cmd)
            idx = cmd.index("--mouth-live-control")
            self.assertEqual(cmd[idx + 1], "control.json")
            self.assertIn("--mouth-auto-request", cmd)
            self.assertIn("--mouth-auto-result", cmd)
        finally:
            os.unlink(rt)

    def test_with_audio_device_spec(self):
        rt = self._make_runtime("# no flags")
        try:
            cmd = build_live_cmd(
                "/b", rt, "loop.mp4", "/mouth", "t.npz", "c.npz", 0,
                audio_device_spec="pa:alsa_input.test",
            )
            self.assertIn("--audio-device-spec", cmd)
            idx = cmd.index("--audio-device-spec")
            self.assertEqual(cmd[idx + 1], "pa:alsa_input.test")
        finally:
            os.unlink(rt)

    def test_uses_resolved_python_executable(self):
        rt = self._make_runtime("# no flags")
        try:
            with mock.patch(
                "mouth_track_gui.actions.resolve_python_subprocess_executable",
                return_value="C:\\venv\\Scripts\\python.exe",
            ):
                cmd = build_live_cmd(
                    "/b", rt, "loop.mp4", "/mouth", "t.npz", "c.npz", 0,
                )
            self.assertEqual(cmd[0], "C:\\venv\\Scripts\\python.exe")
        finally:
            os.unlink(rt)


class PlanTrackAndCalibTests(unittest.TestCase):
    def _make_plan(self, **overrides):
        defaults = dict(
            base_dir="/base",
            video="v.mp4",
            track_npz="track.npz",
            calib_npz="calib.npz",
            open_sprite="open.png",
            mouth_dir="/mouth",
            pad=2.0,
            coverage=0.65,
            audio_device=0,
            audio_device_spec="sd:0",
            smoothing_cutoff=None,
            smoothing_label="Auto",
            mouth_brightness=0.0,
            mouth_saturation=1.0,
            mouth_warmth=0.0,
            mouth_color_strength=0.75,
            mouth_edge_priority=0.85,
            mouth_edge_width_ratio=0.10,
            mouth_inspect_boost=1.0,
        )
        defaults.update(overrides)
        return plan_track_and_calib(**defaults)

    def test_two_steps(self):
        plan = self._make_plan()
        self.assertEqual(plan.total_steps, 2)

    def test_first_step_is_track(self):
        plan = self._make_plan()
        self.assertIn("auto_mouth_track_v2.py", " ".join(plan.steps[0].cmd))
        self.assertTrue(plan.steps[0].allow_soft_stop)
        self.assertFalse(plan.steps[0].skip_on_stop)

    def test_second_step_is_calib(self):
        plan = self._make_plan()
        self.assertIn("calibrate_mouth_track.py", " ".join(plan.steps[1].cmd))
        self.assertTrue(plan.steps[1].skip_on_stop)

    def test_session_init_keys(self):
        plan = self._make_plan()
        for key in ("video", "source_video", "mouth_dir", "coverage", "pad",
                     "audio_device", "audio_device_spec", "smoothing", "mouth_brightness", "mouth_inspect_boost"):
            self.assertIn(key, plan.session_init)

    def test_session_final_keys(self):
        plan = self._make_plan()
        self.assertIn("track", plan.session_final)
        self.assertIn("track_calibrated", plan.session_final)

    def test_smoothing_applied(self):
        plan = self._make_plan(smoothing_cutoff=1.5)
        cmd = plan.steps[0].cmd
        self.assertIn("--smooth-cutoff", cmd)

    def test_first_step_not_skip_on_stop(self):
        """Track step must not be skippable — it needs to run."""
        plan = self._make_plan()
        self.assertFalse(plan.steps[0].skip_on_stop)

    def test_calib_step_skip_on_stop(self):
        """Calib step must be skippable when stop is requested after track."""
        plan = self._make_plan()
        self.assertTrue(plan.steps[1].skip_on_stop)


class PlanCalibOnlyTests(unittest.TestCase):
    def test_one_step(self):
        plan = plan_calib_only(
            base_dir="/b", video="v.mp4",
            track_npz="t.npz", calib_npz="/nonexistent.npz",
            open_sprite="open.png",
            mouth_brightness=0.0,
            mouth_saturation=1.0,
            mouth_warmth=0.0,
            mouth_color_strength=0.75,
            mouth_edge_priority=0.85,
            mouth_edge_width_ratio=0.10,
            mouth_inspect_boost=1.0,
        )
        self.assertEqual(plan.total_steps, 1)

    def test_uses_track_when_no_calib(self):
        plan = plan_calib_only(
            base_dir="/b", video="v.mp4",
            track_npz="t.npz", calib_npz="/nonexistent_calib.npz",
            open_sprite="open.png",
            mouth_brightness=0.0,
            mouth_saturation=1.0,
            mouth_warmth=0.0,
            mouth_color_strength=0.75,
            mouth_edge_priority=0.85,
            mouth_edge_width_ratio=0.10,
            mouth_inspect_boost=1.0,
        )
        self.assertIn("t.npz", plan.steps[0].cmd)

    def test_uses_calib_when_exists(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            calib = f.name
        try:
            plan = plan_calib_only(
                base_dir="/b", video="v.mp4",
                track_npz="t.npz", calib_npz=calib,
                open_sprite="open.png",
                mouth_brightness=0.0,
                mouth_saturation=1.0,
                mouth_warmth=0.0,
                mouth_color_strength=0.75,
                mouth_edge_priority=0.85,
                mouth_edge_width_ratio=0.10,
                mouth_inspect_boost=1.0,
            )
            self.assertIn(calib, plan.steps[0].cmd)
            self.assertTrue(plan.steps[0].pre_log)
        finally:
            os.unlink(calib)

    def test_session_final_keys(self):
        plan = plan_calib_only(
            base_dir="/b", video="v.mp4",
            track_npz="t.npz", calib_npz="/x.npz",
            open_sprite="open.png",
            mouth_brightness=0.0,
            mouth_saturation=1.0,
            mouth_warmth=0.0,
            mouth_color_strength=0.75,
            mouth_edge_priority=0.85,
            mouth_edge_width_ratio=0.10,
            mouth_inspect_boost=1.0,
        )
        for key in ("track", "track_calibrated", "calib"):
            self.assertIn(key, plan.session_final)


class PlanEraseTests(unittest.TestCase):
    def _make_plan(self, **overrides):
        defaults = dict(
            base_dir="/b", video="v.mp4", mouth_dir="/mouth",
            track_npz="t.npz", calib_npz="c.npz",
            erase_track="c.npz", mouthless_mp4="out.mp4",
            coverage=0.65, pad=2.0, audio_device=0,
            audio_device_spec="sd:0",
            erase_shading=True,
        )
        defaults.update(overrides)
        return plan_erase(**defaults)

    def test_one_step(self):
        plan = self._make_plan()
        self.assertEqual(plan.total_steps, 1)

    def test_no_soft_stop(self):
        plan = self._make_plan()
        self.assertFalse(plan.steps[0].allow_soft_stop)

    def test_has_post_actions(self):
        plan = self._make_plan()
        self.assertTrue(len(plan.post_actions) >= 2)
        tags = [a.tag for a in plan.post_actions]
        self.assertIn("export_browser", tags)
        self.assertIn("open_preview", tags)

    def test_post_actions_are_structured(self):
        """PostAction stores paths as tuple elements, not colon-delimited."""
        plan = self._make_plan()
        for pa in plan.post_actions:
            self.assertIsInstance(pa, PostAction)
            self.assertIsInstance(pa.args, tuple)

    def test_post_actions_windows_paths_safe(self):
        """Windows drive-letter paths must survive PostAction round-trip."""
        plan = plan_erase(
            base_dir="C:\\base", video="C:\\data\\v.mp4",
            mouth_dir="C:\\mouth", track_npz="C:\\data\\t.npz",
            calib_npz="C:\\data\\c.npz", erase_track="C:\\data\\c.npz",
            mouthless_mp4="C:\\data\\out.mp4",
            coverage=0.65, pad=2.0, audio_device=0, erase_shading=True,
        )
        export = [a for a in plan.post_actions if a.tag == "export_browser"][0]
        preview = [a for a in plan.post_actions if a.tag == "open_preview"][0]
        self.assertEqual(export.args[0], "C:\\data\\out.mp4")
        self.assertEqual(export.args[1], "C:\\data\\c.npz")
        self.assertEqual(preview.args[0], "C:\\data\\out.mp4")

    def test_session_final_has_mouthless(self):
        plan = self._make_plan()
        self.assertEqual(plan.session_final["video"], "out.mp4")
        self.assertEqual(plan.session_final["source_video"], "v.mp4")
        self.assertEqual(plan.session_final["audio_device_spec"], "sd:0")

    def test_pre_log_has_track_info(self):
        plan = self._make_plan(erase_track="my_track.npz")
        self.assertIn("my_track.npz", plan.steps[0].pre_log)


class PlanLiveTests(unittest.TestCase):
    def _make_runtime(self) -> str:
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8",
        )
        f.write("# --emotion-auto --emotion-preset")
        f.close()
        return f.name

    def test_one_step(self):
        rt = self._make_runtime()
        try:
            plan = plan_live(
                base_dir="/b", runtime_py=rt,
                loop_video="loop.mp4", mouth_dir="/mouth",
                track_npz="t.npz", calib_npz="c.npz",
                device_idx=0, character="default", audio_device_spec="sd:0",
                emotion_preset_label="標準", emotion_preset_key="standard",
                emotion_hud=False,
                mouth_brightness=0.0,
                mouth_saturation=1.0,
                mouth_warmth=0.0,
                mouth_color_strength=0.75,
                mouth_edge_priority=0.85,
                mouth_edge_width_ratio=0.10,
                mouth_inspect_boost=1.0,
                live_color_control_path="control.json",
                auto_color_request_path="auto_req.json",
                auto_color_result_path="auto_res.json",
            )
            self.assertEqual(plan.total_steps, 1)
            self.assertTrue(plan.steps[0].allow_soft_stop)
        finally:
            os.unlink(rt)

    def test_session_init_keys(self):
        rt = self._make_runtime()
        try:
            plan = plan_live(
                base_dir="/b", runtime_py=rt,
                loop_video="loop.mp4", mouth_dir="/mouth",
                track_npz="t.npz", calib_npz="c.npz",
                device_idx=3, character="shizuku", audio_device_spec="pa:alsa_input.test",
                emotion_preset_label="興奮", emotion_preset_key="excited",
                emotion_hud=True,
                mouth_brightness=0.0,
                mouth_saturation=1.0,
                mouth_warmth=0.0,
                mouth_color_strength=0.75,
                mouth_edge_priority=0.85,
                mouth_edge_width_ratio=0.10,
                mouth_inspect_boost=1.0,
                live_color_control_path="control.json",
                auto_color_request_path="auto_req.json",
                auto_color_result_path="auto_res.json",
            )
            self.assertEqual(plan.session_init["audio_device"], 3)
            self.assertEqual(plan.session_init["audio_device_spec"], "pa:alsa_input.test")
            self.assertEqual(plan.session_init["character"], "shizuku")
            self.assertIn("mouth_brightness", plan.session_init)
        finally:
            os.unlink(rt)

    def test_no_completion_msg(self):
        rt = self._make_runtime()
        try:
            plan = plan_live(
                base_dir="/b", runtime_py=rt,
                loop_video="loop.mp4", mouth_dir="/mouth",
                track_npz="t.npz", calib_npz="c.npz",
                device_idx=0, character="default", audio_device_spec="sd:0",
                emotion_preset_label="標準", emotion_preset_key="standard",
                emotion_hud=False,
                mouth_brightness=0.0,
                mouth_saturation=1.0,
                mouth_warmth=0.0,
                mouth_color_strength=0.75,
                mouth_edge_priority=0.85,
                mouth_edge_width_ratio=0.10,
                mouth_inspect_boost=1.0,
                live_color_control_path="control.json",
                auto_color_request_path="auto_req.json",
                auto_color_result_path="auto_res.json",
            )
            self.assertEqual(plan.completion_msg, "")
        finally:
            os.unlink(rt)


if __name__ == "__main__":
    unittest.main()
