"""Action definitions and command building for mouth_track_gui.

Phase 5 of the mouth_track_gui refactoring plan.
Provides pure functions that build subprocess commands and action plans
from resolved context values.  No GUI or subprocess dependencies —
designed for easy unit testing.

Usage from App::

    plan = plan_track_and_calib(base_dir=..., video=..., ...)
    self._execute_plan(plan)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from python_exec import resolve_python_subprocess_executable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionStep:
    """A single command execution step in a workflow."""

    cmd: list[str]
    label: str  # Header label shown in log
    cwd: str | None = None
    expected_outputs: tuple[str, ...] = ()
    allow_soft_stop: bool = True
    error_msg: str = ""  # Shown on failure (with rc appended)
    skip_on_stop: bool = False  # Skip this step if soft stop is pending
    pre_log: str = ""  # Extra log line before running
    progress_label: str = ""  # Short label for progress bar


@dataclass(frozen=True)
class PostAction:
    """A post-plan action (e.g. export browser assets, open preview)."""

    tag: str  # "export_browser" or "open_preview"
    args: tuple[str, ...] = ()  # Paths — safe for Windows drive letters


@dataclass(frozen=True)
class ActionPlan:
    """A complete workflow plan with steps and metadata."""

    name: str
    steps: tuple[ActionStep, ...]
    session_init: dict = field(default_factory=dict)
    session_final: dict = field(default_factory=dict)
    post_actions: tuple[PostAction, ...] = ()
    completion_msg: str = "完了"

    @property
    def total_steps(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# Command builders (pure functions)
# ---------------------------------------------------------------------------

def build_track_cmd(
    base_dir: str,
    video: str,
    track_npz: str,
    pad: float,
    *,
    smoothing_cutoff: float | None = None,
) -> list[str]:
    """Build the auto_mouth_track_v2.py command."""
    cmd = [
        resolve_python_subprocess_executable(),
        os.path.join(base_dir, "auto_mouth_track_v2.py"),
        "--video", video,
        "--out", track_npz,
        "--pad", f"{pad:.2f}",
        "--stride", "1",
        "--det-scale", "1.0",
        "--min-conf", "0.5",
        "--early-stop",
        "--max-tries", "4",
    ]
    if smoothing_cutoff is not None:
        cmd += ["--smooth-cutoff", str(smoothing_cutoff)]
    return cmd


def build_calib_cmd(
    base_dir: str,
    video: str,
    track_input: str,
    open_sprite: str,
    calib_npz: str,
    *,
    mouth_brightness: float,
    mouth_saturation: float,
    mouth_warmth: float,
    mouth_color_strength: float,
    mouth_edge_priority: float,
    mouth_edge_width_ratio: float,
    mouth_inspect_boost: float,
) -> list[str]:
    """Build the calibrate_mouth_track.py command."""
    return [
        resolve_python_subprocess_executable(),
        os.path.join(base_dir, "calibrate_mouth_track.py"),
        "--video", video,
        "--track", track_input,
        "--sprite", open_sprite,
        "--out", calib_npz,
        "--mouth-brightness", str(mouth_brightness),
        "--mouth-saturation", str(mouth_saturation),
        "--mouth-warmth", str(mouth_warmth),
        "--mouth-color-strength", str(mouth_color_strength),
        "--mouth-edge-priority", str(mouth_edge_priority),
        "--mouth-edge-width-ratio", str(mouth_edge_width_ratio),
        "--mouth-inspect-boost", str(mouth_inspect_boost),
    ]


def build_erase_coverage_arg(coverage: float) -> str:
    """Compute the multi-coverage argument string for auto_erase_mouth.py."""
    covs = [max(0.40, min(0.90, coverage + x)) for x in (0.0, 0.10, 0.20)]
    covs = sorted(set(round(x, 2) for x in covs))
    return ",".join(f"{x:.2f}" for x in covs)


def build_erase_cmd(
    base_dir: str,
    video: str,
    erase_track: str,
    mouthless_mp4: str,
    coverage: float,
    erase_shading: bool,
) -> list[str]:
    """Build the auto_erase_mouth.py command."""
    cov_arg = build_erase_coverage_arg(coverage)
    return [
        resolve_python_subprocess_executable(),
        os.path.join(base_dir, "auto_erase_mouth.py"),
        "--video", video,
        "--track", erase_track,
        "--out", mouthless_mp4,
        "--coverage", cov_arg,
        "--try-strict",
        "--keep-audio",
        "--shading", ("plane" if erase_shading else "none"),
    ]


def resolve_runtime_script(base_dir: str) -> str:
    """Return the best available runtime script path."""
    emotion_auto = os.path.join(
        base_dir, "loop_lipsync_runtime_patched_emotion_auto.py",
    )
    if os.path.isfile(emotion_auto):
        return emotion_auto
    return os.path.join(base_dir, "loop_lipsync_runtime_patched.py")


def build_live_cmd(
    base_dir: str,
    runtime_py: str,
    loop_video: str,
    mouth_dir: str,
    track_npz: str,
    calib_npz: str,
    device_idx: int,
    *,
    audio_device_spec: str = "",
    emotion_preset_key: str = "standard",
    emotion_hud: bool = False,
    mouth_brightness: float = 0.0,
    mouth_saturation: float = 1.0,
    mouth_warmth: float = 0.0,
    mouth_color_strength: float = 0.75,
    mouth_edge_priority: float = 0.85,
    mouth_edge_width_ratio: float = 0.10,
    mouth_inspect_boost: float = 1.0,
    mouth_ipc_token: str = "",
    live_color_control_path: str = "",
    auto_color_request_path: str = "",
    auto_color_result_path: str = "",
) -> list[str]:
    """Build the runtime command for live execution."""
    from .services import script_contains

    cmd = [
        resolve_python_subprocess_executable(), runtime_py,
        "--no-auto-last-session",
        "--loop-video", loop_video,
        "--mouth-dir", mouth_dir,
        "--track", track_npz,
        "--track-calibrated", calib_npz,
        "--device", str(device_idx),
    ]
    if audio_device_spec:
        cmd += ["--audio-device-spec", audio_device_spec]

    if script_contains(runtime_py, ["--no-emotion-gui"]):
        cmd.append("--no-emotion-gui")

    if script_contains(runtime_py, ["--mouth-brightness"]):
        cmd += ["--mouth-brightness", str(mouth_brightness)]
    if script_contains(runtime_py, ["--mouth-saturation"]):
        cmd += ["--mouth-saturation", str(mouth_saturation)]
    if script_contains(runtime_py, ["--mouth-warmth"]):
        cmd += ["--mouth-warmth", str(mouth_warmth)]
    if script_contains(runtime_py, ["--mouth-color-strength"]):
        cmd += ["--mouth-color-strength", str(mouth_color_strength)]
    if script_contains(runtime_py, ["--mouth-edge-priority"]):
        cmd += ["--mouth-edge-priority", str(mouth_edge_priority)]
    if script_contains(runtime_py, ["--mouth-edge-width-ratio"]):
        cmd += ["--mouth-edge-width-ratio", str(mouth_edge_width_ratio)]
    if script_contains(runtime_py, ["--mouth-inspect-boost"]):
        cmd += ["--mouth-inspect-boost", str(mouth_inspect_boost)]
    if mouth_ipc_token and script_contains(runtime_py, ["--mouth-ipc-token"]):
        cmd += ["--mouth-ipc-token", mouth_ipc_token]
    if live_color_control_path and script_contains(runtime_py, ["--mouth-live-control"]):
        cmd += ["--mouth-live-control", live_color_control_path]
    if auto_color_request_path and script_contains(runtime_py, ["--mouth-auto-request"]):
        cmd += ["--mouth-auto-request", auto_color_request_path]
    if auto_color_result_path and script_contains(runtime_py, ["--mouth-auto-result"]):
        cmd += ["--mouth-auto-result", auto_color_result_path]

    if script_contains(runtime_py, ["--emotion-auto"]):
        cmd.append("--emotion-auto")
        if script_contains(runtime_py, ["--emotion-preset"]):
            cmd += ["--emotion-preset", emotion_preset_key]
        if script_contains(runtime_py, ["--emotion-hud", "--no-emotion-hud"]):
            cmd.append(
                "--emotion-hud" if emotion_hud else "--no-emotion-hud",
            )
        if script_contains(runtime_py, ["--emotion-silence-db"]):
            cmd += ["--emotion-silence-db", "-65"]
        if script_contains(runtime_py, ["--emotion-min-conf"]):
            # Keep the existing GUI tuning explicit.  The runtime's hidden
            # default is 0.45, but the GUI has been launching emotion-auto
            # with a looser 0.12 threshold.
            cmd += ["--emotion-min-conf", "0.12"]
        if script_contains(runtime_py, ["--emotion-hud-font"]):
            cmd += ["--emotion-hud-font", "28"]
        if script_contains(runtime_py, ["--emotion-hud-alpha"]):
            cmd += ["--emotion-hud-alpha", "0.92"]

    return cmd


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------

def plan_track_and_calib(
    *,
    base_dir: str,
    video: str,
    track_npz: str,
    calib_npz: str,
    open_sprite: str,
    mouth_dir: str,
    pad: float,
    coverage: float,
    audio_device: int,
    smoothing_cutoff: float | None,
    smoothing_label: str,
    mouth_brightness: float,
    mouth_saturation: float,
    mouth_warmth: float,
    mouth_color_strength: float,
    mouth_edge_priority: float,
    mouth_edge_width_ratio: float,
    mouth_inspect_boost: float,
    audio_device_spec: str = "",
) -> ActionPlan:
    """Build an ActionPlan for track + calibrate workflow."""
    return ActionPlan(
        name="解析/キャリブ",
        steps=(
            ActionStep(
                cmd=build_track_cmd(
                    base_dir, video, track_npz, pad,
                    smoothing_cutoff=smoothing_cutoff,
                ),
                label="解析（自動修復つき・最高品質）",
                progress_label="解析",
                cwd=base_dir,
                expected_outputs=(track_npz,),
                allow_soft_stop=True,
                error_msg="解析に失敗しました",
            ),
            ActionStep(
                cmd=build_calib_cmd(
                    base_dir, video, track_npz, open_sprite, calib_npz,
                    mouth_brightness=mouth_brightness,
                    mouth_saturation=mouth_saturation,
                    mouth_warmth=mouth_warmth,
                    mouth_color_strength=mouth_color_strength,
                    mouth_edge_priority=mouth_edge_priority,
                    mouth_edge_width_ratio=mouth_edge_width_ratio,
                    mouth_inspect_boost=mouth_inspect_boost,
                ),
                label="キャリブレーション（画面を閉じると完了）",
                progress_label="キャリブ",
                cwd=base_dir,
                expected_outputs=(calib_npz,),
                allow_soft_stop=True,
                error_msg="キャリブに失敗しました",
                skip_on_stop=True,
            ),
        ),
        session_init={
            "video": video,
            "source_video": video,
            "mouth_dir": mouth_dir,
            "coverage": coverage,
            "pad": pad,
            "audio_device": audio_device,
            "audio_device_spec": audio_device_spec,
            "smoothing": smoothing_label,
            "mouth_brightness": mouth_brightness,
            "mouth_saturation": mouth_saturation,
            "mouth_warmth": mouth_warmth,
            "mouth_color_strength": mouth_color_strength,
            "mouth_edge_priority": mouth_edge_priority,
            "mouth_edge_width_ratio": mouth_edge_width_ratio,
            "mouth_inspect_boost": mouth_inspect_boost,
        },
        session_final={
            "track": track_npz,
            "track_calibrated": calib_npz,
        },
        completion_msg="完了（次は『② 口消し動画生成』）",
    )


def plan_calib_only(
    *,
    base_dir: str,
    video: str,
    track_npz: str,
    calib_npz: str,
    open_sprite: str,
    mouth_brightness: float,
    mouth_saturation: float,
    mouth_warmth: float,
    mouth_color_strength: float,
    mouth_edge_priority: float,
    mouth_edge_width_ratio: float,
    mouth_inspect_boost: float,
) -> ActionPlan:
    """Build an ActionPlan for calibration-only workflow."""
    use_calib = os.path.isfile(calib_npz)
    input_track = calib_npz if use_calib else track_npz
    pre_log = (
        "[info] 既存のキャリブ済みトラックを使用（位置を維持）"
        if use_calib else ""
    )
    return ActionPlan(
        name="キャリブ（やり直し）",
        steps=(
            ActionStep(
                cmd=build_calib_cmd(
                    base_dir, video, input_track, open_sprite, calib_npz,
                    mouth_brightness=mouth_brightness,
                    mouth_saturation=mouth_saturation,
                    mouth_warmth=mouth_warmth,
                    mouth_color_strength=mouth_color_strength,
                    mouth_edge_priority=mouth_edge_priority,
                    mouth_edge_width_ratio=mouth_edge_width_ratio,
                    mouth_inspect_boost=mouth_inspect_boost,
                ),
                label="キャリブレーション（やり直し）",
                progress_label="キャリブ",
                cwd=base_dir,
                expected_outputs=(calib_npz,),
                allow_soft_stop=True,
                error_msg="キャリブに失敗しました",
                pre_log=pre_log,
            ),
        ),
        session_final={
            "track": track_npz,
            "track_path": track_npz,
            "track_calibrated": calib_npz,
            "track_calibrated_path": calib_npz,
            "calib": calib_npz,
        },
    )


def plan_erase(
    *,
    base_dir: str,
    video: str,
    mouth_dir: str,
    track_npz: str,
    calib_npz: str,
    erase_track: str,
    mouthless_mp4: str,
    coverage: float,
    pad: float,
    audio_device: int,
    erase_shading: bool,
    audio_device_spec: str = "",
) -> ActionPlan:
    """Build an ActionPlan for mouth-erase video generation."""
    return ActionPlan(
        name="口消し動画生成",
        steps=(
            ActionStep(
                cmd=build_erase_cmd(
                    base_dir, video, erase_track, mouthless_mp4,
                    coverage, erase_shading,
                ),
                label="口消し動画生成（自動候補→自動選別）",
                progress_label="口消し",
                cwd=base_dir,
                expected_outputs=(mouthless_mp4,),
                allow_soft_stop=False,
                error_msg="口消し動画生成に失敗しました",
                pre_log=f"[info] 口消しに使用する track: {erase_track}",
            ),
        ),
        session_final={
            "video": mouthless_mp4,
            "source_video": video,
            "mouth_dir": mouth_dir,
            "track": track_npz,
            "track_calibrated": calib_npz,
            "coverage": coverage,
            "pad": pad,
            "audio_device": audio_device,
            "audio_device_spec": audio_device_spec,
        },
        post_actions=(
            PostAction("export_browser", (mouthless_mp4, calib_npz)),
            PostAction("open_preview", (mouthless_mp4,)),
        ),
        completion_msg="完了（次は『③ ライブ実行』）",
    )


def plan_live(
    *,
    base_dir: str,
    runtime_py: str,
    loop_video: str,
    mouth_dir: str,
    track_npz: str,
    calib_npz: str,
    device_idx: int,
    character: str,
    emotion_preset_label: str,
    emotion_preset_key: str,
    emotion_hud: bool,
    mouth_brightness: float,
    mouth_saturation: float,
    mouth_warmth: float,
    mouth_color_strength: float,
    mouth_edge_priority: float,
    mouth_edge_width_ratio: float,
    mouth_inspect_boost: float,
    live_color_control_path: str = "",
    auto_color_request_path: str = "",
    auto_color_result_path: str = "",
    mouth_ipc_token: str = "",
    audio_device_spec: str = "",
) -> ActionPlan:
    """Build an ActionPlan for live runtime execution."""
    return ActionPlan(
        name="ライブ実行",
        steps=(
            ActionStep(
                cmd=build_live_cmd(
                    base_dir, runtime_py, loop_video, mouth_dir,
                    track_npz, calib_npz, device_idx,
                    audio_device_spec=audio_device_spec,
                    emotion_preset_key=emotion_preset_key,
                    emotion_hud=emotion_hud,
                    mouth_brightness=mouth_brightness,
                    mouth_saturation=mouth_saturation,
                    mouth_warmth=mouth_warmth,
                    mouth_color_strength=mouth_color_strength,
                    mouth_edge_priority=mouth_edge_priority,
                    mouth_edge_width_ratio=mouth_edge_width_ratio,
                    mouth_inspect_boost=mouth_inspect_boost,
                    mouth_ipc_token=mouth_ipc_token,
                    live_color_control_path=live_color_control_path,
                    auto_color_request_path=auto_color_request_path,
                    auto_color_result_path=auto_color_result_path,
                ),
                label="ライブ実行（qで終了）",
                progress_label="ライブ実行",
                cwd=base_dir,
                allow_soft_stop=True,
            ),
        ),
        session_init={
            "audio_device": device_idx,
            "audio_device_spec": audio_device_spec,
            "character": character,
            "emotion_preset": emotion_preset_label,
            "emotion_hud": emotion_hud,
            "mouth_brightness": mouth_brightness,
            "mouth_saturation": mouth_saturation,
            "mouth_warmth": mouth_warmth,
            "mouth_color_strength": mouth_color_strength,
            "mouth_edge_priority": mouth_edge_priority,
            "mouth_edge_width_ratio": mouth_edge_width_ratio,
            "mouth_inspect_boost": mouth_inspect_boost,
        },
        completion_msg="",
    )
