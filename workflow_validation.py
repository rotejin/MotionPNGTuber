from __future__ import annotations

import os
from dataclasses import dataclass

from lipsync_core import resolve_preferred_track_path


@dataclass(frozen=True)
class WorkflowPaths:
    source_video: str
    out_dir: str
    track_npz: str
    calib_npz: str
    preferred_track: str | None


def format_missing_path_message(label: str, path: str, next_step: str = "") -> str:
    lines = [f"{label} が見つかりません。", f"期待した場所: {path}"]
    if next_step:
        lines.append(next_step)
    return "\n".join(lines)


def format_missing_paths_message(label: str, paths: list[str], next_step: str = "") -> str:
    lines = [f"{label} が見つかりません。"]
    if paths:
        lines.append("確認した場所:")
        for path in paths:
            lines.append(f"- {path}")
    if next_step:
        lines.append(next_step)
    return "\n".join(lines)


def summarize_named_issues(
    header: str,
    items: list[tuple[str, str]],
    *,
    empty_action: str = "",
    tail_hint: str = "",
    limit: int = 3,
) -> str:
    if not items:
        lines = [header]
        if empty_action:
            lines.append(empty_action)
        return "\n".join(lines)

    lines = [header]
    for name, reason in items[: max(1, limit)]:
        lines.append(f"- {name}: {reason}")
    if len(items) > limit:
        lines.append("…")
    if tail_hint:
        lines.append(tail_hint)
    return "\n".join(lines)


def validate_existing_file(path: str, *, empty_message: str, missing_label: str) -> tuple[str | None, str]:
    candidate = str(path or "").strip()
    if not candidate:
        return None, empty_message

    full_path = os.path.abspath(candidate)
    if not os.path.isfile(full_path):
        return None, format_missing_path_message(missing_label, full_path)
    return full_path, ""


def validate_existing_dir(path: str, *, empty_message: str, missing_label: str) -> tuple[str | None, str]:
    candidate = str(path or "").strip()
    if not candidate:
        return None, empty_message

    full_path = os.path.abspath(candidate)
    if not os.path.isdir(full_path):
        return None, format_missing_path_message(missing_label, full_path)
    return full_path, ""


def build_workflow_paths(
    source_video: str,
    *,
    require_track: bool = False,
    require_calibrated: bool = False,
    prefer_calibrated: bool = False,
) -> tuple[WorkflowPaths | None, str]:
    video_path, err = validate_existing_file(
        source_video,
        empty_message="動画を選択してください。",
        missing_label="動画ファイル",
    )
    if video_path is None:
        return None, err

    out_dir = os.path.dirname(video_path)
    track_npz = os.path.join(out_dir, "mouth_track.npz")
    calib_npz = os.path.join(out_dir, "mouth_track_calibrated.npz")

    if require_track and not os.path.isfile(track_npz):
        return None, format_missing_path_message(
            "mouth_track.npz",
            track_npz,
            "先に『① 解析→キャリブ』を実行してください。",
        )

    if require_calibrated and not os.path.isfile(calib_npz):
        return None, format_missing_path_message(
            "mouth_track_calibrated.npz",
            calib_npz,
            "『キャリブのみ（やり直し）』または『① 解析→キャリブ』を実行してください。",
        )

    preferred_track = resolve_preferred_track_path(
        track_npz,
        calib_npz,
        prefer_calibrated=prefer_calibrated,
    )
    if not os.path.isfile(preferred_track):
        preferred_track = None

    return WorkflowPaths(
        source_video=video_path,
        out_dir=out_dir,
        track_npz=track_npz,
        calib_npz=calib_npz,
        preferred_track=preferred_track,
    ), ""
