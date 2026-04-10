#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_mouth_track.py

既存の mouth_track.npz に対して、スプライトのサイズ・位置・回転を
インタラクティブに調整するツール。

使い方:
    python calibrate_mouth_track.py \
        --video "assets/assets10/loop.mp4" \
        --track "assets/assets10/mouth_track.npz" \
        --sprite "assets/assets10/mouth/open.png" \
        --out "assets/assets10/mouth_track_calibrated.npz"

操作:
    - マウス左ドラッグ: 移動
    - マウスホイール: スケール (Ctrl+ホイールで微調整)
    - マウス右ドラッグ: 回転
    - 矢印キー: 微移動
    - +/-: スケール
    - z/x: 回転
    - [/]: フレーム移動（プレビュー用。パラメータ自体は維持されます）
    - r: 中立値にリセット（offset=0, scale=1, rotation=0）
    - Space/Enter: 確定
    - q/Esc: キャンセル
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

from motionpngtuber.image_io import read_image_bgra
from motionpngtuber.mouth_color_adjust import (
    MouthColorAdjust,
    apply_inspect_boost_3ch,
    apply_mouth_color_adjust_4ch,
    clamp_mouth_color_adjust,
)


# Windows の cv2.waitKeyEx が返す矢印キーコード
ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_RIGHT = 2555904
ARROW_DOWN = 2621440


def load_bgra(path: str) -> np.ndarray:
    img_bgra = read_image_bgra(path)
    if img_bgra is None:
        raise FileNotFoundError(path)
    return img_bgra


def warp_sprite_to_quad(src_sprite_4ch: np.ndarray, dst_quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Warp a 4-channel sprite in OpenCV-native order into a destination quad."""
    sh, sw = src_sprite_4ch.shape[:2]
    # OpenCV の射影変換は端点を含む座標系の方がズレが少ない
    src_quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    dst = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(src_quad, dst)
    warped = cv2.warpPerspective(
        src_sprite_4ch,
        M,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def alpha_blend_sprite_over_bgr(dst_bgr: np.ndarray, src_sprite_4ch_full: np.ndarray) -> np.ndarray:
    """Alpha-blend a 4-channel OpenCV-native sprite over a BGR frame."""
    if src_sprite_4ch_full.shape[:2] != dst_bgr.shape[:2]:
        raise ValueError("size mismatch")
    a = (src_sprite_4ch_full[:, :, 3:4].astype(np.float32) / 255.0)
    out = dst_bgr.astype(np.float32) * (1.0 - a) + src_sprite_4ch_full[:, :, :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_quad(img_bgr: np.ndarray, quad: np.ndarray, color=(0, 255, 0), thickness=2):
    q = quad.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img_bgr, [q], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return img_bgr


def quad_center(quad: np.ndarray) -> np.ndarray:
    return quad.mean(axis=0)


def quad_size(quad: np.ndarray) -> tuple[float, float]:
    w = np.linalg.norm(quad[1] - quad[0])
    h = np.linalg.norm(quad[3] - quad[0])
    return float(w), float(h)


def transform_quad(quad: np.ndarray, offset: np.ndarray, scale: float, rotation_deg: float) -> np.ndarray:
    center = quad_center(quad)
    rel = quad - center
    rel = rel * scale
    th = math.radians(rotation_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rel = rel @ R.T
    return (rel + center + offset).astype(np.float32)


def compute_preview_size(src_w: int, src_h: int, max_w: int, max_h: int):
    s = min(max_w / src_w, max_h / src_h, 1.0)
    disp_w = max(2, int(round(src_w * s)))
    disp_h = max(2, int(round(src_h * s)))
    return disp_w, disp_h, s


@dataclass
class DragState:
    dragging: bool = False
    start_xy: tuple[int, int] | None = None
    start_offset: np.ndarray | None = None
    rotating: bool = False
    start_rot_deg: float | None = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True, help="入力 mouth_track.npz")
    ap.add_argument("--sprite", required=True, help="口スプライト (サイズ確認用)")
    ap.add_argument("--out", required=True, help="出力 calibrated npz")
    ap.add_argument("--frame", type=int, default=0, help="開始フレーム (プレビュー用)")
    ap.add_argument("--ui-max-w", type=int, default=720)
    ap.add_argument("--ui-max-h", type=int, default=1280)
    ap.add_argument("--mouth-brightness", type=float, default=0.0)
    ap.add_argument("--mouth-saturation", type=float, default=1.0)
    ap.add_argument("--mouth-warmth", type=float, default=0.0)
    ap.add_argument("--mouth-color-strength", type=float, default=0.75)
    ap.add_argument("--mouth-edge-priority", type=float, default=0.85)
    ap.add_argument("--mouth-edge-width-ratio", type=float, default=0.10)
    ap.add_argument("--mouth-inspect-boost", type=float, default=1.0)
    args = ap.parse_args()
    color_cfg = clamp_mouth_color_adjust(
        MouthColorAdjust(
            brightness=float(args.mouth_brightness),
            saturation=float(args.mouth_saturation),
            warmth=float(args.mouth_warmth),
            color_strength=float(args.mouth_color_strength),
            edge_priority=float(args.mouth_edge_priority),
            edge_width_ratio=float(args.mouth_edge_width_ratio),
            inspect_boost=float(args.mouth_inspect_boost),
        ),
    )

    track = np.load(args.track, allow_pickle=False)
    quads = track["quad"].astype(np.float32).copy()
    N = int(len(quads))
    valid = track["valid"].astype(np.uint8) if "valid" in track.files else np.ones((N,), np.uint8)
    print(f"[info] loaded track: {N} frames")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read_frame(idx: int) -> np.ndarray:
        # track の範囲で clamp（動画の frame count が取れない場合もある）
        idx = int(max(0, min(idx, N - 1)))
        if total_video_frames > 0:
            idx = int(max(0, min(idx, total_video_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx}")
        return frame

    sprite_bgra = apply_mouth_color_adjust_4ch(
        load_bgra(args.sprite),
        color_cfg,
        color_order="BGRA",
    )

    # 既存のキャリブレーション値がある場合の処理
    # NOTE: 保存時に quads には既にキャリブレーションが適用済み。
    #       再キャリブレーション時は 1.0/0/0 から開始する必要がある。
    #       （既存値を再適用すると二重適用になってしまう）
    has_existing_calib = "calib_scale" in track.files or "calib_offset" in track.files
    if has_existing_calib:
        prev_offset = track["calib_offset"].astype(np.float32) if "calib_offset" in track.files else np.zeros(2, np.float32)
        prev_scale = float(track["calib_scale"]) if "calib_scale" in track.files else 1.0
        prev_rotation = float(track["calib_rotation"]) if "calib_rotation" in track.files else 0.0
        print(f"[info] existing calibration found: offset=({prev_offset[0]:.2f}, {prev_offset[1]:.2f}), scale={prev_scale:.4f}, rot={prev_rotation:.2f}deg")
        print(f"[info] starting from neutral (1.0/0/0) since quads already include previous calibration")

    # 常に中立値から開始（quads は既にキャリブレーション適用済みのため）
    offset = np.array([0.0, 0.0], dtype=np.float32)
    scale = 1.0
    rotation = 0.0

    # reset 用に初期値を保持
    init_offset = offset.copy()
    init_scale = float(scale)
    init_rotation = float(rotation)

    frame_idx = int(max(0, min(args.frame, N - 1)))
    base_bgr = read_frame(frame_idx)
    orig_quad = quads[frame_idx].copy()
    orig_w, orig_h = quad_size(orig_quad)

    def set_frame(idx: int):
        nonlocal frame_idx, base_bgr, orig_quad, orig_w, orig_h
        frame_idx = int(max(0, min(idx, N - 1)))
        base_bgr = read_frame(frame_idx)
        orig_quad = quads[frame_idx].copy()
        orig_w, orig_h = quad_size(orig_quad)

    state = DragState()

    disp_w, disp_h, ui_scale = compute_preview_size(vid_w, vid_h, args.ui_max_w, args.ui_max_h)
    win = "Mouth Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    def to_orig(pt_disp: tuple[int, int]) -> tuple[float, float]:
        return float(pt_disp[0]) / ui_scale, float(pt_disp[1]) / ui_scale

    def on_mouse(event, x, y, flags, userdata):
        nonlocal offset, scale, rotation
        if event == cv2.EVENT_LBUTTONDOWN:
            state.dragging = True
            state.start_xy = (x, y)
            state.start_offset = offset.copy()
        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
            state.start_xy = None
            state.start_offset = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.rotating = True
            state.start_xy = (x, y)
            state.start_rot_deg = float(rotation)
        elif event == cv2.EVENT_RBUTTONUP:
            state.rotating = False
            state.start_xy = None
            state.start_rot_deg = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragging and state.start_xy is not None and state.start_offset is not None:
                ox, oy = to_orig((x, y))
                sx, sy = to_orig(state.start_xy)
                dx, dy = (ox - sx), (oy - sy)
                offset = (state.start_offset + np.array([dx, dy], dtype=np.float32)).astype(np.float32)
            elif state.rotating and state.start_xy is not None and state.start_rot_deg is not None:
                dx = float(x - state.start_xy[0])
                rotation = float(state.start_rot_deg) + dx * 0.3
        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
                step = 1.05 if delta > 0 else (1.0 / 1.05)
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    step = 1.01 if delta > 0 else (1.0 / 1.01)
                scale *= step
            except Exception:
                pass

    cv2.setMouseCallback(win, on_mouse)

    print("[calib] L-drag: move | wheel: scale (Ctrl=fine) | R-drag: rotate")
    print("[calib] arrows/WASD: nudge | +/-: scale | z/x: rotate | [/]: frame | v: inspect | r: reset(neutral) | Space/Enter: confirm | q/Esc: quit")

    nudge = 1.0
    inspect_levels = (1.0, 2.0, 3.0, 4.0)
    inspect_boost = min(inspect_levels, key=lambda x: abs(x - float(color_cfg.inspect_boost)))
    try:
        while True:
            transformed_quad = transform_quad(orig_quad, offset, scale, rotation)
            vis = base_bgr.copy()
            warped = warp_sprite_to_quad(sprite_bgra, transformed_quad, vid_w, vid_h)
            vis = alpha_blend_sprite_over_bgr(vis, warped)
            vis = draw_quad(vis, transformed_quad, color=(0, 255, 0), thickness=2)
            vis = apply_inspect_boost_3ch(vis, inspect_boost, color_order="BGR")

            tw, th = quad_size(transformed_quad)
            cv2.putText(
                vis,
                f"frame {frame_idx+1}/{N}  valid={int(valid[frame_idx])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"offset=({offset[0]:.1f}, {offset[1]:.1f}) scale={scale:.3f} rot={rotation:.1f}deg",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"quad size: {tw:.1f}x{th:.1f} (original: {orig_w:.1f}x{orig_h:.1f}) inspect={inspect_boost:.1f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"bri={color_cfg.brightness:.0f} sat={color_cfg.saturation:.2f} warm={color_cfg.warmth:.0f} edge={color_cfg.edge_priority:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vis_disp = cv2.resize(
                vis,
                (disp_w, disp_h),
                interpolation=cv2.INTER_AREA if ui_scale < 1.0 else cv2.INTER_LINEAR,
            )
            cv2.imshow(win, vis_disp)

            key = cv2.waitKeyEx(15)

            if key in (27, ord("q")):
                print("[calib] cancelled")
                cv2.destroyAllWindows()
                return 1  # キャンセル時は非ゼロ

            if key in (13, 32):  # Enter or Space
                break

            step = float(nudge)
            if key in (ARROW_LEFT, ord("a"), ord("A")):
                offset[0] -= step
            elif key in (ARROW_RIGHT, ord("d"), ord("D")):
                offset[0] += step
            elif key in (ARROW_UP, ord("w"), ord("W")):
                offset[1] -= step
            elif key in (ARROW_DOWN, ord("s"), ord("S")):
                offset[1] += step
            elif key in (ord("+"), ord("=")):
                scale *= 1.02
            elif key in (ord("-"), ord("_")):
                scale /= 1.02
            elif key in (ord("z"), ord("Z")):
                rotation -= 1.0
            elif key in (ord("x"), ord("X")):
                rotation += 1.0
            elif key in (ord("["),):
                set_frame(frame_idx - 1)
            elif key in (ord("]"),):
                set_frame(frame_idx + 1)
            elif key in (ord("r"), ord("R")):
                offset = init_offset.copy()
                scale = float(init_scale)
                rotation = float(init_rotation)
            elif key in (ord("v"), ord("V")):
                cur_idx = inspect_levels.index(inspect_boost) if inspect_boost in inspect_levels else 0
                inspect_boost = inspect_levels[(cur_idx + 1) % len(inspect_levels)]
            elif key in (ord(","), ord("<")):
                nudge = max(0.1, nudge / 1.5)
            elif key in (ord("."), ord(">")):
                nudge = min(20.0, nudge * 1.5)

        cv2.destroyAllWindows()

    finally:
        cap.release()

    print(f"[info] applying transform to {N} frames...")
    # ベクトル化（高速）
    quads_center = quads.mean(axis=1, keepdims=True)  # (N,1,2)
    rel = (quads - quads_center) * float(scale)
    th = math.radians(float(rotation))
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rel = rel @ R.T
    calibrated_quads = (rel + quads_center + offset.reshape(1, 1, 2)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_dict = {k: track[k] for k in track.files if k != "quad"}
    save_dict["quad"] = calibrated_quads
    save_dict["calib_offset"] = offset.astype(np.float32)
    save_dict["calib_scale"] = float(scale)
    save_dict["calib_rotation"] = float(rotation)
    np.savez_compressed(args.out, **save_dict)

    print(f"[saved] {args.out}")
    print(f"  offset: ({offset[0]:.2f}, {offset[1]:.2f})")
    print(f"  scale: {scale:.4f}")
    print(f"  rotation: {rotation:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
