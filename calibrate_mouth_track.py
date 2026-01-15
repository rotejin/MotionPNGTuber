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
    編集モード切替（ボタンクリック or キー）:
    - M: 移動モード（左ドラッグで移動）
    - R: 回転モード（左ドラッグで回転）
    - S: 均等スケールモード（左ドラッグで縦横同時拡縮）
    - X: X軸スケールモード（左ドラッグで横方向のみ拡縮）
    - Y: Y軸スケールモード（左ドラッグで縦方向のみ拡縮）

    ビュー操作（右パネル）:
    - ズームスライダー: 画面ズーム（ドラッグで調整）
    - パンボタン: クリックでパンモード切替、左ドラッグで画面移動
    - Homeボタン: ビューをリセット（ズーム・パンを初期状態に）
    - Undo/Redoボタン: 変換操作の取り消し・やり直し

    マウス操作（どのモードでも使用可能）:
    - ホイール: スケール調整
    - 中ボタンドラッグ: 画面移動（パン）
    - 右ドラッグ: 回転
    - Home: ビューをリセット

    キーボード:
    - 矢印キー: 微移動
    - +/-: スケール調整
    - </> : 回転微調整
    - [/]: フレーム移動（プレビュー用）
    - Backspace: 変換値を中立値にリセット
    - Enter/Confirmボタン: 確定
    - q/Esc: キャンセル
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import cv2
import numpy as np


class EditMode(Enum):
    """編集モード"""
    MOVE = auto()       # 移動
    ROTATE = auto()     # 回転
    SCALE_XY = auto()   # 均等スケール
    SCALE_X = auto()    # X軸スケール
    SCALE_Y = auto()    # Y軸スケール
    VIEW_PAN = auto()   # 画面パン（右パネルのボタンで切替）


# Windows の cv2.waitKeyEx が返す矢印キーコード
ARROW_LEFT = 2424832
ARROW_UP = 2490368
ARROW_RIGHT = 2555904
ARROW_DOWN = 2621440

# ボタン設定
BUTTON_HEIGHT = 35
BUTTON_MARGIN = 5
BUTTON_CONFIGS = [
    (EditMode.MOVE, "M", "Move"),
    (EditMode.ROTATE, "R", "Rotate"),
    (EditMode.SCALE_XY, "S", "Scale"),
    (EditMode.SCALE_X, "X", "ScaleX"),
    (EditMode.SCALE_Y, "Y", "ScaleY"),
]

# 右パネル設定
RIGHT_PANEL_WIDTH = 50
ZOOM_SLIDER_HEIGHT = 250  # スライダーを長くした
PAN_BUTTON_SIZE = 30
MAX_HISTORY = 50  # Undo/Redo履歴の最大数


def load_rgba(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("sprite must be 3 or 4 channel image")
    if img.shape[2] == 3:
        a = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        img = np.concatenate([img, a], axis=2)
    return img


def warp_rgba_to_quad(src_rgba: np.ndarray, dst_quad: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    sh, sw = src_rgba.shape[:2]
    src_quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)
    dst = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(src_quad, dst)
    warped = cv2.warpPerspective(
        src_rgba,
        M,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped


def alpha_blend_full(dst_bgr: np.ndarray, src_rgba_full: np.ndarray) -> np.ndarray:
    if src_rgba_full.shape[:2] != dst_bgr.shape[:2]:
        raise ValueError("size mismatch")
    a = (src_rgba_full[:, :, 3:4].astype(np.float32) / 255.0)
    out = dst_bgr.astype(np.float32) * (1.0 - a) + src_rgba_full[:, :, :3].astype(np.float32) * a
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


def transform_quad(quad: np.ndarray, offset: np.ndarray, scale_x: float, scale_y: float, rotation_deg: float) -> np.ndarray:
    """
    四角形を変換する。

    Args:
        quad: 元の四角形 (4, 2)
        offset: オフセット (2,)
        scale_x: X軸スケール
        scale_y: Y軸スケール
        rotation_deg: 回転角度 (度)

    Returns:
        変換後の四角形 (4, 2)
    """
    center = quad_center(quad)
    rel = quad - center
    # 縦横独立スケール
    rel = rel * np.array([scale_x, scale_y], dtype=np.float32)
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
class Button:
    """UIボタン"""
    x: int
    y: int
    width: int
    height: int
    mode: EditMode | None  # Noneはアクションボタン（確定ボタンなど）
    key: str
    label: str
    is_action: bool = False  # アクションボタンの場合True

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def draw(self, img: np.ndarray, is_active: bool):
        # 背景色（BGR形式）
        if self.is_action:
            bg_color = (0, 140, 255)  # オレンジ（アクションボタン）
        elif is_active:
            bg_color = (0, 180, 0)  # 緑（アクティブ）
        else:
            bg_color = (80, 80, 80)  # グレー

        # ボタン描画（枠線を内側に描画して塗りつぶしと揃える）
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.width - 1, self.y + self.height - 1
        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # テキスト
        text = f"[{self.key}] {self.label}"
        font_scale = 0.4
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def create_buttons(img_width: int) -> tuple[List[Button], Button]:
    """モードボタンと確定ボタンを作成"""
    buttons = []
    # 確定ボタン用のスペースを右側に確保
    confirm_width = 120
    available_width = img_width - confirm_width - BUTTON_MARGIN * 2
    total_buttons = len(BUTTON_CONFIGS)
    button_width = (available_width - BUTTON_MARGIN * (total_buttons + 1)) // total_buttons

    for i, (mode, key, label) in enumerate(BUTTON_CONFIGS):
        x = BUTTON_MARGIN + i * (button_width + BUTTON_MARGIN)
        buttons.append(Button(
            x=x,
            y=BUTTON_MARGIN,
            width=button_width,
            height=BUTTON_HEIGHT,
            mode=mode,
            key=key,
            label=label
        ))

    # 確定ボタン（右端に配置）
    confirm_btn = Button(
        x=img_width - confirm_width - BUTTON_MARGIN,
        y=BUTTON_MARGIN,
        width=confirm_width,
        height=BUTTON_HEIGHT,
        mode=None,
        key="Enter",
        label="Confirm",
        is_action=True
    )

    return buttons, confirm_btn


@dataclass
class DragState:
    """ドラッグ状態を保持"""
    dragging: bool = False
    start_xy: tuple[int, int] | None = None
    start_offset: np.ndarray | None = None
    start_scale_x: float | None = None
    start_scale_y: float | None = None
    start_rot_deg: float | None = None
    # パンモード用
    start_view_pan: list | None = None
    # 中ボタンドラッグ（パン用）
    middle_dragging: bool = False
    middle_start_xy: tuple[int, int] | None = None
    middle_start_pan: list | None = None
    # 右ボタンドラッグ（回転用）
    right_dragging: bool = False
    right_start_xy: tuple[int, int] | None = None
    right_start_rot_deg: float | None = None
    # ズームスライダードラッグ
    zoom_slider_dragging: bool = False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True, help="入力 mouth_track.npz")
    ap.add_argument("--sprite", required=True, help="口スプライト (サイズ確認用)")
    ap.add_argument("--out", required=True, help="出力 calibrated npz")
    ap.add_argument("--frame", type=int, default=0, help="開始フレーム (プレビュー用)")
    ap.add_argument("--ui-max-w", type=int, default=720)
    ap.add_argument("--ui-max-h", type=int, default=1280)
    args = ap.parse_args()

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
        idx = int(max(0, min(idx, N - 1)))
        if total_video_frames > 0:
            idx = int(max(0, min(idx, total_video_frames - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {idx}")
        return frame

    spr = load_rgba(args.sprite)

    # 既存のキャリブレーション値がある場合の処理（後方互換性）
    has_existing_calib = "calib_scale" in track.files or "calib_scale_x" in track.files or "calib_offset" in track.files
    if has_existing_calib:
        prev_offset = track["calib_offset"].astype(np.float32) if "calib_offset" in track.files else np.zeros(2, np.float32)
        # 新形式優先、旧形式フォールバック
        if "calib_scale_x" in track.files:
            prev_scale_x = float(track["calib_scale_x"])
            prev_scale_y = float(track["calib_scale_y"])
        else:
            prev_scale = float(track["calib_scale"]) if "calib_scale" in track.files else 1.0
            prev_scale_x = prev_scale
            prev_scale_y = prev_scale
        prev_rotation = float(track["calib_rotation"]) if "calib_rotation" in track.files else 0.0
        print(f"[info] existing calibration found: offset=({prev_offset[0]:.2f}, {prev_offset[1]:.2f}), scale=({prev_scale_x:.4f}, {prev_scale_y:.4f}), rot={prev_rotation:.2f}deg")
        print(f"[info] starting from neutral (1.0/0/0) since quads already include previous calibration")

    # 常に中立値から開始
    offset = np.array([0.0, 0.0], dtype=np.float32)
    scale_x = 1.0
    scale_y = 1.0
    rotation = 0.0
    edit_mode = EditMode.MOVE

    # ビュー制御用変数
    view_zoom = 1.0  # 1.0 = 100%, 2.0 = 200%
    view_pan = [0, 0]  # パンオフセット（表示ピクセル単位）

    # 確定フラグ
    confirmed = False

    # リセット用に初期値を保持
    init_offset = offset.copy()
    init_scale_x = float(scale_x)
    init_scale_y = float(scale_y)
    init_rotation = float(rotation)
    init_view_zoom = 1.0
    init_view_pan = [0, 0]

    # Undo/Redo履歴
    history: list[tuple] = []  # [(offset, scale_x, scale_y, rotation), ...]
    redo_stack: list[tuple] = []

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

    def save_state():
        """現在の状態を履歴に保存"""
        state_tuple = (offset.copy(), float(scale_x), float(scale_y), float(rotation))
        history.append(state_tuple)
        if len(history) > MAX_HISTORY:
            history.pop(0)
        redo_stack.clear()  # 新しい操作でredo履歴をクリア

    def do_undo():
        """ひとつ前の状態に戻す"""
        nonlocal offset, scale_x, scale_y, rotation
        if len(history) > 0:
            # 現在の状態をredo_stackに保存
            redo_stack.append((offset.copy(), float(scale_x), float(scale_y), float(rotation)))
            # 履歴から復元
            prev = history.pop()
            offset[:] = prev[0]
            scale_x = prev[1]
            scale_y = prev[2]
            rotation = prev[3]

    def do_redo():
        """やり直し"""
        nonlocal offset, scale_x, scale_y, rotation
        if len(redo_stack) > 0:
            # 現在の状態をhistoryに保存
            history.append((offset.copy(), float(scale_x), float(scale_y), float(rotation)))
            # redo_stackから復元
            next_state = redo_stack.pop()
            offset[:] = next_state[0]
            scale_x = next_state[1]
            scale_y = next_state[2]
            rotation = next_state[3]

    state = DragState()

    disp_w, disp_h, ui_scale = compute_preview_size(vid_w, vid_h, args.ui_max_w, args.ui_max_h)
    # ボタン用に上部にスペースを追加
    button_area_height = BUTTON_HEIGHT + BUTTON_MARGIN * 2
    # 右パネル用のスペース
    preview_w = disp_w - RIGHT_PANEL_WIDTH  # プレビュー領域の幅

    # 右パネル内のUI要素の位置（ウィンドウ座標系）
    panel_x = preview_w
    slider_x = panel_x + (RIGHT_PANEL_WIDTH - 10) // 2  # スライダー中心
    slider_y_top = button_area_height + 30
    slider_y_bottom = slider_y_top + ZOOM_SLIDER_HEIGHT
    pan_btn_x = panel_x + (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
    pan_btn_y = slider_y_bottom + 20
    # Homeボタン（パンボタン下）
    home_btn_x = panel_x + (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
    home_btn_y = pan_btn_y + PAN_BUTTON_SIZE + 10
    # Undoボタン（Homeボタン下）
    undo_btn_x = panel_x + (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
    undo_btn_y = home_btn_y + PAN_BUTTON_SIZE + 10
    # Redoボタン（Undoボタン下）
    redo_btn_x = panel_x + (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
    redo_btn_y = undo_btn_y + PAN_BUTTON_SIZE + 10

    win = "Mouth Calibration"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    buttons, confirm_btn = create_buttons(preview_w)  # プレビュー幅でボタン作成

    def to_orig(pt_disp: tuple[int, int]) -> tuple[float, float]:
        # ボタンエリアの高さを引いてから変換
        return float(pt_disp[0]) / ui_scale, float(pt_disp[1] - button_area_height) / ui_scale

    def on_mouse(event, x, y, flags, userdata):
        nonlocal offset, scale_x, scale_y, rotation, edit_mode, view_zoom, view_pan, confirmed

        # ズームスライダーの判定
        def is_in_slider(px, py):
            return (slider_x - 15 <= px <= slider_x + 15 and
                    slider_y_top <= py <= slider_y_bottom)

        # パンボタンの判定
        def is_in_pan_btn(px, py):
            return (pan_btn_x <= px <= pan_btn_x + PAN_BUTTON_SIZE and
                    pan_btn_y <= py <= pan_btn_y + PAN_BUTTON_SIZE)

        # Homeボタンの判定
        def is_in_home_btn(px, py):
            return (home_btn_x <= px <= home_btn_x + PAN_BUTTON_SIZE and
                    home_btn_y <= py <= home_btn_y + PAN_BUTTON_SIZE)

        # Undoボタンの判定
        def is_in_undo_btn(px, py):
            return (undo_btn_x <= px <= undo_btn_x + PAN_BUTTON_SIZE and
                    undo_btn_y <= py <= undo_btn_y + PAN_BUTTON_SIZE)

        # Redoボタンの判定
        def is_in_redo_btn(px, py):
            return (redo_btn_x <= px <= redo_btn_x + PAN_BUTTON_SIZE and
                    redo_btn_y <= py <= redo_btn_y + PAN_BUTTON_SIZE)

        # ズームスライダーのドラッグ処理
        if event == cv2.EVENT_LBUTTONDOWN and is_in_slider(x, y):
            state.zoom_slider_dragging = True
            # クリック位置からズーム値を計算
            t = (y - slider_y_top) / (slider_y_bottom - slider_y_top)
            t = max(0.0, min(1.0, t))
            view_zoom = 4.0 - t * 3.75  # 4.0 (top) to 0.25 (bottom)
            return

        if event == cv2.EVENT_LBUTTONUP and state.zoom_slider_dragging:
            state.zoom_slider_dragging = False
            return

        if event == cv2.EVENT_MOUSEMOVE and state.zoom_slider_dragging:
            t = (y - slider_y_top) / (slider_y_bottom - slider_y_top)
            t = max(0.0, min(1.0, t))
            view_zoom = 4.0 - t * 3.75
            return

        # パンボタンクリック処理
        if event == cv2.EVENT_LBUTTONDOWN and is_in_pan_btn(x, y):
            # パンモードの切替
            if edit_mode == EditMode.VIEW_PAN:
                edit_mode = EditMode.MOVE  # パンモード解除
            else:
                edit_mode = EditMode.VIEW_PAN
            return

        # Homeボタンクリック処理
        if event == cv2.EVENT_LBUTTONDOWN and is_in_home_btn(x, y):
            view_zoom = init_view_zoom
            view_pan[0] = init_view_pan[0]
            view_pan[1] = init_view_pan[1]
            return

        # Undoボタンクリック処理
        if event == cv2.EVENT_LBUTTONDOWN and is_in_undo_btn(x, y):
            do_undo()
            return

        # Redoボタンクリック処理
        if event == cv2.EVENT_LBUTTONDOWN and is_in_redo_btn(x, y):
            do_redo()
            return

        # ボタンクリック処理
        if event == cv2.EVENT_LBUTTONDOWN:
            # 確定ボタンを最初にチェック
            if confirm_btn.contains(x, y):
                confirmed = True
                return
            # モードボタンをチェック
            for btn in buttons:
                if btn.contains(x, y):
                    edit_mode = btn.mode
                    return

        # 中ボタンドラッグでパン（どのモードでも使用可能）
        if event == cv2.EVENT_MBUTTONDOWN:
            state.middle_dragging = True
            state.middle_start_xy = (x, y)
            state.middle_start_pan = view_pan.copy()
            return

        if event == cv2.EVENT_MBUTTONUP:
            state.middle_dragging = False
            state.middle_start_xy = None
            state.middle_start_pan = None
            return

        if event == cv2.EVENT_MOUSEMOVE and state.middle_dragging:
            if state.middle_start_xy is not None and state.middle_start_pan is not None:
                dx_px = x - state.middle_start_xy[0]
                dy_px = y - state.middle_start_xy[1]
                view_pan[0] = state.middle_start_pan[0] + dx_px
                view_pan[1] = state.middle_start_pan[1] + dy_px
            return

        # 右ボタンドラッグで回転（どのモードでも使用可能）
        if event == cv2.EVENT_RBUTTONDOWN:
            save_state()  # 回転開始前に状態を保存（Undo用）
            state.right_dragging = True
            state.right_start_xy = (x, y)
            state.right_start_rot_deg = float(rotation)
            return

        if event == cv2.EVENT_RBUTTONUP:
            state.right_dragging = False
            state.right_start_xy = None
            state.right_start_rot_deg = None
            return

        if event == cv2.EVENT_MOUSEMOVE and state.right_dragging:
            if state.right_start_xy is not None and state.right_start_rot_deg is not None:
                dx_px = x - state.right_start_xy[0]
                rotation = state.right_start_rot_deg + dx_px * 0.3
            return

        # ボタンエリア内はドラッグ処理しない
        if y < button_area_height:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # 編集モードの場合、操作開始前の状態を保存（Undo用）
            if edit_mode != EditMode.VIEW_PAN:
                save_state()
            state.dragging = True
            state.start_xy = (x, y)
            state.start_offset = offset.copy()
            state.start_scale_x = scale_x
            state.start_scale_y = scale_y
            state.start_rot_deg = float(rotation)
            state.start_view_pan = view_pan.copy()

        elif event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
            state.start_xy = None
            state.start_offset = None
            state.start_scale_x = None
            state.start_scale_y = None
            state.start_rot_deg = None
            state.start_view_pan = None

        elif event == cv2.EVENT_MOUSEMOVE:
            if state.dragging and state.start_xy is not None:
                dx_px = x - state.start_xy[0]
                dy_px = y - state.start_xy[1]

                if edit_mode == EditMode.MOVE and state.start_offset is not None:
                    ox, oy = to_orig((x, y))
                    sx, sy = to_orig(state.start_xy)
                    dx_orig, dy_orig = (ox - sx), (oy - sy)
                    offset = (state.start_offset + np.array([dx_orig, dy_orig], dtype=np.float32)).astype(np.float32)

                elif edit_mode == EditMode.ROTATE and state.start_rot_deg is not None:
                    rotation = float(state.start_rot_deg) + dx_px * 0.3

                elif edit_mode == EditMode.SCALE_XY and state.start_scale_x is not None:
                    # 上にドラッグ = 拡大、下にドラッグ = 縮小
                    factor = 1.0 - dy_px * 0.005
                    factor = max(0.1, min(5.0, factor))
                    scale_x = state.start_scale_x * factor
                    scale_y = state.start_scale_y * factor

                elif edit_mode == EditMode.SCALE_X and state.start_scale_x is not None:
                    # 右にドラッグ = 拡大、左にドラッグ = 縮小
                    factor = 1.0 + dx_px * 0.005
                    factor = max(0.1, min(5.0, factor))
                    scale_x = state.start_scale_x * factor

                elif edit_mode == EditMode.SCALE_Y and state.start_scale_y is not None:
                    # 上にドラッグ = 拡大、下にドラッグ = 縮小
                    factor = 1.0 - dy_px * 0.005
                    factor = max(0.1, min(5.0, factor))
                    scale_y = state.start_scale_y * factor

                elif edit_mode == EditMode.VIEW_PAN and state.start_view_pan is not None:
                    # ドラッグでパン
                    view_pan[0] = state.start_view_pan[0] + dx_px
                    view_pan[1] = state.start_view_pan[1] + dy_px

        elif event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
                # ホイール = スケール（どのモードでも使用可能、Undo対応）
                save_state()
                step = 1.05 if delta > 0 else (1.0 / 1.05)
                scale_x *= step
                scale_y *= step
            except Exception:
                pass

    cv2.setMouseCallback(win, on_mouse)

    print("[calib] Click buttons or press M/R/S/X/Y to switch edit mode")
    print("[calib] Wheel: scale | Middle-drag: pan | Right-drag: rotate | Slider: zoom")
    print("[calib] +/-: scale | </> : rotate | [/]: frame | Backspace: reset | Enter: confirm | q/Esc: quit")

    nudge = 1.0
    try:
        while True:
            transformed_quad = transform_quad(orig_quad, offset, scale_x, scale_y, rotation)
            vis = base_bgr.copy()
            warped = warp_rgba_to_quad(spr, transformed_quad, vid_w, vid_h)
            vis = alpha_blend_full(vis, warped)
            vis = draw_quad(vis, transformed_quad, color=(0, 255, 0), thickness=2)

            tw, th = quad_size(transformed_quad)

            # 表示用にリサイズ（プレビュー領域の幅を使用）
            vis_resized = cv2.resize(
                vis,
                (preview_w, disp_h),
                interpolation=cv2.INTER_AREA if ui_scale < 1.0 else cv2.INTER_LINEAR,
            )

            # ビューのズームとパンを適用
            if view_zoom != 1.0 or view_pan[0] != 0 or view_pan[1] != 0:
                zoomed_w = int(preview_w * view_zoom)
                zoomed_h = int(disp_h * view_zoom)
                vis_zoomed = cv2.resize(
                    vis_resized,
                    (zoomed_w, zoomed_h),
                    interpolation=cv2.INTER_LINEAR if view_zoom > 1.0 else cv2.INTER_AREA,
                )
                # パンオフセットを含めたクロップ領域を計算（符号反転でドラッグ方向と一致）
                cx = zoomed_w // 2 - int(view_pan[0] * view_zoom)
                cy = zoomed_h // 2 - int(view_pan[1] * view_zoom)
                x1 = cx - preview_w // 2
                y1 = cy - disp_h // 2
                x2 = x1 + preview_w
                y2 = y1 + disp_h

                # 出力キャンバスを作成
                vis_view = np.zeros((disp_h, preview_w, 3), dtype=np.uint8)
                vis_view[:] = (30, 30, 30)  # ダーク背景

                # ソースとデスティネーション領域を計算
                src_x1 = max(0, x1)
                src_y1 = max(0, y1)
                src_x2 = min(zoomed_w, x2)
                src_y2 = min(zoomed_h, y2)

                dst_x1 = src_x1 - x1
                dst_y1 = src_y1 - y1
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                dst_y2 = dst_y1 + (src_y2 - src_y1)

                if src_x2 > src_x1 and src_y2 > src_y1:
                    vis_view[dst_y1:dst_y2, dst_x1:dst_x2] = vis_zoomed[src_y1:src_y2, src_x1:src_x2]

                vis_resized = vis_view

            # ボタンエリアを追加（プレビュー幅）
            button_area = np.zeros((button_area_height, preview_w, 3), dtype=np.uint8)
            button_area[:] = (40, 40, 40)

            # モードボタン描画
            for btn in buttons:
                btn.draw(button_area, btn.mode == edit_mode)

            # 確定ボタン描画
            confirm_btn.draw(button_area, False)

            # プレビュー部分を結合
            preview_part = np.vstack([button_area, vis_resized])

            # 右パネルを作成
            panel_h = button_area_height + disp_h
            right_panel = np.zeros((panel_h, RIGHT_PANEL_WIDTH, 3), dtype=np.uint8)
            right_panel[:] = (50, 50, 50)  # パネル背景

            # ズームラベル
            cv2.putText(right_panel, "Zoom", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # ズームスライダー描画
            slider_track_x = RIGHT_PANEL_WIDTH // 2
            slider_local_top = slider_y_top - 0  # ウィンドウ座標からパネル座標に変換不要（同じ）
            slider_local_bottom = slider_y_bottom - 0
            # スライダートラック
            cv2.line(right_panel, (slider_track_x, slider_local_top), (slider_track_x, slider_local_bottom), (100, 100, 100), 2)
            # スライダーハンドル位置
            t = (4.0 - view_zoom) / 3.75  # view_zoom to position (0=top, 1=bottom)
            t = max(0.0, min(1.0, t))
            handle_y = int(slider_local_top + t * (slider_local_bottom - slider_local_top))
            cv2.circle(right_panel, (slider_track_x, handle_y), 8, (0, 200, 255), -1)
            cv2.circle(right_panel, (slider_track_x, handle_y), 8, (255, 255, 255), 1)

            # ズーム値表示
            zoom_text = f"{int(view_zoom * 100)}%"
            text_size = cv2.getTextSize(zoom_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            cv2.putText(right_panel, zoom_text, ((RIGHT_PANEL_WIDTH - text_size[0]) // 2, slider_local_bottom + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

            # パンボタン描画
            pan_local_y = pan_btn_y - 0
            pan_local_x = (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
            # ボタン背景
            if edit_mode == EditMode.VIEW_PAN:
                pan_color = (0, 180, 0)  # 緑（アクティブ）
            else:
                pan_color = (80, 80, 80)  # グレー
            cv2.rectangle(right_panel, (pan_local_x, pan_local_y),
                         (pan_local_x + PAN_BUTTON_SIZE, pan_local_y + PAN_BUTTON_SIZE), pan_color, -1)
            cv2.rectangle(right_panel, (pan_local_x, pan_local_y),
                         (pan_local_x + PAN_BUTTON_SIZE, pan_local_y + PAN_BUTTON_SIZE), (200, 200, 200), 1)
            # 十字アイコン（パンを示す）
            cx_btn = pan_local_x + PAN_BUTTON_SIZE // 2
            cy_btn = pan_local_y + PAN_BUTTON_SIZE // 2
            cv2.line(right_panel, (cx_btn - 8, cy_btn), (cx_btn + 8, cy_btn), (255, 255, 255), 2)
            cv2.line(right_panel, (cx_btn, cy_btn - 8), (cx_btn, cy_btn + 8), (255, 255, 255), 2)

            # Homeボタン描画
            home_local_x = (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
            home_local_y = home_btn_y
            cv2.rectangle(right_panel, (home_local_x, home_local_y),
                         (home_local_x + PAN_BUTTON_SIZE, home_local_y + PAN_BUTTON_SIZE), (80, 80, 80), -1)
            cv2.rectangle(right_panel, (home_local_x, home_local_y),
                         (home_local_x + PAN_BUTTON_SIZE, home_local_y + PAN_BUTTON_SIZE), (200, 200, 200), 1)
            # Hアイコン（中央寄せ）
            cv2.putText(right_panel, "H", (home_local_x + 9, home_local_y + 21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Undoボタン描画（履歴があれば明るく）
            undo_local_x = (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
            undo_local_y = undo_btn_y
            undo_color = (100, 100, 100) if len(history) > 0 else (60, 60, 60)
            cv2.rectangle(right_panel, (undo_local_x, undo_local_y),
                         (undo_local_x + PAN_BUTTON_SIZE, undo_local_y + PAN_BUTTON_SIZE), undo_color, -1)
            cv2.rectangle(right_panel, (undo_local_x, undo_local_y),
                         (undo_local_x + PAN_BUTTON_SIZE, undo_local_y + PAN_BUTTON_SIZE), (200, 200, 200), 1)
            # <アイコン（中央寄せ）
            undo_text_color = (255, 255, 255) if len(history) > 0 else (120, 120, 120)
            cv2.putText(right_panel, "<", (undo_local_x + 9, undo_local_y + 21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, undo_text_color, 1)

            # Redoボタン描画（redo_stackがあれば明るく）
            redo_local_x = (RIGHT_PANEL_WIDTH - PAN_BUTTON_SIZE) // 2
            redo_local_y = redo_btn_y
            redo_color = (100, 100, 100) if len(redo_stack) > 0 else (60, 60, 60)
            cv2.rectangle(right_panel, (redo_local_x, redo_local_y),
                         (redo_local_x + PAN_BUTTON_SIZE, redo_local_y + PAN_BUTTON_SIZE), redo_color, -1)
            cv2.rectangle(right_panel, (redo_local_x, redo_local_y),
                         (redo_local_x + PAN_BUTTON_SIZE, redo_local_y + PAN_BUTTON_SIZE), (200, 200, 200), 1)
            # >アイコン（中央寄せ）
            redo_text_color = (255, 255, 255) if len(redo_stack) > 0 else (120, 120, 120)
            cv2.putText(right_panel, ">", (redo_local_x + 9, redo_local_y + 21),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, redo_text_color, 1)

            # プレビューと右パネルを水平結合
            vis_disp = np.hstack([preview_part, right_panel])

            # 情報表示（ボタンエリアの下）
            info_y_base = button_area_height + 30
            cv2.putText(
                vis_disp,
                f"frame {frame_idx+1}/{N}  valid={int(valid[frame_idx])}  mode={edit_mode.name}",
                (10, info_y_base),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_disp,
                f"offset=({offset[0]:.1f}, {offset[1]:.1f}) scale=({scale_x:.3f}, {scale_y:.3f}) rot={rotation:.1f}deg",
                (10, info_y_base + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_disp,
                f"quad: {tw:.1f}x{th:.1f}  zoom: {int(view_zoom*100)}%  pan: ({view_pan[0]}, {view_pan[1]})",
                (10, info_y_base + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win, vis_disp)

            key = cv2.waitKeyEx(15)

            # 確定ボタンがクリックされたかチェック
            if confirmed:
                break

            if key in (27, ord("q")):
                print("[calib] cancelled")
                cv2.destroyAllWindows()
                return 1

            if key == 13:  # Enter のみで確定
                break

            # モード切替（大小文字両対応）
            if key in (ord("m"), ord("M")):
                edit_mode = EditMode.MOVE
            elif key in (ord("r"), ord("R")):
                edit_mode = EditMode.ROTATE
            elif key in (ord("s"), ord("S")):
                edit_mode = EditMode.SCALE_XY
            elif key in (ord("x"), ord("X")):
                edit_mode = EditMode.SCALE_X
            elif key in (ord("y"), ord("Y")):
                edit_mode = EditMode.SCALE_Y

            # Home キーでビューをリセット（Windows キーコード）
            HOME_KEY = 2359296
            if key == HOME_KEY:
                view_zoom = init_view_zoom
                view_pan = init_view_pan.copy()

            step = float(nudge)
            # 矢印キーによる移動（Undo対応）
            if key in (ARROW_LEFT, ARROW_RIGHT, ARROW_UP, ARROW_DOWN):
                save_state()
            if key == ARROW_LEFT:
                offset[0] -= step
            elif key == ARROW_RIGHT:
                offset[0] += step
            elif key == ARROW_UP:
                offset[1] -= step
            elif key == ARROW_DOWN:
                offset[1] += step
            elif key in (ord("+"), ord("="), ord("-"), ord("_")):
                # スケール変更（Undo対応）
                save_state()
            if key in (ord("+"), ord("=")):
                if edit_mode == EditMode.SCALE_X:
                    scale_x *= 1.02
                elif edit_mode == EditMode.SCALE_Y:
                    scale_y *= 1.02
                else:
                    scale_x *= 1.02
                    scale_y *= 1.02
            if key in (ord("-"), ord("_")):
                if edit_mode == EditMode.SCALE_X:
                    scale_x /= 1.02
                elif edit_mode == EditMode.SCALE_Y:
                    scale_y /= 1.02
                else:
                    scale_x /= 1.02
                    scale_y /= 1.02
            # 回転キー（Undo対応）
            if key in (ord(","), ord("<"), ord("."), ord(">")):
                save_state()
            if key in (ord(","), ord("<")):  # < で左回転
                rotation -= 1.0
            if key in (ord("."), ord(">")):  # > で右回転
                rotation += 1.0
            if key == ord("["):
                set_frame(frame_idx - 1)
            if key == ord("]"):
                set_frame(frame_idx + 1)
            if key == 8:  # Backspace で変換値リセット（Undo対応）
                save_state()
                offset = init_offset.copy()
                scale_x = float(init_scale_x)
                scale_y = float(init_scale_y)
                rotation = float(init_rotation)

        cv2.destroyAllWindows()

    finally:
        cap.release()

    print(f"[info] applying transform to {N} frames...")
    # ベクトル化（高速）- 縦横独立スケール対応
    quads_center = quads.mean(axis=1, keepdims=True)  # (N,1,2)
    scale_arr = np.array([scale_x, scale_y], dtype=np.float32).reshape(1, 1, 2)
    rel = (quads - quads_center) * scale_arr
    th = math.radians(float(rotation))
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    rel = rel @ R.T
    calibrated_quads = (rel + quads_center + offset.reshape(1, 1, 2)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_dict = {k: track[k] for k in track.files if k != "quad"}
    save_dict["quad"] = calibrated_quads
    save_dict["calib_offset"] = offset.astype(np.float32)
    # 新形式で保存
    save_dict["calib_scale_x"] = float(scale_x)
    save_dict["calib_scale_y"] = float(scale_y)
    # 後方互換用に平均値も保存
    save_dict["calib_scale"] = float((scale_x + scale_y) / 2)
    save_dict["calib_rotation"] = float(rotation)
    np.savez_compressed(args.out, **save_dict)

    print(f"[saved] {args.out}")
    print(f"  offset: ({offset[0]:.2f}, {offset[1]:.2f})")
    print(f"  scale: ({scale_x:.4f}, {scale_y:.4f})")
    print(f"  rotation: {rotation:.2f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
