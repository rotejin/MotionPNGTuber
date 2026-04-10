"""Lightweight pad / erase-range preview logic for mouth_track_gui.

OpenCV/NumPy only — no tkinter dependency.
The goal is to let users compare pad / coverage quickly before running the
heavier mouthless-video generation step.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable

import cv2  # type: ignore
import numpy as np  # type: ignore

from motionpngtuber.image_io import read_image_bgra
from motionpngtuber.mouth_color_adjust import (
    MouthColorAdjust,
    apply_inspect_boost_3ch,
    apply_mouth_color_adjust_4ch,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackData:
    """Loaded and scaled quad track data."""

    quads: np.ndarray       # (N, 4, 2) float32
    valid: np.ndarray       # (N,) bool
    n_frames: int


@dataclass(frozen=True)
class MaskParams:
    """Parameters for mask generation from coverage."""

    mask_scale_x: float
    mask_scale_y: float
    ring_px: int
    dilate_px: int
    feather_px: int
    top_clip_frac: float
    center_y_off: int


@dataclass(frozen=True)
class PreviewSelection:
    """Selection returned from the lightweight adjustment preview."""

    applied: bool
    pad: float
    coverage: float


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def load_and_scale_quads(
    track_path: str,
    vid_w: int,
    vid_h: int,
) -> TrackData:
    """Load track npz, validate shape, and scale quads to video size.

    Raises ValueError on invalid data.
    """
    npz = np.load(track_path, allow_pickle=False)
    if "quad" not in npz:
        raise ValueError("track npz に 'quad' がありません。")
    quads = np.asarray(npz["quad"], dtype=np.float32)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError("quad の形が不正です（(N,4,2) が必要）。")

    N = int(quads.shape[0])
    valid = (
        np.asarray(npz["valid"], dtype=bool)
        if "valid" in npz
        else np.ones((N,), dtype=bool)
    )

    # Scale to current video size
    src_w = int(npz["w"]) if "w" in npz else vid_w
    src_h = int(npz["h"]) if "h" in npz else vid_h
    sx = float(vid_w) / float(max(1, src_w))
    sy = float(vid_h) / float(max(1, src_h))
    quads = quads.copy()
    quads[..., 0] *= sx
    quads[..., 1] *= sy

    return TrackData(quads=quads, valid=valid, n_frames=N)


def fill_invalid_quads(quads: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Hold-fill invalid frames (same as erase_mouth_offline.py default).

    Returns filled quads, or None if all frames are invalid.
    """
    N = quads.shape[0]
    idxs = np.where(valid)[0]
    if len(idxs) == 0:
        return None

    filled = quads.copy()
    last = int(idxs[0])
    for i in range(N):
        if valid[i]:
            last = i
        else:
            filled[i] = filled[last]
    first = int(idxs[0])
    for i in range(first):
        filled[i] = filled[first]

    return filled


# ---------------------------------------------------------------------------
# Patch / mask computation
# ---------------------------------------------------------------------------

def _ensure_even_ge2(n: int) -> int:
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n - 1)


def scale_quad_about_center(quad: np.ndarray, scale: float) -> np.ndarray:
    """Scale a quad around its center while preserving rotation/orientation."""
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    center = q.mean(axis=0, keepdims=True)
    rel = q - center
    return (rel * float(scale) + center).astype(np.float32)


def build_pad_preview_values(
    current_pad: float,
    *,
    step: float = 0.2,
    min_pad: float = 1.2,
    max_pad: float = 3.2,
) -> tuple[float, float, float]:
    """Build three nearby pad candidates for side-by-side comparison.

    The default current pad 2.1 yields (1.9, 2.1, 2.3).
    Near bounds, additional candidates are pulled from the wider neighborhood
    so three unique values are still returned.
    """
    raw = [
        current_pad - step,
        current_pad,
        current_pad + step,
        current_pad - 2.0 * step,
        current_pad + 2.0 * step,
        min_pad,
        max_pad,
    ]
    vals: list[float] = []
    for v in raw:
        vv = round(float(np.clip(v, min_pad, max_pad)), 2)
        if vv not in vals:
            vals.append(vv)
        if len(vals) >= 3:
            break
    while len(vals) < 3:
        vals.append(round(float(np.clip(current_pad, min_pad, max_pad)), 2))
    vals = sorted(vals[:3])
    return float(vals[0]), float(vals[1]), float(vals[2])


def compute_norm_patch_size(
    filled_quads: np.ndarray,
    n_out: int,
) -> tuple[int, int]:
    """Compute normalized patch size (norm_w, norm_h) from filled quads."""
    qsz = filled_quads[:n_out]
    ws = np.linalg.norm(qsz[:, 1, :] - qsz[:, 0, :], axis=1)
    hs = np.linalg.norm(qsz[:, 3, :] - qsz[:, 0, :], axis=1)
    ratio = float(np.median(ws / np.maximum(1e-6, hs)))
    p95w = float(np.percentile(ws, 95))
    oversample = 1.2
    norm_w = _ensure_even_ge2(max(96, int(round(p95w * oversample))))
    ratio_c = max(0.25, min(4.0, ratio))
    norm_h = _ensure_even_ge2(max(64, int(round(norm_w / ratio_c))))
    return norm_w, norm_h


def warp_sprite_to_quad(
    src_sprite_4ch: np.ndarray,
    dst_quad: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Warp a 4-channel sprite in OpenCV-native order into a destination quad."""
    src = np.asarray(src_sprite_4ch)
    if src.ndim != 3 or src.shape[2] not in (3, 4):
        raise ValueError("sprite image must have 3 or 4 channels")
    if src.shape[2] == 3:
        alpha = np.full(src.shape[:2] + (1,), 255, dtype=np.uint8)
        src = np.concatenate([src, alpha], axis=2)

    sh, sw = src.shape[:2]
    src_quad = np.array(
        [[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]],
        dtype=np.float32,
    )
    dst = np.asarray(dst_quad, dtype=np.float32).reshape(4, 2)
    M = cv2.getPerspectiveTransform(src_quad, dst)
    return cv2.warpPerspective(
        src,
        M,
        (int(out_w), int(out_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def warp_rgba_to_quad(
    src_rgba: np.ndarray,
    quad_xy: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Backward-compatible alias kept for tests and existing imports."""
    return warp_sprite_to_quad(src_rgba, quad_xy, out_w, out_h)


def alpha_blend_sprite_over_bgr(
    base_bgr: np.ndarray,
    overlay_sprite_4ch: np.ndarray,
    *,
    opacity: float = 0.75,
) -> np.ndarray:
    """Alpha-blend a 4-channel OpenCV-native sprite over a BGR image."""
    base = np.asarray(base_bgr, dtype=np.float32)
    over = np.asarray(overlay_sprite_4ch)
    if over.ndim != 3 or over.shape[2] < 4:
        return base_bgr.copy()
    alpha = (over[..., 3:4].astype(np.float32) / 255.0) * float(np.clip(opacity, 0.0, 1.0))
    bgr = over[..., :3].astype(np.float32)
    out = base * (1.0 - alpha) + bgr * alpha
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def alpha_blend_rgba_over_bgr(
    base_bgr: np.ndarray,
    overlay_rgba: np.ndarray,
    *,
    opacity: float = 0.75,
) -> np.ndarray:
    """Backward-compatible alias kept for tests and existing imports."""
    return alpha_blend_sprite_over_bgr(base_bgr, overlay_rgba, opacity=opacity)


def resize_for_preview(
    img: np.ndarray,
    *,
    max_w: int = 1800,
    max_h: int = 980,
) -> np.ndarray:
    """Shrink large preview images to fit common displays."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return img
    dst_w = max(2, int(round(w * scale)))
    dst_h = max(2, int(round(h * scale)))
    return cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_AREA)


def build_pad_button_rects(
    panel_width: int,
    panel_count: int,
    *,
    top_y: int = 92,
    button_w: int = 220,
    button_h: int = 40,
) -> list[tuple[int, int, int, int]]:
    """Build one clickable button rect for each preview panel."""
    rects: list[tuple[int, int, int, int]] = []
    bw = max(80, min(int(button_w), max(80, int(panel_width) - 40)))
    bh = max(28, int(button_h))
    for i in range(max(0, int(panel_count))):
        panel_left = i * int(panel_width)
        cx = panel_left + int(panel_width) // 2
        x0 = max(panel_left + 10, cx - bw // 2)
        x1 = min(panel_left + int(panel_width) - 10, x0 + bw)
        x0 = x1 - bw
        rects.append((x0, int(top_y), x1, int(top_y) + bh))
    return rects


def rect_contains(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
    """Return True when a point lies inside a rectangle."""
    x0, y0, x1, y1 = rect
    return int(x0) <= int(x) <= int(x1) and int(y0) <= int(y) <= int(y1)


def compute_mask_params(coverage: float, norm_h: int) -> MaskParams:
    """Compute mask parameters from coverage value (same tuning as erase_mouth_offline.py)."""
    cov = float(np.clip(coverage, 0.0, 1.0))
    return MaskParams(
        mask_scale_x=0.50 + 0.18 * cov,
        mask_scale_y=0.44 + 0.14 * cov,
        ring_px=int(round(16 + 10 * cov)),
        dilate_px=int(round(8 + 8 * cov)),
        feather_px=int(round(18 + 10 * cov)),
        top_clip_frac=float(0.84 - 0.06 * cov),
        center_y_off=int(round(norm_h * (0.05 + 0.01 * cov))),
    )


def make_mouth_mask(
    w: int,
    h: int,
    rx: int,
    ry: int,
    *,
    center_y_offset_px: int = 0,
    top_clip_frac: float = 0.82,
) -> np.ndarray:
    """Create a mouth mask matching erase_mouth_offline.py geometry."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy0 = w // 2, h // 2
    cy = int(np.clip(cy0 + int(center_y_offset_px), 0, h - 1))
    rx2 = int(max(1, min(int(rx), w // 2 - 1)))
    ry2 = int(max(1, min(int(ry), h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx2, ry2), 0.0, 0.0, 360.0, 255, -1)
    top_clip_frac = float(np.clip(top_clip_frac, 0.6, 1.0))
    clip_y = int(round(cy - ry2 * top_clip_frac))
    clip_y = int(np.clip(clip_y, 0, h))
    if clip_y > 0:
        mask[:clip_y, :] = 0
    return mask


def feather_mask(
    mask_u8: np.ndarray,
    dilate_px: int,
    feather_px: int,
) -> np.ndarray:
    """Dilate and feather a uint8 mask, returning a float32 [0,1] mask."""
    m = mask_u8.copy()
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)
    if feather_px > 0:
        k = 2 * int(feather_px) + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


def build_preview_masks(
    norm_w: int,
    norm_h: int,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build inner (feathered) and ring masks in normalized space.

    Returns (inner_f, ring_f) as float32 [0,1] arrays of shape (norm_h, norm_w).
    """
    params = compute_mask_params(coverage, norm_h)
    rx = int((norm_w * params.mask_scale_x) * 0.5)
    ry = int((norm_h * params.mask_scale_y) * 0.5)

    inner_u8 = make_mouth_mask(
        norm_w, norm_h, rx=rx, ry=ry,
        center_y_offset_px=params.center_y_off,
        top_clip_frac=params.top_clip_frac,
    )
    outer_u8 = make_mouth_mask(
        norm_w, norm_h, rx=rx + params.ring_px, ry=ry + params.ring_px,
        center_y_offset_px=params.center_y_off,
        top_clip_frac=params.top_clip_frac,
    )
    ring_u8 = cv2.subtract(outer_u8, inner_u8)

    inner_f = feather_mask(inner_u8, dilate_px=params.dilate_px, feather_px=params.feather_px)
    ring_f = (ring_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    return inner_f, ring_f


# ---------------------------------------------------------------------------
# Interactive preview
# ---------------------------------------------------------------------------

def _read_preview_frame(
    cap: cv2.VideoCapture,
    idx: int,
    *,
    cached_idx: int | None,
    cached_frame: np.ndarray | None,
    next_read_idx: int | None,
) -> tuple[bool, np.ndarray | None, int | None, np.ndarray | None, int | None]:
    """Read preview frame efficiently with cache + sequential-read fast path.

    Strategy:
    - If the requested frame is already cached, reuse it without touching
      the decoder.
    - If the request is the expected next sequential frame, read directly
      without a seek.
    - Otherwise, seek to the requested frame and read once.
    """
    req = int(max(0, idx))

    if cached_frame is not None and cached_idx == req:
        return True, cached_frame, cached_idx, cached_frame, next_read_idx

    if next_read_idx != req:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(req))

    ok, frame = cap.read()
    if not ok or frame is None:
        return False, None, cached_idx, cached_frame, next_read_idx

    cached = frame.copy()
    return True, frame, req, cached, req + 1

def run_erase_range_preview(
    *,
    video: str,
    track_path: str,
    track_npz: str,
    calib_npz: str,
    coverage: float,
    preview_pad: float,
    open_sprite: str | None,
    color_adjust: MouthColorAdjust | None,
    stop_flag: threading.Event,
    log_fn: Callable[[str], None],
    show_error: Callable[[str, str], None],
) -> PreviewSelection:
    """Run the interactive lightweight preview window.

    Shows three nearby pad candidates side-by-side, plus the erase-range
    overlay for the current coverage value. Users can apply the chosen pad /
    coverage back to the GUI without regenerating the mouthless video.
    """
    cov = float(np.clip(coverage, 0.40, 0.90))
    pad_values = build_pad_preview_values(preview_pad)
    preview_pad_safe = max(1e-6, float(preview_pad))
    selected_idx = int(np.argmin([abs(p - float(preview_pad)) for p in pad_values]))
    selection = PreviewSelection(applied=False, pad=float(pad_values[selected_idx]), coverage=cov)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        show_error("エラー", f"動画を開けません: {video}")
        return selection

    try:
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if vid_w <= 0 or vid_h <= 0:
            show_error("エラー", "動画サイズが取得できませんでした。")
            return selection

        track_label = (
            "mouth_track_calibrated.npz"
            if os.path.abspath(track_path) == os.path.abspath(calib_npz)
            else "mouth_track.npz"
        )
        log_fn(f"[info] 軽量見た目確認に使用する track: {track_path}")

        sprite_bgra: np.ndarray | None = None
        if open_sprite and os.path.isfile(open_sprite):
            try:
                sprite_bgra = read_image_bgra(open_sprite)
                if sprite_bgra is not None and color_adjust is not None:
                    sprite_bgra = apply_mouth_color_adjust_4ch(
                        sprite_bgra,
                        color_adjust,
                        color_order="BGRA",
                    )
                if sprite_bgra is not None:
                    log_fn(f"[info] 口PNG重ね表示に使用する open.png: {open_sprite}")
            except Exception as e:
                sprite_bgra = None
                log_fn(f"[warn] open.png の読み込みに失敗したため、口PNG重ね表示なしで続行します: {e}")

        # ---- load track ----
        try:
            track_data = load_and_scale_quads(track_path, vid_w, vid_h)
        except ValueError as e:
            show_error("エラー", str(e))
            return selection

        filled = fill_invalid_quads(track_data.quads, track_data.valid)
        if filled is None:
            show_error("エラー", "track が全フレーム invalid のようです。")
            return selection

        N = track_data.n_frames
        n_out = min(total_frames if total_frames > 0 else N, N)

        # ---- normalized patch size ----
        norm_w, norm_h = compute_norm_patch_size(filled, n_out)

        # ---- build masks ----
        inner_f, ring_f = build_preview_masks(norm_w, norm_h, cov)

        # ---- interactive preview loop ----
        win = "look preview (q/ESC=close, Enter=apply, 1/2/3=pad, r/f=cov, space=play, a/d=step, [ ]=+/-10)"
        paused = True
        idx = 0
        cached_idx: int | None = None
        cached_frame: np.ndarray | None = None
        next_read_idx: int | None = None

        src_pts = np.array(
            [[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]],
            dtype=np.float32,
        )

        red = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        red[:, :, 2] = 255
        yellow = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
        yellow[:, :, 1] = 255
        yellow[:, :, 2] = 255
        selected_color = (0, 255, 0)
        other_color = (255, 200, 0)
        invalid_color = (0, 0, 255)
        applied = False
        inspect_levels = (1.0, 2.0, 3.0, 4.0)
        inspect_boost = 1.0
        if color_adjust is not None:
            inspect_boost = min(inspect_levels, key=lambda x: abs(x - float(color_adjust.inspect_boost)))
        ui_state: dict[str, object] = {
            "selected_idx": selected_idx,
            "pad_button_rects": [],
            "apply_rect": None,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "request_apply": False,
        }

        def _on_mouse(event, x, y, _flags, _param) -> None:
            if event != cv2.EVENT_LBUTTONUP:
                return
            scale_x = float(ui_state.get("scale_x", 1.0) or 1.0)
            scale_y = float(ui_state.get("scale_y", 1.0) or 1.0)
            canvas_x = int(round(float(x) * scale_x))
            canvas_y = int(round(float(y) * scale_y))
            for i, rect in enumerate(ui_state.get("pad_button_rects", [])):
                if rect_contains(rect, canvas_x, canvas_y):
                    ui_state["selected_idx"] = int(i)
                    return
            apply_rect = ui_state.get("apply_rect")
            if isinstance(apply_rect, tuple) and rect_contains(apply_rect, canvas_x, canvas_y):
                ui_state["request_apply"] = True

        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, _on_mouse)

        while True:
            if stop_flag.is_set():
                break
            selected_idx = int(ui_state.get("selected_idx", selected_idx))
            ok, frame, cached_idx, cached_frame, next_read_idx = _read_preview_frame(
                cap,
                idx,
                cached_idx=cached_idx,
                cached_frame=cached_frame,
                next_read_idx=next_read_idx,
            )
            if not ok or frame is None:
                break

            q = filled[idx].astype(np.float32).reshape(4, 2)
            panels: list[np.ndarray] = []
            for pad_idx, pad_value in enumerate(pad_values):
                q_pad = scale_quad_about_center(q, float(pad_value) / preview_pad_safe)
                M = cv2.getPerspectiveTransform(src_pts, q_pad)
                m_inner = cv2.warpPerspective(
                    inner_f,
                    M,
                    (vid_w, vid_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                m_ring = cv2.warpPerspective(
                    ring_f,
                    M,
                    (vid_w, vid_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )

                out = frame.copy()
                if sprite_bgra is not None:
                    warped_sprite = warp_sprite_to_quad(sprite_bgra, q_pad, vid_w, vid_h)
                    out = alpha_blend_sprite_over_bgr(out, warped_sprite, opacity=0.78)

                a_inner = 0.40
                a_ring = 0.22
                out = (
                    out.astype(np.float32) * (1.0 - a_inner * m_inner[..., None])
                    + red.astype(np.float32) * (a_inner * m_inner[..., None])
                ).astype(np.uint8)
                out = (
                    out.astype(np.float32) * (1.0 - a_ring * m_ring[..., None])
                    + yellow.astype(np.float32) * (a_ring * m_ring[..., None])
                ).astype(np.uint8)
                out = apply_inspect_boost_3ch(out, inspect_boost, color_order="BGR")

                panel_color = selected_color if pad_idx == selected_idx else other_color
                thickness = 3 if pad_idx == selected_idx else 2
                pts = q_pad.reshape(-1, 1, 2).astype(np.int32)
                cv2.polylines(out, [pts], True, panel_color, thickness, cv2.LINE_AA)

                panel_info = f"{pad_idx + 1}: pad={pad_value:.2f}"
                if pad_idx == selected_idx:
                    panel_info += " [selected]"
                cv2.putText(
                    out,
                    panel_info,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    out,
                    panel_info,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    panel_color,
                    1,
                    cv2.LINE_AA,
                )
                if not bool(track_data.valid[idx]):
                    cv2.putText(
                        out,
                        "INVALID (filled)",
                        (10, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        invalid_color,
                        2,
                        cv2.LINE_AA,
                    )
                panels.append(out)

            strip = np.concatenate(panels, axis=1)
            canvas = cv2.copyMakeBorder(
                strip,
                146,
                0,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(18, 18, 18),
            )
            info1 = (
                f"frame {idx+1}/{n_out}  cov={cov:.2f}  track={track_label}  "
                f"red=erase  yellow=ring  inspect={inspect_boost:.1f}"
            )
            info2 = (
                "click top buttons or 1/2/3=pad  r/f=coverage v=inspect  Enter=apply  "
                "space=play/pause  a/d=+/-1  [ ]=+/-10"
            )
            info3 = "pad preview is a lightweight approximation based on current track."
            info4 = ""
            if color_adjust is not None:
                info4 = (
                    f"mouth adj  bri={color_adjust.brightness:.0f} sat={color_adjust.saturation:.2f} "
                    f"warm={color_adjust.warmth:.0f} edge={color_adjust.edge_priority:.2f}"
                )
            info_rows = [(24, info1), (52, info2), (80, info3)]
            if info4:
                info_rows.append((108, info4))
            for y, text in info_rows:
                cv2.putText(
                    canvas,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70 if y < 80 else 0.62,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    text,
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70 if y < 80 else 0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            button_labels = ["Tight", "Std", "Wide"]
            pad_button_rects = build_pad_button_rects(vid_w, len(pad_values))
            ui_state["pad_button_rects"] = pad_button_rects
            for pad_idx, rect in enumerate(pad_button_rects):
                x0, y0, x1, y1 = rect
                is_selected = pad_idx == selected_idx
                fill = (60, 110, 60) if is_selected else (70, 70, 70)
                stroke = selected_color if is_selected else other_color
                cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (x0, y0), (x1, y1), stroke, 2, cv2.LINE_AA)
                label = f"{button_labels[min(pad_idx, 2)]}  pad={pad_values[pad_idx]:.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (x0 + 12, y0 + 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            apply_rect = (max(20, canvas.shape[1] - 210), 92, max(20, canvas.shape[1] - 20), 132)
            ui_state["apply_rect"] = apply_rect
            cv2.rectangle(canvas, (apply_rect[0], apply_rect[1]), (apply_rect[2], apply_rect[3]), (80, 55, 20), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (apply_rect[0], apply_rect[1]), (apply_rect[2], apply_rect[3]), (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "Apply to GUI",
                (apply_rect[0] + 28, apply_rect[1] + 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            display = resize_for_preview(canvas)
            ui_state["scale_x"] = float(canvas.shape[1]) / float(max(1, display.shape[1]))
            ui_state["scale_y"] = float(canvas.shape[0]) / float(max(1, display.shape[0]))
            cv2.imshow(win, display)

            delay = max(1, int(round(1000.0 / max(1.0, fps)))) if not paused else 15
            k = cv2.waitKey(delay)
            k8 = k & 0xFF

            selected_idx = int(ui_state.get("selected_idx", selected_idx))
            if bool(ui_state.get("request_apply")):
                applied = True
                ui_state["request_apply"] = False
                break
            if k8 in (ord("q"), 27):
                break
            if k in (10, 13):
                applied = True
                break
            if k8 == ord(" "):
                paused = not paused
                continue
            if k8 in (ord("1"), ord("2"), ord("3")):
                selected_idx = min(len(pad_values) - 1, max(0, k8 - ord("1")))
                paused = True
                continue
            if k8 == ord("r"):
                cov = min(0.90, round(cov + 0.02, 2))
                inner_f, ring_f = build_preview_masks(norm_w, norm_h, cov)
                paused = True
                continue
            if k8 == ord("f"):
                cov = max(0.40, round(cov - 0.02, 2))
                inner_f, ring_f = build_preview_masks(norm_w, norm_h, cov)
                paused = True
                continue
            if k8 == ord("v"):
                cur_idx = inspect_levels.index(inspect_boost) if inspect_boost in inspect_levels else 0
                inspect_boost = inspect_levels[(cur_idx + 1) % len(inspect_levels)]
                paused = True
                continue
            if k8 == ord("a"):
                idx = max(0, idx - 1)
                paused = True
                continue
            if k8 == ord("d"):
                idx = min(n_out - 1, idx + 1)
                paused = True
                continue
            if k8 == ord("["):
                idx = max(0, idx - 10)
                paused = True
                continue
            if k8 == ord("]"):
                idx = min(n_out - 1, idx + 10)
                paused = True
                continue

            if not paused:
                idx += 1
                if idx >= n_out:
                    break

        selection = PreviewSelection(
            applied=applied,
            pad=float(pad_values[selected_idx]),
            coverage=float(cov),
        )
        cv2.destroyWindow(win)
        return selection

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
