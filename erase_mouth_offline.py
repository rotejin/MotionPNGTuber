#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""erase_mouth_offline.py

Level-3 "mouth eraser" (offline, high quality).

Goal
----
Generate a "mouthless" base video by removing the mouth region using the tracked mouth quad.
The key is temporal stability:

1) Warp each frame's mouth patch to a fixed normalized canvas (homography).
2) Build ONE clean-plate patch (texture) in that normalized space (inpaint on a reference frame).
3) For each frame, estimate shading from a ring region around the mouth (also in normalized space)
   and apply it to the clean-plate, so shadows/lighting follow the original frame.
4) Warp the shaded clean-plate back to the original frame and blend with a feathered mask.

This avoids "skin color mismatch" when the face has shading.

Inputs
------
- --video: input video (loop mp4 etc.)
- --track: mouth_track(_calibrated).npz containing quad (N,4,2) and optional valid/confidence/w/h

Outputs
-------
- --out: mouthless video. If ffmpeg is available and --keep-audio is on, audio is kept.

Notes
-----
- Requires: numpy, opencv-python
- Optional: ffmpeg in PATH (for audio mux)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from image_io import write_image_file


def ensure_even_ge2(n: int) -> int:
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n - 1)


def quad_wh(quad: np.ndarray) -> Tuple[float, float]:
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = float(np.linalg.norm(quad[1] - quad[0]))
    h = float(np.linalg.norm(quad[3] - quad[0]))
    return w, h


def quad_bbox(quad: np.ndarray, pad_px: int = 0) -> Tuple[int, int, int, int]:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0 = int(np.floor(q[:, 0].min())) - int(pad_px)
    y0 = int(np.floor(q[:, 1].min())) - int(pad_px)
    x1 = int(np.ceil(q[:, 0].max())) + int(pad_px)
    y1 = int(np.ceil(q[:, 1].max())) + int(pad_px)
    return x0, y0, x1, y1


def make_ellipse_mask(w: int, h: int, rx: int, ry: int) -> np.ndarray:
    """Return uint8 mask (0/255) of a filled ellipse centered in the canvas."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return mask



def make_mouth_mask(
    w: int,
    h: int,
    rx: int,
    ry: int,
    *,
    center_y_offset_px: int = 0,
    top_clip_frac: float = 0.82,
) -> np.ndarray:
    """Mouth-eraser mask specialized for anime faces.

    Compared to a plain centered ellipse, this mask:
    - shifts slightly downward (away from the nose)
    - clips the top part of the ellipse to avoid erasing the nose/philtrum

    top_clip_frac controls how much of the top is clipped:
    - 1.0  : almost no clip (keeps most of the ellipse)
    - 0.75 : clips more (safer for noses, but may expose upper lip)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy0 = w // 2, h // 2
    cy = int(np.clip(cy0 + int(center_y_offset_px), 0, h - 1))
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)

    # Clip the top portion to protect the nose area.
    # Keep the lower region from (cy - ry*top_clip_frac) downward.
    top_clip_frac = float(np.clip(top_clip_frac, 0.6, 1.0))
    clip_y = int(round(cy - ry * top_clip_frac))
    clip_y = int(np.clip(clip_y, 0, h))
    if clip_y > 0:
        mask[:clip_y, :] = 0
    return mask

def feather_mask(mask_u8: np.ndarray, dilate_px: int, feather_px: int) -> np.ndarray:
    """mask_u8(0/255) -> float32 mask (0..1) with dilation + gaussian feather."""
    m = mask_u8.copy()
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        m = cv2.dilate(m, kernel, iterations=1)
    if feather_px > 0:
        k = 2 * int(feather_px) + 1
        m = cv2.GaussianBlur(m, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


def warp_frame_to_norm(frame_bgr: np.ndarray, quad: np.ndarray, norm_w: int, norm_h: int) -> np.ndarray:
    """Extract mouth patch in a fixed normalized space."""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(
        frame_bgr,
        M,
        (int(norm_w), int(norm_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


def warp_norm_to_bbox(patch_bgr: np.ndarray, mask_f: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Warp normalized patch+mask to the quad's bbox area.

    Returns:
      warped_patch_bgr (bh,bw,3), warped_mask_f (bh,bw,1), x0, y0
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0, y0, x1, y1 = quad_bbox(q, pad_px=1)
    bw = max(2, x1 - x0)
    bh = max(2, y1 - y0)

    dst_pts = (q - np.array([x0, y0], dtype=np.float32))
    h, w = patch_bgr.shape[:2]
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    warped_patch = cv2.warpPerspective(
        patch_bgr,
        M,
        dsize=(bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # mask_f is (H,W) float in 0..1. Warp as float then expand to (bh,bw,1)
    warped_mask = cv2.warpPerspective(
        mask_f,
        M,
        dsize=(bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_mask = warped_mask.astype(np.float32).clip(0.0, 1.0)[:, :, None]
    return warped_patch, warped_mask, x0, y0


def alpha_blend_roi(dst_bgr: np.ndarray, src_bgr: np.ndarray, mask_f_1: np.ndarray) -> np.ndarray:
    """Blend src into dst using mask (H,W,1) float 0..1."""
    a = mask_f_1.astype(np.float32)
    out = dst_bgr.astype(np.float32) * (1.0 - a) + src_bgr.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def fit_plane_2d(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit plane v = ax + by + c in normalized coords x,y in [-1,1].

    Returns coeffs [a,b,c].
    """
    # Normalize coordinates for numerical stability
    xn = (xs.astype(np.float32) - (w * 0.5)) / (w * 0.5)
    yn = (ys.astype(np.float32) - (h * 0.5)) / (h * 0.5)
    A = np.stack([xn, yn, np.ones_like(xn)], axis=1)
    b = values.astype(np.float32)
    # Least squares
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs.astype(np.float32)


def eval_plane(coeffs: np.ndarray, w: int, h: int) -> np.ndarray:
    """Evaluate plane ax+by+c on a grid (h,w), with x,y in [-1,1]."""
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    xs = (np.arange(w, dtype=np.float32) - (w * 0.5)) / (w * 0.5)
    ys = (np.arange(h, dtype=np.float32) - (h * 0.5)) / (h * 0.5)
    X, Y = np.meshgrid(xs, ys)
    return (a * X + b * Y + c).astype(np.float32)


class PlaneFitter:
    """Precomputed plane fitter for ring pixels in normalized mouth space.

    This avoids rebuilding the least-squares matrix and meshgrid on every frame.
    The math is equivalent to fit_plane_2d/eval_plane with fixed (xs,ys,w,h).
    """

    def __init__(self, ring_xs: np.ndarray, ring_ys: np.ndarray, w: int, h: int):
        self.w = int(w)
        self.h = int(h)

        xs = ring_xs.astype(np.float32)
        ys = ring_ys.astype(np.float32)

        # Normalize coordinates to [-1, 1] (same as fit_plane_2d/eval_plane)
        xn = (xs - (self.w * 0.5)) / (self.w * 0.5)
        yn = (ys - (self.h * 0.5)) / (self.h * 0.5)

        A = np.stack([xn, yn, np.ones_like(xn)], axis=1).astype(np.float32)  # (K,3)
        # pinv(A): (3,K). Use float64 for stability, cast back to float32.
        self.pinvA = np.linalg.pinv(A.astype(np.float64)).astype(np.float32)

        # Precompute plane grid basis (X,Y) once (same as eval_plane).
        gx = (np.arange(self.w, dtype=np.float32) - (self.w * 0.5)) / (self.w * 0.5)
        gy = (np.arange(self.h, dtype=np.float32) - (self.h * 0.5)) / (self.h * 0.5)
        self.X, self.Y = np.meshgrid(gx, gy)

    @classmethod
    def from_ring(cls, ring_xs: np.ndarray, ring_ys: np.ndarray, w: int, h: int) -> "PlaneFitter":
        return cls(ring_xs=ring_xs, ring_ys=ring_ys, w=w, h=h)

    def fit(self, values: np.ndarray) -> np.ndarray:
        """Fit plane coeffs [a,b,c] for ring values."""
        b = values.astype(np.float32).reshape(-1)
        return (self.pinvA @ b).astype(np.float32)

    def eval(self, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate plane ax+by+c on the full normalized grid (h,w)."""
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
        return (a * self.X + b * self.Y + c).astype(np.float32)


@dataclass
class Track:
    quads: np.ndarray        # (N,4,2) float32, scaled to target size
    valid: np.ndarray        # (N,) bool
    filled: np.ndarray       # (N,4,2) float32
    confidence: Optional[np.ndarray]
    total: int


def load_track(path: str, target_w: int, target_h: int, policy: str = "hold") -> Track:
    npz = np.load(path, allow_pickle=False)
    if "quad" not in npz:
        raise ValueError("track must contain 'quad' (N,4,2)")
    quads = np.asarray(npz["quad"], dtype=np.float32)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError("quad must be (N,4,2)")
    N = int(quads.shape[0])
    valid = np.asarray(npz["valid"], dtype=np.uint8).astype(bool) if "valid" in npz else np.ones((N,), bool)
    if valid.shape[0] != N:
        valid = np.ones((N,), bool)

    src_w = int(npz["w"]) if "w" in npz else target_w
    src_h = int(npz["h"]) if "h" in npz else target_h
    sx = float(target_w) / float(max(1, src_w))
    sy = float(target_h) / float(max(1, src_h))
    quads = quads.copy()
    quads[..., 0] *= sx
    quads[..., 1] *= sy

    conf = np.asarray(npz["confidence"], dtype=np.float32) if "confidence" in npz else None
    if conf is not None and conf.shape[0] != N:
        conf = None

    if policy == "strict":
        filled = quads.copy()
    else:
        # hold fill
        filled = quads.copy()
        idxs = np.where(valid)[0]
        if len(idxs) > 0:
            last = int(idxs[0])
            for i in range(N):
                if valid[i]:
                    last = i
                else:
                    filled[i] = filled[last]
            first = int(idxs[0])
            for i in range(first):
                filled[i] = filled[first]
    return Track(quads=quads, valid=valid, filled=filled, confidence=conf, total=N)


def choose_ref_frame(track: Track) -> int:
    """Pick a stable reference frame for clean-plate generation."""
    if track.total <= 0:
        return 0
    if track.confidence is not None:
        # Prefer highest confidence among valid frames
        mask = track.valid
        if mask.any():
            idx = int(np.argmax(track.confidence * mask.astype(np.float32)))
            return idx
    # fallback: first valid, else 0
    idxs = np.where(track.valid)[0]
    return int(idxs[0]) if len(idxs) else 0


def mux_audio_ffmpeg(video_no_audio: str, src_video: str, out_path: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    # Copy video stream, re-encode audio to AAC (safe default), keep shortest.
    cmd = [
        ffmpeg, "-y",
        "-i", video_no_audio,
        "-i", src_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception:
        return False


def backup_rename_if_exists(path: str) -> str | None:
    """Best-effort backup by renaming (no duplication). Good for large outputs like .mp4.
    Returns backup path if created, else None.
    """
    try:
        if os.path.isfile(path):
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = f"{path}.bak.{ts}"
            os.replace(path, bak)
            return bak
    except Exception:
        return None
    return None


def save_debug_metrics(debug_dir: str, payload: dict, valid: np.ndarray) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    json_path = os.path.join(debug_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # simple plot (valid only) without matplotlib
    W, H = 1200, 260
    M = 40
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (M, M), (W - M, H - M), (220, 220, 220), 1)
    cv2.putText(img, "track valid", (M, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    y = valid.astype(np.float32)
    if y.size > 1:
        xs = np.linspace(M, W - M, num=y.size).astype(np.int32)
        ys = (H - M - (y * (H - 2 * M))).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, (30, 30, 30), 2, cv2.LINE_AA)
    png_path = os.path.join(debug_dir, "metrics.png")
    if not write_image_file(png_path, img):
        raise RuntimeError(f"Failed to save debug image: {png_path}")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video")
    ap.add_argument("--track", default="", help="mouth_track(_calibrated).npz (default: auto in video folder)")
    ap.add_argument("--out", default="", help="output mouthless video (default: *_mouthless.mp4)")
    ap.add_argument("--valid-policy", choices=["hold", "strict"], default="hold")
    ap.add_argument("--ref-frame", type=int, default=-1, help="reference frame index (-1=auto)")
    ap.add_argument("--oversample", type=float, default=1.2, help="normalized patch size multiplier")
    ap.add_argument("--norm-w", type=int, default=0, help="override normalized patch width (0=auto)")
    ap.add_argument("--norm-h", type=int, default=0, help="override normalized patch height (0=auto)")
    ap.add_argument("--coverage", type=float, default=-1.0, help="single knob 0..1 (larger=erase wider). If set >=0, auto-tunes mask/ring/feather for anime faces.")
    ap.add_argument("--mask-scale-x", type=float, default=0.62, help="inner erase mask ellipse scale (x) relative to patch")
    ap.add_argument("--mask-scale-y", type=float, default=0.62, help="inner erase mask ellipse scale (y) relative to patch")
    ap.add_argument("--ring", type=int, default=18, help="ring width in pixels in normalized space")
    ap.add_argument("--dilate", type=int, default=10, help="mask dilation in pixels before feather")
    ap.add_argument("--feather", type=int, default=20, help="mask feather (gaussian radius) in pixels")
    ap.add_argument("--inpaint-radius", type=float, default=5.0, help="cv2.inpaint radius")
    ap.add_argument("--shading", choices=["plane", "none"], default="plane")
    ap.add_argument("--keep-audio", action="store_true", help="try to keep original audio with ffmpeg")
    ap.add_argument(
        "--codec",
        default="mp4v",
        help="fourcc for cv2.VideoWriter (default: mp4v). Use 'auto' to try multiple codecs.",
    )
    ap.add_argument("--max-frames", type=int, default=-1, help="debug: limit frames")
    ap.add_argument("--debug", default="", help="debug output dir (optional)")
    ap.add_argument("--profile", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--profile-out", default="", help=argparse.SUPPRESS)
    args = ap.parse_args()

    profiler = None
    if getattr(args, 'profile', False):
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(args.video)

    video_dir = os.path.dirname(os.path.abspath(args.video))
    if args.track:
        track_path = args.track
    else:
        # auto prefer calibrated
        cand1 = os.path.join(video_dir, "mouth_track_calibrated.npz")
        cand2 = os.path.join(video_dir, "mouth_track.npz")
        track_path = cand1 if os.path.isfile(cand1) else cand2
    if not os.path.isfile(track_path):
        raise FileNotFoundError(f"track not found: {track_path}")

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(video_dir, f"{base}_mouthless.mp4")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Safety: writing to the same path as the input can corrupt the video.
    in_path_abs = os.path.abspath(args.video)
    out_path_abs = os.path.abspath(out_path)
    inplace_replace = (in_path_abs == out_path_abs)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if vid_w <= 0 or vid_h <= 0:
        raise RuntimeError("invalid video size")

    track = load_track(track_path, vid_w, vid_h, policy=args.valid_policy)
    N = track.total

    if total_frames <= 0:
        total_frames = N
    if N <= 0:
        raise RuntimeError("empty track")
    n_out = min(total_frames, N)
    if args.max_frames > 0:
        n_out = min(n_out, int(args.max_frames))

    # Decide normalized patch size
    quads_for_size = track.filled[:n_out]
    ws = np.array([quad_wh(q)[0] for q in quads_for_size], dtype=np.float32)
    hs = np.array([quad_wh(q)[1] for q in quads_for_size], dtype=np.float32)
    ratio = float(np.median(ws / np.maximum(1e-6, hs)))
    p95w = float(np.percentile(ws, 95))
    # derive norm size
    if args.norm_w > 0:
        norm_w = int(args.norm_w)
    else:
        norm_w = int(round(p95w * float(args.oversample)))
    norm_w = ensure_even_ge2(max(96, norm_w))
    if args.norm_h > 0:
        norm_h = int(args.norm_h)
    else:
        norm_h = int(round(norm_w / max(0.25, min(4.0, ratio))))
    norm_h = ensure_even_ge2(max(64, norm_h))

    print(f"[info] video: {vid_w}x{vid_h} fps={fps:.3f} frames={n_out}")
    print(f"[info] track: {os.path.basename(track_path)} N={N} valid={int(track.valid[:n_out].sum())}/{n_out} policy={args.valid_policy}")
    print(f"[info] norm patch: {norm_w}x{norm_h} (ratio~{ratio:.3f}, oversample={args.oversample})")

    # Masks in normalized space
    # For end-users, prefer a single knob `--coverage` (0..1).
    # Larger coverage erases a wider area (helps fast mouth motion). Smaller coverage is safer for noses/cheeks.
    mask_scale_x = float(args.mask_scale_x)
    mask_scale_y = float(args.mask_scale_y)
    ring_px = int(max(2, args.ring))
    dilate_px = int(args.dilate)
    feather_px = int(args.feather)
    inpaint_radius = float(args.inpaint_radius)

    # Nose-guard defaults (anime)
    top_clip_frac = 0.82
    center_y_off = int(round(norm_h * 0.05))

    if float(args.coverage) >= 0.0:
        cov = float(np.clip(float(args.coverage), 0.0, 1.0))
        # Tuned defaults for anime faces (black line art): keep the erase area tight, expand gently with coverage.
        mask_scale_x = 0.50 + 0.18 * cov   # 0.50..0.68
        mask_scale_y = 0.44 + 0.14 * cov   # 0.44..0.58 (vertical grows slower to avoid the nose)
        ring_px = int(round(16 + 10 * cov))  # 16..26
        dilate_px = int(round(8 + 8 * cov))  # 8..16
        feather_px = int(round(18 + 10 * cov))  # 18..28
        inpaint_radius = 4.0 + 4.0 * cov    # 4..8
        # When coverage is larger, clip a bit more at the top to protect the nose.
        top_clip_frac = float(0.84 - 0.06 * cov)  # 0.84..0.78
        center_y_off = int(round(norm_h * (0.05 + 0.01 * cov)))
        print(
            f"[info] coverage={cov:.2f} => mask_scale=({mask_scale_x:.3f},{mask_scale_y:.3f}) "
            f"ring={ring_px} dilate={dilate_px} feather={feather_px} inpaint={inpaint_radius:.1f}"
        )

    rx = int((norm_w * mask_scale_x) * 0.5)
    ry = int((norm_h * mask_scale_y) * 0.5)
    inner_u8 = make_mouth_mask(
        norm_w, norm_h, rx=rx, ry=ry, center_y_offset_px=center_y_off, top_clip_frac=top_clip_frac
    )

    outer_u8 = make_mouth_mask(
        norm_w, norm_h, rx=rx + ring_px, ry=ry + ring_px, center_y_offset_px=center_y_off, top_clip_frac=top_clip_frac
    )
    ring_u8 = cv2.subtract(outer_u8, inner_u8)
    ring_ys, ring_xs = np.where(ring_u8 > 0)

    plane_fitter = PlaneFitter.from_ring(ring_xs, ring_ys, norm_w, norm_h)

    mask_f = feather_mask(inner_u8, dilate_px=dilate_px, feather_px=feather_px)
    # Pick reference frame
    if args.ref_frame >= 0:
        ref_idx = int(max(0, min(args.ref_frame, n_out - 1)))
    else:
        ref_idx = int(max(0, min(choose_ref_frame(track), n_out - 1)))
    print(f"[info] ref_frame={ref_idx}")

    def read_frame_at(idx: int, fallback_range: int = 0, return_idx: bool = False):
        """Read frame at idx with optional fallback to nearby frames on failure."""
        offsets = [0] + [d for i in range(1, fallback_range + 1) for d in (i, -i)]
        last_err = None
        for off in offsets:
            try_idx = max(0, min(n_out - 1, idx + off))
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(try_idx))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                if off != 0:
                    print(f"[warn] frame {idx} unreadable, using fallback frame {try_idx}")
                if bgr.shape[1] != vid_w or bgr.shape[0] != vid_h:
                    bgr = cv2.resize(bgr, (vid_w, vid_h), interpolation=cv2.INTER_AREA)
                return (try_idx, bgr) if return_idx else bgr
            last_err = f"failed to read frame {try_idx}"
        raise RuntimeError(last_err or f"failed to read frame {idx}")

    # Build clean plate (base texture) in normalized space
    # Use fallback_range to try nearby frames if ref_idx is unreadable
    ref_idx_used, ref_frame = read_frame_at(ref_idx, fallback_range=5, return_idx=True)
    if ref_idx_used != ref_idx:
        ref_idx = int(ref_idx_used)
        print(f"[warn] ref_frame changed to {ref_idx} due to read fallback")
    ref_quad = track.filled[ref_idx]
    ref_patch = warp_frame_to_norm(ref_frame, ref_quad, norm_w, norm_h)
    ref_patch_lab = cv2.cvtColor(ref_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    # reference shading model from ring
    if len(ring_xs) < 16:
        raise RuntimeError("ring mask too small; adjust --mask-scale-* / --ring")

    # NOTE: For shading=none, we don't need per-frame planes at all.
    plane_ref_grid = None
    a_ref_mean = 0.0
    b_ref_mean = 0.0
    if args.shading == "plane":
        L_ref = ref_patch_lab[:, :, 0][ring_ys, ring_xs]
        plane_ref = plane_fitter.fit(L_ref)
        plane_ref_grid = plane_fitter.eval(plane_ref)

        a_ref_mean = float(ref_patch_lab[:, :, 1][ring_ys, ring_xs].mean())
        b_ref_mean = float(ref_patch_lab[:, :, 2][ring_ys, ring_xs].mean())

    # Inpaint (remove mouth) on reference patch
    inpaint_mask = inner_u8
    clean_patch = cv2.inpaint(ref_patch, inpaint_mask, inpaintRadius=float(inpaint_radius), flags=cv2.INPAINT_TELEA)
    clean_lab_base = cv2.cvtColor(clean_patch, cv2.COLOR_BGR2LAB).astype(np.float32)

    # debug outputs
    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        for name, image in (
            ("ref_patch.png", ref_patch),
            ("clean_patch.png", clean_patch),
            ("mask_inner.png", inner_u8),
            ("mask_ring.png", ring_u8),
        ):
            out_path = os.path.join(args.debug, name)
            if not write_image_file(out_path, image):
                raise RuntimeError(f"Failed to save debug image: {out_path}")

    # Prepare writer (silent)
    mux_audio = bool(args.keep_audio)
    will_overwrite = os.path.isfile(out_path)
    need_temp = bool(mux_audio) or bool(inplace_replace) or bool(will_overwrite)
    tmp_dir = None
    tmp_out = out_path
    if need_temp:
        tmp_dir = tempfile.mkdtemp(prefix="mouthless_")
        if inplace_replace and not mux_audio:
            print(f"[warn] --out is the same as --video; will write to a temp file and replace in-place: {out_path}")
            tmp_out = os.path.join(tmp_dir, "mouthless_tmp.mp4")
        else:
            if inplace_replace and mux_audio:
                print(f"[warn] --out is the same as --video; will write to a temp file and replace in-place: {out_path}")
            tmp_out = os.path.join(tmp_dir, "video_no_audio.mp4")

    def open_writer(path: str) -> cv2.VideoWriter:
        codec_try = [str(args.codec)]
        if str(args.codec).lower() == "auto":
            codec_try = ["mp4v", "avc1", "H264", "XVID", "MJPG"]
        else:
            # If the requested codec fails, try a small fallback set.
            codec_try += ["mp4v", "XVID", "MJPG"]

        for c in codec_try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            w = cv2.VideoWriter(path, fourcc, fps, (vid_w, vid_h))
            if w.isOpened():
                if c != str(args.codec):
                    print(f"[warn] VideoWriter codec fallback: {args.codec} -> {c}")
                return w

        raise RuntimeError(f"failed to open VideoWriter: {path} codec={args.codec} (tried {codec_try})")

    writer = open_writer(tmp_out)

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

    # plane_ref_grid is prepared above (PlaneFitter)

    # Progress (TTY only; no new CLI flags)
    use_tqdm = sys.stderr.isatty()
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore
    it = range(n_out)
    if use_tqdm and tqdm is not None:
        it = tqdm(it, total=n_out, desc="Erasing", unit="frame", dynamic_ncols=True)

    for i in it:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != vid_w or frame.shape[0] != vid_h:
            frame = cv2.resize(frame, (vid_w, vid_h), interpolation=cv2.INTER_AREA)

        if args.valid_policy == "strict" and (not bool(track.valid[i])):
            # strict: leave original
            writer.write(frame)
            continue

        quad = track.filled[i]
        patch = warp_frame_to_norm(frame, quad, norm_w, norm_h)

        if args.shading == "none":
            out_patch_lab = clean_lab_base.copy()
        else:
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)

            # Fit shading plane on ring for current frame
            L_i = patch_lab[:, :, 0][ring_ys, ring_xs]
            plane_i = plane_fitter.fit(L_i)
            plane_i_grid = plane_fitter.eval(plane_i)

            # Mean chroma shift (a,b)
            a_i_mean = float(patch_lab[:, :, 1][ring_ys, ring_xs].mean())
            b_i_mean = float(patch_lab[:, :, 2][ring_ys, ring_xs].mean())
            da = a_i_mean - a_ref_mean
            db = b_i_mean - b_ref_mean

            out_patch_lab = clean_lab_base.copy()
            out_patch_lab[:, :, 0] = np.clip(out_patch_lab[:, :, 0] + (plane_i_grid - plane_ref_grid), 0, 255)
            out_patch_lab[:, :, 1] = np.clip(out_patch_lab[:, :, 1] + da, 0, 255)
            out_patch_lab[:, :, 2] = np.clip(out_patch_lab[:, :, 2] + db, 0, 255)

        out_patch = cv2.cvtColor(out_patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        warped_patch, warped_mask, x0, y0 = warp_norm_to_bbox(out_patch, mask_f, quad)

        # ROI blend with bounds
        H, W = frame.shape[:2]
        bw, bh = warped_patch.shape[1], warped_patch.shape[0]
        rx0 = max(0, x0)
        ry0 = max(0, y0)
        rx1 = min(W, x0 + bw)
        ry1 = min(H, y0 + bh)
        if rx0 < rx1 and ry0 < ry1:
            sx0 = rx0 - x0
            sy0 = ry0 - y0
            sx1 = sx0 + (rx1 - rx0)
            sy1 = sy0 + (ry1 - ry0)
            roi = frame[ry0:ry1, rx0:rx1]
            src = warped_patch[sy0:sy1, sx0:sx1]
            msk = warped_mask[sy0:sy1, sx0:sx1]
            frame[ry0:ry1, rx0:rx1] = alpha_blend_roi(roi, src, msk)

        # Optional debug overlay every so often
        if args.debug and (i % 120 == 0):
            dbg = frame.copy()
            q = quad.astype(np.int32).reshape(4, 2)
            cv2.polylines(dbg, [q], True, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(dbg, f"{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            out_path = os.path.join(args.debug, f"frame_{i:06d}.png")
            if not write_image_file(out_path, dbg):
                raise RuntimeError(f"Failed to save debug frame: {out_path}")

        writer.write(frame)

        if (tqdm is None or not use_tqdm) and ((i + 1) % 120 == 0 or i == n_out - 1):
            print(f"  {i+1}/{n_out}")

    writer.release()
    cap.release()

    # mux audio if requested
    if mux_audio:
        if will_overwrite:
            backup_rename_if_exists(out_path)
        ok = mux_audio_ffmpeg(tmp_out, args.video, out_path)
        if ok:
            print(f"[info] wrote (with audio): {out_path}")
        else:
            # fallback: keep silent file
            shutil.copyfile(tmp_out, out_path)
            print(f"[warn] ffmpeg mux failed; wrote silent: {out_path}")
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
    else:
        # In-place replace safety (and/or other temp usage)
        if tmp_out != out_path:
            if will_overwrite:
                backup_rename_if_exists(out_path)
            if inplace_replace:
                try:
                    os.replace(tmp_out, out_path)
                except Exception:
                    shutil.copyfile(tmp_out, out_path)
            else:
                shutil.copyfile(tmp_out, out_path)
        if tmp_dir:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        print(f"[info] wrote: {out_path}")

        # Internal metrics (always computed; only saved when --debug is set)
        valid_rate = float(track.valid[:n_out].mean()) if n_out > 0 else 0.0
        if valid_rate < 0.80:
            print(f"[warn] track valid_rate is low ({valid_rate:.2f}). "
                  f"Erase quality may degrade; try improving tracking settings.")

        if args.debug:
            payload = {
                "video": os.path.abspath(args.video),
                "track": os.path.abspath(track_path),
                "out": os.path.abspath(out_path),
                "n_frames": int(n_out),
                "valid_rate": valid_rate,
                "params": {
                    "valid_policy": str(args.valid_policy),
                    "norm_w": int(norm_w),
                    "norm_h": int(norm_h),
                    "oversample": float(args.oversample),
                    "coverage": float(args.coverage),
                    "ring": int(args.ring),
                    "dilate": int(args.dilate),
                    "feather": int(args.feather),
                    "shading": str(args.shading),
                },
            }
            try:
                save_debug_metrics(args.debug, payload, track.valid[:n_out].astype(np.uint8))
                print(f"[debug] metrics: {os.path.join(args.debug, 'metrics.json')}")
            except Exception as e:
                print(f"[warn] failed to write debug metrics: {e}")

        # Profiling output (dev-only)
        if profiler is not None:
            try:
                import io
                import pstats
                profiler.disable()
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                ps.print_stats(30)
                out_txt = s.getvalue()
                if getattr(args, "profile_out", ""):
                    os.makedirs(os.path.dirname(os.path.abspath(args.profile_out)) or ".", exist_ok=True)
                    with open(args.profile_out, "w", encoding="utf-8") as f:
                        f.write(out_txt)
                else:
                    print(out_txt)
            except Exception as e:
                print(f"[warn] profile output failed: {e}")


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
