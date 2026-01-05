#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_track_anime_detector.py

anime-face-detector を使用してアニメ顔の口領域を直接検出するトラッカー。
ORBベースの手法と異なり、フレームごとに独立して検出するため
高速な動きに強い。

インストール:

    pip install dghs-imgutils[gpu]
    # or
    pip install dghs-imgutils


使い方:
    python face_track_anime_detector.py \
        --video "loop.mp4" \
        --out "mouth_track.npz" \
        --debug "mouth_track_debug.mp4" \
        --pad 1.5 \
        --smooth-cutoff 3.0

出力:
    mouth_track.npz (loop_lipsync_runtime.py と互換)
      - quad: (N, 4, 2) float32
      - valid: (N,) uint8
      - fps, w, h, ref_sprite_w, ref_sprite_h, pad
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# anime-face-detector -> dghs-imgutils (dwpose)
try:
    from imgutils.pose import dwpose_estimate
    from imgutils.pose.format import OP18KeyPointSet
    HAS_IMGUTILS = True
except ImportError:
    HAS_IMGUTILS = False
    print("[warn] dghs-imgutils not installed. Run: pip install dghs-imgutils[gpu]")


# ランドマークのインデックス定義 (Standard 68 points)
# 48-59: Mouth Outer Lip (12 points)
# 60-67: Mouth Inner Lip (8 points)
MOUTH_OUTLINE = list(range(48, 60)) 



def ensure_even(n: int) -> int:
    """偶数に丸める。最小値は2を保証（0や1を返すと後段でクラッシュするため）"""
    n = max(2, int(n))
    return n if (n % 2 == 0) else (n - 1)


def one_pole_beta(cutoff_hz: float, fps: float) -> float:
    """EMAのbeta係数を計算 (1-pole lowpass)"""
    return float(1.0 - np.exp(-2.0 * np.pi * float(cutoff_hz) / float(max(1e-6, fps))))


def decompose_quad(quad: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    quadを中心、幅、高さ、角度に分解。
    Returns: (center, width, height, angle_deg)
    """
    center = quad.mean(axis=0)

    # TL->TR ベクトルから幅と角度を計算
    v_top = quad[1] - quad[0]
    width = float(np.linalg.norm(v_top))
    angle_deg = float(np.degrees(np.arctan2(v_top[1], v_top[0])))

    # TL->BL ベクトルから高さを計算
    v_left = quad[3] - quad[0]
    height = float(np.linalg.norm(v_left))

    return center, width, height, angle_deg


def decompose_quads_vectorized(quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized version of decompose_quad for N quads.
    quads: (N, 4, 2) array
    Returns: (centers (N,2), widths (N,), heights (N,), angles (N,))
    """
    # centers: mean of 4 corners
    centers = quads.mean(axis=1)  # (N, 2)

    # TL->TR vector for width and angle
    v_top = quads[:, 1, :] - quads[:, 0, :]  # (N, 2)
    widths = np.linalg.norm(v_top, axis=1)  # (N,)
    angles = np.degrees(np.arctan2(v_top[:, 1], v_top[:, 0]))  # (N,)

    # TL->BL vector for height
    v_left = quads[:, 3, :] - quads[:, 0, :]  # (N, 2)
    heights = np.linalg.norm(v_left, axis=1)  # (N,)

    return centers.astype(np.float32), widths.astype(np.float32), heights.astype(np.float32), angles.astype(np.float32)


def compose_quad(center: np.ndarray, width: float, height: float, angle_deg: float) -> np.ndarray:
    """
    中心、幅、高さ、角度からquadを再構成。
    """
    hw, hh = width / 2, height / 2

    # ローカル座標
    local = np.array([
        [-hw, -hh],
        [+hw, -hh],
        [+hw, +hh],
        [-hw, +hh],
    ], dtype=np.float32)

    # 回転
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    rotated = local @ R.T
    return (rotated + center).astype(np.float32)


def compose_quads_vectorized(centers: np.ndarray, widths: np.ndarray, heights: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """
    Vectorized version of compose_quad for N quads.
    centers: (N, 2), widths: (N,), heights: (N,), angles_deg: (N,)
    Returns: (N, 4, 2) array
    """
    N = len(centers)
    hw = widths / 2  # (N,)
    hh = heights / 2  # (N,)

    # Local coordinates for each quad: (N, 4, 2)
    # TL, TR, BR, BL
    local = np.zeros((N, 4, 2), dtype=np.float32)
    local[:, 0, 0] = -hw
    local[:, 0, 1] = -hh
    local[:, 1, 0] = +hw
    local[:, 1, 1] = -hh
    local[:, 2, 0] = +hw
    local[:, 2, 1] = +hh
    local[:, 3, 0] = -hw
    local[:, 3, 1] = +hh

    # Rotation matrices: (N, 2, 2)
    angles_rad = np.radians(angles_deg)
    cos_a = np.cos(angles_rad)  # (N,)
    sin_a = np.sin(angles_rad)  # (N,)

    # Apply rotation: rotated[i] = local[i] @ R[i].T
    # For each point p in local[i], rotated = [cos*px - sin*py, sin*px + cos*py]
    rotated = np.zeros_like(local)
    rotated[:, :, 0] = cos_a[:, None] * local[:, :, 0] - sin_a[:, None] * local[:, :, 1]
    rotated[:, :, 1] = sin_a[:, None] * local[:, :, 0] + cos_a[:, None] * local[:, :, 1]

    # Translate by center
    result = rotated + centers[:, None, :]  # (N, 4, 2)
    return result.astype(np.float32)


def limit_angle_change(angles: np.ndarray, max_change_deg: float) -> np.ndarray:
    """
    連続するフレーム間の角度変化を制限。
    急激な変化を抑制する。
    """
    if max_change_deg <= 0:
        return angles.copy()
    
    result = angles.copy()
    for i in range(1, len(result)):
        diff = result[i] - result[i-1]
        if abs(diff) > max_change_deg:
            # 変化量を制限
            result[i] = result[i-1] + np.sign(diff) * max_change_deg
    return result


# Cached scipy median filter (None = not checked, False = unavailable, function = available)
_scipy_median_cache: Any = None


def _get_scipy_median():
    """Get cached scipy median filter function, or None if unavailable."""
    global _scipy_median_cache
    if _scipy_median_cache is None:
        try:
            from scipy.ndimage import median_filter
            _scipy_median_cache = median_filter
        except ImportError:
            _scipy_median_cache = False
    return _scipy_median_cache if _scipy_median_cache else None


def median_filter_1d(arr: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """1Dメディアンフィルタ（外れ値除去用）

    Uses scipy.ndimage.median_filter if available (faster), otherwise falls back to numpy.
    """
    scipy_median = _get_scipy_median()
    if scipy_median is not None:
        return scipy_median(arr, size=kernel_size, mode='nearest')

    # Fallback: pure numpy implementation
    N = len(arr)
    result = arr.copy()
    half = kernel_size // 2

    for i in range(N):
        start = max(0, i - half)
        end = min(N, i + half + 1)
        result[i] = np.median(arr[start:end])

    return result


def smooth_quads_zero_phase(quads: np.ndarray, valid: np.ndarray, beta: float, max_angle_change: float = 15.0) -> np.ndarray:
    """
    valid=1のフレームのみを使って、zero-phase EMAで平滑化。
    角度は分離してスムージング + メディアンフィルタで外れ値除去。

    max_angle_change: 1フレームあたりの最大角度変化(度)。0で無制限。
    """
    N = len(quads)
    if N == 0:
        return quads.copy()

    out = quads.copy()

    # まずinvalid区間を線形補間で埋める
    out = interpolate_invalid_quads(out, valid)

    # quadを分解 (vectorized for performance)
    centers, widths, heights, angles = decompose_quads_vectorized(out)

    # 角度のunwrap（-180→180のジャンプを滑らかに）
    angles_unwrapped = np.unwrap(np.radians(angles))
    angles_unwrapped = np.degrees(angles_unwrapped)

    # 角度にメディアンフィルタ（1-2フレームの外れ値を除去）
    angles_filtered = median_filter_1d(angles_unwrapped, kernel_size=3)

    # 各成分をスムージング
    def smooth_1d(arr, beta):
        out = arr.copy()
        # forward
        for i in range(1, len(out)):
            out[i] = out[i-1] + beta * (out[i] - out[i-1])
        # backward (zero-phase)
        for i in range(len(out)-2, -1, -1):
            out[i] = out[i+1] + beta * (out[i] - out[i+1])
        return out

    cx_smooth = smooth_1d(centers[:, 0], beta)
    cy_smooth = smooth_1d(centers[:, 1], beta)
    w_smooth = smooth_1d(widths, beta)
    h_smooth = smooth_1d(heights, beta)
    angle_smooth = smooth_1d(angles_filtered, beta)

    # 角度変化の制限を適用
    if max_angle_change > 0:
        angle_smooth = limit_angle_change(angle_smooth, max_angle_change)

    # 再構成 (vectorized for performance)
    centers_smooth = np.column_stack([cx_smooth, cy_smooth]).astype(np.float32)
    result = compose_quads_vectorized(centers_smooth, w_smooth, h_smooth, angle_smooth)

    return result.astype(np.float32)


def interpolate_invalid_quads(quads: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """valid==0 の区間を前後フレームから補間して埋める。

    以前は corner そのままの線形補間でしたが、回転を含む場合に歪みやすいので、
    (center, width, height, angle) に分解して補間 → 再構成する方式に変更。
    """
    N = len(quads)
    if N == 0:
        return quads

    res = quads.copy()
    v = valid.astype(np.uint8).reshape(-1)

    def angle_shortest_diff(a0: float, a1: float) -> float:
        # [-180, 180) の最短差分
        return float(((a1 - a0 + 180.0) % 360.0) - 180.0)

    i = 0
    while i < N:
        if v[i] == 1:
            i += 1
            continue

        # invalid区間の開始
        start = i
        while i < N and v[i] == 0:
            i += 1
        end = i  # [start, end) is invalid

        prev_ok = start - 1 if start > 0 and v[start - 1] == 1 else -1
        next_ok = end if end < N and v[end] == 1 else -1

        if prev_ok >= 0 and next_ok >= 0:
            # (center, w, h, angle) で補間
            c0, w0, h0, a0 = decompose_quads_vectorized(quads[prev_ok:prev_ok + 1])
            c1, w1, h1, a1 = decompose_quads_vectorized(quads[next_ok:next_ok + 1])
            c0 = c0[0]; c1 = c1[0]
            w0 = float(w0[0]); w1 = float(w1[0])
            h0 = float(h0[0]); h1 = float(h1[0])
            a0 = float(a0[0]); a1 = float(a1[0])

            da = angle_shortest_diff(a0, a1)
            gap = float(next_ok - prev_ok)
            for j in range(start, end):
                t = float(j - prev_ok) / gap
                c = (1.0 - t) * c0 + t * c1
                w = (1.0 - t) * w0 + t * w1
                h = (1.0 - t) * h0 + t * h1
                a = a0 + t * da
                res[j] = compose_quad(c.astype(np.float32), float(w), float(h), float(a))
        elif prev_ok >= 0:
            # 前の値をhold
            for j in range(start, end):
                res[j] = quads[prev_ok]
        elif next_ok >= 0:
            # 後の値をhold
            for j in range(start, end):
                res[j] = quads[next_ok]
        # 両方ない場合はそのまま

    return res

def mouth_quad_from_landmarks(
    keypoints: np.ndarray,
    *,
    bbox: Optional[np.ndarray] = None,
    sprite_aspect: float = 1.0,
    pad: float = 2.1,
    min_mouth_w_ratio: float = 0.0,
    min_mouth_w_px: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    28点のランドマークから口のquadを生成。

    keypoints: (28, 3) - x, y, confidence
    bbox: Optional[np.ndarray] - [x1, y1, x2, y2, conf] (hybridモード用)
    sprite_aspect: スプライトのwidth/height比
    pad: quad を口より少し大きめに取るパディング係数
    min_mouth_w_ratio: 口幅の最小値（顔幅に対する比率）
    min_mouth_w_px: 口幅の最小値（ピクセル）

    Returns:
        quad: (4, 2) - [TL, TR, BR, BL]
        confidence: 口ランドマークの平均信頼度
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]
    xy = mouth_pts[:, :2].astype(np.float32)
    conf = float(mouth_pts[:, 2].mean())
    cx, cy = xy.mean(axis=0)

    angle_deg = estimate_face_rotation(keypoints)
    ang = np.deg2rad(angle_deg)
    ca, sa = float(np.cos(ang)), float(np.sin(ang))
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)  # local->global

    rel = xy - np.array([cx, cy], dtype=np.float32)
    rel_local = rel @ R          # row vec => rotate by -angle
    x_min, y_min = rel_local.min(axis=0)
    x_max, y_max = rel_local.max(axis=0)
    w_lm = float(x_max - x_min)
    h_lm = float(y_max - y_min)

    w_floor = 0.0
    if bbox is not None:
        face_w = float(max(1.0, float(bbox[2]) - float(bbox[0])))
        w_floor = face_w * float(min_mouth_w_ratio)

    w0 = max(w_lm, w_floor, float(min_mouth_w_px))
    asp = max(0.25, min(4.0, float(sprite_aspect)))
    h0 = max(h_lm, w0 / asp)

    w = w0 * float(pad)
    h = h0 * float(pad)
    hw, hh = w / 2.0, h / 2.0
    quad_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    quad = (quad_local @ R.T) + np.array([cx, cy], dtype=np.float32)
    return quad.astype(np.float32), conf


def _mouth_wh_local_rotated(keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    口4点を、顔の傾き(目の傾き)を用いてローカル座標に回してから
    口のw/h（ローカル軸）を推定する。
    Returns:
        c(2,), R(2,2), w0, h0, conf
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]  # (4, 3)
    xy = mouth_pts[:, :2].astype(np.float32)
    conf = float(mouth_pts[:, 2].mean())

    c = xy.mean(axis=0).astype(np.float32)

    angle_deg = estimate_face_rotation(keypoints)
    ang = np.radians(angle_deg)
    ca = float(np.cos(ang))
    sa = float(np.sin(ang))
    R = np.array([[ca, -sa],
                  [sa,  ca]], dtype=np.float32)

    # 既存コードの回転規約に合わせる:
    # quad_local -> global で (quad_local @ R.T) を使っているので
    # global -> local は (p @ R) を使う
    xy_local = (xy - c) @ R
    x_min, y_min = xy_local.min(axis=0)
    x_max, y_max = xy_local.max(axis=0)
    w0 = float(x_max - x_min)
    h0 = float(y_max - y_min)
    return c, R, w0, h0, conf


def mouth_quad_from_mouth_landmarks_rotated(
    keypoints: np.ndarray,
    sprite_aspect: float = 1.0,
    pad: float = 2.1,
    min_w_px: float = 8.0,
    min_h_px: float = 6.0,
    height_floor_scale: float = 0.75,
) -> Tuple[np.ndarray, float]:
    """
    口4点の広がりからw/hを推定し、顔傾きに合わせて回転したquadを作る（bbox依存を弱める）。
    小さすぎる閉じ口対策として、高さに下限を持たせる。
    """
    c, R, w0, h0, conf = _mouth_wh_local_rotated(keypoints)

    w = float(max(min_w_px, w0)) * float(pad)
    h = float(max(min_h_px, h0)) * float(pad)

    asp = float(np.clip(float(sprite_aspect), 0.25, 4.0))
    h = float(max(h, (w / asp) * float(height_floor_scale)))

    hw, hh = w * 0.5, h * 0.5
    quad_local = np.array([
        [-hw, -hh],  # TL
        [+hw, -hh],  # TR
        [+hw, +hh],  # BR
        [-hw, +hh],  # BL
    ], dtype=np.float32)
    quad = (quad_local @ R.T) + c
    return quad.astype(np.float32), float(conf)


def mouth_quad_auto(
    bbox: np.ndarray,
    keypoints: np.ndarray,
    sprite_aspect: float,
    pad: float,
    auto_min_mouth_ratio: float,
    auto_min_mouth_px: float,
) -> Tuple[np.ndarray, float]:
    """
    通常は mouth-landmarks 依存、口が小さすぎるときだけ bbox 依存にフォールバック。
    """
    c, R, w0, h0, conf0 = _mouth_wh_local_rotated(keypoints)
    face_w = float(max(1.0, float(bbox[2] - bbox[0])))

    too_small = (w0 < float(auto_min_mouth_px)) or ((w0 / face_w) < float(auto_min_mouth_ratio))
    if too_small:
        quad, conf = mouth_quad_from_face_bbox_and_landmarks(
            bbox, keypoints,
            sprite_aspect=float(sprite_aspect),
            pad=float(pad),
        )
        return quad, float(conf)
    else:
        quad, conf = mouth_quad_from_mouth_landmarks_rotated(
            keypoints,
            sprite_aspect=float(sprite_aspect),
            pad=float(pad),
        )
        return quad, float(conf)


def estimate_face_rotation(keypoints: np.ndarray) -> float:
    """
    ランドマークから顔の傾き角度（度）を推定。
    
    Standard 68 points:
      Right Eye: 36-41 (digits are standard 0-indexed)
      Left Eye: 42-47
    """
    # 左目のランドマーク (42-47) -> Subject's Left (Viewer's Right)
    # But usually 36-41 is Right Eye (Viewer's Left) and 42-47 is Left Eye (Viewer's Right)
    # Let's verify standard 68 ordering. 
    # (Typically 0-16 jaw, 17-21 R-Brow, 22-26 L-Brow, 27-35 Nose, 36-41 R-Eye, 42-47 L-Eye)
    # R-Eye center vs L-Eye center
    
    # 36-41: Right Eye (Subject's Right / Viewer's Left)
    # 42-47: Left Eye (Subject's Left / Viewer's Right)
    
    left_eye = keypoints[42:48, :2]  # indices 42..47
    left_eye_center = left_eye.mean(axis=0)
    
    # 右目のランドマーク (36-41)
    right_eye = keypoints[36:42, :2] # indices 36..41
    right_eye_center = right_eye.mean(axis=0)
    
    # 目の中心を結ぶベクトルから角度を計算
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)


def mouth_quad_from_face_bbox_and_landmarks(
    bbox: np.ndarray,
    keypoints: np.ndarray,
    sprite_aspect: float = 1.0,
    pad: float = 2.1
) -> Tuple[np.ndarray, float]:
    """
    口ランドマークを中心として、顔bboxのサイズを参照してquadを生成。
    顔の傾きに応じてquadを回転させる。

    bbox: [x1, y1, x2, y2, conf]
    keypoints: (28, 3)
    sprite_aspect: スプライトのwidth/height比
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]
    xy = mouth_pts[:, :2]
    conf = mouth_pts[:, 2].mean()

    # 口の中心
    cx, cy = xy.mean(axis=0)

    # 顔bboxのサイズから口quadのサイズを推定
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]

    # 口の幅は顔幅の約40%, 高さはアスペクト比から計算
    mouth_w = face_w * 0.40 * pad
    mouth_h = mouth_w / sprite_aspect

    hw, hh = mouth_w / 2, mouth_h / 2
    
    # まず水平なquadを作成（中心を原点として）
    quad_local = np.array([
        [-hw, -hh],  # TL
        [+hw, -hh],  # TR
        [+hw, +hh],  # BR
        [-hw, +hh],  # BL
    ], dtype=np.float32)
    
    # 顔の傾きを取得して回転
    angle_deg = estimate_face_rotation(keypoints)
    angle_rad = np.radians(angle_deg)
    
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ], dtype=np.float32)
    
    # 回転を適用
    quad_rotated = (quad_local @ R.T)
    
    # 口の中心に移動
    quad = quad_rotated + np.array([cx, cy], dtype=np.float32)

    return quad, float(conf)


def draw_quad(img: np.ndarray, quad: np.ndarray, color=(0, 255, 0), thickness=2):
    pts = quad.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for p in quad.astype(np.int32):
        cv2.circle(img, tuple(p), 3, color, -1)


def draw_landmarks(img: np.ndarray, keypoints: np.ndarray, color=(255, 0, 255), radius=2):
    """28点のランドマークを描画"""
    for i, kp in enumerate(keypoints):
        x, y, conf = kp
        if conf > 0.3:
            cv2.circle(img, (int(x), int(y)), radius, color, -1)
            # 口のランドマークは強調
            if i in MOUTH_OUTLINE:
                cv2.circle(img, (int(x), int(y)), radius + 2, (0, 255, 255), 2)


def backup_copy_if_exists(path: str) -> str | None:
    """Best-effort backup (copy) of an existing small output (e.g., .npz)."""
    try:
        if os.path.isfile(path):
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = f"{path}.bak.{ts}"
            shutil.copy2(path, bak)
            return bak
    except Exception:
        return None
    return None


def _quad_width_px(quads: np.ndarray) -> np.ndarray:
    """Approx width (px) of each quad frame."""
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        return np.zeros((int(quads.shape[0]),), dtype=np.float32)
    p0, p1, p2, p3 = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    w_top = np.linalg.norm(p1 - p0, axis=1)
    w_bot = np.linalg.norm(p2 - p3, axis=1)
    return (0.5 * (w_top + w_bot)).astype(np.float32)


def _center_jitter_px(quads: np.ndarray) -> np.ndarray:
    """Per-frame jitter = L2 distance of quad center vs previous frame."""
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        return np.zeros((int(quads.shape[0]),), dtype=np.float32)
    centers = quads.mean(axis=1)
    d = np.linalg.norm(np.diff(centers, axis=0), axis=1).astype(np.float32)
    return np.concatenate([np.zeros((1,), np.float32), d])


def calc_track_metrics(quads: np.ndarray, valid: np.ndarray, conf: np.ndarray, *, min_conf: float) -> dict:
    valid_b = valid.astype(bool)
    n = int(valid_b.size)
    vr = float(valid_b.mean()) if n > 0 else 0.0
    conf_v = conf[valid_b] if (conf is not None and np.any(valid_b)) else np.array([], np.float32)
    jitter = _center_jitter_px(quads)
    w = _quad_width_px(quads)
    w_med = float(np.median(w[valid_b])) if np.any(valid_b) else 0.0
    jitter_p95 = float(np.quantile(jitter[1:][valid_b[1:]], 0.95)) if (jitter.size > 1 and np.any(valid_b[1:])) else 0.0
    jitter_ratio_p95 = float(jitter_p95 / max(1.0, w_med)) if w_med > 0 else float(jitter_p95)
    return {
        "n_frames": int(n),
        "n_valid": int(valid_b.sum()),
        "valid_rate": vr,
        "mean_conf": float(conf_v.mean()) if conf_v.size else None,
        "p10_conf": float(np.quantile(conf_v, 0.10)) if conf_v.size else None,
        "min_conf_threshold": float(min_conf),
        "quad_width_median_px": w_med,
        "jitter_p95_px": jitter_p95,
        "jitter_p95_over_width": jitter_ratio_p95,
    }


def save_metrics_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_metrics_png(path: str, series: dict[str, np.ndarray], title: str = "") -> None:
    """Write a tiny diagnostic plot without matplotlib (OpenCV draw)."""
    W, H = 1200, 360
    M = 40
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (M, M), (W - M, H - M), (220, 220, 220), 1)
    if title:
        cv2.putText(img, title[:80], (M, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    colors = [(30, 30, 30), (0, 128, 255), (255, 0, 128)]
    labels = list(series.keys())
    N = 0
    for v in series.values():
        N = max(N, int(len(v)))
    if N <= 1:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        cv2.imwrite(path, img)
        return

    def norm_y(name: str, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float32)
        if y.size == 0:
            return y
        if name.lower() == "valid":
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.quantile(y, 0.05))
            hi = float(np.quantile(y, 0.95))
            if not np.isfinite(lo):
                lo = float(np.min(y))
            if not np.isfinite(hi):
                hi = float(np.max(y))
            if abs(hi - lo) < 1e-6:
                hi = lo + 1.0
        y = (y - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    for k, name in enumerate(labels):
        y = np.asarray(series[name], np.float32)
        if y.size <= 1:
            continue
        y = norm_y(name, y)
        xs = np.linspace(M, W - M, num=y.size).astype(np.int32)
        ys = (H - M - (y * (H - 2 * M))).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colors[k % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(img, name, (M + 10 + 140 * k, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colors[k % len(colors)], 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, img)



def main() -> int:
    if not HAS_IMGUTILS:
        print("[error] dghs-imgutils is required.")
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="入力動画")
    ap.add_argument("--out", required=True, help="出力 mouth_track.npz")
    ap.add_argument("--debug", default="", help="デバッグ動画出力 (optional)")
    ap.add_argument("--model", default="yolov3", choices=["yolov3"], help="検出モデル")
    ap.add_argument("--device", default="cuda:0", help="cpu / cuda:N / auto (try cuda then cpu)")
    ap.add_argument("--quality", default="custom", choices=["max", "high", "normal", "fast", "custom"], help="解析品質プリセット (customでdet-scale/strideを使用)")
    ap.add_argument("--det-scale", type=float, default=1.0, help="解析時の入力縮小倍率 (1.0=元のまま)")
    ap.add_argument("--stride", type=int, default=1, help="解析を何フレーム毎に実行するか (1=全フレーム)")
    ap.add_argument("--pad", type=float, default=2.1, help="口quadのパディング係数")
    ap.add_argument("--sprite-aspect", type=float, default=1.0, help="スプライトのアスペクト比 (w/h)")
    ap.add_argument("--quad-mode", default="hybrid", choices=["hybrid", "mouth", "bbox"],
                    help="口quad生成方式: hybrid=口点ベース+最低サイズfloor(推奨) / mouth=口点のみ / bbox=顔bbox依存(従来)")
    ap.add_argument("--min-mouth-w-ratio", type=float, default=0.12,
                    help="quad-mode=hybrid時: 口幅の最小値（顔幅に対する比率）")
    ap.add_argument("--min-mouth-w-px", type=float, default=16.0,
                    help="quad-mode=hybrid時: 口幅の最小値（ピクセル）")
    ap.add_argument("--min-conf", type=float, default=0.5, help="最小検出信頼度")
    ap.add_argument("--smooth-cutoff", type=float, default=3.0, help="平滑化のカットオフ周波数 (Hz). 0で無効")
    ap.add_argument("--max-angle-change", type=float, default=15.0, help="1フレームあたりの最大角度変化 (度). 0で無制限")
    ap.add_argument("--valid-min-ratio", type=float, default=0.05, help="有効フレーム率がこれ未満なら平滑化をスキップ（誤補間防止）")
    ap.add_argument("--valid-min-count", type=int, default=5, help="有効フレーム数がこれ未満なら平滑化をスキップ（誤補間防止）")
    ap.add_argument("--ref-sprite-w", type=int, default=128, help="参照スプライト幅 (互換性用)")
    ap.add_argument("--ref-sprite-h", type=int, default=85, help="参照スプライト高さ (互換性用)")
    args = ap.parse_args()

    print(f"[info] detector: dghs-imgutils (dwpose)")
    # No explicit detector initialization needed for dwpose function call, 
    # but models will be downloaded on first run.

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] failed to open video: {args.video}")
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"[info] video: {vid_w}x{vid_h} @ {fps:.2f}fps, {n_frames} frames")

    quality = args.quality
    if quality != "custom":
        presets = {
            "max": (1.0, 1),
            "high": (0.75, 1),
            "normal": (0.5, 2),
            "fast": (0.5, 3),
        }
        det_scale, stride = presets[quality]
        print(f"[info] quality preset: {quality} (det_scale={det_scale}, stride={stride})")
    else:
        det_scale = float(args.det_scale)
        if det_scale <= 0:
            print("[warn] det-scale must be > 0. Using 1.0")
            det_scale = 1.0
        stride = max(1, int(args.stride))
        if stride != int(args.stride) or args.stride < 1:
            print(f"[warn] stride must be >= 1. Using {stride}")
    if det_scale != 1.0:
        det_w = max(2, int(round(vid_w * det_scale)))
        det_h = max(2, int(round(vid_h * det_scale)))
        det_interp = cv2.INTER_AREA if det_scale < 1.0 else cv2.INTER_LINEAR
        print(f"[info] detection scale: {det_scale:.2f} ({det_w}x{det_h})")
    else:
        det_w, det_h = vid_w, vid_h
        det_interp = cv2.INTER_LINEAR
    det_inv = 1.0 / det_scale
    if stride != 1:
        print(f"[info] detection stride: {stride} (process every {stride} frames)")

    # 出力配列
    quads = np.zeros((n_frames, 4, 2), dtype=np.float32)
    valid = np.zeros((n_frames,), dtype=np.uint8)
    confidences = np.zeros((n_frames,), dtype=np.float32)

    # デバッグライター
    debug_writer = None
    if args.debug:
        os.makedirs(os.path.dirname(args.debug) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(
            args.debug, fourcc, fps,
            (ensure_even(vid_w), ensure_even(vid_h))
        )
        if not debug_writer.isOpened():
            print(f"[warn] Failed to open debug video writer: {args.debug}", file=sys.stderr)
            debug_writer = None

    det_count = 0
    print("[info] processing frames...")
    # Progress (TTY only; no new CLI flags)
    use_tqdm = sys.stderr.isatty()
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        tqdm = None  # type: ignore
    it = range(n_frames)
    if use_tqdm and tqdm is not None:
        it = tqdm(it, total=n_frames, desc="Tracking", unit="frame", dynamic_ncols=True)

    for i in it:
        ok, frame = cap.read()
        if not ok:
            break

        # anime-face-detector で検出 (stride対応)
        # 戻り値: [{'bbox': [x1,y1,x2,y2,conf], 'keypoints': (28,3)}, ...]
        do_detect = (i % stride == 0)
        preds = []
        if do_detect:
            det_count += 1
            det_frame = frame
            # cv2 reads as BGR, imgutils expects RGB?
            # dwpose_estimate takes "Input image". Assuming file path or PIL Image or numpy array.
            # If numpy array, usually expects RGB if it's from PIL. cv2 is BGR.
            # Let's convert to RGB to be safe, though many libs handle it.
            # imgutils functions often take file paths or PIL images.
            det_frame_rgb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB)
            
            if det_scale != 1.0:
                det_frame_rgb = cv2.resize(det_frame_rgb, (det_w, det_h), interpolation=det_interp)

            # Detect faces/pose
            # Returns List[OP18KeyPointSet]
            from PIL import Image
            dw_preds = dwpose_estimate(Image.fromarray(det_frame_rgb))
            
            # Convert to local unified format
            preds = []
            for pred in dw_preds:
                # pred is OP18KeyPointSet
                # pred.face is the face keypoints (numpy array)
                # We need to construct a bbox and keypoints array
                
                # Check if face exists
                # OP18KeyPointSet doesn't seem to have a flag, but if .face is valid points...
                # The doc said "OP18_FACE_MIN = 24". 
                # pred.all is (N, 3). 
                
                # Extract face points (68 points).
                # Indices 24 to 91 inclusive (68 points).
                face_kps = pred.all[24:92].copy() # (68, 3)
                
                # Filter out low confidence points? Or just take them.
                # Assuming valid if sum of scores > threshold?
                # For now take all.
                
                # Calculate bbox from face keypoints
                # (x, y, conf)
                xs = face_kps[:, 0]
                ys = face_kps[:, 1]
                
                # Filter out points with 0 confidence if needed?
                # Ideally we want the "Face BBox".
                # If keypoints are all 0, it's invalid.
                if face_kps[:, 2].max() < 0.05:
                    continue
                
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                conf = face_kps[:, 2].mean()
                
                bbox = np.array([x1, y1, x2, y2, conf], dtype=np.float32)
                
                # Scale back if needed
                if det_scale != 1.0:
                    bbox[:4] *= det_inv
                    face_kps[:, :2] *= det_inv
                
                preds.append({'bbox': bbox, 'keypoints': face_kps})


        quad = None
        conf = 0.0

        if len(preds) > 0:
            # 最も大きい(面積が大きい)顔を選択
            best_pred = max(preds, key=lambda p: (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1]))
            bbox = best_pred['bbox']
            keypoints = best_pred['keypoints']

            # 顔検出の信頼度チェック
            if bbox[4] >= args.min_conf:
                if args.quad_mode == "bbox":
                    quad, conf = mouth_quad_from_face_bbox_and_landmarks(
                        bbox, keypoints,
                        sprite_aspect=args.sprite_aspect,
                        pad=args.pad
                    )
                elif args.quad_mode == "mouth":
                    quad, conf = mouth_quad_from_landmarks(
                        keypoints, bbox=None, sprite_aspect=args.sprite_aspect, pad=args.pad
                    )
                else:
                    # hybrid: 口点ベース + 最低サイズfloor
                    quad, conf = mouth_quad_from_landmarks(
                        keypoints, bbox=bbox,
                        sprite_aspect=args.sprite_aspect, pad=args.pad,
                        min_mouth_w_ratio=args.min_mouth_w_ratio,
                        min_mouth_w_px=args.min_mouth_w_px
                    )

        if quad is not None and conf >= args.min_conf:
            quads[i] = quad
            valid[i] = 1
            confidences[i] = conf
        else:
            # 検出失敗: 前フレームの値を仮保持 (後で補間)
            if i > 0:
                quads[i] = quads[i - 1]
            valid[i] = 0
            confidences[i] = 0.0

        # デバッグ描画
        if debug_writer is not None:
            dbg = frame.copy()

            # ランドマーク描画
            if len(preds) > 0:
                for pred in preds:
                    draw_landmarks(dbg, pred['keypoints'])

            # quad描画
            color = (0, 255, 0) if valid[i] else (0, 0, 255)
            draw_quad(dbg, quads[i], color=color, thickness=2)

            # 情報テキスト
            cv2.putText(
                dbg,
                f"frame {i}  valid={int(valid[i])}  conf={confidences[i]:.2f}  faces={len(preds)}  det={int(do_detect)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA
            )

            debug_writer.write(dbg[:ensure_even(vid_h), :ensure_even(vid_w)])

        # 進捗表示
        if (i + 1) % 100 == 0 or i == n_frames - 1:
            attempted = det_count if stride != 1 else (i + 1)
            valid_rate = valid[:i+1].sum() / max(1, attempted)
            msg = f"  frame {i+1}/{n_frames}  valid_rate={valid_rate:.1%}"
            if stride != 1:
                msg += f"  detected={det_count}"
            print(msg)

    cap.release()
    if debug_writer is not None:
        debug_writer.release()
    # ---- post process ----
    # NOTE: valid は『検出に成功した生のフレーム』を表す。平滑化しても上書きしない。
    valid_raw = valid.copy()
    n_valid_raw = int(valid_raw.sum())
    attempted = det_count if stride != 1 else n_frames
    valid_rate_raw = n_valid_raw / max(1, attempted)

    do_smooth = (
        args.smooth_cutoff > 0
        and n_valid_raw >= int(args.valid_min_count)
        and valid_rate_raw >= float(args.valid_min_ratio)
    )

    if do_smooth:
        beta = one_pole_beta(args.smooth_cutoff, fps)
        print(f"[info] smoothing with cutoff={args.smooth_cutoff}Hz, beta={beta:.4f}, max_angle_change={args.max_angle_change}deg")
        quads = smooth_quads_zero_phase(quads, valid_raw, beta, args.max_angle_change)
    else:
        if args.smooth_cutoff > 0:
            print(
                f"[warn] skip smoothing: valid={n_valid_raw}/{n_frames} ({valid_rate_raw:.1%}) "
                f"< min_count={args.valid_min_count} or min_ratio={args.valid_min_ratio:.1%}"
            )

    # 統計
    print(f"[stat] raw_valid={n_valid_raw}/{n_frames} ({valid_rate_raw:.1%})")
    # 保存する valid は raw のまま
    valid = valid_raw

    # 保存
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    backup_copy_if_exists(args.out)
    np.savez_compressed(
        args.out,
        quad=quads.astype(np.float32),
        valid=valid.astype(np.uint8),
        confidence=confidences.astype(np.float32),
        fps=float(fps),
        w=int(vid_w),
        h=int(vid_h),
        ref_sprite_w=int(args.ref_sprite_w),
        ref_sprite_h=int(args.ref_sprite_h),
        pad=float(args.pad),
        det_scale=float(det_scale),
        det_stride=int(stride),
    )
    print(f"saved: {args.out}")
    # Internal metrics (always computed; only saved when --debug is set)
    met = calc_track_metrics(quads, valid, confidences, min_conf=float(args.min_conf))

    # Minimal warning only when something looks off (avoid noisy UX)
    if met["valid_rate"] < 0.80:
        print(f"[warn] tracking quality low: valid_rate={met['valid_rate']:.2f}. "
              f"Try det-scale↑ or stride↓ (or min-conf↓) for this video.")
    elif met["jitter_p95_over_width"] > 0.10:
        print(f"[warn] tracking looks jittery: jitter/width(p95)={met['jitter_p95_over_width']:.3f}. "
              f"Consider smoothing or stricter min-conf.")

    if args.debug:
        dbg_prefix = os.path.splitext(args.debug)[0]
        json_path = dbg_prefix + ".metrics.json"
        png_path = dbg_prefix + ".metrics.png"
        payload = {
            "video": os.path.abspath(args.video),
            "out": os.path.abspath(args.out),
            "metrics": met,
            "params": {
                "quality": str(args.quality),
                "det_scale": float(det_scale),
                "stride": int(stride),
                "pad": float(args.pad),
                "min_conf": float(args.min_conf),
                "smooth_cutoff": float(args.smooth_cutoff),
            },
        }
        try:
            save_metrics_json(json_path, payload)
            save_metrics_png(
                png_path,
                {
                    "confidence": confidences.astype(np.float32),
                    "valid": valid.astype(np.float32),
                    "jitter_px": _center_jitter_px(quads).astype(np.float32),
                },
                title="track metrics",
            )
            print(f"[debug] metrics: {json_path}")
        except Exception as e:
            print(f"[warn] failed to write debug metrics: {e}")


    if args.debug:
        print(f"debug: {args.debug}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
