#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_erase_tuner_gui.py

目的
----
アニメ画像（静止画）に対して、
1) 顔検出＋口ランドマークから口quad推定（anime-face-detector）
2) 口消し処理（erase_mouth_offline.py のロジック相当：正規化パッチ上でinpaint→逆変換→フェザー合成）
をGUIで素早く試行錯誤できるツール。

特徴
- 画像1枚で「口消し」のパラメータ（coverage / mask / ring / feather / inpaint等）を調整し、結果を即プレビュー
- フォルダ読み込み（Prev/Next）で複数キャラを連続テスト
- 検出が不安定なときは det-scale / min-conf / pad / sprite-aspect を調整

依存
- opencv-python
- numpy
- pillow
- anime-face-detector（mmdet/mmpose 依存）
- 可能なら同じフォルダに erase_mouth_offline.py があること（無い場合は最小実装で動くようにfallback）

使い方
------
python mouth_erase_tuner_gui.py
"""

from __future__ import annotations

import os
import sys
import json
import math
import glob
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ----------------------------
# Optional import: drag & drop (tkinterdnd2)
# ----------------------------
_HAS_TK_DND = False
_TK_DND_IMPORT_ERR = None
try:
    # pip install tkinterdnd2
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _HAS_TK_DND = True
except Exception as e:
    _HAS_TK_DND = False
    _TK_DND_IMPORT_ERR = e
    DND_FILES = None  # type: ignore
    TkinterDnD = None  # type: ignore

# Base Tk class (with optional DnD support)
BaseTk = TkinterDnD.Tk if _HAS_TK_DND else tk.Tk

import numpy as np

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV (cv2) が必要です: pip install opencv-python") from e

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise RuntimeError("Pillow が必要です: pip install pillow") from e


# ----------------------------
# Optional import: erase_mouth_offline helpers
# ----------------------------
_ERASE_HELPERS_OK = False
_ERASE_IMPORT_ERR = None
try:
    # 同フォルダ優先
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if _BASE_DIR not in sys.path:
        sys.path.insert(0, _BASE_DIR)
    import erase_mouth_offline as emo  # type: ignore
    _ERASE_HELPERS_OK = True
except Exception as e:
    _ERASE_HELPERS_OK = False
    _ERASE_IMPORT_ERR = e
    emo = None  # type: ignore


# ----------------------------
# Optional import: anime-face-detector
# ----------------------------
_HAS_ANIME_DETECTOR = False
_ANIME_IMPORT_ERR = None
try:
    from anime_face_detector import create_detector  # type: ignore
    _HAS_ANIME_DETECTOR = True
except Exception as e:
    _HAS_ANIME_DETECTOR = False
    _ANIME_IMPORT_ERR = e
    create_detector = None  # type: ignore


LAST_SESSION_FILE = ".mouth_erase_tuner_last_session.json"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# face landmarks indices (same as face_track_anime_detector.py)
MOUTH_OUTLINE = [24, 25, 26, 27]  # 口の4点


def _imread_jp(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Windowsで日本語パスを含む画像を読み込む。
    cv2.imread()は日本語パスを正しく処理できないため、numpy.fromfile()とcv2.imdecode()を使用。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None


def _imwrite_jp(path: str, img: np.ndarray, params: Optional[List[int]] = None) -> bool:
    """
    Windowsで日本語パスを含む画像を保存する。
    cv2.imwrite()は日本語パスを正しく処理できないため、cv2.imencode()とnumpy.ndarray.tofile()を使用。
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.jpg' or ext == '.jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        elif ext == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
        else:
            encode_params = []
        
        if params is not None:
            encode_params = params
        
        success, data = cv2.imencode(ext, img, encode_params)
        if not success:
            return False
        data.tofile(path)
        return True
    except Exception:
        return False


def _ensure_even_ge2(n: int) -> int:
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n + 1)


def _quad_wh(quad: np.ndarray) -> Tuple[float, float]:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = float(np.linalg.norm(q[1] - q[0]))
    h = float(np.linalg.norm(q[3] - q[0]))
    return w, h


def _draw_quad(img_bgr: np.ndarray, quad: np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = img_bgr.copy()
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=int(thickness), lineType=cv2.LINE_AA)
    return out


def _estimate_face_rotation_deg(keypoints: np.ndarray) -> float:
    """
    face_track_anime_detector.py と同等の簡易推定。
    - 左目(11-16)と右目(17-22)中心を結ぶ角度
    """
    if keypoints.shape[0] < 23:
        return 0.0
    left_eye = keypoints[11:17, :2]
    right_eye = keypoints[17:23, :2]
    le = left_eye.mean(axis=0)
    re = right_eye.mean(axis=0)
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return float(np.degrees(np.arctan2(dy, dx)))


def _mouth_quad_from_face_bbox_and_landmarks(
    bbox: np.ndarray,
    keypoints: np.ndarray,
    sprite_aspect: float = 1.0,
    pad: float = 2.1
) -> Tuple[np.ndarray, float]:
    """
    face_track_anime_detector.py のロジックを踏襲。
    bbox: [x1,y1,x2,y2,conf]
    keypoints: (28,3) 28点ランドマーク想定
    Returns:
        quad (4,2), conf (mouth pts平均)
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]
    xy = mouth_pts[:, :2]
    conf = float(mouth_pts[:, 2].mean())

    cx, cy = xy.mean(axis=0)

    x1, y1, x2, y2 = bbox[:4].astype(np.float32)
    face_w = float(max(1.0, x2 - x1))
    face_h = float(max(1.0, y2 - y1))

    # 口のサイズは顔サイズから決め打ち寄り（アニメ向け）
    # pad をかけて大きめに
    mouth_w = face_w * 0.35 * float(pad)
    mouth_h = (mouth_w / max(0.25, min(4.0, float(sprite_aspect)))) * 0.75

    # 顔傾きを推定して回転
    angle_deg = _estimate_face_rotation_deg(keypoints)
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    hw, hh = mouth_w / 2.0, mouth_h / 2.0
    quad_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    quad = (quad_local @ R.T) + np.array([cx, cy], dtype=np.float32)
    return quad.astype(np.float32), conf


def _mouth_quad_from_landmarks(
    keypoints: np.ndarray,
    *,
    bbox: Optional[np.ndarray] = None,
    sprite_aspect: float = 1.0,
    pad: float = 2.1,
    min_mouth_w_ratio: float = 0.0,
    min_mouth_w_px: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    face_track_anime_detector.py と同じロジック。
    hybridモード（口点ベース + 最低サイズfloor）を実装。
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]
    xy = mouth_pts[:, :2].astype(np.float32)
    conf = float(mouth_pts[:, 2].mean())
    cx, cy = xy.mean(axis=0)

    angle_deg = _estimate_face_rotation_deg(keypoints)
    ang = math.radians(float(angle_deg))
    ca, sa = math.cos(ang), math.sin(ang)
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


def _mouth_quad_from_mouth_landmarks(
    keypoints: np.ndarray,
    sprite_aspect: float = 1.0,
    pad: float = 2.1,
    min_w_px: float = 8.0,
    min_h_px: float = 6.0,
) -> Tuple[np.ndarray, float]:
    """
    口ランドマーク(4点)から口の幅・高さを推定してquadを作る（顔サイズ依存を弱める）。
    - 回転は目ランドマークから推定した顔傾きを使用。
    - 口が閉じて高さが極端に小さい場合は sprite_aspect を下限として高さを確保。
    """
    mouth_pts = keypoints[MOUTH_OUTLINE]
    xy = mouth_pts[:, :2].astype(np.float32)
    conf = float(mouth_pts[:, 2].mean())

    c = xy.mean(axis=0)

    angle_deg = _estimate_face_rotation_deg(keypoints)
    ang = math.radians(float(angle_deg))
    ca, sa = math.cos(ang), math.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float32)

    # row-vector なので逆回転は @R
    xy_local = (xy - c) @ R

    x_min, y_min = xy_local.min(axis=0)
    x_max, y_max = xy_local.max(axis=0)
    w = float(max(min_w_px, x_max - x_min))
    h = float(max(min_h_px, y_max - y_min))

    w *= float(pad)
    h *= float(pad)

    asp = float(max(0.25, min(4.0, float(sprite_aspect))))
    h = float(max(h, w / asp))

    hw, hh = w / 2.0, h / 2.0
    quad_local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    quad = (quad_local @ R.T) + c.astype(np.float32)
    return quad.astype(np.float32), conf


def _order_polygon_points_ccw(pts: np.ndarray) -> np.ndarray:
    """点を重心まわりの角度で並べ替え（CCW / 画像座標でもOK）。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    return pts[order]


def _make_mouth_polygon_mask(
    w: int,
    h: int,
    pts_norm: np.ndarray,
    expand_x: float = 1.0,
    expand_y: float = 1.0,
    center_y_offset_px: int = 0,
    top_clip_frac: float = 0.82,
) -> np.ndarray:
    """
    口ランドマーク点列からポリゴンmaskを生成（0/255のuint8）。
    pts_norm: (K,2) normalized patch座標上の点
    """
    mask = np.zeros((int(h), int(w)), dtype=np.uint8)
    pts = np.asarray(pts_norm, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 3:
        return mask

    c = pts.mean(axis=0)
    d = pts - c
    d[:, 0] *= float(expand_x)
    d[:, 1] *= float(expand_y)
    pts2 = c + d
    pts2[:, 1] += float(center_y_offset_px)

    pts2 = _order_polygon_points_ccw(pts2)
    pts_i = np.round(pts2).astype(np.int32)
    pts_i[:, 0] = np.clip(pts_i[:, 0], 0, int(w) - 1)
    pts_i[:, 1] = np.clip(pts_i[:, 1], 0, int(h) - 1)
    cv2.fillConvexPoly(mask, pts_i, 255)

    top_clip_frac = float(np.clip(top_clip_frac, 0.0, 1.0))
    if top_clip_frac > 0:
        y_min = float(pts2[:, 1].min())
        y_max = float(pts2[:, 1].max())
        cy = float((y_min + y_max) * 0.5)
        ry = float(max(1.0, (y_max - y_min) * 0.5))
        clip_y = int(round(cy - ry * top_clip_frac))
        if clip_y > 0:
            mask[:clip_y, :] = 0
    return mask


# ----------------------------
# Fallback erase helpers (when erase_mouth_offline.py not importable)
# ----------------------------
def _make_mouth_mask(w: int, h: int, rx: int, ry: int, center_y_offset_px: int = 0, top_clip_frac: float = 0.82) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2 + int(center_y_offset_px)
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)

    top_clip_frac = float(np.clip(top_clip_frac, 0.6, 1.0))
    clip_y = int(round(cy - ry * top_clip_frac))
    clip_y = int(np.clip(clip_y, 0, h))
    if clip_y > 0:
        mask[:clip_y, :] = 0
    return mask


def _feather_mask(inner_u8: np.ndarray, dilate_px: int = 10, feather_px: int = 20) -> np.ndarray:
    m = inner_u8.copy()
    if dilate_px > 0:
        k = int(max(1, dilate_px))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ensure_even_ge2(k) + 1, _ensure_even_ge2(k) + 1))
        m = cv2.dilate(m, kernel, iterations=1)
    m = m.astype(np.float32) / 255.0
    if feather_px > 0:
        k = int(max(1, feather_px))
        k = _ensure_even_ge2(2 * k + 1)  # odd size
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = np.clip(m, 0.0, 1.0)
    return m.astype(np.float32)


def _warp_frame_to_norm(frame_bgr: np.ndarray, quad: np.ndarray, norm_w: int, norm_h: int) -> np.ndarray:
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(frame_bgr, H, (norm_w, norm_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    return patch


def _quad_bbox(quad: np.ndarray, pad_px: int = 0) -> Tuple[int, int, int, int]:
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0 = int(np.floor(q[:, 0].min())) - int(pad_px)
    y0 = int(np.floor(q[:, 1].min())) - int(pad_px)
    x1 = int(np.ceil(q[:, 0].max())) + int(pad_px)
    y1 = int(np.ceil(q[:, 1].max())) + int(pad_px)
    return x0, y0, x1, y1


def _warp_norm_to_bbox(patch_bgr: np.ndarray, mask_f: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    正規化パッチを、quadのbbox範囲にだけ逆ワープして返す（パフォーマンス用）
    """
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    x0, y0, x1, y1 = _quad_bbox(q, pad_px=2)
    w = int(max(2, x1 - x0))
    h = int(max(2, y1 - y0))

    dst = q - np.array([x0, y0], dtype=np.float32)
    src = np.array([[0, 0], [patch_bgr.shape[1] - 1, 0], [patch_bgr.shape[1] - 1, patch_bgr.shape[0] - 1], [0, patch_bgr.shape[0] - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)

    warped_patch = cv2.warpPerspective(patch_bgr, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    warped_mask = cv2.warpPerspective(mask_f.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_mask = np.clip(warped_mask, 0.0, 1.0).astype(np.float32)
    return warped_patch, warped_mask, x0, y0


def _alpha_blend_roi(dst_bgr: np.ndarray, src_bgr: np.ndarray, alpha_f: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = dst_bgr.copy()
    H, W = out.shape[:2]
    h, w = src_bgr.shape[:2]
    xA = max(0, x0); yA = max(0, y0)
    xB = min(W, x0 + w); yB = min(H, y0 + h)
    if xA >= xB or yA >= yB:
        return out
    sx0 = xA - x0; sy0 = yA - y0
    sx1 = sx0 + (xB - xA); sy1 = sy0 + (yB - yA)

    roi = out[yA:yB, xA:xB].astype(np.float32)
    src = src_bgr[sy0:sy1, sx0:sx1].astype(np.float32)
    a = alpha_f[sy0:sy1, sx0:sx1].astype(np.float32)[..., None]
    out[yA:yB, xA:xB] = np.clip(src * a + roi * (1.0 - a), 0, 255).astype(np.uint8)
    return out


def _get_erase_helpers():
    """emo が使えればそちら優先、無理ならfallback"""
    if _ERASE_HELPERS_OK and emo is not None:
        def _alpha_blend_roi_ext(dst_bgr: np.ndarray, src_bgr: np.ndarray, mask_f: np.ndarray, x0: int, y0: int) -> np.ndarray:
            """Blend src patch into full dst frame at (x0,y0) using mask.

            erase_mouth_offline.alpha_blend_roi() は ROI 同士（同サイズ）のみで blend する実装なので、
            GUI側では bbox / 画面外はみ出しの境界処理を行ってから ROI を渡す。
            """
            out = dst_bgr.copy()
            H, W = out.shape[:2]
            bh, bw = src_bgr.shape[:2]

            rx0 = max(0, int(x0))
            ry0 = max(0, int(y0))
            rx1 = min(W, int(x0) + bw)
            ry1 = min(H, int(y0) + bh)
            if rx0 >= rx1 or ry0 >= ry1:
                return out

            sx0 = rx0 - int(x0)
            sy0 = ry0 - int(y0)
            sx1 = sx0 + (rx1 - rx0)
            sy1 = sy0 + (ry1 - ry0)

            roi = out[ry0:ry1, rx0:rx1]
            src = src_bgr[sy0:sy1, sx0:sx1]
            msk = mask_f[sy0:sy1, sx0:sx1]
            if msk.ndim == 2:
                msk = msk[:, :, None]

            out[ry0:ry1, rx0:rx1] = emo.alpha_blend_roi(roi, src, msk)
            return out

        return {
            "make_mouth_mask": emo.make_mouth_mask,
            "feather_mask": emo.feather_mask,
            "warp_frame_to_norm": emo.warp_frame_to_norm,
            "warp_norm_to_bbox": emo.warp_norm_to_bbox,
            "alpha_blend_roi": _alpha_blend_roi_ext,
            "fit_plane_2d": getattr(emo, "fit_plane_2d", None),
            "eval_plane": getattr(emo, "eval_plane", None),
            "ensure_even_ge2": emo.ensure_even_ge2,
        }
    return {
        "make_mouth_mask": _make_mouth_mask,
        "feather_mask": _feather_mask,
        "warp_frame_to_norm": _warp_frame_to_norm,
        "warp_norm_to_bbox": _warp_norm_to_bbox,
        "alpha_blend_roi": _alpha_blend_roi,
        "fit_plane_2d": None,
        "eval_plane": None,
        "ensure_even_ge2": _ensure_even_ge2,
    }




def _fit_plane_2d_fallback(L: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    """
    L ~ a*x + b*y + c を最小二乗で当てる簡易版。
    xs,ys は画素座標（0..w-1 / 0..h-1）の配列。
    """
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    L = L.astype(np.float32)
    A = np.stack([xs, ys, np.ones_like(xs)], axis=1)  # (N,3)
    # lstsq
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    a, b, c = float(coef[0]), float(coef[1]), float(coef[2])
    return a, b, c


def _eval_plane_fallback(a: float, b: float, c: float, w: int, h: int) -> np.ndarray:
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return (a * X + b * Y + c).astype(np.float32)
@dataclass
class EraseParams:
    # detector
    model: str = "yolov3"
    device: str = "auto"
    det_scale: float = 1.0
    min_conf: float = 0.5
    sprite_aspect: float = 1.0
    pad: float = 2.1

    # quad generation
    quad_mode: str = "hybrid"
    min_mouth_w_ratio: float = 0.12
    min_mouth_w_px: float = 16.0
    # 後方互換性のため残す（quad_mode="mouth"相当）
    quad_from_landmarks: bool = True

    # mask shape
    # True: 口ランドマーク形状(点列)からポリゴンmaskを作る（楕円より"口形"に寄せる）
    mask_from_landmarks: bool = False

    # norm patch sizing
    oversample: float = 1.2
    norm_w: int = 0
    norm_h: int = 0

    # auto coverage
    use_coverage: bool = True
    coverage: float = 0.60  # 0..1

    # manual mask params
    mask_scale_x: float = 0.62
    mask_scale_y: float = 0.62
    ring_px: int = 18
    dilate_px: int = 10
    feather_px: int = 20
    inpaint_radius: float = 5.0

    # nose guard
    top_clip_frac: float = 0.82
    center_y_off_frac: float = 0.05  # norm_h * frac

    # ring match (shade/chroma)
    match_ring: bool = True

    # visualization
    draw_quad: bool = True


def coverage_to_params(norm_h: int, cov: float) -> Dict[str, Any]:
    """
    erase_mouth_offline.py の coverage チューニングと同等。
    cov: 0..1
    """
    cov = float(np.clip(cov, 0.0, 1.0))
    mask_scale_x = 0.50 + 0.18 * cov   # 0.50..0.68
    mask_scale_y = 0.44 + 0.14 * cov   # 0.44..0.58
    ring_px = int(round(16 + 10 * cov))  # 16..26
    dilate_px = int(round(8 + 8 * cov))  # 8..16
    feather_px = int(round(18 + 10 * cov))  # 18..28
    inpaint_radius = 4.0 + 4.0 * cov    # 4..8
    top_clip_frac = float(0.84 - 0.06 * cov)  # 0.84..0.78
    center_y_off = float(0.05 + 0.01 * cov)  # frac
    return dict(
        mask_scale_x=mask_scale_x,
        mask_scale_y=mask_scale_y,
        ring_px=ring_px,
        dilate_px=dilate_px,
        feather_px=feather_px,
        inpaint_radius=inpaint_radius,
        top_clip_frac=top_clip_frac,
        center_y_off_frac=center_y_off,
    )


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def u8_to_pil_gray(u8: np.ndarray) -> Image.Image:
    if u8.ndim == 2:
        return Image.fromarray(u8)
    return Image.fromarray(u8[:, :, 0])


def f32_to_pil_gray01(f: np.ndarray) -> Image.Image:
    g = np.clip(f, 0.0, 1.0)
    u8 = (g * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(u8)


def fit_to_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    if nw < 1: nw = 1
    if nh < 1: nh = 1
    return img.resize((nw, nh), Image.BILINEAR)


class MouthEraseTunerGUI(BaseTk):
    def __init__(self):
        super().__init__()
        self.title("Mouth Erase Tuner (静止画)")
        self.geometry("1200x750")

        self._helpers = _get_erase_helpers()

        self.params = EraseParams()
        self._detector = None
        self._detector_device_actual = None

        # image list
        self.current_path: str = ""
        self.folder_paths: List[str] = []
        self.folder_index: int = -1

        # last detection
        self._preds: List[Dict[str, Any]] = []
        self._selected_face_idx = 0
        self._quad: Optional[np.ndarray] = None
        self._mouth_conf: float = 0.0
        self._mouth_xy: Optional[np.ndarray] = None

        self._orig_bgr: Optional[np.ndarray] = None
        self._out_bgr: Optional[np.ndarray] = None
        self._patch_bgr: Optional[np.ndarray] = None
        self._clean_bgr: Optional[np.ndarray] = None
        self._inner_mask_u8: Optional[np.ndarray] = None
        self._feather_mask_f: Optional[np.ndarray] = None

        self._update_job = None

        self._build_ui()
        self._setup_dnd()
        self._load_last_session()

        if not _HAS_ANIME_DETECTOR:
            self._log(f"[warn] anime-face-detector が見つかりません。検出機能は使えません。\n{_ANIME_IMPORT_ERR}\n")
        if not _ERASE_HELPERS_OK:
            self._log(f"[warn] erase_mouth_offline.py を import できませんでした。簡易実装で動作します。\n{_ERASE_IMPORT_ERR}\n")

    # ---------- UI ----------
    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        # ---- left: file controls ----
        file_box = ttk.LabelFrame(left, text="画像", padding=8)
        file_box.pack(fill="x", pady=6)

        ttk.Button(file_box, text="画像を開く", command=self.open_image).pack(fill="x", pady=2)
        ttk.Button(file_box, text="フォルダを開く", command=self.open_folder).pack(fill="x", pady=2)

        nav = ttk.Frame(file_box)
        nav.pack(fill="x", pady=2)
        ttk.Button(nav, text="◀ Prev", command=self.prev_image).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(nav, text="Next ▶", command=self.next_image).pack(side="left", expand=True, fill="x", padx=2)

        self.path_var = tk.StringVar(value="")
        ttk.Label(file_box, textvariable=self.path_var, wraplength=320, foreground="#444").pack(fill="x", pady=(6, 2))

        # ---- left: detection settings ----
        det_box = ttk.LabelFrame(left, text="検出設定 (anime-face-detector)", padding=8)
        det_box.pack(fill="x", pady=6)

        self.device_var = tk.StringVar(value=self.params.device)
        self.det_scale_var = tk.DoubleVar(value=self.params.det_scale)
        self.min_conf_var = tk.DoubleVar(value=self.params.min_conf)
        self.sprite_aspect_var = tk.DoubleVar(value=self.params.sprite_aspect)
        self.pad_var = tk.DoubleVar(value=self.params.pad)
        self.quad_mode_var = tk.StringVar(value=self.params.quad_mode)
        self.min_mouth_w_ratio_var = tk.DoubleVar(value=self.params.min_mouth_w_ratio)
        self.min_mouth_w_px_var = tk.DoubleVar(value=self.params.min_mouth_w_px)
        self.quad_from_landmarks_var = tk.BooleanVar(value=bool(self.params.quad_from_landmarks))
        self.mask_from_landmarks_var = tk.BooleanVar(value=bool(self.params.mask_from_landmarks))

        # device
        dev_row = ttk.Frame(det_box); dev_row.pack(fill="x", pady=2)
        ttk.Label(dev_row, text="device", width=10).pack(side="left")
        ttk.Combobox(dev_row, textvariable=self.device_var, values=["auto", "cpu", "cuda:0", "cuda:1"], state="readonly", width=12).pack(side="left")
        ttk.Button(dev_row, text="検出器再作成", command=self._reset_detector).pack(side="left", padx=6)

        # det_scale
        self._slider(det_box, "det_scale", self.det_scale_var, 0.5, 1.5, 0.01)
        self._slider(det_box, "min_conf", self.min_conf_var, 0.0, 0.99, 0.01)
        self._slider(det_box, "sprite_aspect", self.sprite_aspect_var, 0.7, 3.0, 0.05)
        self._slider(det_box, "pad", self.pad_var, 1.0, 3.0, 0.05)

        # quad_mode
        quad_mode_row = ttk.Frame(det_box); quad_mode_row.pack(fill="x", pady=2)
        ttk.Label(quad_mode_row, text="quad_mode", width=10).pack(side="left")
        ttk.Combobox(quad_mode_row, textvariable=self.quad_mode_var, values=["hybrid", "mouth", "bbox"], state="readonly", width=12).pack(side="left")
        self._slider(det_box, "min_mouth_w_ratio", self.min_mouth_w_ratio_var, 0.0, 0.40, 0.01)
        self._slider(det_box, "min_mouth_w_px", self.min_mouth_w_px_var, 0.0, 64.0, 1.0)

        opt_row = ttk.Frame(det_box); opt_row.pack(fill="x", pady=4)
        ttk.Checkbutton(
            opt_row,
            text="quad: 口ランドマークから推定",
            variable=self.quad_from_landmarks_var,
            command=self.detect_and_preview,
        ).pack(side="left")
        ttk.Checkbutton(
            opt_row,
            text="mask: 口ランドマーク形状(ポリゴン)",
            variable=self.mask_from_landmarks_var,
            command=self.detect_and_preview,
        ).pack(side="left", padx=8)

        # faces dropdown
        face_row = ttk.Frame(det_box); face_row.pack(fill="x", pady=4)
        ttk.Label(face_row, text="face", width=10).pack(side="left")
        self.face_var = tk.StringVar(value="(not detected)")
        self.face_combo = ttk.Combobox(face_row, textvariable=self.face_var, values=[], state="readonly", width=28)
        self.face_combo.pack(side="left")
        self.face_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_select_face())

        ttk.Button(det_box, text="検出→口消しプレビュー", command=self.detect_and_preview).pack(fill="x", pady=(6, 2))

        # ---- left: erase settings ----
        erase_box = ttk.LabelFrame(left, text="口消し設定 (normalized inpaint)", padding=8)
        erase_box.pack(fill="x", pady=6)

        self.use_coverage_var = tk.BooleanVar(value=self.params.use_coverage)
        self.coverage_var = tk.DoubleVar(value=self.params.coverage)

        cov_row = ttk.Frame(erase_box); cov_row.pack(fill="x", pady=(2, 2))
        ttk.Checkbutton(
            cov_row,
            text="coverage で自動チューニング",
            variable=self.use_coverage_var,
            command=self._schedule_update
        ).pack(side="left")
        self._slider(erase_box, "coverage", self.coverage_var, 0.0, 1.0, 0.01)

        # manual params
        self.mask_scale_x_var = tk.DoubleVar(value=self.params.mask_scale_x)
        self.mask_scale_y_var = tk.DoubleVar(value=self.params.mask_scale_y)
        self.ring_px_var = tk.IntVar(value=self.params.ring_px)
        self.dilate_px_var = tk.IntVar(value=self.params.dilate_px)
        self.feather_px_var = tk.IntVar(value=self.params.feather_px)
        self.match_ring_var = tk.BooleanVar(value=self.params.match_ring)
        self.inpaint_radius_var = tk.DoubleVar(value=self.params.inpaint_radius)
        self.top_clip_frac_var = tk.DoubleVar(value=self.params.top_clip_frac)
        self.center_y_off_frac_var = tk.DoubleVar(value=self.params.center_y_off_frac)

        self._slider(erase_box, "mask_scale_x", self.mask_scale_x_var, 0.35, 0.95, 0.01)
        self._slider(erase_box, "mask_scale_y", self.mask_scale_y_var, 0.30, 0.95, 0.01)
        self._slider(erase_box, "ring_px", self.ring_px_var, 0, 60, 1)
        self._slider(erase_box, "dilate_px", self.dilate_px_var, 0, 60, 1)
        self._slider(erase_box, "feather_px", self.feather_px_var, 0, 80, 1)
        self._slider(erase_box, "inpaint_radius", self.inpaint_radius_var, 0.0, 15.0, 0.5)
        ttk.Checkbutton(erase_box, text="ringから色/陰影を合わせる", variable=self.match_ring_var, command=self._schedule_update).pack(anchor="w", pady=(4,2))

        ttk.Separator(erase_box).pack(fill="x", pady=6)
        self._slider(erase_box, "top_clip_frac", self.top_clip_frac_var, 0.60, 1.0, 0.01)
        self._slider(erase_box, "center_y_off_frac", self.center_y_off_frac_var, 0.00, 0.20, 0.01)

        # norm patch size
        ttk.Separator(erase_box).pack(fill="x", pady=6)
        self.oversample_var = tk.DoubleVar(value=self.params.oversample)
        self.norm_w_var = tk.IntVar(value=self.params.norm_w)
        self.norm_h_var = tk.IntVar(value=self.params.norm_h)
        self._slider(erase_box, "oversample", self.oversample_var, 0.8, 2.0, 0.05)
        self._entry_row(erase_box, "norm_w(0=auto)", self.norm_w_var)
        self._entry_row(erase_box, "norm_h(0=auto)", self.norm_h_var)

        # viz
        viz_box = ttk.LabelFrame(left, text="表示", padding=8)
        viz_box.pack(fill="x", pady=6)
        self.draw_quad_var = tk.BooleanVar(value=self.params.draw_quad)
        ttk.Checkbutton(viz_box, text="quadを描画", variable=self.draw_quad_var, command=self._schedule_update).pack(anchor="w")
        ttk.Button(viz_box, text="画像として保存", command=self.save_output).pack(fill="x", pady=(6, 2))
        ttk.Button(viz_box, text="設定をJSON保存", command=self.save_settings_json).pack(fill="x", pady=2)
        ttk.Button(viz_box, text="設定をJSON読込", command=self.load_settings_json).pack(fill="x", pady=2)

        # log
        log_box = ttk.LabelFrame(left, text="ログ", padding=8)
        log_box.pack(fill="both", expand=True, pady=6)
        self.log = tk.Text(log_box, width=42, height=10)
        self.log.pack(fill="both", expand=True)

        # ---- right: preview notebook ----
        nb = ttk.Notebook(right)
        nb.grid(row=0, column=0, sticky="nsew")

        tab_res = ttk.Frame(nb)
        tab_mask = ttk.Frame(nb)
        tab_patch = ttk.Frame(nb)

        nb.add(tab_res, text="結果")
        nb.add(tab_mask, text="マスク")
        nb.add(tab_patch, text="パッチ")

        # result: before/after
        tab_res.columnconfigure(0, weight=1)
        tab_res.columnconfigure(1, weight=1)
        tab_res.rowconfigure(0, weight=1)
        self.canvas_before = tk.Canvas(tab_res, bg="#222")
        self.canvas_after = tk.Canvas(tab_res, bg="#222")
        self.canvas_before.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.canvas_after.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # mask tab
        tab_mask.columnconfigure(0, weight=1)
        tab_mask.columnconfigure(1, weight=1)
        tab_mask.rowconfigure(0, weight=1)
        self.canvas_mask_inner = tk.Canvas(tab_mask, bg="#222")
        self.canvas_mask_feather = tk.Canvas(tab_mask, bg="#222")
        self.canvas_mask_inner.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.canvas_mask_feather.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # patch tab
        tab_patch.columnconfigure(0, weight=1)
        tab_patch.columnconfigure(1, weight=1)
        tab_patch.rowconfigure(0, weight=1)
        self.canvas_patch = tk.Canvas(tab_patch, bg="#222")
        self.canvas_clean = tk.Canvas(tab_patch, bg="#222")
        self.canvas_patch.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.canvas_clean.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # store PhotoImage refs
        self._tk_imgs = {}

        # bind change events (debounced)
        for v in [
            self.device_var, self.det_scale_var, self.min_conf_var, self.sprite_aspect_var, self.pad_var,
            self.quad_mode_var, self.min_mouth_w_ratio_var, self.min_mouth_w_px_var,
            self.use_coverage_var, self.coverage_var,
            self.mask_scale_x_var, self.mask_scale_y_var, self.ring_px_var, self.dilate_px_var,
            self.feather_px_var, self.inpaint_radius_var, self.top_clip_frac_var, self.center_y_off_frac_var,
            self.oversample_var, self.norm_w_var, self.norm_h_var,
            self.match_ring_var,
            self.draw_quad_var,
        ]:
            try:
                v.trace_add("write", lambda *_: self._schedule_update())
            except Exception:
                pass

    # ---------- drag & drop ----------
    def _setup_dnd(self):
        """Drag & Drop を有効化（tkinterdnd2 が入っている場合のみ）"""
        if not _HAS_TK_DND:
            # 依存を入れれば有効になる。無くてもGUI自体は動く
            self._log("[info] Drag&Dropを使うには: pip install tkinterdnd2\n")
            return
        try:
            widgets = [
                self,
                self.canvas_before, self.canvas_after,
                self.canvas_mask_inner, self.canvas_mask_feather,
                self.canvas_patch, self.canvas_clean,
            ]
            for w in widgets:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop_files)
            self._log("[info] Drag&Drop enabled: 画像/フォルダをウィンドウにドロップできます\n")
        except Exception as e:
            self._log(f"[warn] Drag&Dropの初期化に失敗: {e}\n")

    def _on_drop_files(self, event):
        """DND_FILES を受け取って画像/フォルダを開く"""
        try:
            raw = self.tk.splitlist(event.data)
        except Exception:
            raw = [event.data]

        paths: List[str] = []
        for p in raw:
            p = str(p).strip()
            if not p:
                continue

            # file:///C:/... 形式を通常パスへ
            if p.startswith("file://"):
                try:
                    import urllib.parse
                    p2 = urllib.parse.unquote(p)
                    if p2.startswith("file:///"):
                        p2 = p2[8:]
                    else:
                        p2 = p2[7:]
                    # Windows drive: /C:/... → C:/...
                    if os.name == "nt" and len(p2) >= 3 and p2[0] == "/" and p2[2] == ":":
                        p2 = p2[1:]
                    p = p2
                except Exception:
                    pass

            p = p.strip("{}")
            paths.append(os.path.abspath(p))

        if not paths:
            return

        # フォルダが混ざっていればフォルダ優先
        for p in paths:
            if os.path.isdir(p):
                self.load_folder(p)
                return

        img_paths = [p for p in paths if os.path.isfile(p) and p.lower().endswith(IMG_EXTS)]
        if not img_paths:
            messagebox.showwarning("対応外", "画像ファイルまたはフォルダをドロップしてください。")
            return

        # 複数画像ドロップなら、それをリストとしてPrev/Next可能に
        if len(img_paths) >= 2:
            self.folder_paths = sorted(img_paths)
            self.folder_index = 0
            self.load_image(self.folder_paths[self.folder_index])
        else:
            self.load_image(img_paths[0])

    def _slider(self, parent, name: str, var, vmin: float, vmax: float, step: float):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=name, width=16).pack(side="left")
        scale = ttk.Scale(row, from_=vmin, to=vmax, orient="horizontal", variable=var, length=180)
        scale.pack(side="left", padx=6)
        val = ttk.Label(row, width=8, anchor="e")
        val.pack(side="left")

        def _update_label(*_):
            try:
                if isinstance(var, tk.IntVar):
                    val.configure(text=str(int(var.get())))
                else:
                    # stepが整数なら整数表示、それ以外は小数
                    if float(step).is_integer():
                        val.configure(text=f"{float(var.get()):.0f}")
                    else:
                        val.configure(text=f"{float(var.get()):.2f}")
            except Exception:
                pass

        _update_label()
        try:
            var.trace_add("write", _update_label)
        except Exception:
            pass

    def _entry_row(self, parent, label: str, var):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        e = ttk.Entry(row, textvariable=var, width=10)
        e.pack(side="left", padx=6)
        return e

    def _log(self, s: str):
        self.log.insert("end", s)
        self.log.see("end")

    # ---------- session ----------
    def _save_last_session(self):
        data = {
            "current_path": self.current_path,
            "folder_paths": self.folder_paths[:2000],
            "folder_index": self.folder_index,
            "params": self._collect_params_dict(),
        }
        try:
            with open(LAST_SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_last_session(self):
        try:
            if not os.path.isfile(LAST_SESSION_FILE):
                return
            with open(LAST_SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data.get("params"), dict):
                self._apply_params_dict(data["params"])
            if data.get("current_path") and os.path.isfile(data["current_path"]):
                self.load_image(data["current_path"])
            elif data.get("folder_paths"):
                self.folder_paths = [p for p in data["folder_paths"] if os.path.isfile(p)]
                self.folder_index = int(data.get("folder_index", -1))
                if 0 <= self.folder_index < len(self.folder_paths):
                    self.load_image(self.folder_paths[self.folder_index])
        except Exception as e:
            self._log(f"[warn] failed to load last session: {e}\n")

    # ---------- file ops ----------
    def open_image(self):
        if sys.platform == "darwin":  # Mac
            p = filedialog.askopenfilename(title="画像を選択")
        else:  # Windows/Linux
            p = filedialog.askopenfilename(
                title="画像を選択",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All", "*.*")]
            )
        if not p:
            return
        self.load_image(p)

    def open_folder(self):
        d = filedialog.askdirectory(title="画像フォルダを選択")
        if not d:
            return
        self.load_folder(d)

    def load_folder(self, d: str):
        d = os.path.abspath(d)
        paths = []
        for ext in IMG_EXTS:
            paths.extend(glob.glob(os.path.join(d, f"*{ext}")))
            paths.extend(glob.glob(os.path.join(d, f"*{ext.upper()}")))
        paths = sorted(list(set(paths)))
        if not paths:
            messagebox.showwarning("画像なし", "このフォルダに画像が見つかりませんでした。")
            return
        self.folder_paths = paths
        self.folder_index = 0
        self.load_image(self.folder_paths[self.folder_index])

    def prev_image(self):
        if not self.folder_paths:
            return
        self.folder_index = (self.folder_index - 1) % len(self.folder_paths)
        self.load_image(self.folder_paths[self.folder_index])

    def next_image(self):
        if not self.folder_paths:
            return
        self.folder_index = (self.folder_index + 1) % len(self.folder_paths)
        self.load_image(self.folder_paths[self.folder_index])

    def load_image(self, path: str):
        path = os.path.abspath(path)
        bgr = _imread_jp(path, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("エラー", f"画像を読み込めませんでした:\n{path}")
            return
        self.current_path = path
        self.path_var.set(path)
        self._orig_bgr = bgr
        self._out_bgr = None
        self._preds = []
        self._quad = None
        self._mouth_conf = 0.0
        self.face_combo["values"] = []
        self.face_var.set("(not detected)")
        self._render_all()
        self._save_last_session()

        # 画像を変えたら自動で検出/プレビュー。
        # ※起動直後（mainloop前）に重い初期化で固まらないよう after_idle で遅延実行する
        self.after_idle(self.detect_and_preview)

    def save_output(self):
        if self._out_bgr is None:
            messagebox.showinfo("未生成", "まだ結果がありません。")
            return
        if sys.platform == "darwin":  # Mac
            out_path = filedialog.asksaveasfilename(
                title="保存先",
                defaultextension=".png",
            )
        else:  # Windows/Linux
            out_path = filedialog.asksaveasfilename(
                title="保存先",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("All", "*.*")]
            )
        if not out_path:
            return
        try:
            if not _imwrite_jp(out_path, self._out_bgr):
                raise RuntimeError("画像の保存に失敗しました")
            self._log(f"[ok] saved: {out_path}\n")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗しました:\n{e}")

    # ---------- settings json ----------
    def save_settings_json(self):
        if sys.platform == "darwin":  # Mac
            out_path = filedialog.asksaveasfilename(
                title="設定を保存(JSON)",
                defaultextension=".json",
            )
        else:  # Windows/Linux
            out_path = filedialog.asksaveasfilename(
                title="設定を保存(JSON)",
                defaultextension=".json",
                filetypes=[("JSON", "*.json")]
            )
        if not out_path:
            return
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(self._collect_params_dict(), f, ensure_ascii=False, indent=2)
            self._log(f"[ok] settings saved: {out_path}\n")
        except Exception as e:
            messagebox.showerror("エラー", f"保存に失敗:\n{e}")

    def load_settings_json(self):
        if sys.platform == "darwin":  # Mac
            p = filedialog.askopenfilename(title="設定(JSON)を読み込む")
        else:  # Windows/Linux
            p = filedialog.askopenfilename(title="設定(JSON)を読み込む", filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                raise ValueError("json is not an object")
            self._apply_params_dict(d)
            self._log(f"[ok] settings loaded: {p}\n")
            self._schedule_update()
        except Exception as e:
            messagebox.showerror("エラー", f"読み込みに失敗:\n{e}")

    # ---------- detector ----------
    def _reset_detector(self):
        self._detector = None
        self._detector_device_actual = None
        self._log("[info] detector reset\n")
        self._schedule_update()

    def _get_detector(self):
        if not _HAS_ANIME_DETECTOR:
            raise RuntimeError("anime-face-detector がインストールされていません。")
        if self._detector is not None:
            return self._detector

        dev = self.device_var.get().strip() or "auto"
        self._log(f"[info] creating detector (model=yolov3, device={dev})...\n")

        # face_track_anime_detector.py と同じ fallback 方針
        device_try = ["cuda:0", "cpu"] if dev == "auto" else [dev] + (["cpu"] if dev.startswith("cuda") else [])
        last_err = None
        for d in device_try:
            try:
                det = create_detector("yolov3", device=d)  # type: ignore
                self._detector = det
                self._detector_device_actual = d
                if d != dev:
                    self._log(f"[info] detector fallback: using device={d}\n")
                return det
            except Exception as e:
                last_err = e
                self._log(f"[warn] detector init failed on {d}: {e}\n")
        raise RuntimeError(f"detector init failed: {last_err}")

    def detect_and_preview(self):
        if self._orig_bgr is None:
            return
        try:
            self._collect_params_from_ui()
            det = self._get_detector()
            bgr = self._orig_bgr

            det_scale = float(self.params.det_scale)
            if det_scale <= 0:
                det_scale = 1.0
            if det_scale != 1.0:
                h, w = bgr.shape[:2]
                dw = max(2, int(round(w * det_scale)))
                dh = max(2, int(round(h * det_scale)))
                det_frame = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA if det_scale < 1.0 else cv2.INTER_LINEAR)
            else:
                det_frame = bgr

            preds = det(det_frame)  # list[dict]
            if det_scale != 1.0 and preds:
                inv = 1.0 / det_scale
                scaled = []
                for p in preds:
                    bbox = np.asarray(p.get("bbox", []), dtype=np.float32).copy()
                    kps = np.asarray(p.get("keypoints", []), dtype=np.float32).copy()
                    if bbox.shape[0] >= 4:
                        bbox[:4] *= inv
                    if kps.ndim == 2 and kps.shape[1] >= 2:
                        kps[:, :2] *= inv
                    scaled.append({"bbox": bbox, "keypoints": kps})
                preds = scaled

            self._preds = preds or []
            if not self._preds:
                self._quad = None
                self._out_bgr = None
                self._log("[warn] no face detected\n")
                self._render_all()
                return

            # populate faces list
            items = []
            areas = []
            for i, p in enumerate(self._preds):
                bbox = np.asarray(p["bbox"], dtype=np.float32)
                if bbox.shape[0] < 4:
                    area = 0.0
                    conf = 0.0
                else:
                    area = float(max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
                    conf = float(bbox[4]) if bbox.shape[0] >= 5 else 0.0
                areas.append(area)
                items.append(f"#{i} area={area:.0f} conf={conf:.2f}")

            best_idx = int(np.argmax(np.asarray(areas))) if areas else 0
            self._selected_face_idx = best_idx
            self.face_combo["values"] = items
            self.face_var.set(items[best_idx])

            self._compute_quad_from_selected()
            self._run_erase_preview()
            self._save_last_session()
        except Exception as e:
            self._log(f"[error] detect_and_preview: {e}\n")
            self._log(traceback.format_exc() + "\n")
            messagebox.showerror("エラー", f"検出に失敗しました:\n{e}")

    def _on_select_face(self):
        if not self._preds:
            return
        # face_var から idx 抽出
        s = self.face_var.get().strip()
        if s.startswith("#"):
            try:
                idx = int(s.split()[0][1:])
                self._selected_face_idx = idx
                self._compute_quad_from_selected()
                self._run_erase_preview()
                self._save_last_session()
            except Exception:
                pass

    def _compute_quad_from_selected(self):
        idx = int(self._selected_face_idx)
        idx = max(0, min(idx, len(self._preds) - 1))
        p = self._preds[idx]
        bbox = np.asarray(p["bbox"], dtype=np.float32)
        kps = np.asarray(p["keypoints"], dtype=np.float32)
        if bbox.shape[0] < 5:
            # bbox[4] のconfが無い場合は仮で1
            bbox = np.concatenate([bbox[:4], np.array([1.0], dtype=np.float32)], axis=0)
        if bbox[4] < float(self.params.min_conf):
            self._log(f"[warn] face conf too low: {bbox[4]:.2f} < min_conf {self.params.min_conf:.2f}\n")

        # mouth landmarks（mask用にも保持）
        try:
            self._mouth_xy = kps[MOUTH_OUTLINE, :2].astype(np.float32).copy()
        except Exception:
            self._mouth_xy = None

        mode = self.params.quad_mode
        if mode == "bbox":
            quad, mconf = _mouth_quad_from_face_bbox_and_landmarks(
                bbox=bbox,
                keypoints=kps,
                sprite_aspect=float(self.params.sprite_aspect),
                pad=float(self.params.pad),
            )
        elif mode == "mouth":
            quad, mconf = _mouth_quad_from_landmarks(
                keypoints=kps, bbox=None, sprite_aspect=float(self.params.sprite_aspect), pad=float(self.params.pad)
            )
        else:
            # hybrid: 口点ベース + 最低サイズfloor
            quad, mconf = _mouth_quad_from_landmarks(
                keypoints=kps, bbox=bbox, sprite_aspect=float(self.params.sprite_aspect), pad=float(self.params.pad),
                min_mouth_w_ratio=self.params.min_mouth_w_ratio,
                min_mouth_w_px=self.params.min_mouth_w_px,
            )
        self._quad = quad
        self._mouth_conf = float(mconf)
        self._log(f"[info] mouth quad computed (mouth_conf={self._mouth_conf:.2f})\n")

    # ---------- erase ----------
    def _run_erase_preview(self):
        if self._orig_bgr is None or self._quad is None:
            self._out_bgr = None
            self._render_all()
            return

        self._collect_params_from_ui()
        bgr = self._orig_bgr
        quad = self._quad.astype(np.float32)

        # derive norm size
        wq, hq = _quad_wh(quad)
        ratio = float(wq / max(1e-6, hq))
        overs = float(self.params.oversample)

        if int(self.params.norm_w) > 0:
            norm_w = int(self.params.norm_w)
        else:
            norm_w = int(round(wq * overs))
        norm_w = int(self._helpers["ensure_even_ge2"](max(96, norm_w)))

        if int(self.params.norm_h) > 0:
            norm_h = int(self.params.norm_h)
        else:
            norm_h = int(round(norm_w / max(0.25, min(4.0, ratio))))
        norm_h = int(self._helpers["ensure_even_ge2"](max(64, norm_h)))

        # apply coverage auto-tune
        if bool(self.params.use_coverage):
            tuned = coverage_to_params(norm_h, float(self.params.coverage))
            mask_scale_x = float(tuned["mask_scale_x"])
            mask_scale_y = float(tuned["mask_scale_y"])
            ring_px = int(tuned["ring_px"])
            dilate_px = int(tuned["dilate_px"])
            feather_px = int(tuned["feather_px"])
            inpaint_radius = float(tuned["inpaint_radius"])
            top_clip_frac = float(tuned["top_clip_frac"])
            center_y_off_frac = float(tuned["center_y_off_frac"])
        else:
            mask_scale_x = float(self.params.mask_scale_x)
            mask_scale_y = float(self.params.mask_scale_y)
            ring_px = int(self.params.ring_px)
            dilate_px = int(self.params.dilate_px)
            feather_px = int(self.params.feather_px)
            inpaint_radius = float(self.params.inpaint_radius)
            top_clip_frac = float(self.params.top_clip_frac)
            center_y_off_frac = float(self.params.center_y_off_frac)

        center_y_off_px = int(round(norm_h * center_y_off_frac))

        rx = int((norm_w * mask_scale_x) * 0.5)
        ry = int((norm_h * mask_scale_y) * 0.5)

        make_mask = self._helpers["make_mouth_mask"]
        feather_mask = self._helpers["feather_mask"]
        warp_to_norm = self._helpers["warp_frame_to_norm"]
        warp_back = self._helpers["warp_norm_to_bbox"]
        blend_roi = self._helpers["alpha_blend_roi"]

        using_landmark_mask = False
        inner_u8 = None
        if bool(self.params.mask_from_landmarks) and (self._mouth_xy is not None):
            try:
                src = quad.astype(np.float32)
                dst = np.array([[0, 0], [norm_w - 1, 0], [norm_w - 1, norm_h - 1], [0, norm_h - 1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                pts = self._mouth_xy.reshape(-1, 1, 2).astype(np.float32)
                pts_norm = cv2.perspectiveTransform(pts, M).reshape(-1, 2)

                expand_x = 1.0 + 2.0 * (float(mask_scale_x) - 0.50)
                expand_y = 1.0 + 2.0 * (float(mask_scale_y) - 0.44)
                expand_x = float(np.clip(expand_x, 0.8, 2.5))
                expand_y = float(np.clip(expand_y, 0.8, 2.5))

                inner_u8 = _make_mouth_polygon_mask(
                    norm_w, norm_h,
                    pts_norm=pts_norm,
                    expand_x=expand_x,
                    expand_y=expand_y,
                    center_y_offset_px=center_y_off_px,
                    top_clip_frac=top_clip_frac,
                )
                using_landmark_mask = True
            except Exception:
                inner_u8 = None

        if inner_u8 is None:
            inner_u8 = make_mask(norm_w, norm_h, rx=rx, ry=ry, center_y_offset_px=center_y_off_px, top_clip_frac=top_clip_frac)

        mask_f = feather_mask(inner_u8, dilate_px=int(dilate_px), feather_px=int(feather_px))

        patch = warp_to_norm(bgr, quad, norm_w, norm_h)

        # inpaint on normalized patch
        clean = cv2.inpaint(patch, inner_u8, inpaintRadius=float(inpaint_radius), flags=cv2.INPAINT_TELEA)

        # Optional: match shading/chroma from a ring around the mouth (ring_px is meaningful here)
        if bool(self.params.match_ring) and int(ring_px) > 0:
            try:
                if using_landmark_mask:
                    k = int(ring_px)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                    outer_u8 = cv2.dilate(inner_u8, kernel, iterations=1)
                    top_clip_frac2 = float(np.clip(top_clip_frac, 0.0, 1.0))
                    if top_clip_frac2 > 0:
                        ys2, xs2 = np.where(outer_u8 > 0)
                        if ys2.size > 0:
                            y_min = float(ys2.min()); y_max = float(ys2.max())
                            cy = float((y_min + y_max) * 0.5)
                            ry2 = float(max(1.0, (y_max - y_min) * 0.5))
                            clip_y = int(round(cy - ry2 * top_clip_frac2))
                            if clip_y > 0:
                                outer_u8[:clip_y, :] = 0
                else:
                    outer_u8 = make_mask(
                        norm_w, norm_h, rx=rx + int(ring_px), ry=ry + int(ring_px),
                        center_y_offset_px=center_y_off_px, top_clip_frac=top_clip_frac
                    )
                ring_u8 = cv2.subtract(outer_u8, inner_u8)
                ring_ys, ring_xs = np.where(ring_u8 > 0)
                if ring_xs.size >= 32:
                    patch_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                    clean_lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)

                    # a,b mean matching
                    a_p = float(patch_lab[:, :, 1][ring_ys, ring_xs].mean())
                    b_p = float(patch_lab[:, :, 2][ring_ys, ring_xs].mean())
                    a_c = float(clean_lab[:, :, 1][ring_ys, ring_xs].mean())
                    b_c = float(clean_lab[:, :, 2][ring_ys, ring_xs].mean())
                    da = a_p - a_c
                    db = b_p - b_c
                    clean_lab[:, :, 1] = np.clip(clean_lab[:, :, 1] + da, 0, 255)
                    clean_lab[:, :, 2] = np.clip(clean_lab[:, :, 2] + db, 0, 255)

                    # L plane matching (gradient)
                    fit_plane = self._helpers.get("fit_plane_2d", None)
                    eval_plane = self._helpers.get("eval_plane", None)
                    if callable(fit_plane) and callable(eval_plane):
                        Lp = patch_lab[:, :, 0][ring_ys, ring_xs]
                        Lc = clean_lab[:, :, 0][ring_ys, ring_xs]
                        plane_p = fit_plane(Lp, ring_xs, ring_ys, norm_w, norm_h)
                        plane_c = fit_plane(Lc, ring_xs, ring_ys, norm_w, norm_h)
                        grid_p = eval_plane(plane_p, norm_w, norm_h)
                        grid_c = eval_plane(plane_c, norm_w, norm_h)
                        clean_lab[:, :, 0] = np.clip(clean_lab[:, :, 0] + (grid_p - grid_c), 0, 255)
                    else:
                        # fallback: mean L matching only
                        Lp = float(patch_lab[:, :, 0][ring_ys, ring_xs].mean())
                        Lc = float(clean_lab[:, :, 0][ring_ys, ring_xs].mean())
                        clean_lab[:, :, 0] = np.clip(clean_lab[:, :, 0] + (Lp - Lc), 0, 255)

                    clean = cv2.cvtColor(clean_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            except Exception:
                # matching is best-effort
                pass

        warped_patch, warped_mask, x0, y0 = warp_back(clean, mask_f, quad)
        out = blend_roi(bgr, warped_patch, warped_mask, x0, y0)

        if bool(self.draw_quad_var.get()):
            out = _draw_quad(out, quad, color=(0, 255, 0), thickness=2)
            before = _draw_quad(bgr, quad, color=(0, 255, 0), thickness=2)
        else:
            before = bgr

        self._out_bgr = out
        self._patch_bgr = patch
        self._clean_bgr = clean
        self._inner_mask_u8 = inner_u8
        self._feather_mask_f = mask_f

        self._log(f"[erase] norm={norm_w}x{norm_h} rx={rx} ry={ry} ring={ring_px} dilate={dilate_px} feather={feather_px} inpaint={inpaint_radius:.1f}\n")
        self._render_all()

    # ---------- rendering ----------
    def _render_canvas(self, canvas: tk.Canvas, pil_img: Image.Image, key: str):
        canvas.update_idletasks()
        cw = max(1, int(canvas.winfo_width()))
        ch = max(1, int(canvas.winfo_height()))
        fitted = fit_to_box(pil_img, cw, ch)
        tk_img = ImageTk.PhotoImage(fitted)
        self._tk_imgs[key] = tk_img  # keep ref
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")

    def _render_all(self):
        if self._orig_bgr is None:
            return

        before = self._orig_bgr
        after = self._out_bgr if self._out_bgr is not None else self._orig_bgr

        # Result tab
        self._render_canvas(self.canvas_before, bgr_to_pil(before), "before")
        self._render_canvas(self.canvas_after, bgr_to_pil(after), "after")

        # Mask tab
        if self._inner_mask_u8 is not None:
            self._render_canvas(self.canvas_mask_inner, u8_to_pil_gray(self._inner_mask_u8), "mask_inner")
        else:
            self.canvas_mask_inner.delete("all")
        if self._feather_mask_f is not None:
            self._render_canvas(self.canvas_mask_feather, f32_to_pil_gray01(self._feather_mask_f), "mask_feather")
        else:
            self.canvas_mask_feather.delete("all")

        # Patch tab
        if self._patch_bgr is not None:
            self._render_canvas(self.canvas_patch, bgr_to_pil(self._patch_bgr), "patch")
        else:
            self.canvas_patch.delete("all")
        if self._clean_bgr is not None:
            self._render_canvas(self.canvas_clean, bgr_to_pil(self._clean_bgr), "clean")
        else:
            self.canvas_clean.delete("all")

    # ---------- updates ----------
    def _schedule_update(self):
        if self._update_job is not None:
            try:
                self.after_cancel(self._update_job)
            except Exception:
                pass
        self._update_job = self.after(180, self._apply_update_now)

    def _apply_update_now(self):
        self._update_job = None
        if self._orig_bgr is None:
            return
        # pad/sprite_aspect/quad_mode/min_floor が変わると quad 自体が変わるので
        # 検出済み(predsあり)なら _compute_quad_from_selected() を先に呼んでから _run_erase_preview()
        self._collect_params_from_ui()
        if self._preds and len(self._preds) > 0:
            self._compute_quad_from_selected()
            self._run_erase_preview()
        elif self._quad is not None:
            self._run_erase_preview()
        else:
            self._render_all()
        self._save_last_session()

    # ---------- params ----------
    def _collect_params_from_ui(self):
        self.params.device = str(self.device_var.get())
        self.params.det_scale = float(self.det_scale_var.get())
        self.params.min_conf = float(self.min_conf_var.get())
        self.params.sprite_aspect = float(self.sprite_aspect_var.get())
        self.params.pad = float(self.pad_var.get())
        self.params.quad_mode = str(self.quad_mode_var.get())
        self.params.min_mouth_w_ratio = float(self.min_mouth_w_ratio_var.get())
        self.params.min_mouth_w_px = float(self.min_mouth_w_px_var.get())
        self.params.quad_from_landmarks = bool(self.quad_from_landmarks_var.get())
        self.params.mask_from_landmarks = bool(self.mask_from_landmarks_var.get())

        self.params.use_coverage = bool(self.use_coverage_var.get())
        self.params.coverage = float(self.coverage_var.get())

        self.params.mask_scale_x = float(self.mask_scale_x_var.get())
        self.params.mask_scale_y = float(self.mask_scale_y_var.get())
        self.params.ring_px = int(self.ring_px_var.get())
        self.params.dilate_px = int(self.dilate_px_var.get())
        self.params.feather_px = int(self.feather_px_var.get())
        self.params.inpaint_radius = float(self.inpaint_radius_var.get())
        self.params.top_clip_frac = float(self.top_clip_frac_var.get())
        self.params.center_y_off_frac = float(self.center_y_off_frac_var.get())

        self.params.oversample = float(self.oversample_var.get())
        self.params.norm_w = int(self.norm_w_var.get() or 0)
        self.params.norm_h = int(self.norm_h_var.get() or 0)

        self.params.match_ring = bool(self.match_ring_var.get())

        self.params.draw_quad = bool(self.draw_quad_var.get())

    def _collect_params_dict(self) -> Dict[str, Any]:
        self._collect_params_from_ui()
        return {
            "device": self.params.device,
            "det_scale": self.params.det_scale,
            "min_conf": self.params.min_conf,
            "sprite_aspect": self.params.sprite_aspect,
            "pad": self.params.pad,
            "quad_mode": self.params.quad_mode,
            "min_mouth_w_ratio": self.params.min_mouth_w_ratio,
            "min_mouth_w_px": self.params.min_mouth_w_px,
            "quad_from_landmarks": bool(self.params.quad_from_landmarks),
            "mask_from_landmarks": bool(self.params.mask_from_landmarks),
            "oversample": self.params.oversample,
            "norm_w": self.params.norm_w,
            "norm_h": self.params.norm_h,
            "use_coverage": self.params.use_coverage,
            "coverage": self.params.coverage,
            "mask_scale_x": self.params.mask_scale_x,
            "mask_scale_y": self.params.mask_scale_y,
            "ring_px": self.params.ring_px,
            "dilate_px": self.params.dilate_px,
            "feather_px": self.params.feather_px,
            "inpaint_radius": self.params.inpaint_radius,
            "top_clip_frac": self.params.top_clip_frac,
            "center_y_off_frac": self.params.center_y_off_frac,
            "match_ring": self.params.match_ring,
            "draw_quad": self.params.draw_quad,
        }

    def _apply_params_dict(self, d: Dict[str, Any]):
        def _set(var, key, cast):
            if key in d:
                try:
                    var.set(cast(d[key]))
                except Exception:
                    pass

        _set(self.device_var, "device", str)
        _set(self.det_scale_var, "det_scale", float)
        _set(self.min_conf_var, "min_conf", float)
        _set(self.sprite_aspect_var, "sprite_aspect", float)
        _set(self.pad_var, "pad", float)
        _set(self.quad_mode_var, "quad_mode", str)
        _set(self.min_mouth_w_ratio_var, "min_mouth_w_ratio", float)
        _set(self.min_mouth_w_px_var, "min_mouth_w_px", float)
        _set(self.quad_from_landmarks_var, "quad_from_landmarks", bool)
        _set(self.mask_from_landmarks_var, "mask_from_landmarks", bool)

        _set(self.oversample_var, "oversample", float)
        _set(self.norm_w_var, "norm_w", int)
        _set(self.norm_h_var, "norm_h", int)

        _set(self.use_coverage_var, "use_coverage", bool)
        _set(self.coverage_var, "coverage", float)

        _set(self.mask_scale_x_var, "mask_scale_x", float)
        _set(self.mask_scale_y_var, "mask_scale_y", float)
        _set(self.ring_px_var, "ring_px", int)
        _set(self.dilate_px_var, "dilate_px", int)
        _set(self.feather_px_var, "feather_px", int)
        _set(self.inpaint_radius_var, "inpaint_radius", float)
        _set(self.top_clip_frac_var, "top_clip_frac", float)
        _set(self.center_y_off_frac_var, "center_y_off_frac", float)

        _set(self.match_ring_var, "match_ring", bool)
        _set(self.draw_quad_var, "draw_quad", bool)


def main():
    app = MouthEraseTunerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
