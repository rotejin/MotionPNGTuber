from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

ColorOrder3 = Literal["BGR", "RGB"]
ColorOrder4 = Literal["BGRA", "RGBA"]


@dataclass(frozen=True)
class MouthColorAdjust:
    brightness: float = 0.0
    saturation: float = 1.0
    warmth: float = 0.0
    color_strength: float = 0.75
    edge_priority: float = 0.85
    edge_width_ratio: float = 0.10
    inspect_boost: float = 1.0


def clamp_mouth_color_adjust(cfg: MouthColorAdjust) -> MouthColorAdjust:
    return MouthColorAdjust(
        brightness=float(np.clip(float(cfg.brightness), -32.0, 32.0)),
        saturation=float(np.clip(float(cfg.saturation), 0.70, 1.50)),
        warmth=float(np.clip(float(cfg.warmth), -24.0, 24.0)),
        color_strength=float(np.clip(float(cfg.color_strength), 0.0, 1.0)),
        edge_priority=float(np.clip(float(cfg.edge_priority), 0.0, 1.0)),
        edge_width_ratio=float(np.clip(float(cfg.edge_width_ratio), 0.02, 0.20)),
        inspect_boost=float(np.clip(float(cfg.inspect_boost), 1.0, 4.0)),
    )


def alpha_bbox_from_mask(alpha_u8: np.ndarray) -> tuple[int, int, int, int] | None:
    alpha = np.asarray(alpha_u8)
    if alpha.ndim != 2:
        raise ValueError("alpha_u8 must be 2D")
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def build_edge_weight(alpha_u8: np.ndarray, edge_width_ratio: float) -> np.ndarray:
    alpha = np.asarray(alpha_u8)
    if alpha.ndim != 2:
        raise ValueError("alpha_u8 must be 2D")
    bbox = alpha_bbox_from_mask(alpha)
    out = np.zeros(alpha.shape, dtype=np.float32)
    if bbox is None:
        return out
    x0, y0, x1, y1 = bbox
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    edge_ratio = float(np.clip(edge_width_ratio, 0.02, 0.20))
    edge_px = max(1, int(round(min(bw, bh) * edge_ratio)))

    mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
    k = max(1, edge_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inner = cv2.erode(mask, kernel, iterations=1)
    ring = cv2.subtract(mask, inner)

    blur_k = max(3, edge_px * 2 + 1)
    if blur_k % 2 == 0:
        blur_k += 1
    weight = cv2.GaussianBlur(ring, (blur_k, blur_k), sigmaX=0)
    return np.clip(weight.astype(np.float32) / 255.0, 0.0, 1.0)


def _mean_color_3ch(
    img: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    arr = np.asarray(img)
    w = np.asarray(weight, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must be HxWx3")
    if w.ndim != 2 or w.shape != arr.shape[:2]:
        raise ValueError("weight must be HxW matching img")
    mask = w > 1e-6
    count = int(mask.sum())
    if count <= 0:
        return None
    weighted = arr.astype(np.float32) * w[..., None]
    denom = float(w[mask].sum())
    if denom <= 1e-6:
        return None
    mean = weighted.sum(axis=(0, 1)) / denom
    return mean.astype(np.float32), count


def sample_colored_edge_mean_4ch(
    img: np.ndarray,
    *,
    edge_width_ratio: float,
    color_order: ColorOrder4,
    alpha_threshold: int = 24,
) -> tuple[np.ndarray, int] | None:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("img must be HxWx4")
    alpha = arr[..., 3].astype(np.uint8)
    opaque = alpha >= int(alpha_threshold)
    if not np.any(opaque):
        return None
    edge_weight = build_edge_weight(alpha, edge_width_ratio).astype(np.float32)
    alpha_weight = (alpha.astype(np.float32) / 255.0)
    weight = edge_weight * alpha_weight
    weight[~opaque] = 0.0
    rgb = arr[..., :3]
    return _mean_color_3ch(rgb, weight)


def sample_background_ring_mean_3ch(
    frame: np.ndarray,
    alpha_u8: np.ndarray,
    x0: int,
    y0: int,
    *,
    edge_width_ratio: float,
    color_order: ColorOrder3,
    alpha_threshold: int = 24,
) -> tuple[np.ndarray, int] | None:
    arr = np.asarray(frame)
    alpha = np.asarray(alpha_u8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("frame must be HxWx3")
    if alpha.ndim != 2:
        raise ValueError("alpha_u8 must be HxW")
    bbox = alpha_bbox_from_mask(np.where(alpha >= int(alpha_threshold), alpha, 0).astype(np.uint8))
    if bbox is None:
        return None
    px0 = max(0, int(x0))
    py0 = max(0, int(y0))
    px1 = min(arr.shape[1], int(x0) + alpha.shape[1])
    py1 = min(arr.shape[0], int(y0) + alpha.shape[0])
    if px0 >= px1 or py0 >= py1:
        return None
    ax0 = px0 - int(x0)
    ay0 = py0 - int(y0)
    ax1 = ax0 + (px1 - px0)
    ay1 = ay0 + (py1 - py0)
    alpha_crop = alpha[ay0:ay1, ax0:ax1]
    bbox = alpha_bbox_from_mask(np.where(alpha_crop >= int(alpha_threshold), alpha_crop, 0).astype(np.uint8))
    if bbox is None:
        return None
    bx0, by0, bx1, by1 = bbox
    bw = max(1, bx1 - bx0)
    bh = max(1, by1 - by0)
    edge_px = max(1, int(round(min(bw, bh) * float(np.clip(edge_width_ratio, 0.02, 0.20)))))
    occ = np.where(alpha_crop >= int(alpha_threshold), 255, 0).astype(np.uint8)
    inner_px = max(1, edge_px)
    outer_px = max(inner_px + 1, edge_px * 2 + 1)
    k_inner = inner_px * 2 + 1
    k_outer = outer_px * 2 + 1
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_inner, k_inner))
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_outer, k_outer))
    inner = cv2.dilate(occ, inner_kernel, iterations=1)
    outer = cv2.dilate(occ, outer_kernel, iterations=1)
    ring = cv2.subtract(outer, inner).astype(np.float32) / 255.0
    if not np.any(ring > 0):
        return None
    frame_crop = arr[py0:py1, px0:px1]
    return _mean_color_3ch(frame_crop, ring)


def _bgr_to_lab(img: np.ndarray, color_order: ColorOrder3) -> np.ndarray:
    code = cv2.COLOR_BGR2LAB if color_order == "BGR" else cv2.COLOR_RGB2LAB
    return cv2.cvtColor(img, code)


def _lab_to_bgr(img: np.ndarray, color_order: ColorOrder3) -> np.ndarray:
    code = cv2.COLOR_LAB2BGR if color_order == "BGR" else cv2.COLOR_LAB2RGB
    return cv2.cvtColor(img, code)


def _bgr_to_hsv(img: np.ndarray, color_order: ColorOrder3) -> np.ndarray:
    code = cv2.COLOR_BGR2HSV if color_order == "BGR" else cv2.COLOR_RGB2HSV
    return cv2.cvtColor(img, code)


def _hsv_to_bgr(img: np.ndarray, color_order: ColorOrder3) -> np.ndarray:
    code = cv2.COLOR_HSV2BGR if color_order == "BGR" else cv2.COLOR_HSV2RGB
    return cv2.cvtColor(img, code)


def _channel_indices(color_order: ColorOrder3) -> tuple[int, int]:
    if color_order == "BGR":
        return 2, 0
    return 0, 2


def _mean_to_rgb(mean: np.ndarray, color_order: ColorOrder3) -> np.ndarray:
    out = np.asarray(mean, dtype=np.float32).reshape(3)
    if color_order == "RGB":
        return out
    return out[[2, 1, 0]]


def _rgb_to_lab_triplet(rgb: np.ndarray) -> np.ndarray:
    pix = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
    return cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32).reshape(3)


def estimate_auto_mouth_color_adjust(
    current_cfg: MouthColorAdjust,
    *,
    bg_mean: np.ndarray,
    mouth_mean: np.ndarray,
    color_order: ColorOrder3,
) -> tuple[MouthColorAdjust, dict[str, float]]:
    cfg = clamp_mouth_color_adjust(current_cfg)
    bg_rgb = _mean_to_rgb(bg_mean, color_order)
    mouth_rgb = _mean_to_rgb(mouth_mean, color_order)
    bg_lab = _rgb_to_lab_triplet(bg_rgb)
    mouth_lab = _rgb_to_lab_triplet(mouth_rgb)

    delta_l = float(bg_lab[0] - mouth_lab[0])
    brightness = float(np.clip(delta_l * 0.55, -32.0, 32.0))

    rb_bg = float(bg_rgb[0] - bg_rgb[2])
    rb_mouth = float(mouth_rgb[0] - mouth_rgb[2])
    delta_rb = rb_bg - rb_mouth
    warmth = float(np.clip(delta_rb * 0.35, -24.0, 24.0))

    bg_chroma = float(np.hypot(bg_lab[1] - 128.0, bg_lab[2] - 128.0))
    mouth_chroma = float(np.hypot(mouth_lab[1] - 128.0, mouth_lab[2] - 128.0))
    if mouth_chroma < 1.0 or bg_chroma < 1.0:
        saturation = 1.0
        sat_ratio = 1.0
    else:
        ratio = bg_chroma / max(mouth_chroma, 1.0)
        sat_ratio = float(np.clip(1.0 + 0.35 * (ratio - 1.0), 0.75, 1.25))
        saturation = float(np.clip(sat_ratio, 0.70, 1.50))

    delta_e = float(np.linalg.norm(bg_lab - mouth_lab))
    if delta_e >= 10.0:
        suggested_strength = 0.65 + min(delta_e, 60.0) / 60.0 * 0.25
        color_strength = float(np.clip(suggested_strength, 0.0, 1.0))
    else:
        color_strength = float(cfg.color_strength)

    new_cfg = clamp_mouth_color_adjust(
        MouthColorAdjust(
            brightness=brightness,
            saturation=saturation,
            warmth=warmth,
            color_strength=color_strength,
            edge_priority=cfg.edge_priority,
            edge_width_ratio=cfg.edge_width_ratio,
            inspect_boost=cfg.inspect_boost,
        ),
    )
    debug = {
        "bg_r": float(bg_rgb[0]),
        "bg_g": float(bg_rgb[1]),
        "bg_b": float(bg_rgb[2]),
        "mouth_r": float(mouth_rgb[0]),
        "mouth_g": float(mouth_rgb[1]),
        "mouth_b": float(mouth_rgb[2]),
        "delta_l": delta_l,
        "delta_rb": float(delta_rb),
        "delta_e": delta_e,
        "sat_ratio": float(sat_ratio),
    }
    return new_cfg, debug


def apply_basic_color_adjust_3ch(
    img: np.ndarray,
    cfg: MouthColorAdjust,
    *,
    color_order: ColorOrder3,
) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must be HxWx3")
    cfg = clamp_mouth_color_adjust(cfg)
    out = arr.astype(np.uint8).copy()

    if cfg.brightness != 0.0:
        lab = _bgr_to_lab(out, color_order).astype(np.float32)
        lab[..., 0] = np.clip(lab[..., 0] + float(cfg.brightness), 0.0, 255.0)
        out = _lab_to_bgr(lab.astype(np.uint8), color_order)

    if cfg.saturation != 1.0:
        hsv = _bgr_to_hsv(out, color_order).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * float(cfg.saturation), 0.0, 255.0)
        out = _hsv_to_bgr(hsv.astype(np.uint8), color_order)

    if cfg.warmth != 0.0:
        warm = out.astype(np.float32)
        idx_r, idx_b = _channel_indices(color_order)
        warm[..., idx_r] = np.clip(warm[..., idx_r] + float(cfg.warmth), 0.0, 255.0)
        warm[..., idx_b] = np.clip(warm[..., idx_b] - float(cfg.warmth), 0.0, 255.0)
        out = warm.astype(np.uint8)

    return out


def apply_mouth_color_adjust_4ch(
    img: np.ndarray,
    cfg: MouthColorAdjust,
    *,
    color_order: ColorOrder4,
) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("img must be HxWx4")
    cfg = clamp_mouth_color_adjust(cfg)
    base = arr.astype(np.uint8).copy()
    rgb = base[..., :3]
    alpha = base[..., 3]
    adjusted = apply_basic_color_adjust_3ch(
        rgb,
        cfg,
        color_order="BGR" if color_order == "BGRA" else "RGB",
    ).astype(np.float32)
    base_rgb = rgb.astype(np.float32)
    edge_weight = build_edge_weight(alpha, cfg.edge_width_ratio)[..., None]
    mix = float(cfg.color_strength) * (
        (1.0 - float(cfg.edge_priority)) + float(cfg.edge_priority) * edge_weight
    )
    mixed = np.clip(base_rgb * (1.0 - mix) + adjusted * mix, 0.0, 255.0).astype(np.uint8)
    out = np.empty_like(base)
    out[..., :3] = mixed
    out[..., 3] = alpha
    return out


def apply_inspect_boost_3ch(
    img: np.ndarray,
    boost: float,
    *,
    color_order: ColorOrder3,
) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("img must be HxWx3")
    boost_f = float(boost)
    if boost_f <= 1.0:
        return arr.astype(np.uint8).copy()
    lab = _bgr_to_lab(arr.astype(np.uint8), color_order).astype(np.float32)
    lab[..., 1] = np.clip(128.0 + (lab[..., 1] - 128.0) * boost_f, 0.0, 255.0)
    lab[..., 2] = np.clip(128.0 + (lab[..., 2] - 128.0) * boost_f, 0.0, 255.0)
    return _lab_to_bgr(lab.astype(np.uint8), color_order)
