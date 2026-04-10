"""
lipsync_core.py

共通コアモジュール: 口パク合成に必要なクラス・関数を集約。

- loop_lipsync_runtime_patched_emotion_auto.py
- multi_video_live_gui.py

の両方から使用される。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from collections import deque

import cv2
import numpy as np
from PIL import Image


# ============================================================
# Utility Functions
# ============================================================

def one_pole_beta(cutoff_hz: float, update_hz: int) -> float:
    """One-pole lowpass filter coefficient."""
    return float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz / update_hz))


def load_rgba(path: str) -> np.ndarray:
    """Load image as RGBA numpy array."""
    im = Image.open(path).convert("RGBA")
    return np.array(im, dtype=np.uint8)


def open_video_capture(path: str, retries: int = 25, delay_sec: float = 0.08) -> cv2.VideoCapture:
    """
    OpenCV VideoCapture を「少し待ってリトライ」しながら開く。
    Windows で書き出し直後の mp4 を開くと失敗することがあるための対策。
    """
    backend_names = ["CAP_FFMPEG", "CAP_MSMF", "CAP_DSHOW"]
    backends: list[int | None] = []
    for name in backend_names:
        v = getattr(cv2, name, None)
        if isinstance(v, int):
            backends.append(v)
    backends.append(None)  # default

    for _ in range(max(1, retries)):
        for be in backends:
            cap = None
            try:
                cap = cv2.VideoCapture(path) if be is None else cv2.VideoCapture(path, be)
            except Exception:
                continue

            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return cap
                cap.release()
            else:
                try:
                    cap.release()
                except Exception:
                    pass
        time.sleep(delay_sec)

    raise RuntimeError(f"Failed to open video (after retries): {path}")


def probe_video_size(path: str) -> tuple[int, int] | None:
    """Get video dimensions (width, height) or None if failed."""
    try:
        cap = open_video_capture(path, retries=8, delay_sec=0.05)
    except Exception:
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if w > 0 and h > 0:
            return (w, h)
    finally:
        cap.release()
    return None


def probe_video_fps(path: str) -> float | None:
    """Get video FPS or None if failed."""
    try:
        cap = open_video_capture(path, retries=8, delay_sec=0.05)
    except Exception:
        return None
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        if fps > 0:
            return fps
    finally:
        cap.release()
    return None


def probe_video_frame_count(path: str) -> int | None:
    """Get video frame count or None if failed."""
    try:
        cap = open_video_capture(path, retries=8, delay_sec=0.05)
    except Exception:
        return None
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if count > 0:
            return count
    finally:
        cap.release()
    return None


def alpha_blit_rgb_safe(dst_rgb: np.ndarray, src_rgba: np.ndarray, x: int, y: int) -> None:
    """dst_rgb(H,W,3) に src_rgba(h,w,4) を (x,y) にアルファ合成（はみ出し安全）"""
    H, W = dst_rgb.shape[:2]
    h, w = src_rgba.shape[:2]

    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, W)
    y1 = min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return

    sx0 = x0 - x
    sy0 = y0 - y
    sx1 = sx0 + (x1 - x0)
    sy1 = sy0 + (y1 - y0)

    roi = dst_rgb[y0:y1, x0:x1]
    src = src_rgba[sy0:sy1, sx0:sx1]

    a = src[..., 3:4].astype(np.uint16)
    inv = (255 - a).astype(np.uint16)
    out = (src[..., :3].astype(np.uint16) * a + roi.astype(np.uint16) * inv) // 255
    roi[:] = out.astype(np.uint8)


def warp_rgba_to_quad(src_rgba: np.ndarray, quad_xy: np.ndarray) -> tuple[np.ndarray, int, int]:
    """RGBA画像をquad(4,2)へ射影変換して、(patch_rgba, x0, y0) を返す。"""
    quad = quad_xy.astype(np.float32)
    xs = quad[:, 0]
    ys = quad[:, 1]
    x0 = int(np.floor(xs.min()))
    y0 = int(np.floor(ys.min()))
    x1 = int(np.ceil(xs.max()))
    y1 = int(np.ceil(ys.max()))
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    dst_pts = (quad - np.array([x0, y0], dtype=np.float32))
    h, w = src_rgba.shape[:2]
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    patch = cv2.warpPerspective(
        src_rgba,
        M,
        dsize=(bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return patch, x0, y0


def alpha_bbox(rgba: np.ndarray, thresh: int = 8) -> tuple | None:
    """アルファチャンネルから非透明領域のbboxを取得"""
    a = rgba[:, :, 3]
    ys, xs = np.where(a > thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def resolve_preferred_track_path(
    base_track: str,
    calibrated_track: str = "",
    prefer_calibrated: bool = True,
) -> str:
    """Prefer calibrated track when available, otherwise fall back to base track."""
    if prefer_calibrated and calibrated_track and os.path.isfile(calibrated_track):
        return calibrated_track
    return base_track


class AudioChunkBuffer:
    """Bounded audio chunk buffer optimized for sliding-window inference."""

    def __init__(self, max_samples: int):
        self.max_samples = max(1, int(max_samples))
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._chunks.clear()
        self._size = 0

    def append(self, chunk: np.ndarray) -> None:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if arr.size <= 0:
            return

        # The audio callback may reuse the source buffer, so keep an owned copy.
        if arr.size >= self.max_samples:
            self._chunks.clear()
            self._chunks.append(arr[-self.max_samples:].copy())
            self._size = self.max_samples
            return

        self._chunks.append(arr.copy())
        self._size += int(arr.size)
        self._trim_left()

    def tail(self, n_samples: int) -> np.ndarray:
        n = max(0, min(int(n_samples), self._size))
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)

        parts: list[np.ndarray] = []
        remaining = n
        for chunk in reversed(self._chunks):
            if remaining <= 0:
                break
            if chunk.size <= remaining:
                parts.append(chunk)
                remaining -= int(chunk.size)
            else:
                parts.append(chunk[-remaining:])
                remaining = 0

        parts.reverse()
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        if len(parts) == 1:
            return parts[0].copy()
        return np.concatenate(parts, axis=0)

    def _trim_left(self) -> None:
        while self._size > self.max_samples and self._chunks:
            overflow = self._size - self.max_samples
            head = self._chunks[0]
            if overflow >= head.size:
                self._chunks.popleft()
                self._size -= int(head.size)
                continue

            self._chunks[0] = head[overflow:]
            self._size -= int(overflow)
            break


# ============================================================
# MouthTrack Class
# ============================================================

@dataclass
class MouthTrack:
    """frame_idx -> quad(4,2) を返す（ロード時にスケール適用）。"""

    quads: np.ndarray          # (N,4,2) float32 (元)
    valid: np.ndarray          # (N,) bool
    quads_filled: np.ndarray   # (N,4,2) float32 (hold用に穴埋め済み)
    has_any_valid: bool
    total: int
    policy: str                # "hold" or "strict"
    src_w: int
    src_h: int
    calibrated: bool

    @staticmethod
    def _bbox_to_quad(bb: np.ndarray) -> np.ndarray:
        x, y, w, h = bb[:, 0], bb[:, 1], bb[:, 2], bb[:, 3]
        quad = np.stack(
            [
                np.stack([x, y], axis=1),
                np.stack([x + w, y], axis=1),
                np.stack([x + w, y + h], axis=1),
                np.stack([x, y + h], axis=1),
            ],
            axis=1,
        ).astype(np.float32)
        return quad

    @staticmethod
    def _make_filled(quads: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, bool]:
        """validがある場合、前後でhold補間（前方埋め→先頭側を後方埋め）"""
        N = int(quads.shape[0])
        if N <= 0:
            return quads.copy(), False
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            return quads.copy(), False

        filled = quads.copy()
        last = idxs[0]
        for i in range(0, N):
            if valid[i]:
                last = i
            else:
                filled[i] = filled[last]
        first = idxs[0]
        for i in range(0, first):
            filled[i] = filled[first]
        return filled, True

    @staticmethod
    def load(path: str, target_w: int, target_h: int, policy: str = "hold") -> "MouthTrack | None":
        if not path or (not os.path.isfile(path)):
            return None

        npz = np.load(path, allow_pickle=False)

        if "quad" in npz:
            quad = np.asarray(npz["quad"], dtype=np.float32)
            if quad.ndim != 3 or quad.shape[1:] != (4, 2):
                raise ValueError("mouth_track.npz: quad must be (N,4,2)")
        elif "bbox" in npz:
            bb = np.asarray(npz["bbox"], dtype=np.float32)
            if bb.ndim != 2 or bb.shape[1] != 4:
                raise ValueError("mouth_track.npz: bbox must be (N,4)")
            quad = MouthTrack._bbox_to_quad(bb)
        else:
            raise ValueError("mouth_track.npz must contain 'quad' or 'bbox'")

        N = int(quad.shape[0])
        if "valid" in npz:
            valid = np.asarray(npz["valid"], dtype=np.uint8).astype(bool)
            if valid.shape[0] != N:
                valid = np.ones((N,), dtype=bool)
        else:
            valid = np.ones((N,), dtype=bool)

        src_w = int(npz["w"]) if "w" in npz else target_w
        src_h = int(npz["h"]) if "h" in npz else target_h

        sx = float(target_w) / float(max(1, src_w))
        sy = float(target_h) / float(max(1, src_h))
        quad = quad.copy()
        quad[..., 0] *= sx
        quad[..., 1] *= sy

        filled, has_any = MouthTrack._make_filled(quad, valid)
        calibrated = ("calib_offset" in npz) or ("calib_scale" in npz) or ("calib_rotation" in npz)

        return MouthTrack(
            quads=quad,
            valid=valid,
            quads_filled=filled,
            has_any_valid=has_any,
            total=N,
            policy=policy,
            src_w=src_w,
            src_h=src_h,
            calibrated=calibrated,
        )

    @staticmethod
    def load_with_transform(
        path: str,
        base_w: int,
        base_h: int,
        video_w: int,
        video_h: int,
        policy: str = "hold"
    ) -> "MouthTrack | None":
        """
        Load track and apply scale-to-fill + center-crop transformation.

        video_w, video_h: 実際の動画サイズ
        base_w, base_h: 出力基準サイズ
        """
        if not path or (not os.path.isfile(path)):
            return None

        npz = np.load(path, allow_pickle=False)

        if "quad" in npz:
            quad = np.asarray(npz["quad"], dtype=np.float32)
            if quad.ndim != 3 or quad.shape[1:] != (4, 2):
                raise ValueError("mouth_track.npz: quad must be (N,4,2)")
        elif "bbox" in npz:
            bb = np.asarray(npz["bbox"], dtype=np.float32)
            if bb.ndim != 2 or bb.shape[1] != 4:
                raise ValueError("mouth_track.npz: bbox must be (N,4)")
            quad = MouthTrack._bbox_to_quad(bb)
        else:
            raise ValueError("mouth_track.npz must contain 'quad' or 'bbox'")

        N = int(quad.shape[0])
        if "valid" in npz:
            valid = np.asarray(npz["valid"], dtype=np.uint8).astype(bool)
            if valid.shape[0] != N:
                valid = np.ones((N,), dtype=bool)
        else:
            valid = np.ones((N,), dtype=bool)

        track_w = int(npz["w"]) if "w" in npz else video_w
        track_h = int(npz["h"]) if "h" in npz else video_h

        # Step 1: トラック座標を動画座標系に変換
        sx_track = float(video_w) / float(max(1, track_w))
        sy_track = float(video_h) / float(max(1, track_h))
        quad = quad.copy()
        quad[..., 0] *= sx_track
        quad[..., 1] *= sy_track

        # Step 2: scale-to-fill + center-crop の変換
        # s = max(base_w / video_w, base_h / video_h)
        s = max(base_w / max(1, video_w), base_h / max(1, video_h))
        rw = round(video_w * s)
        rh = round(video_h * s)
        dx = (rw - base_w) / 2.0
        dy = (rh - base_h) / 2.0

        # quad' = quad * s - (dx, dy)
        quad[..., 0] = quad[..., 0] * s - dx
        quad[..., 1] = quad[..., 1] * s - dy

        # quad中心が完全に画面外のフレームは無効化
        for i in range(N):
            cx = quad[i, :, 0].mean()
            cy = quad[i, :, 1].mean()
            if cx < 0 or cx >= base_w or cy < 0 or cy >= base_h:
                valid[i] = False

        filled, has_any = MouthTrack._make_filled(quad, valid)
        calibrated = ("calib_offset" in npz) or ("calib_scale" in npz) or ("calib_rotation" in npz)

        return MouthTrack(
            quads=quad,
            valid=valid,
            quads_filled=filled,
            has_any_valid=has_any,
            total=N,
            policy=policy,
            src_w=track_w,
            src_h=track_h,
            calibrated=calibrated,
        )

    def get_quad(self, frame_idx: int) -> np.ndarray | None:
        if self.total <= 0:
            return None
        if not self.has_any_valid:
            return None
        idx = int(frame_idx) % self.total
        if self.policy == "strict":
            if not bool(self.valid[idx]):
                return None
            return self.quads[idx]
        return self.quads_filled[idx]


# ============================================================
# BgVideo Class
# ============================================================

class BgVideo:
    """OpenCV VideoCaptureでmp4をループ再生してRGBフレームを返す + frame_idx を持つ"""

    def __init__(
        self,
        path: str,
        w: int,
        h: int,
        scale_mode: str = "fit",  # "fit" (aspect維持) or "fill" (scale-to-fill + center-crop)
    ):
        self.path = path
        self.cap = open_video_capture(path, retries=25, delay_sec=0.08)

        # 実際の動画サイズを取得
        self.src_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
        self.src_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)

        self.scale_mode = scale_mode
        self.target_w = w
        self.target_h = h

        if scale_mode == "fill":
            # scale-to-fill + center-crop
            self.w = w
            self.h = h
            s = max(w / max(1, self.src_w), h / max(1, self.src_h))
            self._fill_rw = round(self.src_w * s)
            self._fill_rh = round(self.src_h * s)
            self._fill_dx = (self._fill_rw - w) // 2
            self._fill_dy = (self._fill_rh - h) // 2
        else:
            # fit: アスペクト比維持
            req_aspect = w / max(1, h)
            src_aspect = self.src_w / max(1, self.src_h)
            if abs(req_aspect - src_aspect) > 0.01:
                self.w = w
                self.h = int(round(w / src_aspect))
            else:
                self.w = w
                self.h = h

        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps and fps > 1e-3 else 30.0

        nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.total_frames = nframes if nframes > 0 else 0
        self.frame_idx = -1

        self._acc = 0.0
        self._last_t = time.perf_counter()
        self._cached: np.ndarray | None = None

        fr = self._read_one()
        if fr is None:
            raise RuntimeError(f"Failed to read first frame: {path}")
        self._cached = fr

    def _read_one(self) -> np.ndarray | None:
        ret, bgr = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = -1
            ret, bgr = self.cap.read()
            if not ret:
                return None

        self.frame_idx += 1
        if self.total_frames:
            self.frame_idx %= self.total_frames

        if self.scale_mode == "fill":
            # scale-to-fill + center-crop
            if bgr.shape[1] != self._fill_rw or bgr.shape[0] != self._fill_rh:
                bgr = cv2.resize(bgr, (self._fill_rw, self._fill_rh), interpolation=cv2.INTER_AREA)
            # center-crop
            bgr = bgr[self._fill_dy:self._fill_dy + self.h, self._fill_dx:self._fill_dx + self.w]
        else:
            # fit mode
            if bgr.shape[1] != self.w or bgr.shape[0] != self.h:
                bgr = cv2.resize(bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def get_frame(self, now: float) -> np.ndarray:
        dt = now - self._last_t
        self._last_t = now
        self._acc += dt * self.fps
        n = int(self._acc)
        if n > 0:
            self._acc -= n
            for _ in range(n):
                fr = self._read_one()
                if fr is not None:
                    self._cached = fr
        assert self._cached is not None
        return self._cached

    def reset(self) -> None:
        """タイミングとフレームインデックスをリセット（動画切替時用）"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = -1
        self._acc = 0.0
        self._last_t = time.perf_counter()
        # 最初のフレームを読み込み
        fr = self._read_one()
        if fr is not None:
            self._cached = fr

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


# ============================================================
# Mouth Sprite Loading
# ============================================================

def load_mouth_sprites(mouth_dir: str, full_w: int, full_h: int) -> dict[str, np.ndarray]:
    """
    mouth_dir から口スプライトを読み込む。

    互換性:
    - 旧仕様: closed/half/open/u/e の5枚が必要
    - 新仕様: 最低 open.png があれば動作（不足分は open.png から自動生成して代用）
    """
    required = {"open": "open.png"}
    optional = {
        "closed": "closed.png",
        "half": "half.png",
        "u": "u.png",
        "e": "e.png",
    }

    def crop_full_canvas(rgba: np.ndarray, key: str) -> np.ndarray:
        if rgba.shape[0] == full_h and rgba.shape[1] == full_w:
            bbox = alpha_bbox(rgba)
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                rgba = rgba[y0:y1, x0:x1].copy()
        return rgba

    def variant_from_open(open_rgba: np.ndarray, key: str) -> np.ndarray:
        h, w = open_rgba.shape[:2]
        if key == "open":
            return open_rgba

        if key == "half":
            sx, sy = 1.00, 0.65
        elif key == "closed":
            sx, sy = 1.00, 0.25
        elif key == "u":
            sx, sy = 0.88, 0.55
        elif key == "e":
            sx, sy = 1.08, 0.55
        else:
            sx, sy = 1.00, 1.00

        # open.png から派生形状を自動生成する場合、e 形状のように横幅が元画像を
        # 上回るケースがある。元キャンバス外にはみ出すとブロードキャスト代入が
        # 失敗するため、貼り付け前に必ずキャンバス内へ収める。
        rw = max(2, min(int(round(w * sx)), w))
        rh = max(2, min(int(round(h * sy)), h))
        small = cv2.resize(open_rgba, (rw, rh), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        x0 = max(0, (w - rw) // 2)
        y0 = max(0, (h - rh) // 2)
        paste_w = min(rw, w - x0)
        paste_h = min(rh, h - y0)
        canvas[y0:y0 + paste_h, x0:x0 + paste_w] = small[:paste_h, :paste_w]

        if key == "closed":
            canvas[..., 3] = (canvas[..., 3].astype(np.float32) * 0.85).astype(np.uint8)

        return canvas

    sprites: dict[str, np.ndarray] = {}

    open_path = os.path.join(mouth_dir, required["open"])
    if not os.path.isfile(open_path):
        raise FileNotFoundError(f"mouth sprite not found (required): {open_path}")
    open_rgba = crop_full_canvas(load_rgba(open_path), "open")

    sprites["open"] = open_rgba

    for key, fn in optional.items():
        p = os.path.join(mouth_dir, fn)
        if os.path.isfile(p):
            rgba = crop_full_canvas(load_rgba(p), key)
            sprites[key] = rgba
        else:
            sprites[key] = variant_from_open(open_rgba, key)

    for k in ["closed", "half", "open", "u", "e"]:
        if k not in sprites:
            sprites[k] = open_rgba

    sizes = {(v.shape[1], v.shape[0]) for v in sprites.values()}
    if len(sizes) != 1:
        tw = max(s[0] for s in sizes)
        th = max(s[1] for s in sizes)
        out: dict[str, np.ndarray] = {}
        for k, im in sprites.items():
            if (im.shape[1], im.shape[0]) != (tw, th):
                out[k] = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            else:
                out[k] = im
        sprites = out

    return sprites


def _is_mouth_set_dir(p: str) -> bool:
    """Return True if `p` looks like a mouth sprite-set directory."""
    return os.path.isfile(os.path.join(p, "open.png"))


def discover_mouth_sets(mouth_dir: str) -> dict[str, str]:
    """
    Discover mouth sprite sets.

    Supported layouts:
      A) Single set (backward compatible):
         mouth_dir/
            open.png (required)
            closed.png / half.png / u.png / e.png (optional)

      B) Emotion sets:
         mouth_dir/
            Happy/ (open.png required)
            Sad/
            Angry/
            ...

    Returns:
        dict: {emotion_name: directory_path}
    """
    mouth_dir = os.path.abspath(mouth_dir)

    sets: dict[str, str] = {}

    if _is_mouth_set_dir(mouth_dir):
        sets["Default"] = mouth_dir

    if not os.path.isdir(mouth_dir):
        return sets

    for name in sorted(os.listdir(mouth_dir)):
        p = os.path.join(mouth_dir, name)
        if os.path.isdir(p) and _is_mouth_set_dir(p):
            sets[name] = p

    return sets


# ============================================================
# Emotion Mapping Utilities
# ============================================================

def _norm_token(s: str) -> str:
    return "".join(str(s).strip().lower().split())


def pick_mouth_set_for_label(set_names: list[str], label: str) -> str | None:
    """Map canonical emotion label -> available mouth set folder name."""
    if not set_names:
        return None
    label = str(label).strip().lower()

    syn = {
        "neutral": ["neutral", "default", "normal", "通常", "平常", "無感情", "ニュートラル"],
        "happy": ["happy", "smile", "嬉", "楽", "ハッピー", "笑"],
        "angry": ["angry", "mad", "怒", "イライラ"],
        "sad": ["sad", "cry", "泣", "悲"],
        "excited": ["excited", "excite", "fun", "party", "興奮", "ワクワク"],
    }
    wants = syn.get(label, [label])

    low = {_norm_token(n): n for n in set_names}
    for w in wants:
        key = _norm_token(w)
        if key in low:
            return low[key]

    for n in set_names:
        nlow = _norm_token(n)
        for w in wants:
            if _norm_token(w) in nlow:
                return n
    return None


def infer_label_from_set_name(set_name: str) -> str:
    n = _norm_token(set_name)
    if "happy" in n or "smile" in n or "嬉" in set_name or "楽" in set_name or "笑" in set_name:
        return "happy"
    if "angry" in n or "mad" in n or "怒" in set_name:
        return "angry"
    if "sad" in n or "cry" in n or "悲" in set_name or "泣" in set_name:
        return "sad"
    if "excited" in n or "excite" in n or "fun" in n or "興奮" in set_name or "ワクワク" in set_name:
        return "excited"
    if "neutral" in n or "default" in n or "normal" in n or "通常" in set_name or "平常" in set_name:
        return "neutral"
    return "neutral"


# ============================================================
# Emotion HUD Formatting
# ============================================================

EMOJI_BY_LABEL = {
    "neutral": "😐",
    "happy": "😊",
    "angry": "😠",
    "sad": "😢",
    "excited": "🤩",
}


def format_emotion_hud_text(label: str) -> str:
    label = str(label).strip().lower()
    emoji = EMOJI_BY_LABEL.get(label, "🙂")
    return f"{emoji} {label}"
