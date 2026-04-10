#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_sprite_extractor.py

動画から口スプライト（5種類のPNG）を自動抽出するコアモジュール。

機能:
1. 動画の全フレームから口quadを検出
2. 口の位置が安定しているフレーム群（クラスタ）を特定
3. 5種類の口形状を自動選別（open, closed, half, e, u）
4. 楕円マスク＋フェザーで透過PNG出力

使い方（コマンドライン）:
    python mouth_sprite_extractor.py --video loop.mp4 --out mouth/

使い方（モジュールとして）:
    from mouth_sprite_extractor import MouthSpriteExtractor
    extractor = MouthSpriteExtractor(video_path)
    extractor.analyze()
    extractor.extract_sprites(output_dir, feather_px=15)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from image_io import write_image_file
from python_exec import resolve_python_subprocess_executable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MouthFrameInfo:
    """1フレームの口情報"""
    # 既存フィールド
    frame_idx: int
    quad: np.ndarray          # (4, 2) float32
    center: np.ndarray        # (2,) float32 - 口の中心座標
    width: float              # 口quadの幅
    height: float             # 口quadの高さ
    confidence: float         # 検出信頼度
    valid: bool               # 検出が有効か

    # 新規フィールド（全てデフォルト値あり - 後方互換性のため）
    inner_darkness: float = 0.0       # 口内部の暗さ (0.0-1.0)
    opening_ratio: float = 0.0        # 開口度 (0.0-1.0)
    horizontal_stretch: float = 0.0   # 横への伸び (0.0-1.0)
    vertical_compression: float = 0.0  # 縦の圧縮 (0.0-1.0)
    lip_curvature: float = 0.0        # 唇の曲率 (-1.0〜1.0)
    score_open: float = 0.0           # openタイプのスコア
    score_closed: float = 0.0         # closedタイプのスコア
    score_half: float = 0.0           # halfタイプのスコア
    score_e: float = 0.0              # eタイプのスコア
    score_u: float = 0.0              # uタイプのスコア


@dataclass
class MouthTypeSelection:
    """5種類の口の選択結果"""
    open_idx: int             # 口を大きく開けたフレーム
    closed_idx: int           # 口を閉じたフレーム
    half_idx: int             # 半開きフレーム
    e_idx: int                # 横長の口フレーム
    u_idx: int                # すぼめた口フレーム
    
    def as_dict(self) -> Dict[str, int]:
        return {
            "open": self.open_idx,
            "closed": self.closed_idx,
            "half": self.half_idx,
            "e": self.e_idx,
            "u": self.u_idx,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quad_center(quad: np.ndarray) -> np.ndarray:
    """quadの中心座標を計算"""
    return quad.mean(axis=0).astype(np.float32)


def quad_wh(quad: np.ndarray) -> Tuple[float, float]:
    """quadの幅と高さを計算"""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    w = float(np.linalg.norm(quad[1] - quad[0]))
    h = float(np.linalg.norm(quad[3] - quad[0]))
    return w, h


def ensure_even_ge2(n: int) -> int:
    """偶数に丸める（最小2）"""
    n = int(n)
    if n < 2:
        return 2
    return n if (n % 2 == 0) else (n - 1)


# ---------------------------------------------------------------------------
# Track loading (compatible with face_track_anime_detector.py output)
# ---------------------------------------------------------------------------

def load_track_data(track_path: str, target_w: int, target_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mouth_track.npz からデータを読み込む。
    
    Returns:
        quads: (N, 4, 2) float32
        valid: (N,) bool
        confidence: (N,) float32 or None
    """
    npz = np.load(track_path, allow_pickle=False)
    
    if "quad" not in npz:
        raise ValueError("track must contain 'quad' (N,4,2)")
    
    quads = np.asarray(npz["quad"], dtype=np.float32)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError("quad must be (N,4,2)")
    
    N = int(quads.shape[0])
    
    # valid配列
    if "valid" in npz:
        valid = np.asarray(npz["valid"], dtype=np.uint8).astype(bool)
        if valid.shape[0] != N:
            valid = np.ones((N,), bool)
    else:
        valid = np.ones((N,), bool)
    
    # スケーリング
    src_w = int(npz["w"]) if "w" in npz else target_w
    src_h = int(npz["h"]) if "h" in npz else target_h
    sx = float(target_w) / float(max(1, src_w))
    sy = float(target_h) / float(max(1, src_h))
    quads = quads.copy()
    quads[..., 0] *= sx
    quads[..., 1] *= sy
    
    # confidence配列
    if "confidence" in npz:
        confidence = np.asarray(npz["confidence"], dtype=np.float32)
        if confidence.shape[0] != N:
            confidence = np.ones((N,), dtype=np.float32)
    else:
        confidence = np.ones((N,), dtype=np.float32)
    
    return quads, valid, confidence


# ---------------------------------------------------------------------------
# Position-aware clustering
# ---------------------------------------------------------------------------

def find_stable_position_cluster(
    centers: np.ndarray,
    valid: np.ndarray,
    distance_threshold: float = 50.0,
) -> np.ndarray:
    """
    口の中心座標が安定しているフレーム群を特定する。
    
    最も多くのフレームが集まっている位置クラスタを見つけ、
    そのクラスタに属するフレームのマスクを返す。
    
    Args:
        centers: (N, 2) 口の中心座標
        valid: (N,) 有効フレームマスク
        distance_threshold: クラスタ判定の距離閾値（ピクセル）
    
    Returns:
        cluster_mask: (N,) bool - クラスタに属するフレーム
    """
    N = len(centers)
    valid_indices = np.where(valid)[0]
    
    if len(valid_indices) < 5:
        # フレームが少なすぎる場合はすべて使用
        return valid.copy()
    
    valid_centers = centers[valid_indices]
    
    # 各有効フレームについて、近くにいくつのフレームがあるかカウント
    counts = np.zeros(len(valid_indices), dtype=np.int32)
    for i, c in enumerate(valid_centers):
        dists = np.linalg.norm(valid_centers - c, axis=1)
        counts[i] = np.sum(dists <= distance_threshold)
    
    # 最も密なフレームを基準にクラスタを構成
    best_idx = np.argmax(counts)
    best_center = valid_centers[best_idx]
    
    # 基準点からの距離でクラスタを決定
    dists_from_best = np.linalg.norm(valid_centers - best_center, axis=1)
    cluster_valid_mask = dists_from_best <= distance_threshold
    
    # 元のインデックスに変換
    cluster_mask = np.zeros(N, dtype=bool)
    for i, orig_idx in enumerate(valid_indices):
        if cluster_valid_mask[i]:
            cluster_mask[orig_idx] = True
    
    return cluster_mask


# ---------------------------------------------------------------------------
# Mouth type selection
# ---------------------------------------------------------------------------

def select_5_mouth_types(
    mouth_frames: List[MouthFrameInfo],
    cluster_mask: np.ndarray,
) -> MouthTypeSelection:
    """
    5種類の口タイプを自動選別する。
    
    選別基準:
    - open: 口の高さが最大
    - closed: 口の高さが最小
    - half: 高さが中央値付近
    - e: 幅/高さ比が最大（横長）
    - u: 幅が最小かつ高さが中程度（すぼめた口）
    
    Args:
        mouth_frames: 全フレームの口情報
        cluster_mask: 位置クラスタに属するフレームのマスク
    
    Returns:
        MouthTypeSelection
    """
    # クラスタ内の有効フレームのみを対象
    candidates = [
        mf for mf in mouth_frames 
        if mf.valid and cluster_mask[mf.frame_idx]
    ]
    
    if len(candidates) < 5:
        # フォールバック: 全有効フレームを使用
        candidates = [mf for mf in mouth_frames if mf.valid]
    
    if len(candidates) == 0:
        raise ValueError("No valid mouth frames found")
    
    # 各種メトリクス
    heights = np.array([mf.height for mf in candidates])
    widths = np.array([mf.width for mf in candidates])
    aspect_ratios = widths / np.maximum(heights, 1e-6)
    
    used_indices = set()
    
    def pick_best(scores: np.ndarray, maximize: bool = True) -> int:
        """最良のフレームを選択（既に使用済みは除外）"""
        sorted_indices = np.argsort(scores)
        if maximize:
            sorted_indices = sorted_indices[::-1]
        
        for idx in sorted_indices:
            frame_idx = candidates[idx].frame_idx
            if frame_idx not in used_indices:
                used_indices.add(frame_idx)
                return frame_idx
        
        # フォールバック（すべて使用済みの場合）
        return candidates[sorted_indices[0]].frame_idx
    
    # 1. open: 高さ最大
    open_idx = pick_best(heights, maximize=True)
    
    # 2. closed: 高さ最小
    closed_idx = pick_best(heights, maximize=False)
    
    # 3. half: 高さが中央値付近
    median_height = np.median(heights)
    half_scores = -np.abs(heights - median_height)  # 中央に近いほど高スコア
    half_idx = pick_best(half_scores, maximize=True)
    
    # 4. e: 横長（幅/高さ比が大きい）
    e_idx = pick_best(aspect_ratios, maximize=True)
    
    # 5. u: すぼめた口（幅が小さく、高さは中程度）
    # 幅が小さいものを優先、高さは極端でないもの
    height_median_dist = np.abs(heights - median_height)
    u_scores = -widths - 0.5 * height_median_dist  # 幅小さく、高さは中央寄り
    u_idx = pick_best(u_scores, maximize=True)
    
    return MouthTypeSelection(
        open_idx=open_idx,
        closed_idx=closed_idx,
        half_idx=half_idx,
        e_idx=e_idx,
        u_idx=u_idx,
    )


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def make_ellipse_mask(w: int, h: int, rx: int, ry: int) -> np.ndarray:
    """楕円マスクを生成（0/255）"""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    rx = int(max(1, min(rx, w // 2 - 1)))
    ry = int(max(1, min(ry, h // 2 - 1)))
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return mask


def feather_mask(mask_u8: np.ndarray, feather_px: int) -> np.ndarray:
    """マスクにフェザー（グラデーション）を適用"""
    if feather_px <= 0:
        return (mask_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)
    
    k = 2 * int(feather_px) + 1
    m = cv2.GaussianBlur(mask_u8, (k, k), sigmaX=0)
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Sprite extraction
# ---------------------------------------------------------------------------

def warp_frame_to_norm(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    norm_w: int,
    norm_h: int,
) -> np.ndarray:
    """フレームから口パッチを正規化空間に変換"""
    src = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([
        [0, 0],
        [norm_w - 1, 0],
        [norm_w - 1, norm_h - 1],
        [0, norm_h - 1],
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(
        frame_bgr,
        M,
        (int(norm_w), int(norm_h)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return patch


def extract_mouth_sprite(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    feather_px: int = 15,
    mask_scale: float = 0.85,
) -> np.ndarray:
    """
    フレームから口スプライトを抽出する。
    
    Args:
        frame_bgr: 入力フレーム（BGR）
        quad: 口のquad (4, 2)
        unified_w: 出力幅
        unified_h: 出力高さ
        feather_px: フェザー幅（ピクセル）
        mask_scale: マスクの楕円サイズ（0.0-1.0）
    
    Returns:
        rgba: (H, W, 4) uint8 - 透過PNG用
    """
    # 正規化空間に変換
    patch_bgr = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)
    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    
    # 楕円マスク生成
    rx = int((unified_w * mask_scale) * 0.5)
    ry = int((unified_h * mask_scale) * 0.5)
    mask_u8 = make_ellipse_mask(unified_w, unified_h, rx, ry)
    
    # フェザー適用
    mask_f = feather_mask(mask_u8, feather_px)
    
    # RGBA画像を生成
    rgba = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = patch_rgb
    rgba[:, :, 3] = (mask_f * 255).astype(np.uint8)
    
    return rgba


def compute_unified_size(
    mouth_frames: List[MouthFrameInfo],
    selected_indices: List[int],
    padding: float = 1.1,
) -> Tuple[int, int]:
    """
    選択されたフレームの口がすべて収まるサイズを計算。
    
    Args:
        mouth_frames: 全フレームの口情報
        selected_indices: 選択されたフレームインデックス
        padding: 余白係数
    
    Returns:
        (width, height)
    """
    idx_to_mf = {mf.frame_idx: mf for mf in mouth_frames}
    
    max_w = 0.0
    max_h = 0.0
    for idx in selected_indices:
        if idx in idx_to_mf:
            mf = idx_to_mf[idx]
            max_w = max(max_w, mf.width)
            max_h = max(max_h, mf.height)
    
    # パディングを適用して偶数に丸める
    w = ensure_even_ge2(int(max_w * padding))
    h = ensure_even_ge2(int(max_h * padding))
    
    return w, h


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class MouthSpriteExtractor:
    """口スプライト抽出器"""
    
    def __init__(self, video_path: str, track_path: str = ""):
        """
        Args:
            video_path: 入力動画のパス
            track_path: mouth_track.npz のパス（空の場合は自動検索）
        """
        self.video_path = video_path
        self.track_path = track_path
        
        # 動画情報
        self.vid_w = 0
        self.vid_h = 0
        self.fps = 0.0
        self.n_frames = 0
        
        # 解析結果
        self.mouth_frames: List[MouthFrameInfo] = []
        self.cluster_mask: Optional[np.ndarray] = None
        self.selection: Optional[MouthTypeSelection] = None
        self.unified_size: Optional[Tuple[int, int]] = None
        
        self._load_video_info()
    
    def _load_video_info(self):
        """動画の情報を取得"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        self.vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
    
    def _find_track_path(self) -> str:
        """トラックファイルを自動検索"""
        if self.track_path and os.path.isfile(self.track_path):
            return self.track_path
        
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        candidates = [
            os.path.join(video_dir, "mouth_track_calibrated.npz"),
            os.path.join(video_dir, "mouth_track.npz"),
        ]
        
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        
        return ""
    
    def _run_face_detector(self, callback=None) -> str:
        """face_track_anime_detector.py を実行してトラッキング"""
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        track_out = os.path.join(video_dir, "mouth_track.npz")
        
        # 既存ファイルがあれば使用
        if os.path.isfile(track_out):
            return track_out
        
        # face_track_anime_detector.py を実行
        script_dir = os.path.dirname(os.path.abspath(__file__))
        detector_script = os.path.join(script_dir, "face_track_anime_detector.py")
        
        if not os.path.isfile(detector_script):
            raise FileNotFoundError(f"Detector script not found: {detector_script}")
        
        cmd = [
            resolve_python_subprocess_executable(),
            detector_script,
            "--video", self.video_path,
            "--out", track_out,
            "--device", "auto",
        ]
        
        if callback:
            callback("Running face detector...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Face detector failed: {result.stderr}")
        
        return track_out
    
    def analyze(self, callback=None, position_threshold: float = 50.0) -> None:
        """
        動画を解析して口情報を取得。
        
        Args:
            callback: 進捗コールバック関数（文字列を受け取る）
            position_threshold: 位置クラスタリングの閾値（ピクセル）
        """
        # トラックファイルを取得
        track_path = self._find_track_path()
        if not track_path:
            if callback:
                callback("Track file not found. Running face detector...")
            track_path = self._run_face_detector(callback)
        
        if callback:
            callback(f"Loading track: {os.path.basename(track_path)}")
        
        # トラックデータ読み込み
        quads, valid, confidence = load_track_data(
            track_path, self.vid_w, self.vid_h
        )
        
        # フレーム情報を構築
        self.mouth_frames = []
        for i in range(len(quads)):
            quad = quads[i]
            center = quad_center(quad)
            w, h = quad_wh(quad)
            
            mf = MouthFrameInfo(
                frame_idx=i,
                quad=quad,
                center=center,
                width=w,
                height=h,
                confidence=float(confidence[i]),
                valid=bool(valid[i]),
            )
            self.mouth_frames.append(mf)
        
        if callback:
            callback(f"Analyzed {len(self.mouth_frames)} frames")
        
        # 位置クラスタリング
        centers = np.array([mf.center for mf in self.mouth_frames])
        valid_array = np.array([mf.valid for mf in self.mouth_frames])
        
        self.cluster_mask = find_stable_position_cluster(
            centers, valid_array, distance_threshold=position_threshold
        )
        
        cluster_count = int(self.cluster_mask.sum())
        if callback:
            callback(f"Found stable cluster with {cluster_count} frames")
        
        # 5種類の口を選別
        self.selection = select_5_mouth_types(self.mouth_frames, self.cluster_mask)
        
        # 統一サイズを計算
        selected_indices = list(self.selection.as_dict().values())
        self.unified_size = compute_unified_size(
            self.mouth_frames, selected_indices
        )
        
        if callback:
            callback(f"Selected frames: {self.selection.as_dict()}")
            callback(f"Unified size: {self.unified_size[0]}x{self.unified_size[1]}")
    
    def get_preview_sprites(self, feather_px: int = 15) -> Dict[str, np.ndarray]:
        """
        プレビュー用のスプライトを取得。
        
        Args:
            feather_px: フェザー幅
        
        Returns:
            {"open": rgba, "closed": rgba, ...}
        """
        if self.selection is None or self.unified_size is None:
            raise RuntimeError("Must call analyze() first")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")
        
        unified_w, unified_h = self.unified_size
        idx_to_mf = {mf.frame_idx: mf for mf in self.mouth_frames}
        
        sprites = {}
        for name, frame_idx in self.selection.as_dict().items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            
            mf = idx_to_mf[frame_idx]
            rgba = extract_mouth_sprite(
                frame, mf.quad, unified_w, unified_h, feather_px
            )
            sprites[name] = rgba
        
        cap.release()
        return sprites
    
    def extract_sprites(
        self,
        output_dir: str,
        feather_px: int = 15,
        callback=None,
    ) -> Dict[str, str]:
        """
        スプライトを抽出してファイルに保存。
        
        Args:
            output_dir: 出力ディレクトリ
            feather_px: フェザー幅
            callback: 進捗コールバック
        
        Returns:
            {"open": path, "closed": path, ...}
        """
        if self.selection is None or self.unified_size is None:
            raise RuntimeError("Must call analyze() first")
        
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # スプライトを取得
        sprites = self.get_preview_sprites(feather_px)
        
        # ファイルに保存
        saved_paths = {}
        for name, rgba in sprites.items():
            filename = f"{name}.png"
            filepath = os.path.join(output_dir, filename)
            
            # RGBA画像として保存
            ok = write_image_file(
                filepath,
                cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
            )
            if not ok:
                raise RuntimeError(f"Failed to save sprite: {filepath}")
            saved_paths[name] = filepath
            
            if callback:
                callback(f"Saved: {filename}")
        
        return saved_paths


# ---------------------------------------------------------------------------
# Utility functions for output directory
# ---------------------------------------------------------------------------

def get_unique_output_dir(base_dir: str) -> str:
    """
    一意の出力ディレクトリ名を生成。
    
    base_dir が存在しなければそのまま返す。
    存在すれば "_001", "_002" 等のサフィックスを付ける。
    """
    if not os.path.exists(base_dir):
        return base_dir
    
    i = 1
    while True:
        new_dir = f"{base_dir}_{i:03d}"
        if not os.path.exists(new_dir):
            return new_dir
        i += 1


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Extract mouth sprites from video"
    )
    ap.add_argument("--video", required=True, help="Input video file")
    ap.add_argument("--track", default="", help="mouth_track.npz (optional)")
    ap.add_argument("--out", default="", help="Output directory (default: mouth/ next to video)")
    ap.add_argument("--feather", type=int, default=15, help="Feather width in pixels")
    ap.add_argument("--position-threshold", type=float, default=50.0,
                    help="Position clustering threshold in pixels")
    args = ap.parse_args()
    
    if not os.path.isfile(args.video):
        print(f"[error] Video not found: {args.video}")
        return 1
    
    # 出力ディレクトリ
    if args.out:
        output_dir = args.out
    else:
        video_dir = os.path.dirname(os.path.abspath(args.video))
        output_dir = os.path.join(video_dir, "mouth")
    
    output_dir = get_unique_output_dir(output_dir)
    
    def log(msg: str):
        print(f"[info] {msg}")
    
    try:
        extractor = MouthSpriteExtractor(args.video, args.track)
        log(f"Video: {extractor.vid_w}x{extractor.vid_h} @ {extractor.fps:.2f}fps")
        
        extractor.analyze(callback=log, position_threshold=args.position_threshold)
        
        saved = extractor.extract_sprites(output_dir, args.feather, callback=log)
        
        log(f"Output directory: {output_dir}")
        log(f"Saved {len(saved)} sprites")
        
        return 0
        
    except Exception as e:
        print(f"[error] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
