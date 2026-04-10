#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_crop_estimator.py

自動切り抜きパラメータ推定モジュール。
複数の正規化パッチから最適なcrop値（マージン）を推定する。

bounds表現: マージン（絶対座標ではなく、端からのピクセル数）
- top: 上端から削るピクセル数
- bottom: 下端から削るピクセル数
- left: 左端から削るピクセル数
- right: 右端から削るピクセル数

merge規則: 最小マージンを採用（口が切れないように）
クランプ: パッチサイズの0-40%に制限
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class AutoCropEstimator:
    """自動切り抜きパラメータ推定器"""

    def __init__(self, norm_size: Tuple[int, int]):
        """
        Args:
            norm_size: (width, height) 正規化パッチのサイズ
        """
        self.norm_w, self.norm_h = norm_size

    def estimate_crop_params(self, patches: List[np.ndarray]) -> Dict[str, int]:
        """
        複数の正規化パッチから最適なcrop値（マージン）を推定

        Args:
            patches: 正規化パッチのリスト（BGR画像）

        Returns:
            マージン値 {"top": int, "bottom": int, "left": int, "right": int}
        """
        if not patches:
            return self._get_default_margins()

        all_margins: List[Dict[str, int]] = []

        for patch in patches:
            if patch is None or patch.size == 0:
                continue

            margins = self._detect_mouth_boundary_canny(patch)
            if margins is None:
                margins = self._get_default_margins()
            all_margins.append(margins)

        if not all_margins:
            return self._get_default_margins()

        return self._merge_margins(all_margins)

    def _detect_mouth_boundary_canny(self, patch: np.ndarray) -> Optional[Dict[str, int]]:
        """
        口の境界を検出し、マージンとして返す

        アニメ画像用に最適化（複合アプローチ）:
        - V値の閾値で口内部（暗い部分）を検出
        - 赤/ピンク検出（彩度が高く色相が赤系）
        - 最大連結成分を抽出してノイズを除去

        Args:
            patch: 正規化パッチ（BGR画像）

        Returns:
            マージン値、または検出失敗時はNone
        """
        h, w = patch.shape[:2]

        # HSV変換
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]  # 色相 (0-180)
        s_channel = hsv[:, :, 1]  # 彩度 (0-255)
        v_channel = hsv[:, :, 2]  # 明度 (0-255)

        # 方法1: V値で暗い部分（口内部）を検出
        _, mouth_interior = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY_INV)

        # 方法2: 赤/ピンク検出（彩度が高く、色相が赤系）
        # OpenCVのHは0-180（赤は0付近または180付近）
        red_mask1 = (h_channel < 15) & (s_channel > 60)
        red_mask2 = (h_channel > 165) & (s_channel > 60)
        red_mask = ((red_mask1 | red_mask2) * 255).astype(np.uint8)

        # 統合（グレースケール閾値は除外 - エッジノイズの原因）
        mouth_mask = cv2.bitwise_or(mouth_interior, red_mask)

        # モルフォロジー処理でノイズ除去と穴埋め
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mouth_mask = cv2.morphologyEx(mouth_mask, cv2.MORPH_CLOSE, kernel)
        mouth_mask = cv2.morphologyEx(mouth_mask, cv2.MORPH_OPEN, kernel)

        # エッジ領域を除外（5%マージン）
        edge_margin = int(min(w, h) * 0.05)
        edge_mask = np.zeros_like(mouth_mask)
        edge_mask[edge_margin:h-edge_margin, edge_margin:w-edge_margin] = 255
        mouth_mask = cv2.bitwise_and(mouth_mask, edge_mask)

        # 連結成分解析で最大の口領域を抽出
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mouth_mask, connectivity=8
        )

        if num_labels <= 1:
            # 口が検出されなかった（背景のみ）
            return None

        # 背景（ラベル0）を除いた最大の連結成分を選択
        # パッチ中央に近い大きな成分を優先
        center_y, center_x = h // 2, w // 2
        best_label = None
        best_score = -1

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 10:  # 小さすぎるノイズは無視
                continue

            cx, cy = centroids[label]
            # 中心からの距離（正規化）
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            dist_ratio = dist_from_center / max(max_dist, 1)

            # スコア = 面積 * (1 - 中心からの距離比)
            # 大きくて中央に近いほど高スコア
            score = area * (1.0 - 0.5 * dist_ratio)

            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            return None

        # 最大成分のみを抽出
        best_mask = (labels == best_label).astype(np.uint8) * 255

        # 口領域が存在する行/列を特定
        rows_with_mouth = np.any(best_mask > 0, axis=1)
        cols_with_mouth = np.any(best_mask > 0, axis=0)

        if not np.any(rows_with_mouth) or not np.any(cols_with_mouth):
            return None

        # 口が存在する範囲
        row_indices = np.where(rows_with_mouth)[0]
        col_indices = np.where(cols_with_mouth)[0]

        first_row = row_indices[0]
        last_row = row_indices[-1]
        first_col = col_indices[0]
        last_col = col_indices[-1]

        # パディングを追加（口を切らないように）
        padding = int(min(self.norm_w, self.norm_h) * 0.05)

        # マージンを計算（口領域の外側がマージン）
        top_margin = max(0, first_row - padding)
        bottom_margin = max(0, self.norm_h - 1 - last_row - padding)
        left_margin = max(0, first_col - padding)
        right_margin = max(0, self.norm_w - 1 - last_col - padding)

        return {
            "top": int(top_margin),
            "bottom": int(bottom_margin),
            "left": int(left_margin),
            "right": int(right_margin),
        }

    def _get_default_margins(self) -> Dict[str, int]:
        """
        固定マージンフォールバック（パッチサイズの10%）

        Returns:
            デフォルトのマージン値
        """
        margin_x = int(self.norm_w * 0.1)
        margin_y = int(self.norm_h * 0.1)
        return {
            "top": margin_y,
            "bottom": margin_y,
            "left": margin_x,
            "right": margin_x,
        }

    def _merge_margins(self, all_margins: List[Dict[str, int]]) -> Dict[str, int]:
        """
        最小マージンを採用（口が切れないように）

        Args:
            all_margins: 全パッチのマージンリスト

        Returns:
            マージされたマージン値（クランプ済み）
        """
        # 最大マージン（パッチサイズの40%）
        max_margin_x = int(self.norm_w * 0.4)
        max_margin_y = int(self.norm_h * 0.4)

        # 最小マージンを採用
        merged_top = min(m["top"] for m in all_margins)
        merged_bottom = min(m["bottom"] for m in all_margins)
        merged_left = min(m["left"] for m in all_margins)
        merged_right = min(m["right"] for m in all_margins)

        # クランプ [0, max_margin]
        return {
            "top": min(max(0, merged_top), max_margin_y),
            "bottom": min(max(0, merged_bottom), max_margin_y),
            "left": min(max(0, merged_left), max_margin_x),
            "right": min(max(0, merged_right), max_margin_x),
        }

    def apply_crop(self, patch: np.ndarray, margins: Dict[str, int]) -> np.ndarray:
        """
        マージンを適用してパッチを切り抜き

        Args:
            patch: 正規化パッチ（BGR or RGBA）
            margins: マージン値

        Returns:
            切り抜かれたパッチ
        """
        h, w = patch.shape[:2]

        top = margins.get("top", 0)
        bottom = margins.get("bottom", 0)
        left = margins.get("left", 0)
        right = margins.get("right", 0)

        # 座標を計算（範囲チェック）
        y1 = min(top, h - 1)
        y2 = max(h - bottom, y1 + 1)
        x1 = min(left, w - 1)
        x2 = max(w - right, x1 + 1)

        return patch[y1:y2, x1:x2].copy()

    def margins_to_crop_rect(
        self,
        margins: Dict[str, int],
    ) -> Tuple[int, int, int, int]:
        """
        マージンを切り抜き矩形に変換

        Args:
            margins: マージン値

        Returns:
            (x, y, width, height) の切り抜き矩形
        """
        top = margins.get("top", 0)
        bottom = margins.get("bottom", 0)
        left = margins.get("left", 0)
        right = margins.get("right", 0)

        x = left
        y = top
        width = self.norm_w - left - right
        height = self.norm_h - top - bottom

        # 最小サイズを保証
        width = max(width, 1)
        height = max(height, 1)

        return (x, y, width, height)
