#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_feature_analyzer.py

口パッチから特徴量を抽出するモジュール。
正規化された口パッチ画像から、分類に必要な特徴量を計算する。

特徴量:
- inner_darkness: 口内部の暗さ (0.0-1.0)
- opening_ratio: 開口度 (0.0-1.0)
- horizontal_stretch: 横への伸び (0.0-1.0)
- vertical_compression: 縦の圧縮 (0.0-1.0)
- lip_curvature: 唇の曲率 (-1.0〜1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class MouthFeatures:
    """口パッチから抽出した特徴量"""
    inner_darkness: float = 0.0      # 口内部の暗さ (0.0-1.0)
    opening_ratio: float = 0.0       # 開口度 (0.0-1.0)
    horizontal_stretch: float = 0.0  # 横への伸び (0.0-1.0)
    vertical_compression: float = 0.0  # 縦の圧縮 (0.0-1.0)
    lip_curvature: float = 0.0       # 唇の曲率 (-1.0〜1.0)


class MouthFeatureAnalyzer:
    """口パッチから特徴量を抽出するアナライザ"""

    def __init__(self, norm_size: Tuple[int, int]):
        """
        Args:
            norm_size: (width, height) 正規化パッチのサイズ
        """
        self.norm_w, self.norm_h = norm_size

    def analyze_frame(self, patch_bgr: np.ndarray) -> MouthFeatures:
        """
        正規化された口パッチから特徴量を抽出

        Args:
            patch_bgr: warp_frame_to_normで生成した正規化パッチ (BGR)

        Returns:
            MouthFeatures: 抽出された特徴量
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return MouthFeatures()

        # HSV変換
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]  # 色相 (0-180)
        s_channel = hsv[:, :, 1]  # 彩度 (0-255)
        v_channel = hsv[:, :, 2]  # 明度 (0-255)

        # 口領域検出（複合アプローチ）
        binary = self._detect_mouth_region(h_channel, s_channel, v_channel)

        # 各特徴量を計算
        inner_darkness = self._calc_inner_darkness(v_channel)
        opening_ratio = self._calc_opening_ratio(binary)
        horizontal_stretch = self._calc_horizontal_stretch(binary)
        vertical_compression = self._calc_vertical_compression(v_channel)
        lip_curvature = self._calc_lip_curvature(binary)

        # 彩度ベースの開口度も計算（明るい口用）
        saturation_opening = self._calc_saturation_opening(s_channel, h_channel)

        # opening_ratioの最大値を採用（両方の検出方法のうち良い方）
        final_opening = max(opening_ratio, saturation_opening)

        return MouthFeatures(
            inner_darkness=inner_darkness,
            opening_ratio=final_opening,
            horizontal_stretch=horizontal_stretch,
            vertical_compression=vertical_compression,
            lip_curvature=lip_curvature,
        )

    def _detect_mouth_region(
        self,
        h_channel: np.ndarray,
        s_channel: np.ndarray,
        v_channel: np.ndarray,
    ) -> np.ndarray:
        """
        口領域を検出（複合アプローチ）

        2つの方法を組み合わせ:
        1. V < 80: 暗い口内部（一部のアニメスタイル用）
        2. 赤/ピンク検出: 彩度が高く色相が赤系（明るい口用）

        Args:
            h_channel: 色相チャンネル (0-180)
            s_channel: 彩度チャンネル (0-255)
            v_channel: 明度チャンネル (0-255)

        Returns:
            口領域の二値マスク
        """
        # 方法1: V < 80（暗い口内部）
        _, dark_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY_INV)

        # 方法2: 赤/ピンク検出（彩度が高く、色相が赤系）
        # OpenCVのHは0-180（赤は0付近または180付近）
        # 赤: H < 15 または H > 165、かつ S > 60
        red_mask1 = (h_channel < 15) & (s_channel > 60)
        red_mask2 = (h_channel > 165) & (s_channel > 60)
        red_mask = ((red_mask1 | red_mask2) * 255).astype(np.uint8)

        # 両方を統合（OR）
        binary = cv2.bitwise_or(dark_mask, red_mask)

        return binary

    def _calc_saturation_opening(
        self,
        s_channel: np.ndarray,
        h_channel: np.ndarray,
    ) -> float:
        """
        彩度ベースの開口度を計算（明るい赤/ピンクの口用）

        中央領域の彩度が高いピクセル（赤/ピンク）の比率

        Args:
            s_channel: 彩度チャンネル
            h_channel: 色相チャンネル

        Returns:
            開口度 (0.0-1.0)
        """
        h, w = s_channel.shape

        # 中央50%領域
        y1 = h // 4
        y2 = h - h // 4
        x1 = w // 4
        x2 = w - w // 4

        if y2 <= y1 or x2 <= x1:
            return 0.0

        center_s = s_channel[y1:y2, x1:x2]
        center_h = h_channel[y1:y2, x1:x2]

        if center_s.size == 0:
            return 0.0

        # 赤/ピンクの検出（彩度 > 60、色相が赤系）
        red_mask1 = (center_h < 15) & (center_s > 60)
        red_mask2 = (center_h > 165) & (center_s > 60)
        red_pixels = np.sum(red_mask1 | red_mask2)
        total_pixels = center_s.size

        saturation_opening = float(red_pixels) / float(total_pixels)

        # クランプ [0.0, 1.0]
        return float(np.clip(saturation_opening, 0.0, 1.0))

    def _calc_inner_darkness(self, v_channel: np.ndarray) -> float:
        """
        口内部の暗さを計算

        中央50%領域の平均明度を計算し、暗さに変換
        """
        h, w = v_channel.shape

        # 中央50%領域
        y1 = h // 4
        y2 = h - h // 4
        x1 = w // 4
        x2 = w - w // 4

        if y2 <= y1 or x2 <= x1:
            return 0.0

        center_region = v_channel[y1:y2, x1:x2]
        if center_region.size == 0:
            return 0.0

        # 正規化: mean_v / 255.0
        mean_v = float(np.mean(center_region))

        # 暗さに変換: inner_darkness = 1.0 - (mean_v / 255.0)
        inner_darkness = 1.0 - (mean_v / 255.0)

        # クランプ [0.0, 1.0]
        return float(np.clip(inner_darkness, 0.0, 1.0))

    def _calc_opening_ratio(self, binary: np.ndarray) -> float:
        """
        開口度を計算

        中央領域の暗いピクセル数 / 総ピクセル数
        """
        h, w = binary.shape

        # 中央領域
        y1 = h // 4
        y2 = h - h // 4
        x1 = w // 4
        x2 = w - w // 4

        if y2 <= y1 or x2 <= x1:
            return 0.0

        center_region = binary[y1:y2, x1:x2]
        if center_region.size == 0:
            return 0.0

        # 白ピクセル（二値化で反転済みなので口内部は白）の割合
        dark_pixels = np.sum(center_region > 0)
        total_pixels = center_region.size

        opening_ratio = float(dark_pixels) / float(total_pixels)

        # クランプ [0.0, 1.0]
        return float(np.clip(opening_ratio, 0.0, 1.0))

    def _calc_horizontal_stretch(self, binary: np.ndarray) -> float:
        """
        横への伸びを計算

        二値化領域の水平方向最大幅 / 垂直方向最大高さ
        """
        # 前景ピクセルの座標を取得
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            # 欠損時処理: horizontal_stretch = 0.0
            return 0.0

        # 行方向（y）と列方向（x）の範囲
        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        width = float(x_coords.max() - x_coords.min() + 1)
        height = float(y_coords.max() - y_coords.min() + 1)

        if height == 0:
            # 欠損時処理: horizontal_stretch = 0.0
            return 0.0

        aspect = width / height

        # 正規化: min(w/h, 3.0) / 3.0 でスケール（最大3:1を想定）
        horizontal_stretch = min(aspect, 3.0) / 3.0

        # クランプ [0.0, 1.0]
        return float(np.clip(horizontal_stretch, 0.0, 1.0))

    def _calc_vertical_compression(self, v_channel: np.ndarray) -> float:
        """
        縦の圧縮度を計算

        パッチ上25%領域と下25%領域のV平均値の差
        """
        h, w = v_channel.shape

        # 上25%と下25%領域
        top_h = max(1, h // 4)
        bottom_h = max(1, h // 4)

        top_region = v_channel[:top_h, :]
        bottom_region = v_channel[-bottom_h:, :]

        if top_region.size == 0 or bottom_region.size == 0:
            return 0.0

        top_mean = float(np.mean(top_region))
        bottom_mean = float(np.mean(bottom_region))

        # 差を計算
        diff = abs(top_mean - bottom_mean)

        # 正規化: diff / 255.0
        # 圧縮度に変換: vertical_compression = 1.0 - (diff / 255.0)
        vertical_compression = 1.0 - (diff / 255.0)

        # クランプ [0.0, 1.0]
        return float(np.clip(vertical_compression, 0.0, 1.0))

    def _calc_lip_curvature(self, binary: np.ndarray) -> float:
        """
        唇の曲率を計算

        二値化領域の左端/右端（口角）のy座標と、中央のy座標を比較
        正の値: 口角が上がっている（横広がり「い」の形状）
        負の値: 口角が下がっている（すぼめ「う」の形状）
        """
        h, w = binary.shape

        # 前景ピクセルの座標を取得
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) == 0:
            # 欠損時処理: lip_curvature = 0.0
            return 0.0

        y_coords = coords[:, 0]
        x_coords = coords[:, 1]

        # 左端、右端、中央の領域を定義
        x_min = x_coords.min()
        x_max = x_coords.max()
        x_range = x_max - x_min

        if x_range < 3:
            # 幅が狭すぎる場合
            return 0.0

        # 左10%、右10%、中央20%の領域
        left_thresh = x_min + x_range * 0.1
        right_thresh = x_max - x_range * 0.1
        center_left = x_min + x_range * 0.4
        center_right = x_min + x_range * 0.6

        # 各領域のy座標を取得
        left_mask = x_coords <= left_thresh
        right_mask = x_coords >= right_thresh
        center_mask = (x_coords >= center_left) & (x_coords <= center_right)

        left_y = y_coords[left_mask] if np.any(left_mask) else np.array([])
        right_y = y_coords[right_mask] if np.any(right_mask) else np.array([])
        center_y = y_coords[center_mask] if np.any(center_mask) else np.array([])

        if len(left_y) == 0 or len(right_y) == 0 or len(center_y) == 0:
            # 欠損時処理: lip_curvature = 0.0
            return 0.0

        # 各領域の中央y座標
        left_y_mean = np.mean(left_y)
        right_y_mean = np.mean(right_y)
        center_y_mean = np.mean(center_y)

        # 口角の平均y座標
        corner_y_avg = (left_y_mean + right_y_mean) / 2.0

        # 曲率を計算: (center_y - corner_y_avg) / (norm_h / 2)
        # 正の値 = 中央が下（y座標が大きい）= 口角が上がっている
        curvature = (center_y_mean - corner_y_avg) / (self.norm_h / 2.0)

        # クランプ [-1.0, 1.0]
        return float(np.clip(curvature, -1.0, 1.0))
