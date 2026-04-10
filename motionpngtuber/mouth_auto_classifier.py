#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_auto_classifier.py

口タイプの自動分類モジュール。
特徴量に基づいて5種類の口タイプ（open, closed, half, e, u）を自動選択する。

スコア計算:
- 全スコアは正規化された無次元量のみを使用
- 絶対的なheight/widthは使用しない
- opening_ratio, inner_darkness等の正規化特徴量を使用
- アスペクト比（width/height）は無次元量なので使用可
"""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from motionpngtuber.mouth_sprite_extractor import MouthFrameInfo


class MouthAutoClassifier:
    """口タイプの自動分類器"""

    def calculate_type_scores(
        self,
        mf: "MouthFrameInfo",
        opening_stats: dict = None,
    ) -> None:
        """
        各口タイプのスコアを計算してMouthFrameInfoに設定

        全スコアは正規化された無次元量のみを使用:
        - 絶対的なheight/widthは使用しない
        - opening_ratio, inner_darkness等の正規化特徴量を使用
        - quadは常に正方形のため、aspect_ratioは使用しない

        Args:
            mf: 特徴量が設定されたMouthFrameInfo
            opening_stats: opening_ratioの統計情報 {min, max, median, range}
        """
        if opening_stats is None:
            opening_stats = {"min": 0.0, "max": 1.0, "median": 0.5, "range": 1.0}

        open_min = opening_stats["min"]
        open_max = opening_stats["max"]
        open_range = opening_stats["range"]
        open_median = opening_stats["median"]

        # 正規化された開口度（0=最小、1=最大）
        if open_range > 0.001:
            normalized_opening = (mf.opening_ratio - open_min) / open_range
        else:
            normalized_opening = 0.5

        # open: 最も大きく開いた口を検出
        # opening_ratioが高いほどスコアが高い
        mf.score_open = normalized_opening

        # closed: 最も閉じた口を検出
        # opening_ratioが低いほどスコアが高い
        mf.score_closed = 1.0 - normalized_opening

        # half: 開口度が中間（0.3〜0.7の範囲）に近いほど高スコア
        # 中央値への近さで評価
        mid_target = 0.5  # 正規化空間での中央
        distance_from_mid = abs(normalized_opening - mid_target)
        # 中間からの距離が0.2以内なら高スコア、それ以上は急激に下がる
        mf.score_half = max(0, 1.0 - distance_from_mid * 3.0)

        # e: 「い」の形 - 中程度の開き + 横に広がった口
        # half同様の開口度制約 + 形状特徴
        e_opening_score = max(0, 1.0 - distance_from_mid * 2.5)
        e_shape_score = mf.horizontal_stretch
        mf.score_e = e_opening_score * 0.6 + e_shape_score * 0.4

        # u: 「う」の形 - 中程度の開き + すぼめた口（横幅が狭い）
        # half同様の開口度制約 + 形状特徴（横幅が狭い = 1 - horizontal_stretch）
        u_opening_score = max(0, 1.0 - distance_from_mid * 2.5)
        u_shape_score = 1.0 - mf.horizontal_stretch  # 横幅が狭いほど高スコア
        mf.score_u = u_opening_score * 0.6 + u_shape_score * 0.4

    def auto_select_5_types(
        self,
        mouth_frames: List["MouthFrameInfo"],
    ) -> Dict[str, int]:
        """
        5種類の口タイプを自動選択

        スコア計算:
        - opening_ratioの統計情報（min, max, median, range）を計算
        - 正規化された開口度を使用してスコアを計算
        - half/e/u は「中程度の開き」を持つフレームから選択

        同点時の優先規則:
        - confidence（検出信頼度）が高いフレームを優先
        - それでも同点の場合はframe_idxが小さいフレームを優先

        Args:
            mouth_frames: 特徴量が設定されたMouthFrameInfoのリスト

        Returns:
            選択されたフレームのインデックス {"open": idx, "closed": idx, ...}
            有効フレームが不足している場合は選択可能なタイプのみ返す
        """
        # 有効フレームのみを対象
        valid_frames = [mf for mf in mouth_frames if mf.valid]

        if len(valid_frames) == 0:
            # 有効フレームが0件の場合は空のDictを返す
            return {}

        # opening_ratioの統計情報を計算
        opening_ratios = [mf.opening_ratio for mf in valid_frames]
        opening_stats = {
            "min": float(np.min(opening_ratios)),
            "max": float(np.max(opening_ratios)),
            "median": float(np.median(opening_ratios)),
            "range": float(np.max(opening_ratios) - np.min(opening_ratios)),
        }

        # 各フレームのスコアを計算
        for mf in valid_frames:
            self.calculate_type_scores(mf, opening_stats)

        # 各タイプで最高スコアのフレームを選択（重複排除）
        selected: Dict[str, int] = {}
        used_indices: set = set()

        # 選択順序: open, closed, e, u, half
        type_score_attrs = [
            ("open", "score_open"),
            ("closed", "score_closed"),
            ("e", "score_e"),
            ("u", "score_u"),
            ("half", "score_half"),
        ]

        for type_name, score_attr in type_score_attrs:
            # 未使用の候補をスコア順にソート
            candidates = sorted(
                [mf for mf in valid_frames if mf.frame_idx not in used_indices],
                key=lambda mf: (
                    getattr(mf, score_attr),  # スコア（高い順）
                    mf.confidence,             # 信頼度（高い順）
                    -mf.frame_idx,             # フレーム番号（小さい順）
                ),
                reverse=True
            )

            if candidates:
                best = candidates[0]
                selected[type_name] = best.frame_idx
                used_indices.add(best.frame_idx)

        return selected

    def classify_single_frame(self, mf: "MouthFrameInfo") -> str:
        """
        単一フレームの口タイプを分類

        Args:
            mf: 特徴量が設定されたMouthFrameInfo

        Returns:
            口タイプ名 ("open", "closed", "half", "e", "u")
        """
        self.calculate_type_scores(mf)

        # 最高スコアのタイプを返す
        scores = {
            "open": mf.score_open,
            "closed": mf.score_closed,
            "e": mf.score_e,
            "u": mf.score_u,
            "half": mf.score_half,
        }

        return max(scores, key=lambda k: scores[k])
