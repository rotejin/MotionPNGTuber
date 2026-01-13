#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_sprite_extractor_gui.py

動画から口スプライト（5種類のPNG）を自動抽出するGUIツール。

機能:
1. 動画を選択（ドラッグ&ドロップ対応）
2. 解析ボタンで口トラッキング→10枚の候補を表示（口の開き具合順）
3. 候補に1-5の数字を入力して手動割り当て
4. 切り取り範囲（上下左右）とフェザー幅を別々に調整
5. 「更新」ボタンでプレビュー更新、「出力」ボタンでPNG保存

使い方:
    python mouth_sprite_extractor_gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Tkinter imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Optional: drag & drop support
_HAS_TK_DND = False
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    _HAS_TK_DND = True
except Exception:
    _HAS_TK_DND = False

# PIL for image display in Tkinter
try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    print("[warn] PIL not installed. Preview will be limited.")

# Core extractor module
from mouth_sprite_extractor import (
    MouthSpriteExtractor,
    MouthFrameInfo,
    get_unique_output_dir,
    quad_wh,
    warp_frame_to_norm,
    make_ellipse_mask,
    feather_mask,
    ensure_even_ge2,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Mouth Sprite Extractor"
CANDIDATE_COUNT = 10  # 候補フレーム数
THUMB_SIZE = 70       # サムネイルサイズ
PREVIEW_SIZE = 150    # プレビューサイズ（1.5倍に拡大）
DEFAULT_FEATHER = 15
DEFAULT_CROP = 0      # デフォルトの切り取り余白
MAX_CROP = 100        # 切り取り範囲の最大値
MAX_FEATHER = 40


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def create_checkerboard(w: int, h: int, cell_size: int = 10) -> np.ndarray:
    """透過表示用のチェッカーボード背景を生成"""
    board = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            if (x // cell_size + y // cell_size) % 2 == 0:
                board[y:y+cell_size, x:x+cell_size] = 200
            else:
                board[y:y+cell_size, x:x+cell_size] = 255
    return board


def composite_on_checkerboard(rgba: np.ndarray) -> np.ndarray:
    """RGBAをチェッカーボード上に合成"""
    h, w = rgba.shape[:2]
    board = create_checkerboard(w, h)
    
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3].astype(np.float32)
    
    result = board.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    return result.astype(np.uint8)


def numpy_to_photoimage(img_bgr: np.ndarray, size: int) -> Optional["ImageTk.PhotoImage"]:
    """BGR numpy配列をPhotoImageに変換"""
    if not _HAS_PIL:
        return None
    
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    
    # アスペクト比を維持してリサイズ
    w, h = img.size
    scale = min(size / max(w, 1), size / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    return ImageTk.PhotoImage(img)


def extract_sprite_with_crop(
    frame_bgr: np.ndarray,
    quad: np.ndarray,
    unified_w: int,
    unified_h: int,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    feather_px: int = 15,
) -> np.ndarray:
    """
    切り取り範囲とフェザーを適用してスプライトを抽出。
    """
    # 正規化空間に変換
    patch = warp_frame_to_norm(frame_bgr, quad, unified_w, unified_h)
    
    # 切り取り範囲を適用した楕円マスク
    # 楕円の中心をずらし、半径を調整
    # 上を削る = 楕円を下にずらす、下を削る = 楕円を上にずらす
    cx = unified_w // 2 + (crop_right - crop_left) // 2
    cy = unified_h // 2 + (crop_top - crop_bottom) // 2  # 方向を修正
    rx = (unified_w - crop_left - crop_right) // 2
    ry = (unified_h - crop_top - crop_bottom) // 2
    
    rx = max(1, min(rx, unified_w // 2 - 1))
    ry = max(1, min(ry, unified_h // 2 - 1))
    
    mask = np.zeros((unified_h, unified_w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    
    # フェザー適用
    mask_f = feather_mask(mask, feather_px)
    
    # RGBA画像を生成
    rgba = np.zeros((unified_h, unified_w, 4), dtype=np.uint8)
    rgba[:, :, :3] = patch
    rgba[:, :, 3] = (mask_f * 255).astype(np.uint8)
    
    return rgba


# ---------------------------------------------------------------------------
# Main GUI Application
# ---------------------------------------------------------------------------

class MouthSpriteExtractorApp(tk.Tk if not _HAS_TK_DND else TkinterDnD.Tk):
    """口スプライト抽出GUIアプリケーション"""
    
    def __init__(self):
        super().__init__()
        
        self.title(APP_TITLE)
        self.geometry("950x850")
        self.resizable(True, True)
        
        # State
        self.video_path: str = ""
        self.extractor: Optional[MouthSpriteExtractor] = None
        self.candidate_frames: List[MouthFrameInfo] = []  # 候補フレーム（開き具合順）
        self.candidate_images: List[Optional["ImageTk.PhotoImage"]] = []
        self.assignments: Dict[int, int] = {}  # candidate_idx -> mouth_type (1-5)
        self.preview_sprites: Dict[str, np.ndarray] = {}
        self.preview_images: Dict[str, "ImageTk.PhotoImage"] = {}
        self.unified_size: Optional[Tuple[int, int]] = None
        self.is_analyzing = False
        
        # Cached video capture for preview
        self._cached_cap: Optional[cv2.VideoCapture] = None
        
        # Log queue for thread-safe logging
        self.log_queue: queue.Queue[str] = queue.Queue()
        
        # Build UI
        self._build_ui()
        
        # Start log polling
        self._poll_logs()
    
    def _build_ui(self):
        """UIを構築"""
        # Main frame with padding
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Video selection ---
        video_frame = ttk.LabelFrame(main_frame, text="動画ファイル", padding=5)
        video_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.video_var = tk.StringVar()
        video_entry = ttk.Entry(video_frame, textvariable=self.video_var, state="readonly")
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        video_btn = ttk.Button(video_frame, text="選択...", command=self._on_select_video)
        video_btn.pack(side=tk.RIGHT)
        
        # Drag & drop support
        if _HAS_TK_DND:
            video_entry.drop_target_register(DND_FILES)
            video_entry.dnd_bind("<<Drop>>", self._on_drop_video)
        
        # --- Analyze button ---
        self.analyze_btn = ttk.Button(
            main_frame, text="解析", command=self._on_analyze
        )
        self.analyze_btn.pack(fill=tk.X, pady=(0, 10))
        
        # --- Candidate frames area ---
        cand_frame = ttk.LabelFrame(main_frame, text="候補フレーム（口の開き具合順）- 1-5の数字で割り当て", padding=5)
        cand_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scrollable candidate area
        cand_canvas = tk.Canvas(cand_frame, height=130)
        cand_canvas.pack(fill=tk.X, expand=True)
        
        self.cand_inner = ttk.Frame(cand_canvas)
        cand_canvas.create_window((0, 0), window=self.cand_inner, anchor=tk.NW)
        
        # Candidate slots
        self.cand_labels: List[ttk.Label] = []
        self.cand_entries: List[ttk.Entry] = []
        self.cand_vars: List[tk.StringVar] = []
        self.cand_frame_labels: List[ttk.Label] = []
        
        for i in range(CANDIDATE_COUNT):
            col_frame = ttk.Frame(self.cand_inner)
            col_frame.pack(side=tk.LEFT, padx=3)
            
            # サムネイル
            thumb_label = ttk.Label(col_frame, text="", width=10, anchor=tk.CENTER, relief=tk.SUNKEN)
            thumb_label.pack()
            self.cand_labels.append(thumb_label)
            
            # フレーム番号
            frame_label = ttk.Label(col_frame, text="", font=("", 8))
            frame_label.pack()
            self.cand_frame_labels.append(frame_label)
            
            # 割り当て入力
            var = tk.StringVar()
            entry = ttk.Entry(col_frame, textvariable=var, width=3, justify=tk.CENTER)
            entry.pack()
            self.cand_vars.append(var)
            self.cand_entries.append(entry)
        
        # 凡例
        legend_frame = ttk.Frame(cand_frame)
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Label(legend_frame, text="1=open  2=closed  3=half  4=e  5=u", font=("", 9)).pack()
        
        # --- Crop settings ---
        crop_frame = ttk.LabelFrame(main_frame, text="切り取り範囲（余白を削る）", padding=5)
        crop_frame.pack(fill=tk.X, pady=(0, 10))
        
        crop_grid = ttk.Frame(crop_frame)
        crop_grid.pack()
        
        self.crop_vars = {
            "top": tk.IntVar(value=DEFAULT_CROP),
            "bottom": tk.IntVar(value=DEFAULT_CROP),
            "left": tk.IntVar(value=DEFAULT_CROP),
            "right": tk.IntVar(value=DEFAULT_CROP),
        }
        
        # 上
        ttk.Label(crop_grid, text="上:").grid(row=0, column=0, sticky=tk.E)
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["top"], 
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=1)
        self.crop_labels = {}
        self.crop_labels["top"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["top"].grid(row=0, column=2)
        
        # 下
        ttk.Label(crop_grid, text="下:").grid(row=1, column=0, sticky=tk.E)
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["bottom"],
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=1)
        self.crop_labels["bottom"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["bottom"].grid(row=1, column=2)
        
        # 左
        ttk.Label(crop_grid, text="左:").grid(row=0, column=3, sticky=tk.E, padx=(20, 0))
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["left"],
                  orient=tk.HORIZONTAL, length=100).grid(row=0, column=4)
        self.crop_labels["left"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["left"].grid(row=0, column=5)
        
        # 右
        ttk.Label(crop_grid, text="右:").grid(row=1, column=3, sticky=tk.E, padx=(20, 0))
        ttk.Scale(crop_grid, from_=0, to=MAX_CROP, variable=self.crop_vars["right"],
                  orient=tk.HORIZONTAL, length=100).grid(row=1, column=4)
        self.crop_labels["right"] = ttk.Label(crop_grid, text="0px", width=5)
        self.crop_labels["right"].grid(row=1, column=5)
        
        # --- Feather slider ---
        feather_frame = ttk.LabelFrame(main_frame, text="フェザー幅", padding=5)
        feather_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.feather_var = tk.IntVar(value=DEFAULT_FEATHER)
        self.feather_slider = ttk.Scale(
            feather_frame,
            from_=0,
            to=MAX_FEATHER,
            orient=tk.HORIZONTAL,
            variable=self.feather_var,
        )
        self.feather_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.feather_label = ttk.Label(feather_frame, text=f"{DEFAULT_FEATHER}px", width=6)
        self.feather_label.pack(side=tk.RIGHT)
        
        # --- Update button ---
        self.update_btn = ttk.Button(
            main_frame, text="プレビュー更新", command=self._on_update_preview, state=tk.DISABLED
        )
        self.update_btn.pack(fill=tk.X, pady=(0, 10))
        
        # --- Preview area ---
        preview_frame = ttk.LabelFrame(main_frame, text="出力プレビュー", padding=5)
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(expand=True)
        
        self.preview_labels: Dict[str, ttk.Label] = {}
        self.out_frame_labels: Dict[str, ttk.Label] = {}
        
        mouth_names = ["open", "closed", "half", "e", "u"]
        
        for i, name in enumerate(mouth_names):
            col_frame = ttk.Frame(preview_inner)
            col_frame.grid(row=0, column=i, padx=5, pady=5)
            
            preview_label = ttk.Label(
                col_frame,
                text="(未選択)",
                width=12,
                anchor=tk.CENTER,
                relief=tk.SUNKEN,
            )
            preview_label.pack()
            self.preview_labels[name] = preview_label
            
            name_label = ttk.Label(col_frame, text=name, font=("", 9, "bold"))
            name_label.pack()
            
            frame_label = ttk.Label(col_frame, text="", font=("", 8))
            frame_label.pack()
            self.out_frame_labels[name] = frame_label
        
        # --- Output button ---
        self.output_btn = ttk.Button(
            main_frame, text="出力", command=self._on_output, state=tk.DISABLED
        )
        self.output_btn.pack(fill=tk.X, pady=(0, 10))
        
        # --- Log area ---
        log_frame = ttk.LabelFrame(main_frame, text="ログ", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_frame, height=5, state=tk.DISABLED, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)
    
    def log(self, message: str):
        """ログにメッセージを追加（スレッドセーフ）"""
        self.log_queue.put(message)
    
    def _poll_logs(self):
        """ログキューをポーリングしてUIを更新"""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
            except queue.Empty:
                break
        
        # Update crop labels
        for key in self.crop_vars:
            val = self.crop_vars[key].get()
            self.crop_labels[key].configure(text=f"{val}px")
        
        # Update feather label
        self.feather_label.configure(text=f"{self.feather_var.get()}px")
        
        self.after(100, self._poll_logs)
    
    def _on_select_video(self):
        """動画ファイルを選択"""
        if sys.platform == "darwin":  # Mac
            path = filedialog.askopenfilename(title="動画ファイルを選択")
        else:  # Windows/Linux
            path = filedialog.askopenfilename(
                title="動画ファイルを選択",
                filetypes=[
                    ("動画ファイル", "*.mp4 *.avi *.mov *.mkv *.webm"),
                    ("すべてのファイル", "*.*"),
                ],
            )
        if path:
            self._set_video(path)
    
    def _on_drop_video(self, event):
        """ドラッグ&ドロップで動画を設定"""
        path = event.data
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        
        if os.path.isfile(path):
            self._set_video(path)
    
    def _set_video(self, path: str):
        """動画パスを設定"""
        self.video_path = path
        self.video_var.set(path)
        self.extractor = None
        self.candidate_frames = []
        self._clear_candidates()
        self._clear_preview()
        self.update_btn.configure(state=tk.DISABLED)
        self.output_btn.configure(state=tk.DISABLED)
        
        # Close cached capture
        if self._cached_cap:
            self._cached_cap.release()
            self._cached_cap = None
        
        self.log(f"動画を選択: {os.path.basename(path)}")
    
    def _clear_candidates(self):
        """候補表示をクリア"""
        self.candidate_images = []
        for i in range(CANDIDATE_COUNT):
            self.cand_labels[i].configure(image="", text="")
            self.cand_frame_labels[i].configure(text="")
            self.cand_vars[i].set("")
    
    def _clear_preview(self):
        """プレビューをクリア"""
        self.preview_images = {}
        for name, label in self.preview_labels.items():
            label.configure(image="", text="(未選択)")
            self.out_frame_labels[name].configure(text="")
    
    def _on_analyze(self):
        """解析を実行"""
        if not self.video_path:
            messagebox.showwarning("警告", "動画ファイルを選択してください")
            return
        
        if self.is_analyzing:
            return
        
        self.is_analyzing = True
        self.analyze_btn.configure(state=tk.DISABLED)
        self.update_btn.configure(state=tk.DISABLED)
        self.output_btn.configure(state=tk.DISABLED)
        self._clear_candidates()
        self._clear_preview()
        
        thread = threading.Thread(target=self._analyze_worker, daemon=True)
        thread.start()
    
    def _analyze_worker(self):
        """解析ワーカースレッド"""
        try:
            self.log("解析を開始...")
            
            self.extractor = MouthSpriteExtractor(self.video_path)
            self.extractor.analyze(callback=self.log)
            
            # バリエーションのある候補を選択
            valid_frames = [mf for mf in self.extractor.mouth_frames if mf.valid]
            
            if len(valid_frames) == 0:
                self.log("エラー: 有効なフレームがありません")
                return
            
            # 各メトリクスでソート
            heights = np.array([mf.height for mf in valid_frames])
            widths = np.array([mf.width for mf in valid_frames])
            aspect_ratios = widths / np.maximum(heights, 1e-6)
            
            # 選択済みインデックス（重複防止）
            selected_indices = set()
            candidates = []
            
            def pick_by_score(scores, count, maximize=True, label=""):
                """スコアに基づいて候補を選択"""
                sorted_idx = np.argsort(scores)
                if maximize:
                    sorted_idx = sorted_idx[::-1]
                
                picked = 0
                for idx in sorted_idx:
                    if idx not in selected_indices and picked < count:
                        selected_indices.add(idx)
                        candidates.append((valid_frames[idx], label))
                        picked += 1
                    if picked >= count:
                        break
            
            # 2枚ずつ各カテゴリから選択
            pick_by_score(heights, 2, maximize=True, label="open候補")  # 開いた口
            pick_by_score(heights, 2, maximize=False, label="closed候補")  # 閉じた口
            
            # 中間の高さ（half候補）
            median_h = np.median(heights)
            half_scores = -np.abs(heights - median_h)
            pick_by_score(half_scores, 2, maximize=True, label="half候補")
            
            # 横長の口（e候補）
            pick_by_score(aspect_ratios, 2, maximize=True, label="e候補")
            
            # すぼめた口（u候補）- 幅が小さい
            pick_by_score(widths, 2, maximize=False, label="u候補")
            
            # 候補フレームを設定
            self.candidate_frames = [mf for mf, _ in candidates]
            
            self.log(f"候補選択: open候補2枚, closed候補2枚, half候補2枚, e候補2枚, u候補2枚")
            
            # 統一サイズを計算（全有効フレームから）
            if valid_frames:
                max_w = max(mf.width for mf in valid_frames)
                max_h = max(mf.height for mf in valid_frames)
                self.unified_size = (
                    ensure_even_ge2(int(max_w * 1.1)),
                    ensure_even_ge2(int(max_h * 1.1)),
                )
            
            # サムネイル生成
            self.log("候補フレームのサムネイルを生成中...")
            self._generate_thumbnails()
            
            self.after(0, self._update_candidates_ui)
            self.after(0, lambda: self.update_btn.configure(state=tk.NORMAL))
            
            self.log(f"解析完了！{len(self.candidate_frames)}件の候補を表示")
            self.log("1-5の数字を入力して割り当ててください")
            
        except Exception as e:
            self.log(f"エラー: {e}")
            traceback.print_exc()
        
        finally:
            self.is_analyzing = False
            self.after(0, lambda: self.analyze_btn.configure(state=tk.NORMAL))
    
    def _get_video_capture(self) -> cv2.VideoCapture:
        """キャッシュされたVideoCaptureを取得"""
        if self._cached_cap is None or not self._cached_cap.isOpened():
            self._cached_cap = cv2.VideoCapture(self.video_path)
        return self._cached_cap
    
    def _generate_thumbnails(self):
        """候補フレームのサムネイルを生成"""
        if not self.candidate_frames or not self.unified_size:
            return
        
        cap = self._get_video_capture()
        unified_w, unified_h = self.unified_size
        
        self.candidate_images = []
        for mf in self.candidate_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                self.candidate_images.append(None)
                continue
            
            # 正規化空間に変換
            patch = warp_frame_to_norm(frame, mf.quad, unified_w, unified_h)
            
            # PhotoImageに変換
            photo = numpy_to_photoimage(patch, THUMB_SIZE)
            self.candidate_images.append(photo)
    
    def _update_candidates_ui(self):
        """候補UIを更新"""
        for i, mf in enumerate(self.candidate_frames):
            if i < len(self.candidate_images) and self.candidate_images[i]:
                self.cand_labels[i].configure(image=self.candidate_images[i], text="")
            self.cand_frame_labels[i].configure(text=f"F:{mf.frame_idx}")
    
    def _on_update_preview(self):
        """プレビューを更新"""
        if not self.candidate_frames or not self.unified_size:
            messagebox.showwarning("警告", "先に解析を実行してください")
            return
        
        # 割り当てを解析
        assignments = {}  # mouth_type (1-5) -> candidate_idx
        mouth_names = {1: "open", 2: "closed", 3: "half", 4: "e", 5: "u"}
        
        # 全角→半角変換テーブル
        fullwidth_to_halfwidth = str.maketrans("１２３４５６７８９０", "1234567890")
        
        for i, var in enumerate(self.cand_vars):
            val = var.get().strip()
            # 全角数字を半角に変換
            val = val.translate(fullwidth_to_halfwidth)
            if val.isdigit():
                num = int(val)
                if 1 <= num <= 5:
                    if num in assignments:
                        self.log(f"警告: {num}が重複しています")
                    assignments[num] = i
        
        if len(assignments) == 0:
            messagebox.showwarning("警告", "1-5の数字を入力して割り当ててください")
            return
        
        # 取得パラメータ
        crop_top = self.crop_vars["top"].get()
        crop_bottom = self.crop_vars["bottom"].get()
        crop_left = self.crop_vars["left"].get()
        crop_right = self.crop_vars["right"].get()
        feather_px = self.feather_var.get()
        
        unified_w, unified_h = self.unified_size
        cap = self._get_video_capture()
        
        self.preview_sprites = {}
        self._clear_preview()
        
        for num, cand_idx in assignments.items():
            name = mouth_names[num]
            mf = self.candidate_frames[cand_idx]
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(mf.frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            
            # スプライト抽出
            rgba = extract_sprite_with_crop(
                frame, mf.quad, unified_w, unified_h,
                crop_top, crop_bottom, crop_left, crop_right, feather_px
            )
            self.preview_sprites[name] = rgba
            
            # プレビュー表示
            composited = composite_on_checkerboard(rgba)
            photo = numpy_to_photoimage(composited, PREVIEW_SIZE)
            if photo:
                self.preview_images[name] = photo
                self.preview_labels[name].configure(image=photo, text="")
                self.out_frame_labels[name].configure(text=f"F:{mf.frame_idx}")
        
        self.output_btn.configure(state=tk.NORMAL)
        self.log(f"プレビュー更新完了 ({len(self.preview_sprites)}枚)")
    
    def _on_output(self):
        """出力を実行"""
        if not self.preview_sprites:
            messagebox.showwarning("警告", "先にプレビューを更新してください")
            return
        
        # 出力先ディレクトリを決定
        video_dir = os.path.dirname(os.path.abspath(self.video_path))
        base_output = os.path.join(video_dir, "mouth")
        output_dir = get_unique_output_dir(base_output)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for name, rgba in self.preview_sprites.items():
                filepath = os.path.join(output_dir, f"{name}.png")
                # rgbaはBGR+Aなので、そのまま保存（cv2.imwriteはBGRA形式を期待）
                cv2.imwrite(filepath, rgba)
                self.log(f"保存: {name}.png")
            
            self.log(f"出力完了: {output_dir}")
            
            if messagebox.askyesno("完了", f"出力が完了しました。\n{output_dir}\n\nフォルダを開きますか？"):
                os.startfile(output_dir)
            
        except Exception as e:
            self.log(f"出力エラー: {e}")
            messagebox.showerror("エラー", str(e))
    
    def destroy(self):
        """クリーンアップ"""
        if self._cached_cap:
            self._cached_cap.release()
        super().destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """メイン関数"""
    app = MouthSpriteExtractorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
