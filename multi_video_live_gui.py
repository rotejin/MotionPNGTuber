"""
multi_video_live_gui.py

複数の口トラッキング済み動画をGUIボタンで切り替えながらライブ実行できるツール。

使用方法:
    uv run python multi_video_live_gui.py
"""

from __future__ import annotations
import io
import json
import os
import platform
import queue
import sys
import threading
import time
import uuid
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Callable

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
except ImportError:
    raise ImportError("tkinter is required for this GUI application")


def _fatal_dependency(msg: str) -> None:
    try:
        messagebox.showerror("依存関係エラー", msg)
    except Exception:
        print(msg, file=sys.stderr)
    raise SystemExit(1)


try:
    import cv2
except Exception:
    _fatal_dependency("OpenCV (cv2) が必要です。pip install opencv-python を実行してください。")

try:
    import numpy as np
except Exception:
    _fatal_dependency("NumPy が必要です。pip install numpy を実行してください。")

try:
    import sounddevice as sd
except Exception:
    _fatal_dependency("sounddevice が必要です。pip install sounddevice を実行してください。")

try:
    from PIL import Image, ImageTk
except Exception:
    _fatal_dependency("Pillow が必要です。pip install pillow を実行してください。")


# Import from shared core module
from lipsync_core import (
    BgVideo,
    MouthTrack,
    load_mouth_sprites,
    discover_mouth_sets,
    probe_video_size,
    probe_video_fps,
    probe_video_frame_count,
    alpha_blit_rgb_safe,
    warp_rgba_to_quad,
    one_pole_beta,
    pick_mouth_set_for_label,
    infer_label_from_set_name,
    format_emotion_hud_text,
)

# Optional: emotion analyzer
try:
    from realtime_emotion_audio import RealtimeEmotionAnalyzer
    HAS_EMOTION_AUDIO = True
except ImportError:
    RealtimeEmotionAnalyzer = None
    HAS_EMOTION_AUDIO = False

# Optional: psutil for memory check
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Optional: pyvirtualcam
try:
    import pyvirtualcam
    HAS_VCAM = True
except ImportError:
    HAS_VCAM = False

HERE = os.path.abspath(os.path.dirname(__file__))
SESSION_FILE = os.path.join(HERE, ".multi_video_live_session.json")
__VERSION__ = "v1.0"

# ============================================================
# Constants
# ============================================================

FPS_WARNING_MIN = 0.5   # FPS差の警告最小値
FPS_WARNING_RATIO = 0.03  # FPS差の警告比率
FPS_INVALID_MIN = 2.0   # FPS差の無効化最小値
FPS_INVALID_RATIO = 0.10  # FPS差の無効化比率
FRAME_DIFF_WARNING_RATIO = 0.02  # フレーム差がこれを超えたら警告
FRAME_DIFF_INVALID_RATIO = 0.10  # フレーム差がこれを超えたら無効化
DEFAULT_MAX_SETS = 8  # psutilなしの場合のデフォルト上限
THUMBNAIL_SIZE = (120, 80)
MAX_RENDER_FPS = 60
MAX_LOG_LINES = 200  # ログ表示の上限行数

EMOTION_PRESET_PARAMS = {
    "stable": dict(smooth_alpha=0.18, min_hold_sec=0.75, cand_stable_sec=0.30, switch_margin=0.14),
    "standard": dict(smooth_alpha=0.25, min_hold_sec=0.45, cand_stable_sec=0.22, switch_margin=0.10),
    "snappy": dict(smooth_alpha=0.35, min_hold_sec=0.25, cand_stable_sec=0.12, switch_margin=0.06),
}

DEFAULT_TRANSITION_SEC = 0.35
TRANSITION_EFFECTS = {
    "クロスフェード": "crossfade",
}

# Load mode constants
LOAD_MODE_FULL = "full"
LOAD_MODE_LRU = "lru"
LOAD_MODE_LABELS = {
    "全件ロード": LOAD_MODE_FULL,
    "メモリ節約": LOAD_MODE_LRU,
}
LOAD_MODE_LABEL_LIST = list(LOAD_MODE_LABELS.keys())
LRU_MAX_SETS = 4


# ============================================================
# Data Classes
# ============================================================

@dataclass
class VideoSet:
    """設定情報としての動画セット"""
    id: str
    label: str
    folder_path: str
    video_path: str
    track_path: str
    mouth_dir: str
    thumbnail_data: bytes | None = None
    is_valid: bool = True
    warnings: list[str] = field(default_factory=list)
    fps: float = 30.0
    frame_count: int = 0
    video_size: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "folder_path": self.folder_path,
            "video_path": self.video_path,
            "track_path": self.track_path,
            "mouth_dir": self.mouth_dir,
            "warnings": self.warnings,
        }

    @staticmethod
    def from_dict(d: dict) -> "VideoSet":
        return VideoSet(
            id=d.get("id", str(uuid.uuid4())),
            label=d.get("label", ""),
            folder_path=d.get("folder_path", ""),
            video_path=d.get("video_path", ""),
            track_path=d.get("track_path", ""),
            mouth_dir=d.get("mouth_dir", ""),
            warnings=d.get("warnings", []),
        )


@dataclass
class LoadedVideoSet:
    """事前ロード済みの動画セット（メモリ上に保持）"""
    set_id: str
    bg_video: BgVideo
    mouth_track: MouthTrack | None
    mouth_sprites: dict[str, dict[str, np.ndarray]]  # emotion -> shape -> RGBA
    base_size: tuple[int, int]
    duration_sec: float = 0.0


# ============================================================
# Utility Functions
# ============================================================


def _safe_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _safe_int(
    v, default: int, min_v: int | None = None, max_v: int | None = None
) -> int:
    try:
        if isinstance(v, str):
            v = v.split(":", 1)[0].strip()
        iv = int(float(v)) if isinstance(v, str) else int(v)
    except Exception:
        return default
    if min_v is not None:
        iv = max(min_v, iv)
    if max_v is not None:
        iv = min(max_v, iv)
    return iv


def _safe_float(
    v, default: float, min_v: float | None = None, max_v: float | None = None
) -> float:
    try:
        fv = float(v)
    except Exception:
        return default
    if min_v is not None:
        fv = max(min_v, fv)
    if max_v is not None:
        fv = min(max_v, fv)
    return fv

def get_recommended_max_sets() -> int:
    """PCスペックから推奨最大セット数を算出"""
    if not HAS_PSUTIL:
        return DEFAULT_MAX_SETS
    try:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        # 1セットあたり約200MB想定、利用可能メモリの50%まで使用
        return max(2, int((available_mb * 0.5) / 200))
    except Exception:
        return DEFAULT_MAX_SETS


def generate_thumbnail(video_path: str, size: tuple[int, int] = THUMBNAIL_SIZE) -> bytes | None:
    """動画の最初のフレームからサムネイルを生成"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img.thumbnail(size, Image.Resampling.LANCZOS)

        # アスペクト比を維持しつつセンタリング
        thumb = Image.new("RGB", size, (40, 40, 40))
        offset = ((size[0] - img.width) // 2, (size[1] - img.height) // 2)
        thumb.paste(img, offset)

        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[thumbnail] failed: {e}")
        return None


def find_video_file(folder: str) -> str | None:
    """フォルダ内の動画ファイルを検索（フォールバック対応）
    
    優先順位:
    1. loop_mouthless.mp4 (従来の固定名)
    2. loop.mp4 (従来の固定名)
    3. *_mouthless.mp4 (任意の名前で_mouthlessが付くもの)
    4. その他の.mp4ファイル (上記が見つからない場合)
    """
    # 1. 従来の固定名を優先
    fixed_candidates = ["loop_mouthless.mp4", "loop.mp4"]
    for name in fixed_candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    
    # 2. フォルダ内のmp4ファイルを検索
    try:
        files = os.listdir(folder)
    except OSError:
        return None
    
    mp4_files = [f for f in files if f.lower().endswith(".mp4")]
    if not mp4_files:
        return None
    
    # 3. _mouthless.mp4 を優先
    mouthless_files = [f for f in mp4_files if f.lower().endswith("_mouthless.mp4")]
    if mouthless_files:
        # 複数ある場合は名前順で最初のものを選択
        mouthless_files.sort()
        return os.path.join(folder, mouthless_files[0])
    
    # 4. その他のmp4ファイル（フォールバック）
    mp4_files.sort()
    return os.path.join(folder, mp4_files[0])


def find_track_file(folder: str) -> str | None:
    """フォルダ内のトラックファイルを検索（フォールバック対応）"""
    candidates = ["mouth_track_calibrated.npz", "mouth_track.npz"]
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    return None


def find_mouth_dir(folder: str, common_mouth_dir: str = "") -> str | None:
    """mouth/フォルダを検索（フォールバック対応）"""
    local = os.path.join(folder, "mouth")
    if os.path.isdir(local) and _is_mouth_root(local):
        return local
    if common_mouth_dir and os.path.isdir(common_mouth_dir):
        return common_mouth_dir
    return None


def _is_mouth_root(path: str) -> bool:
    """mouthセットとして読み込めるディレクトリかを簡易判定"""
    if os.path.isfile(os.path.join(path, "open.png")):
        return True
    try:
        for name in os.listdir(path):
            sub = os.path.join(path, name)
            if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "open.png")):
                return True
    except Exception:
        return False
    return False


# ============================================================
# VideoSetManager
# ============================================================

class VideoSetManager:
    """動画セットの管理"""

    def __init__(self):
        self.sets: dict[str, VideoSet] = {}
        self.thumbnail_cache: dict[str, ImageTk.PhotoImage] = {}
        self._photo_refs: dict[str, ImageTk.PhotoImage] = {}  # GC対策

    def add_set(
        self,
        folder_path: str,
        label: str = "",
        common_mouth_dir: str = "",
        base_fps: float | None = None,
        base_size: tuple[int, int] | None = None,
        allow_duplicate: bool = False,
    ) -> tuple[VideoSet | None, list[str]]:
        """動画セットを追加"""
        warnings = []
        folder_path = os.path.abspath(folder_path)

        if not os.path.isdir(folder_path):
            return None, [f"フォルダが存在しません: {folder_path}"]

        # ファイル検索
        video_path = find_video_file(folder_path)
        if not video_path:
            return None, [f"動画ファイルが見つかりません: {folder_path}"]

        if "loop.mp4" in video_path and "mouthless" not in video_path:
            warnings.append("loop_mouthless.mp4 ではなく loop.mp4 を使用")

        track_path = find_track_file(folder_path)
        if not track_path:
            return None, [f"トラックファイルが見つかりません: {folder_path}"]

        if "calibrated" not in track_path:
            warnings.append("calibrated でないトラックファイルを使用")

        # 共通mouth_dirは呼び出し元で検証済みの前提
        mouth_dir = find_mouth_dir(folder_path, common_mouth_dir)
        if not mouth_dir:
            warnings.append("mouth/ フォルダが見つかりません（共通mouth_dirも未設定）")

        # 動画情報取得
        video_size = probe_video_size(video_path) or (0, 0)
        fps = probe_video_fps(video_path)
        if not fps or fps <= 0:
            warnings.append("FPS取得に失敗（差分チェックをスキップ）")
            fps = 0.0
        frame_count = probe_video_frame_count(video_path) or 0
        if frame_count <= 0:
            warnings.append("フレーム数取得に失敗（差分チェックをスキップ）")

        # FPS差チェック
        is_valid = True
        if base_fps is not None and base_fps > 0 and fps > 0:
            fps_diff = abs(fps - base_fps)
            warn_th = max(FPS_WARNING_MIN, base_fps * FPS_WARNING_RATIO)
            invalid_th = max(FPS_INVALID_MIN, base_fps * FPS_INVALID_RATIO)
            if fps_diff > invalid_th:
                warnings.append(f"FPS差が大きすぎます（ベース: {base_fps:.1f}, 追加: {fps:.1f}）")
                is_valid = False
            elif fps_diff > warn_th:
                warnings.append(f"FPSが異なります（ベース: {base_fps:.1f}, 追加: {fps:.1f}）")

        # トラックフレーム数チェック
        if track_path and frame_count > 0:
            try:
                npz = np.load(track_path, allow_pickle=False)
                track_frames = len(npz.get("quad", npz.get("bbox", [])))
                if track_frames > 0:
                    diff = abs(track_frames - frame_count)
                    if diff > frame_count * FRAME_DIFF_INVALID_RATIO:
                        warnings.append(f"トラック/動画のフレーム数差が大きすぎます（トラック: {track_frames}, 動画: {frame_count}）")
                        is_valid = False
                    if diff > max(2, frame_count * FRAME_DIFF_WARNING_RATIO):
                        warnings.append(f"フレーム数が異なります（トラック: {track_frames}, 動画: {frame_count}）")
            except Exception:
                pass

        # ラベル設定
        label_auto = not label
        if label_auto:
            label = os.path.basename(folder_path)

        # 重複チェック/自動番号付け
        existing_count = sum(1 for vs in self.sets.values() if vs.folder_path == folder_path)
        if existing_count > 0:
            if not allow_duplicate:
                return None, [f"既に追加済みです: {folder_path}"]
            if label_auto:
                label = f"{label} ({existing_count + 1})"

        # VideoSet作成
        if not mouth_dir:
            is_valid = False

        vs = VideoSet(
            id=str(uuid.uuid4()),
            label=label,
            folder_path=folder_path,
            video_path=video_path,
            track_path=track_path,
            mouth_dir=mouth_dir or "",
            warnings=warnings,
            is_valid=is_valid,
            fps=fps,
            frame_count=frame_count,
            video_size=video_size,
        )

        self.sets[vs.id] = vs
        return vs, warnings

    def reorder_sets(self, ordered_ids: list[str]) -> None:
        """セットの並び順を更新"""
        new_sets: dict[str, VideoSet] = {}
        for sid in ordered_ids:
            vs = self.sets.get(sid)
            if vs:
                new_sets[sid] = vs
        for sid, vs in self.sets.items():
            if sid not in new_sets:
                new_sets[sid] = vs
        self.sets = new_sets

    def remove_set(self, set_id: str) -> None:
        """動画セットを削除"""
        if set_id in self.sets:
            del self.sets[set_id]
        if set_id in self.thumbnail_cache:
            del self.thumbnail_cache[set_id]
        if set_id in self._photo_refs:
            del self._photo_refs[set_id]

    def auto_detect_sets(
        self,
        parent_folder: str,
        common_mouth_dir: str = "",
        base_fps: float | None = None,
    ) -> list[tuple[VideoSet, list[str]]]:
        """親フォルダからサブフォルダを自動検出して追加"""
        results = []
        parent_folder = os.path.abspath(parent_folder)

        if not os.path.isdir(parent_folder):
            return results

        # 直下のサブフォルダのみ走査
        for name in sorted(os.listdir(parent_folder)):
            sub = os.path.join(parent_folder, name)
            if not os.path.isdir(sub):
                continue

            # 動画とトラックの両方があるか確認
            if find_video_file(sub) and find_track_file(sub):
                vs, warnings = self.add_set(sub, common_mouth_dir=common_mouth_dir, base_fps=base_fps)
                if vs:
                    results.append((vs, warnings))

        return results

    def get_photo_image(self, set_id: str) -> ImageTk.PhotoImage | None:
        """サムネイルのPhotoImageを取得（キャッシュ対応）"""
        if set_id in self.thumbnail_cache:
            return self.thumbnail_cache[set_id]

        vs = self.sets.get(set_id)
        if not vs or not vs.thumbnail_data:
            return None

        try:
            img = Image.open(io.BytesIO(vs.thumbnail_data))
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[set_id] = photo
            self._photo_refs[set_id] = photo  # GC対策
            return photo
        except Exception:
            return None

    def generate_thumbnails_async(self, callback: Callable[[str], None] | None = None) -> threading.Thread:
        """バックグラウンドでサムネイル生成"""
        def worker():
            for set_id, vs in list(self.sets.items()):
                if vs.thumbnail_data is None and vs.video_path:
                    vs.thumbnail_data = generate_thumbnail(vs.video_path)
                    if callback:
                        callback(set_id)

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        return th


# ============================================================
# VideoSetLoader
# ============================================================

class VideoSetLoader:
    """動画セットの事前ロードを管理（LRU対応）"""

    def __init__(self, base_size: tuple[int, int], base_fps: float):
        self.base_size = base_size
        self.base_fps = base_fps
        # OrderedDict for LRU tracking (most recently used at end)
        self.loaded: OrderedDict[str, LoadedVideoSet] = OrderedDict()
        self._lock = threading.Lock()
        # Registry for VideoSet objects (set by caller for LRU mode)
        self.set_registry: dict[str, VideoSet] = {}
        # Log callback for GUI notification (set by caller)
        self.log_callback: Callable[[str], None] | None = None

    def _log(self, msg: str) -> None:
        """ログ出力（コールバックがあればGUIへ、なければコンソールへ）"""
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def load_all(
        self,
        sets: list[VideoSet],
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> list[str]:
        """全セットをロード（エラーのあったセットIDのリストを返す）"""
        errors = []
        total = len(sets)

        for i, vs in enumerate(sets):
            if progress_callback:
                progress_callback(vs.label, i + 1, total)

            try:
                loaded = self.load_one(vs)
                if loaded:
                    with self._lock:
                        self.loaded[vs.id] = loaded
                else:
                    errors.append(vs.id)
            except Exception as e:
                print(f"[loader] failed to load {vs.label}: {e}")
                errors.append(vs.id)

        return errors

    def load_one(self, video_set: VideoSet) -> LoadedVideoSet | None:
        """1セットをロード"""
        if not video_set.is_valid:
            return None

        try:
            # BgVideo (scale-to-fill mode)
            bg_video = BgVideo(
                video_set.video_path,
                self.base_size[0],
                self.base_size[1],
                scale_mode="fill"
            )

            # MouthTrack with transformation
            video_size = video_set.video_size if video_set.video_size[0] > 0 else probe_video_size(video_set.video_path)
            if video_size:
                mouth_track = MouthTrack.load_with_transform(
                    video_set.track_path,
                    self.base_size[0],
                    self.base_size[1],
                    video_size[0],
                    video_size[1],
                )
            else:
                mouth_track = MouthTrack.load(
                    video_set.track_path,
                    self.base_size[0],
                    self.base_size[1],
                )

            # Mouth sprites
            mouth_sprites: dict[str, dict[str, np.ndarray]] = {}
            if video_set.mouth_dir:
                sets_dirs = discover_mouth_sets(video_set.mouth_dir)
                full_w, full_h = self.base_size
                if video_size and video_size[0] > 0 and video_size[1] > 0:
                    full_w, full_h = video_size
                for name, path in sets_dirs.items():
                    try:
                        mouth_sprites[name] = load_mouth_sprites(path, full_w, full_h)
                    except Exception as e:
                        print(f"[loader] failed to load mouth set {name}: {e}")

            if not mouth_sprites:
                print(f"[loader] no mouth sprites loaded for {video_set.label}")
                return None

            fps = float(video_set.fps or 0.0)
            frame_count = int(video_set.frame_count or 0)
            if fps <= 0:
                fps = float(bg_video.fps or 0.0)
            if frame_count <= 0:
                frame_count = int(bg_video.total_frames or 0)
            if frame_count <= 0 and mouth_track:
                frame_count = int(mouth_track.total or 0)
            duration_sec = (frame_count / fps) if (frame_count > 0 and fps > 0) else 0.0

            return LoadedVideoSet(
                set_id=video_set.id,
                bg_video=bg_video,
                mouth_track=mouth_track,
                mouth_sprites=mouth_sprites,
                base_size=self.base_size,
                duration_sec=duration_sec,
            )

        except Exception as e:
            print(f"[loader] error loading {video_set.label}: {e}")
            return None

    def unload(self, set_id: str) -> None:
        """セットをアンロード"""
        with self._lock:
            if set_id in self.loaded:
                try:
                    self.loaded[set_id].bg_video.close()
                except Exception:
                    pass
                del self.loaded[set_id]

    def unload_all(self) -> None:
        """全セットをアンロード"""
        with self._lock:
            for loaded in self.loaded.values():
                try:
                    loaded.bg_video.close()
                except Exception:
                    pass
            self.loaded.clear()

    def get(self, set_id: str) -> LoadedVideoSet | None:
        """ロード済みセットを取得"""
        with self._lock:
            return self.loaded.get(set_id)

    def touch(self, set_id: str) -> None:
        """LRU順序を更新（最近使用したものを末尾へ）"""
        with self._lock:
            if set_id in self.loaded:
                self.loaded.move_to_end(set_id)

    def evict_lru(self, pinned_ids: set[str], max_sets: int = LRU_MAX_SETS) -> int:
        """LRU上限を超えた分をアンロード（pinされていないものを古い順に）

        Returns:
            アンロードしたセット数
        """
        evicted = 0
        with self._lock:
            while len(self.loaded) > max_sets:
                # Find oldest non-pinned set
                to_evict = None
                for sid in self.loaded.keys():
                    if sid not in pinned_ids:
                        to_evict = sid
                        break

                if to_evict is None:
                    # All remaining sets are pinned, cannot evict more
                    break

                # Unload the set
                try:
                    self.loaded[to_evict].bg_video.close()
                except Exception:
                    pass
                del self.loaded[to_evict]
                evicted += 1
                self._log(f"[loader] LRU退避: {to_evict}")

        return evicted

    def ensure_loaded(self, set_id: str, pinned_ids: set[str]) -> bool:
        """セットがロード済みか確認し、未ロードならロードする（LRU対応）

        Args:
            set_id: ロードするセットのID
            pinned_ids: 退避対象外のセットID集合

        Returns:
            True: ロード済み or ロード成功
            False: ロード失敗
        """
        # Already loaded?
        with self._lock:
            if set_id in self.loaded:
                self.loaded.move_to_end(set_id)
                return True

        # Need to load - first make room if necessary
        pinned_with_new = pinned_ids | {set_id}
        self.evict_lru(pinned_with_new, LRU_MAX_SETS - 1)  # -1 to make room for new

        # Get VideoSet from registry
        video_set = self.set_registry.get(set_id)
        if video_set is None:
            self._log(f"[loader] エラー: {set_id} がレジストリにありません")
            return False

        # Load
        self._log(f"[loader] オンデマンド読み込み中: {video_set.label}")
        loaded = self.load_one(video_set)
        if loaded is None:
            self._log(f"[loader] 読み込み失敗: {video_set.label}")
            return False

        with self._lock:
            self.loaded[set_id] = loaded

        # Double-check eviction after load
        self.evict_lru(pinned_with_new, LRU_MAX_SETS)

        return True

    def loaded_count(self) -> int:
        """ロード済みセット数を取得"""
        with self._lock:
            return len(self.loaded)


# ============================================================
# LiveRunner
# ============================================================

class LiveRunner:
    """ライブ実行エンジン"""

    def __init__(
        self,
        loader: VideoSetLoader,
        audio_device: int | None,
        emotion_preset: str = "standard",
        show_hud: bool = True,
        use_vcam: bool = False,
        emotion_auto: bool = True,
        allow_high_fps: bool = False,
        auto_cycle: bool = False,
        cycle_order: list[str] | None = None,
        transition_enabled: bool = False,
        transition_type: str = "crossfade",
        transition_sec: float = DEFAULT_TRANSITION_SEC,
        frame_queue: "queue.Queue[np.ndarray] | None" = None,
        preview_max_fps: float = 30.0,
        load_mode: str = LOAD_MODE_FULL,
    ):
        self.loader = loader
        self.audio_device = audio_device
        self.emotion_preset = emotion_preset
        self.show_hud = show_hud
        self.use_vcam = use_vcam and HAS_VCAM
        self.emotion_auto = emotion_auto and HAS_EMOTION_AUDIO
        self.allow_high_fps = allow_high_fps
        self.auto_cycle = auto_cycle
        self.cycle_order = list(cycle_order or [])
        self.load_mode = load_mode
        self._cycle_index = 0
        self._cycle_started_t: float | None = None
        self._cycle_duration_s = 0.0

        self.transition_enabled = transition_enabled
        self.transition_type = self._normalize_transition_type(transition_type)
        self.transition_sec = max(0.0, float(transition_sec or 0.0))
        self.frame_queue = frame_queue
        self.preview_max_fps = max(1.0, float(preview_max_fps))
        self._last_frame_send_t = 0.0
        self._trans_from_id: str | None = None
        self._trans_to_id: str | None = None
        self._trans_start_t = 0.0
        self._trans_dur_s = 0.0
        self._trans_from_frame: np.ndarray | None = None
        self._trans_from_last_idx: int | None = None
        self._trans_from_frozen = False
        self._pending_switch_id: str | None = None

        self.current_set_id: str | None = None
        self.running = False
        self.switch_lock = threading.Lock()
        self._next_set_id: str | None = None
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Audio state
        self._audio_stream = None
        self._feat_q: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=256)

        # Emotion auto
        self._emo_analyzer = None
        self._emo_audio_q: queue.Queue[np.ndarray] | None = None

    @staticmethod
    def _normalize_transition_type(value: str) -> str:
        if not value:
            return "none"
        lowered = str(value).strip().lower()
        if lowered in {"crossfade", "cross", "fade"}:
            return "crossfade"
        if "クロスフェード" in str(value):
            return "crossfade"
        return "none"

    def start(self, initial_set_id: str) -> None:
        """ライブ開始"""
        if self.running:
            return

        self.current_set_id = initial_set_id
        self._sync_cycle_index(initial_set_id)
        self._cycle_started_t = None
        self.running = True
        self.stop_event.clear()

        self._thread = threading.Thread(target=self._main_loop, daemon=True)
        self._thread.start()

    def switch_to(self, set_id: str) -> None:
        """動画切り替え（瞬時）"""
        with self.switch_lock:
            self._next_set_id = set_id

    def stop(self) -> None:
        """ライブ停止"""
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.running = False

    def _sync_cycle_index(self, set_id: str | None) -> None:
        if not set_id or not self.cycle_order:
            return
        try:
            self._cycle_index = self.cycle_order.index(set_id)
        except ValueError:
            self._cycle_index = 0

    def _reset_cycle_timer(self, now: float, loaded: LoadedVideoSet | None) -> None:
        if not self.auto_cycle:
            return
        self._cycle_started_t = now
        self._cycle_duration_s = float(loaded.duration_sec or 0.0) if loaded else 0.0

    def _next_cycle_id(self) -> str | None:
        if not self.cycle_order:
            return None
        if self.current_set_id and self.current_set_id in self.cycle_order:
            idx = self.cycle_order.index(self.current_set_id)
        else:
            idx = self._cycle_index
        next_idx = (idx + 1) % len(self.cycle_order)
        self._cycle_index = next_idx
        return self.cycle_order[next_idx]

    def _main_loop(self) -> None:
        """メインレンダリングループ"""
        base_w, base_h = self.loader.base_size
        render_fps = self.loader.base_fps
        if not self.allow_high_fps:
            render_fps = min(render_fps, MAX_RENDER_FPS)
        if render_fps <= 0:
            render_fps = 30.0
        if self.auto_cycle and not self.cycle_order:
            self.cycle_order = list(self.loader.loaded.keys())
            self._sync_cycle_index(self.current_set_id)

        def _get_input_samplerate(device: int | None) -> int:
            try:
                info = sd.query_devices(device, "input")
                sr = int(float(info.get("default_samplerate", 0)) or 0)
                return sr if sr > 0 else 48000
            except Exception:
                return 48000

        # Audio setup
        samplerate = _get_input_samplerate(self.audio_device)
        hop = int(samplerate / 100)
        window = np.hanning(hop).astype(np.float32)
        freqs = np.fft.rfftfreq(hop, d=1.0 / samplerate)

        # Emotion auto setup
        if self.emotion_auto:
            self._emo_audio_q = queue.Queue(maxsize=256)
            params = EMOTION_PRESET_PARAMS.get(self.emotion_preset, EMOTION_PRESET_PARAMS["standard"])
            try:
                self._emo_analyzer = RealtimeEmotionAnalyzer(sr=samplerate, **params)
            except Exception as e:
                print(f"[emotion-auto] init failed: {e}")
                self._emo_analyzer = None

        def audio_cb(indata, frames, time_info, status):
            x = indata.astype(np.float32)
            if x.ndim == 2:
                x = x.mean(axis=1)
            if len(x) < hop:
                x = np.pad(x, (0, hop - len(x)))
            elif len(x) > hop:
                x = x[:hop]
            rms_raw = float(np.sqrt(np.mean(x * x) + 1e-12))
            w = x * window
            mag = np.abs(np.fft.rfft(w)) + 1e-9
            centroid = float((freqs * mag).sum() / mag.sum())
            centroid = float(np.clip(centroid / (samplerate * 0.5), 0.0, 1.0))
            try:
                self._feat_q.put_nowait((rms_raw, centroid))
            except queue.Full:
                pass
            if self._emo_audio_q is not None:
                try:
                    self._emo_audio_q.put_nowait(x)
                except queue.Full:
                    pass

        try:
            self._audio_stream = sd.InputStream(
                samplerate=samplerate,
                channels=1,
                blocksize=hop,
                dtype="float32",
                callback=audio_cb,
                device=self.audio_device,
                latency="low",
            )
        except Exception as e:
            print(f"[audio] failed to open device {self.audio_device}: {e}")
            # Fallback to default device
            try:
                samplerate = _get_input_samplerate(None)
                hop = int(samplerate / 100)
                window = np.hanning(hop).astype(np.float32)
                freqs = np.fft.rfftfreq(hop, d=1.0 / samplerate)
                self._audio_stream = sd.InputStream(
                    samplerate=samplerate,
                    channels=1,
                    blocksize=hop,
                    dtype="float32",
                    callback=audio_cb,
                    latency="low",
                )
                print("[audio] using default device")
            except Exception as e2:
                print(f"[audio] failed to open default device: {e2}")
                self._audio_stream = None

        # Virtual cam
        cam = None
        if self.use_vcam:
            try:
                cam = pyvirtualcam.Camera(width=base_w, height=base_h, fps=int(render_fps), print_fps=False)
                print(f"[vcam] started: {cam.device}")
            except Exception as e:
                print(f"[vcam] failed: {e}")
                cam = None

        # Audio state
        beta = one_pole_beta(8.0, 100)
        noise = 1e-4
        peak = 1e-3
        peak_decay = 0.995
        silence_gate_rms = 0.002
        rms_smooth_q = deque(maxlen=3)
        env_lp = 0.0
        env_hist = deque(maxlen=1000)
        cent_hist = deque(maxlen=1000)
        TALK_TH, HALF_TH, OPEN_TH = 0.06, 0.30, 0.52
        U_TH, E_TH = 0.16, 0.20

        current_open_shape = "open"
        last_vowel_change_t = -999.0
        e_prev2, e_prev1 = 0.0, 0.0
        mouth_shape_now = "closed"

        # Emotion state
        current_emotion = "Default"
        neutral_set = "Default"
        emo_buf = np.zeros((0,), dtype=np.float32)
        emo_window_len = int(samplerate * 0.25)
        emo_last_eval = 0.0

        # Preview frame send interval
        frame_send_interval = 1.0 / self.preview_max_fps

        t0 = time.perf_counter()
        next_frame_t = time.perf_counter()

        if self._audio_stream:
            self._audio_stream.start()

        def resolve_mouth_set(loaded_set: LoadedVideoSet, preferred_label: str) -> dict[str, np.ndarray]:
            if preferred_label and preferred_label in loaded_set.mouth_sprites:
                return loaded_set.mouth_sprites[preferred_label]
            if "Default" in loaded_set.mouth_sprites:
                return loaded_set.mouth_sprites["Default"]
            if "Neutral" in loaded_set.mouth_sprites:
                return loaded_set.mouth_sprites["Neutral"]
            if loaded_set.mouth_sprites:
                return next(iter(loaded_set.mouth_sprites.values()))
            return {}

        def render_set_frame(
            loaded_set: LoadedVideoSet,
            mouth_dict: dict[str, np.ndarray],
            mouth_shape: str,
        ) -> np.ndarray:
            frame_rgb = loaded_set.bg_video.get_frame(now).copy()
            frame_idx = loaded_set.bg_video.frame_idx
            spr = mouth_dict.get(mouth_shape, mouth_dict.get("closed", None))
            if spr is not None:
                quad = loaded_set.mouth_track.get_quad(frame_idx) if loaded_set.mouth_track else None
                if quad is not None:
                    patch, x0, y0 = warp_rgba_to_quad(spr, quad)
                    alpha_blit_rgb_safe(frame_rgb, patch, x0, y0)
                else:
                    x = base_w // 2 - spr.shape[1] // 2
                    y = int(base_h * 0.58) - spr.shape[0] // 2
                    alpha_blit_rgb_safe(frame_rgb, spr, x, y)
            return frame_rgb

        def blend_frames(frame_a: np.ndarray, frame_b: np.ndarray, alpha: float) -> np.ndarray:
            if alpha <= 0.0:
                return frame_a
            if alpha >= 1.0:
                return frame_b
            return cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)

        def remaining_sec_to_loop_end(loaded_set: LoadedVideoSet | None) -> float | None:
            if not loaded_set:
                return None
            total = int(loaded_set.bg_video.total_frames or 0)
            fps = float(loaded_set.bg_video.fps or 0.0)
            if total <= 0 or fps <= 0.0:
                return None
            idx = int(loaded_set.bg_video.frame_idx)
            if idx < 0:
                idx = 0
            remaining_frames = max(0, total - 1 - idx)
            return float(remaining_frames) / fps

        def start_switch(next_id: str, now_t: float) -> None:
            # LRU mode: ensure target is loaded
            if self.load_mode == LOAD_MODE_LRU:
                pinned_ids = {next_id}
                if self.current_set_id:
                    pinned_ids.add(self.current_set_id)
                if self._trans_from_id:
                    pinned_ids.add(self._trans_from_id)
                if self._trans_to_id:
                    pinned_ids.add(self._trans_to_id)

                if not self.loader.ensure_loaded(next_id, pinned_ids):
                    self.loader._log(f"[live] 切替失敗（読み込みエラー）: {next_id}")
                    return

            loaded_next = self.loader.get(next_id)
            if not loaded_next:
                self.loader._log(f"[live] 切替失敗（未ロード）: {next_id}")
                return
            loaded_next.bg_video.reset()
            if (self.transition_enabled and self.transition_sec > 0.0
                    and self.current_set_id and self.current_set_id != next_id):
                self._trans_from_id = self.current_set_id
                self._trans_to_id = next_id
                self._trans_start_t = now_t
                self._trans_dur_s = self.transition_sec
                self._trans_from_frame = None
                self._trans_from_last_idx = None
                self._trans_from_frozen = False
            else:
                self._trans_from_id = None
                self._trans_to_id = None
                self._trans_dur_s = 0.0
                self._trans_from_frame = None
                self._trans_from_last_idx = None
                self._trans_from_frozen = False
            self.current_set_id = next_id
            self._sync_cycle_index(next_id)
            self._reset_cycle_timer(now_t, loaded_next)
            print(f"[live] switched to: {next_id}")

        try:
            while not self.stop_event.is_set():
                now = time.perf_counter()
                t = now - t0

                # Handle video switch request
                requested_id = None
                with self.switch_lock:
                    if self._next_set_id is not None:
                        requested_id = self._next_set_id
                        self._next_set_id = None

                if requested_id and requested_id != self.current_set_id:
                    defer_ok = (self.transition_enabled and self.transition_sec > 0.0
                                and self.current_set_id)
                    if defer_ok:
                        current_loaded = self.loader.get(self.current_set_id)
                        remain = remaining_sec_to_loop_end(current_loaded)
                        if remain is not None and remain > self.transition_sec:
                            self._pending_switch_id = requested_id
                        else:
                            self._pending_switch_id = None
                            start_switch(requested_id, now)
                    else:
                        self._pending_switch_id = None
                        start_switch(requested_id, now)

                # Get current loaded set
                loaded = self.loader.get(self.current_set_id) if self.current_set_id else None
                if not loaded:
                    time.sleep(0.1)
                    continue

                if self._pending_switch_id:
                    remain = remaining_sec_to_loop_end(loaded)
                    if remain is None or remain <= self.transition_sec:
                        pending_id = self._pending_switch_id
                        self._pending_switch_id = None
                        start_switch(pending_id, now)
                        loaded = self.loader.get(self.current_set_id)
                        if not loaded:
                            time.sleep(0.1)
                            continue
                if self.auto_cycle and self._cycle_started_t is None:
                    self._reset_cycle_timer(now, loaded)

                if self.auto_cycle and self._cycle_duration_s > 0 and self._next_set_id is None:
                    cycle_trigger = self._cycle_duration_s
                    if self.transition_enabled and self.transition_sec > 0.0:
                        cycle_trigger = max(0.0, self._cycle_duration_s - self.transition_sec)
                    if (now - self._cycle_started_t) >= cycle_trigger:
                        next_id = self._next_cycle_id()
                        if next_id and next_id != self.current_set_id:
                            with self.switch_lock:
                                if self._next_set_id is None and self._pending_switch_id is None:
                                    self._next_set_id = next_id
                                    self._cycle_started_t = now
                        else:
                            self._cycle_started_t = now

                # Determine emotion set
                emotions = list(loaded.mouth_sprites.keys())
                if current_emotion not in loaded.mouth_sprites:
                    if "Default" in loaded.mouth_sprites:
                        current_emotion = "Default"
                    elif "Neutral" in loaded.mouth_sprites:
                        current_emotion = "Neutral"
                    elif emotions:
                        current_emotion = emotions[0]
                    neutral_set = pick_mouth_set_for_label(emotions, "neutral") or current_emotion

                mouth = loaded.mouth_sprites.get(current_emotion, {})
                if not mouth and emotions:
                    mouth = loaded.mouth_sprites[emotions[0]]

                # Emotion auto updates
                if self.emotion_auto and self._emo_analyzer and self._emo_audio_q:
                    while True:
                        try:
                            emo_buf = np.concatenate([emo_buf, self._emo_audio_q.get_nowait()])
                        except queue.Empty:
                            break
                    max_len = int(samplerate * 1.2)
                    if emo_buf.size > max_len:
                        emo_buf = emo_buf[-max_len:]
                    if (now - emo_last_eval) >= 0.10 and emo_buf.size >= emo_window_len:
                        emo_last_eval = now
                        xwin = emo_buf[-emo_window_len:]
                        try:
                            lab, info = self._emo_analyzer.update(xwin)
                        except Exception:
                            lab, info = None, {}
                        if lab is not None:
                            rms_db = float(info.get("rms_db", -120.0))
                            voiced = float(info.get("voiced", 0.0)) >= 0.5
                            if rms_db < -65.0:
                                target_set = neutral_set
                            elif not voiced:
                                target_set = None
                            else:
                                target_set = pick_mouth_set_for_label(emotions, str(lab).lower()) or neutral_set
                            if target_set and target_set in loaded.mouth_sprites and target_set != current_emotion:
                                current_emotion = target_set
                                mouth = loaded.mouth_sprites[current_emotion]

                # Audio updates
                while True:
                    try:
                        rms_raw, cent = self._feat_q.get_nowait()
                    except queue.Empty:
                        break

                    if rms_raw < noise + 0.0005:
                        noise = 0.99 * noise + 0.01 * rms_raw
                    else:
                        noise = 0.999 * noise + 0.001 * rms_raw

                    peak = max(rms_raw, peak * peak_decay, noise + silence_gate_rms)
                    denom = max(peak - noise, silence_gate_rms)
                    rms_norm = float(np.clip((rms_raw - noise) / denom, 0.0, 1.0) ** 0.5)

                    if rms_raw < noise + silence_gate_rms:
                        rms_norm = 0.0

                    rms_smooth_q.append(rms_norm)
                    rms_sm = float(np.mean(rms_smooth_q))

                    env_lp = env_lp + beta * (rms_sm - env_lp)
                    env = float(np.clip(0.75 * env_lp + 0.25 * rms_sm, 0.0, 1.0))

                    env_hist.append(env)
                    cent_hist.append(float(cent))

                    # Adaptive thresholds
                    if len(env_hist) > 300 and len(env_hist) % 100 == 0:
                        vals = np.array(env_hist, dtype=np.float32)
                        k = max(1, int(0.2 * len(vals)))
                        noise_floor_env = float(np.median(np.sort(vals)[:k]))
                        TALK_TH = float(np.clip(noise_floor_env + 0.05, 0.03, 0.18))
                        talk_vals = vals[vals > TALK_TH]
                        if len(talk_vals) > 20:
                            HALF_TH = float(np.percentile(talk_vals, 25))
                            OPEN_TH = float(np.percentile(talk_vals, 58))
                            HALF_TH = max(HALF_TH, TALK_TH + 0.02)
                            OPEN_TH = max(OPEN_TH, HALF_TH + 0.05)
                            cents = np.array(cent_hist, dtype=np.float32)
                            open_mask = vals >= OPEN_TH
                            cent_open = cents[open_mask] if open_mask.sum() > 20 else cents[vals > TALK_TH]
                            if len(cent_open) > 20:
                                U_TH = float(np.percentile(cent_open, 20))
                                E_TH = float(np.percentile(cent_open, 80))

                    # Mouth level
                    if env < HALF_TH:
                        mouth_level = "closed"
                    elif env < OPEN_TH:
                        mouth_level = "half"
                    else:
                        mouth_level = "open"

                    # Vowel selection
                    if mouth_level == "open":
                        is_peak = (e_prev2 < e_prev1) and (e_prev1 >= env) and (e_prev1 > OPEN_TH + 0.02)
                        if is_peak and (t - last_vowel_change_t) >= 0.12:
                            if len(cent_hist) >= 5:
                                cm = float(np.mean(list(cent_hist)[-5:]))
                            else:
                                cm = float(cent)
                            if cm < U_TH:
                                current_open_shape = "u"
                            elif cm > E_TH:
                                current_open_shape = "e"
                            else:
                                current_open_shape = "open"
                            last_vowel_change_t = t
                        mouth_shape_now = current_open_shape
                    elif mouth_level == "half":
                        mouth_shape_now = "half"
                    else:
                        mouth_shape_now = "closed"

                    e_prev2, e_prev1 = e_prev1, env

                # Render frame
                mouth_current = mouth or resolve_mouth_set(loaded, current_emotion)
                frame_current = render_set_frame(loaded, mouth_current, mouth_shape_now)

                frame_rgb = frame_current
                if (self.transition_enabled and self._trans_from_id and self._trans_to_id):
                    if self._trans_to_id != self.current_set_id:
                        self._trans_from_id = None
                        self._trans_to_id = None
                        self._trans_dur_s = 0.0
                        self._trans_from_frame = None
                        self._trans_from_last_idx = None
                        self._trans_from_frozen = False
                    else:
                        if self._trans_dur_s <= 0.0:
                            self._trans_from_id = None
                            self._trans_to_id = None
                            self._trans_from_frame = None
                            self._trans_from_last_idx = None
                            self._trans_from_frozen = False
                        else:
                            progress = (now - self._trans_start_t) / self._trans_dur_s
                            if progress >= 1.0:
                                self._trans_from_id = None
                                self._trans_to_id = None
                                self._trans_dur_s = 0.0
                                self._trans_from_frame = None
                                self._trans_from_last_idx = None
                                self._trans_from_frozen = False
                            else:
                                progress = float(np.clip(progress, 0.0, 1.0))
                                eased = progress * progress * (3.0 - 2.0 * progress)
                                loaded_from = self.loader.get(self._trans_from_id)
                                if loaded_from:
                                    mouth_from = resolve_mouth_set(loaded_from, current_emotion)
                                    if self._trans_from_frozen and self._trans_from_frame is not None:
                                        frame_from = self._trans_from_frame
                                    else:
                                        prev_idx = self._trans_from_last_idx
                                        frame_from = render_set_frame(loaded_from, mouth_from, mouth_shape_now)
                                        cur_idx = loaded_from.bg_video.frame_idx
                                        if (prev_idx is not None and loaded_from.bg_video.total_frames
                                                and cur_idx < prev_idx):
                                            self._trans_from_frozen = True
                                            if self._trans_from_frame is not None:
                                                frame_from = self._trans_from_frame
                                        else:
                                            self._trans_from_frame = frame_from
                                            self._trans_from_last_idx = cur_idx
                                    frame_rgb = blend_frames(frame_from, frame_current, eased)
                                else:
                                    self._trans_from_id = None
                                    self._trans_to_id = None
                                    self._trans_dur_s = 0.0
                                    self._trans_from_frame = None
                                    self._trans_from_last_idx = None
                                    self._trans_from_frozen = False

                # HUD overlay
                if self.show_hud:
                    frame_disp = frame_rgb.copy()
                    hud_text = format_emotion_hud_text(infer_label_from_set_name(current_emotion))
                    cv2.putText(frame_disp, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame_disp, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    frame_disp = frame_rgb

                # Send preview frame to queue (fps limited, scaled)
                if self.frame_queue is not None:
                    now_send = time.perf_counter()
                    if (now_send - self._last_frame_send_t) >= frame_send_interval:
                        self._last_frame_send_t = now_send
                        # Scale down before sending to reduce GUI thread load
                        preview_scale = 0.5
                        prev_w = int(base_w * preview_scale)
                        prev_h = int(base_h * preview_scale)
                        preview_rgb = cv2.resize(frame_disp, (prev_w, prev_h), interpolation=cv2.INTER_AREA)
                        try:
                            # Drop old frame if queue is full
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put_nowait(preview_rgb)
                        except queue.Full:
                            pass

                # Virtual cam
                if cam:
                    cam.send(frame_disp)
                    cam.sleep_until_next_frame()

                # Pacing
                next_frame_t += 1.0 / render_fps
                sleep_s = next_frame_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_frame_t = time.perf_counter()

        finally:
            if self._audio_stream:
                self._audio_stream.stop()
                self._audio_stream.close()
            if cam:
                cam.close()
            self.running = False


# ============================================================
# MultiVideoLiveApp
# ============================================================

class MultiVideoLiveApp(tk.Tk):
    """メインGUIアプリケーション"""

    def __init__(self):
        super().__init__()

        self.title(f"マルチ動画ライブ実行 {__VERSION__}")
        self.geometry("800x700")

        self.set_manager = VideoSetManager()
        self.loader: VideoSetLoader | None = None
        self.live_runner: LiveRunner | None = None

        self.selected_set_id: str | None = None
        self.base_fps: float = 30.0
        self.base_size: tuple[int, int] = (1440, 2560)

        # Preview state
        self._frame_queue: "queue.Queue[np.ndarray] | None" = None
        self._preview_image: ImageTk.PhotoImage | None = None
        self._preview_polling_id: str | None = None
        self._preview_window: tk.Toplevel | None = None
        self._preview_label: ttk.Label | None = None

        self._build_ui()
        self._load_session()

    def _build_ui(self) -> None:
        """UIを構築"""
        # Main container
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        # Settings section
        settings_frame = ttk.LabelFrame(main, text="設定", padding=10)
        settings_frame.pack(fill="x", pady=(0, 10))

        # Parent folder
        folder_frame = ttk.Frame(settings_frame)
        folder_frame.pack(fill="x", pady=2)
        ttk.Label(folder_frame, text="親フォルダ:").pack(side="left")
        self.parent_folder_var = tk.StringVar()
        self.parent_folder_entry = ttk.Entry(folder_frame, textvariable=self.parent_folder_var, width=40)
        self.parent_folder_entry.pack(side="left", padx=5)
        ttk.Button(folder_frame, text="選択...", command=self._on_select_parent_folder).pack(side="left")
        ttk.Button(folder_frame, text="自動検出", command=self._on_auto_detect).pack(side="left", padx=5)

        # Common mouth dir
        mouth_frame = ttk.Frame(settings_frame)
        mouth_frame.pack(fill="x", pady=2)
        ttk.Label(mouth_frame, text="共通mouth:").pack(side="left")
        self.common_mouth_var = tk.StringVar()
        ttk.Entry(mouth_frame, textvariable=self.common_mouth_var, width=40).pack(side="left", padx=5)
        ttk.Button(mouth_frame, text="選択...", command=self._on_select_common_mouth).pack(side="left")

        # Audio device
        audio_frame = ttk.Frame(settings_frame)
        audio_frame.pack(fill="x", pady=2)
        ttk.Label(audio_frame, text="オーディオ入力:").pack(side="left")
        self.audio_device_var = tk.StringVar()
        self.audio_combo = ttk.Combobox(audio_frame, textvariable=self.audio_device_var, width=40, state="readonly")
        self.audio_combo.pack(side="left", padx=5)
        ttk.Button(audio_frame, text="再読込", command=self._refresh_audio_devices).pack(side="left")

        # Emotion preset
        emo_frame = ttk.Frame(settings_frame)
        emo_frame.pack(fill="x", pady=2)
        ttk.Label(emo_frame, text="感情オート:").pack(side="left")
        self.emotion_preset_var = tk.StringVar(value="standard")
        self.emotion_combo = ttk.Combobox(
            emo_frame,
            textvariable=self.emotion_preset_var,
            values=["stable", "standard", "snappy"],
            width=15,
            state="readonly"
        )
        self.emotion_combo.pack(side="left", padx=5)
        self.emotion_hud_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(emo_frame, text="HUD表示", variable=self.emotion_hud_var).pack(side="left", padx=5)
        self.use_vcam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(emo_frame, text="仮想カメラ", variable=self.use_vcam_var).pack(side="left", padx=5)
        self.allow_high_fps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(emo_frame, text="FPS上限解除", variable=self.allow_high_fps_var).pack(side="left", padx=5)
        self.auto_cycle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(emo_frame, text="自動巡回", variable=self.auto_cycle_var).pack(side="left", padx=5)

        # Transition effects
        transition_frame = ttk.Frame(settings_frame)
        transition_frame.pack(fill="x", pady=2)
        ttk.Label(transition_frame, text="切替演出:").pack(side="left")
        self.transition_enable_var = tk.BooleanVar(value=True)
        self.transition_enable_chk = ttk.Checkbutton(
            transition_frame,
            text="有効",
            variable=self.transition_enable_var,
            command=self._on_transition_toggle,
        )
        self.transition_enable_chk.pack(side="left", padx=5)
        self.transition_type_var = tk.StringVar(value=list(TRANSITION_EFFECTS.keys())[0])
        self.transition_combo = ttk.Combobox(
            transition_frame,
            textvariable=self.transition_type_var,
            values=list(TRANSITION_EFFECTS.keys()),
            width=12,
            state="readonly",
        )
        self.transition_combo.pack(side="left", padx=5)
        self.transition_sec_var = tk.DoubleVar(value=DEFAULT_TRANSITION_SEC)
        self.transition_scale = ttk.Scale(
            transition_frame,
            from_=0.10,
            to=1.00,
            orient="horizontal",
            variable=self.transition_sec_var,
            command=self._on_transition_sec_change,
            length=160,
        )
        self.transition_scale.pack(side="left", padx=5)
        self.transition_sec_label = ttk.Label(
            transition_frame,
            text=f"{DEFAULT_TRANSITION_SEC:.2f}s",
            width=6,
            anchor="e",
        )
        self.transition_sec_label.pack(side="left", padx=5)

        # Load mode
        load_frame = ttk.Frame(settings_frame)
        load_frame.pack(fill="x", pady=2)
        ttk.Label(load_frame, text="ロード方式:").pack(side="left")
        self.load_mode_var = tk.StringVar(value=LOAD_MODE_LABEL_LIST[0])
        self.load_mode_combo = ttk.Combobox(
            load_frame,
            textvariable=self.load_mode_var,
            values=LOAD_MODE_LABEL_LIST,
            width=12,
            state="readonly",
        )
        self.load_mode_combo.pack(side="left", padx=5)

        # Video sets section
        sets_frame = ttk.LabelFrame(main, text="動画セット", padding=10)
        sets_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Buttons
        btn_frame = ttk.Frame(sets_frame)
        btn_frame.pack(fill="x", pady=(0, 5))
        ttk.Button(btn_frame, text="+追加", command=self._on_add_set).pack(side="left")
        ttk.Button(btn_frame, text="選択中を削除", command=self._on_remove_set).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="←左へ", command=self._on_move_left).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="右へ→", command=self._on_move_right).pack(side="left")

        # Scrollable canvas for video sets
        canvas_frame = ttk.Frame(sets_frame)
        canvas_frame.pack(fill="both", expand=True)

        self.sets_canvas = tk.Canvas(canvas_frame, height=200, bg="#333")
        self.sets_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.sets_canvas.xview)
        scrollbar.pack(side="bottom", fill="x")
        self.sets_canvas.configure(xscrollcommand=scrollbar.set)

        self.sets_inner = ttk.Frame(self.sets_canvas)
        self.sets_canvas.create_window((0, 0), window=self.sets_inner, anchor="nw")
        self.sets_inner.bind("<Configure>", lambda e: self.sets_canvas.configure(scrollregion=self.sets_canvas.bbox("all")))

        # Execution section
        exec_frame = ttk.LabelFrame(main, text="実行", padding=10)
        exec_frame.pack(fill="x", pady=(0, 10))

        self.start_btn = ttk.Button(exec_frame, text="▶ ライブ開始", command=self._on_start_live)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(exec_frame, text="■ 停止", command=self._on_stop_live, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        self.status_var = tk.StringVar(value="待機中")
        ttk.Label(exec_frame, textvariable=self.status_var).pack(side="left", padx=20)

        # Log section
        log_frame = ttk.LabelFrame(main, text="ログ", padding=10)
        log_frame.pack(fill="both", expand=True)

        log_header = ttk.Frame(log_frame)
        log_header.pack(fill="x")
        ttk.Button(log_header, text="ログクリア", command=self._clear_log).pack(side="right")

        self.log_text = tk.Text(log_frame, height=8, state="disabled", bg="#222", fg="#eee")
        self.log_text.pack(fill="both", expand=True)

        # Initialize
        self._refresh_audio_devices()
        self._on_transition_sec_change(str(self.transition_sec_var.get()))
        self._on_transition_toggle()

        # Window close handler
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _apply_common_mouth_dir(self, common_dir: str) -> None:
        """共通mouth_dirを未設定セットに適用"""
        if not common_dir:
            return
        changed = False
        for vs in self.set_manager.sets.values():
            if vs.mouth_dir:
                continue
            if any("mouth/ フォルダが見つかりません" in w for w in vs.warnings):
                vs.warnings = [w for w in vs.warnings if "mouth/ フォルダ" not in w]
            vs.mouth_dir = common_dir
            # 他の無効要因が無ければ有効化
            invalid_flags = any(
                ("FPS差が大きすぎます" in w or
                 "トラック/動画のフレーム数差が大きすぎます" in w or
                 "読み込み失敗" in w)
                for w in vs.warnings
            )
            if not invalid_flags:
                vs.is_valid = True
            changed = True
        if changed:
            self._update_sets_display()

    def _transition_label_to_key(self, label: str) -> str:
        return TRANSITION_EFFECTS.get(label, "crossfade")

    def _transition_key_to_label(self, key: str) -> str:
        for label, value in TRANSITION_EFFECTS.items():
            if key == value or key == label:
                return label
        return list(TRANSITION_EFFECTS.keys())[0]

    def _load_mode_label_to_key(self, label: str) -> str:
        return LOAD_MODE_LABELS.get(label, LOAD_MODE_FULL)

    def _load_mode_key_to_label(self, key: str) -> str:
        for label, value in LOAD_MODE_LABELS.items():
            if key == value or key == label:
                return label
        return LOAD_MODE_LABEL_LIST[0]

    def _on_transition_toggle(self) -> None:
        enabled = bool(self.transition_enable_var.get())
        self.transition_combo.configure(state="readonly" if enabled else "disabled")
        self.transition_scale.configure(state="normal" if enabled else "disabled")

    def _on_transition_sec_change(self, value: str) -> None:
        try:
            sec = float(value)
        except Exception:
            sec = float(self.transition_sec_var.get() or 0.0)
        self.transition_sec_label.configure(text=f"{sec:.2f}s")

    def _log(self, msg: str) -> None:
        """ログに出力（上限超過時は先頭から削除）"""
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        # 上限チェック
        line_count = int(self.log_text.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            excess = line_count - MAX_LOG_LINES
            self.log_text.delete("1.0", f"{excess + 1}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        """ログをクリア"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _open_preview_window(self) -> None:
        """プレビュー用の独立ウィンドウを開く"""
        if self._preview_window is not None:
            return

        # Calculate preview size based on base_size and preview_scale
        preview_scale = 0.5
        prev_w = int(self.base_size[0] * preview_scale)
        prev_h = int(self.base_size[1] * preview_scale)

        self._preview_window = tk.Toplevel(self)
        self._preview_window.title("プレビュー")
        self._preview_window.geometry(f"{prev_w}x{prev_h}")
        self._preview_window.protocol("WM_DELETE_WINDOW", self._on_preview_window_close)

        self._preview_label = ttk.Label(self._preview_window, anchor="center")
        self._preview_label.pack(fill="both", expand=True)

    def _close_preview_window(self) -> None:
        """プレビューウィンドウを閉じる"""
        if self._preview_window is not None:
            try:
                self._preview_window.destroy()
            except Exception:
                pass
            self._preview_window = None
            self._preview_label = None
        self._preview_image = None

    def _on_preview_window_close(self) -> None:
        """プレビューウィンドウが閉じられた時の処理"""
        # プレビューを閉じたらライブも停止
        self._on_stop_live()

    def _start_preview_polling(self) -> None:
        """プレビューポーリングを開始"""
        self._stop_preview_polling()
        self._open_preview_window()
        self._poll_preview()

    def _stop_preview_polling(self) -> None:
        """プレビューポーリングを停止しプレビューをクリア"""
        if self._preview_polling_id is not None:
            try:
                self.after_cancel(self._preview_polling_id)
            except Exception:
                pass
            self._preview_polling_id = None
        self._close_preview_window()

    def _poll_preview(self) -> None:
        """プレビューフレームをポーリングして描画"""
        if self._frame_queue is None:
            return

        try:
            # Get the latest frame (discard older ones)
            frame_rgb = None
            while True:
                try:
                    frame_rgb = self._frame_queue.get_nowait()
                except queue.Empty:
                    break

            if frame_rgb is not None and self._preview_label is not None:
                # Frame is already scaled by LiveRunner, just convert to ImageTk
                pil_img = Image.fromarray(frame_rgb)
                self._preview_image = ImageTk.PhotoImage(pil_img)
                self._preview_label.configure(image=self._preview_image)

        except Exception as e:
            print(f"[preview] error: {e}")

        # Schedule next poll (15ms = ~66fps max polling rate)
        self._preview_polling_id = self.after(15, self._poll_preview)

    def _refresh_audio_devices(self) -> None:
        """オーディオデバイス一覧を更新"""
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append(f"{i}: {dev['name']}")
        except Exception as e:
            self._log(f"オーディオデバイス取得エラー: {e}")

        self.audio_combo["values"] = devices
        if devices and not self.audio_device_var.get():
            self.audio_device_var.set(devices[0])

    def _update_sets_display(self) -> None:
        """動画セットパネルを更新"""
        # Clear existing
        for widget in self.sets_inner.winfo_children():
            widget.destroy()

        # Create cards for each set
        for set_id, vs in self.set_manager.sets.items():
            card = ttk.Frame(self.sets_inner, padding=5)
            card.pack(side="left", padx=5)

            # Thumbnail
            photo = self.set_manager.get_photo_image(set_id)
            if photo:
                lbl = ttk.Label(card, image=photo)
                lbl.pack()
            else:
                placeholder = ttk.Label(card, text="[読込中]", width=15, anchor="center")
                placeholder.pack()

            # Label
            label_text = vs.label[:12] + "..." if len(vs.label) > 12 else vs.label
            ttk.Label(card, text=label_text).pack()

            # Selection indicator
            if set_id == self.selected_set_id:
                ttk.Label(card, text="★選択中", foreground="gold").pack()

            # Warnings
            if vs.warnings:
                ttk.Label(card, text="⚠", foreground="orange").pack()
            if not vs.is_valid:
                ttk.Label(card, text="無効", foreground="red").pack()

            # Click handler
            card.bind("<Button-1>", lambda e, sid=set_id: self._on_select_set(sid))
            for child in card.winfo_children():
                child.bind("<Button-1>", lambda e, sid=set_id: self._on_select_set(sid))

    def _on_select_set(self, set_id: str) -> None:
        """動画セット選択"""
        vs = self.set_manager.sets.get(set_id)
        if not vs:
            return
        if not vs.is_valid and self.live_runner and self.live_runner.running:
            self._log("無効なセットはライブ中に切り替えできません")
            return
        self.selected_set_id = set_id
        self._update_sets_display()

        if self.live_runner and self.live_runner.running:
            self.live_runner.switch_to(set_id)
            self._log(f"切り替え: {vs.label}")

    def _on_select_parent_folder(self) -> None:
        """親フォルダ選択"""
        folder = filedialog.askdirectory(title="親フォルダを選択")
        if folder:
            self.parent_folder_var.set(folder)

    def _on_select_common_mouth(self) -> None:
        """共通mouth_dir選択"""
        folder = filedialog.askdirectory(title="共通mouth_dirを選択")
        if not folder:
            return
        folder = os.path.abspath(folder)
        if not _is_mouth_root(folder):
            self._log(f"警告: 選択したフォルダに口スプライトが見つかりません: {folder}")
            messagebox.showwarning("警告", "口スプライト (open.png) が見つかりません")
            return
        self.common_mouth_var.set(folder)
        self._log(f"共通mouth_dir: {folder}")
        # 既存セットにも適用するか確認
        if self.set_manager.sets:
            apply_all = messagebox.askyesno(
                "確認",
                "既存の全セットにもこの共通mouth_dirを適用しますか？"
            )
            if apply_all:
                for vs in self.set_manager.sets.values():
                    vs.mouth_dir = folder
                    vs.is_valid = True
                self._log("全セットに共通mouth_dirを適用しました")
        self._update_sets_display()
        self._save_session()

    def _on_add_set(self) -> None:
        """動画セット追加"""
        if self.live_runner and self.live_runner.running:
            messagebox.showwarning("警告", "ライブ中はセットを追加できません")
            return

        folder = filedialog.askdirectory(title="動画セットフォルダを選択")
        if not folder:
            return

        # 最初のセットの場合はbase_fps/base_sizeを設定
        base_fps = self.base_fps if self.set_manager.sets else None

        vs, warnings = self.set_manager.add_set(
            folder,
            common_mouth_dir=self.common_mouth_var.get(),
            base_fps=base_fps,
            allow_duplicate=True,
        )

        if vs:
            # 最初のセットの場合はbase_fps/base_sizeを記録
            if len(self.set_manager.sets) == 1:
                self.base_fps = vs.fps
                self.base_size = vs.video_size if vs.video_size[0] > 0 else (1440, 2560)

            self._log(f"追加: {vs.label}")
            if warnings:
                for w in warnings:
                    self._log(f"  警告: {w}")

            if not self.selected_set_id:
                self.selected_set_id = vs.id

            # Generate thumbnail in background
            self.set_manager.generate_thumbnails_async(
                callback=lambda sid: self.after(0, self._update_sets_display)
            )
            self._update_sets_display()
            self._save_session()
        else:
            messagebox.showerror("エラー", "\n".join(warnings))

    def _on_remove_set(self) -> None:
        """選択中のセットを削除"""
        if self.live_runner and self.live_runner.running:
            messagebox.showwarning("警告", "ライブ中はセットを削除できません")
            return

        if not self.selected_set_id:
            return

        vs = self.set_manager.sets.get(self.selected_set_id)
        if vs:
            self._log(f"削除: {vs.label}")
            self.set_manager.remove_set(self.selected_set_id)
            self.selected_set_id = None

            # Select first remaining set
            if self.set_manager.sets:
                self.selected_set_id = list(self.set_manager.sets.keys())[0]

            self._update_sets_display()
            self._save_session()

    def _move_selected_set(self, delta: int) -> None:
        """選択中セットの並び替え"""
        if self.live_runner and self.live_runner.running:
            messagebox.showwarning("警告", "ライブ中は並び替えできません")
            return
        if not self.selected_set_id:
            return
        ids = list(self.set_manager.sets.keys())
        try:
            idx = ids.index(self.selected_set_id)
        except ValueError:
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(ids):
            return
        ids[idx], ids[new_idx] = ids[new_idx], ids[idx]
        self.set_manager.reorder_sets(ids)
        self._update_sets_display()
        self._save_session()

    def _on_move_left(self) -> None:
        """選択中セットを左へ移動"""
        self._move_selected_set(-1)

    def _on_move_right(self) -> None:
        """選択中セットを右へ移動"""
        self._move_selected_set(1)

    def _on_auto_detect(self) -> None:
        """自動検出"""
        if self.live_runner and self.live_runner.running:
            messagebox.showwarning("警告", "ライブ中は自動検出できません")
            return

        parent = self.parent_folder_var.get()
        if not parent or not os.path.isdir(parent):
            messagebox.showerror("エラー", "親フォルダを選択してください")
            return

        base_fps = self.base_fps if self.set_manager.sets else None
        results = self.set_manager.auto_detect_sets(
            parent,
            common_mouth_dir=self.common_mouth_var.get(),
            base_fps=base_fps,
        )

        if results:
            # Update base_fps/base_size from first detected set if no existing sets
            if len(self.set_manager.sets) == len(results):
                first_vs = results[0][0]
                self.base_fps = first_vs.fps
                self.base_size = first_vs.video_size if first_vs.video_size[0] > 0 else (1440, 2560)

            for vs, warnings in results:
                self._log(f"検出: {vs.label}")
                if warnings:
                    for w in warnings:
                        self._log(f"  警告: {w}")

            if not self.selected_set_id and self.set_manager.sets:
                self.selected_set_id = list(self.set_manager.sets.keys())[0]

            self.set_manager.generate_thumbnails_async(
                callback=lambda sid: self.after(0, self._update_sets_display)
            )
            self._update_sets_display()
            self._save_session()
        else:
            self._log("検出されたセットはありません")

    def _on_start_live(self) -> None:
        """ライブ開始"""
        if not self.set_manager.sets:
            messagebox.showerror("エラー", "動画セットを追加してください")
            return

        # Recompute base_size/base_fps from selected set (fallback to first usable set)
        selected_vs = self.set_manager.sets.get(self.selected_set_id) if self.selected_set_id in self.set_manager.sets else None
        base_candidate = selected_vs
        if not (base_candidate and base_candidate.mouth_dir and os.path.isdir(base_candidate.mouth_dir)):
            base_candidate = next(
                (vs for vs in self.set_manager.sets.values()
                 if vs.mouth_dir and os.path.isdir(vs.mouth_dir)),
                None
            )
        if base_candidate:
            base_size = base_candidate.video_size
            if base_size[0] <= 0 or base_size[1] <= 0:
                base_size = probe_video_size(base_candidate.video_path) or base_size
            if base_size[0] <= 0 or base_size[1] <= 0:
                base_size = self.base_size if self.base_size[0] > 0 else (1440, 2560)
            base_fps = base_candidate.fps
            if base_fps <= 0:
                base_fps = probe_video_fps(base_candidate.video_path) or 30.0
            self.base_size = base_size
            self.base_fps = base_fps
            if not self.allow_high_fps_var.get() and self.base_fps > MAX_RENDER_FPS:
                self._log(f"警告: base_fps {self.base_fps:.1f} > {MAX_RENDER_FPS}、render_fpsは{MAX_RENDER_FPS}に制限")

        # Revalidate sets against current base_fps
        for vs in self.set_manager.sets.values():
            vs.warnings = [w for w in vs.warnings if "FPS" not in w]
            non_fps_invalid = False
            if not (vs.mouth_dir and os.path.isdir(vs.mouth_dir)):
                non_fps_invalid = True
            if any("トラック/動画のフレーム数差が大きすぎます" in w for w in vs.warnings):
                non_fps_invalid = True
            if any("読み込み失敗" in w for w in vs.warnings):
                non_fps_invalid = True
            if vs.fps <= 0:
                vs.warnings.append("FPS取得に失敗（差分チェックをスキップ）")
            elif self.base_fps > 0:
                fps_diff = abs(vs.fps - self.base_fps)
                warn_th = max(FPS_WARNING_MIN, self.base_fps * FPS_WARNING_RATIO)
                invalid_th = max(FPS_INVALID_MIN, self.base_fps * FPS_INVALID_RATIO)
                if fps_diff > invalid_th:
                    vs.warnings.append(f"FPS差が大きすぎます（ベース: {self.base_fps:.1f}, 追加: {vs.fps:.1f}）")
                    non_fps_invalid = True
                elif fps_diff > warn_th:
                    vs.warnings.append(f"FPSが異なります（ベース: {self.base_fps:.1f}, 追加: {vs.fps:.1f}）")
            vs.is_valid = not non_fps_invalid

        valid_sets = [vs for vs in self.set_manager.sets.values() if vs.is_valid]
        if not valid_sets:
            messagebox.showerror("エラー", "有効な動画セットがありません")
            return
        if (not self.selected_set_id or
                self.selected_set_id not in self.set_manager.sets or
                not self.set_manager.sets[self.selected_set_id].is_valid):
            self.selected_set_id = valid_sets[0].id

        self._update_sets_display()

        # Parse audio device
        audio_device = None
        try:
            audio_str = self.audio_device_var.get()
            if audio_str:
                audio_device = int(audio_str.split(":")[0])
        except Exception:
            pass
        if audio_device is None:
            self._log("オーディオ入力: 既定デバイスを使用")

        # Get load mode
        load_mode = self._load_mode_label_to_key(self.load_mode_var.get())
        recommended_max = get_recommended_max_sets()
        current_count = len(valid_sets)

        # Warning dialog for full mode with too many sets
        if load_mode == LOAD_MODE_FULL and current_count > recommended_max:
            msg = f"推奨: {recommended_max}セット以下 / 現在: {current_count}セット。\n全件ロードを続行しますか？"
            if not messagebox.askyesno("確認", msg):
                return

        # Disable UI
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.audio_combo.configure(state="disabled")
        self.status_var.set("読み込み中...")

        def loading_worker():
            try:
                # Create loader
                self.loader = VideoSetLoader(self.base_size, self.base_fps)

                # Set registry and log callback for LRU mode
                self.loader.set_registry = {vs.id: vs for vs in valid_sets}
                self.loader.log_callback = lambda msg: self.after(0, lambda m=msg: self._log(m))

                # Log load mode info
                load_mode_label = "全件ロード" if load_mode == LOAD_MODE_FULL else "メモリ節約"
                self.after(0, lambda: self._log(f"ロード方式: {load_mode_label}"))
                self.after(0, lambda: self._log(f"推奨: {recommended_max}セット以下 / 現在: {current_count}セット"))
                if load_mode == LOAD_MODE_LRU:
                    self.after(0, lambda: self._log(f"LRU上限: {LRU_MAX_SETS}セット"))

                if load_mode == LOAD_MODE_FULL:
                    # Full mode: load all valid sets
                    errors = self.loader.load_all(
                        valid_sets,
                        progress_callback=lambda label, i, n: self.after(
                            0, lambda: self.status_var.set(f"読み込み中: {label} ({i}/{n})")
                        )
                    )

                    if errors:
                        for sid in errors:
                            vs = self.set_manager.sets.get(sid)
                            if vs:
                                if "読み込み失敗" not in vs.warnings:
                                    vs.warnings.append("読み込み失敗")
                                vs.is_valid = False
                                self.after(0, lambda v=vs: self._log(f"読み込み失敗: {v.label}"))
                        self.after(0, self._update_sets_display)

                    loaded_ids = list(self.loader.loaded.keys())
                    if not loaded_ids:
                        self.after(0, lambda: self._log("ロード成功したセットがありません"))
                        self.after(0, self._on_live_stopped)
                        return
                    if self.selected_set_id not in loaded_ids:
                        self.selected_set_id = loaded_ids[0]

                else:
                    # LRU mode: only load selected set initially
                    self.after(0, lambda: self.status_var.set(f"読み込み中: {self.selected_set_id}"))
                    selected_vs = next((vs for vs in valid_sets if vs.id == self.selected_set_id), None)
                    if selected_vs is None:
                        selected_vs = valid_sets[0]
                        self.selected_set_id = selected_vs.id

                    loaded = self.loader.load_one(selected_vs)
                    if loaded is None:
                        self.after(0, lambda: self._log(f"読み込み失敗: {selected_vs.label}"))
                        self.after(0, self._on_live_stopped)
                        return
                    self.loader.loaded[selected_vs.id] = loaded

                    loaded_ids = [vs.id for vs in valid_sets]  # All valid IDs for cycle order

                # Create frame queue for preview
                self._frame_queue = queue.Queue(maxsize=2)

                # Start live runner
                self.live_runner = LiveRunner(
                    loader=self.loader,
                    audio_device=audio_device,
                    emotion_preset=self.emotion_preset_var.get(),
                    show_hud=self.emotion_hud_var.get(),
                    use_vcam=self.use_vcam_var.get(),
                    allow_high_fps=self.allow_high_fps_var.get(),
                    auto_cycle=self.auto_cycle_var.get(),
                    cycle_order=loaded_ids,
                    transition_enabled=self.transition_enable_var.get(),
                    transition_type=self._transition_label_to_key(self.transition_type_var.get()),
                    transition_sec=self.transition_sec_var.get(),
                    frame_queue=self._frame_queue,
                    preview_max_fps=30.0,
                    load_mode=load_mode,
                )

                self.live_runner.start(self.selected_set_id)
                self.after(0, self._start_preview_polling)
                self.after(0, lambda: self.status_var.set("ライブ中"))
                self.after(0, lambda: self._log("ライブ開始"))

                # Monitor for stop
                while self.live_runner and self.live_runner.running:
                    time.sleep(0.5)

                self.after(0, self._on_live_stopped)

            except Exception as e:
                self.after(0, lambda: self._log(f"エラー: {e}"))
                self.after(0, self._on_live_stopped)

        threading.Thread(target=loading_worker, daemon=True).start()

    def _on_stop_live(self) -> None:
        """ライブ停止"""
        if self.live_runner:
            self.live_runner.stop()

    def _on_live_stopped(self) -> None:
        """ライブ停止後のUI更新"""
        # Stop preview polling
        self._stop_preview_polling()
        self._frame_queue = None

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.audio_combo.configure(state="readonly")
        self.status_var.set("待機中")
        self._log("ライブ停止")

        if self.loader:
            self.loader.unload_all()
            self.loader = None
        self.live_runner = None

    def _load_session(self) -> None:
        """セッション復元"""
        try:
            if not os.path.isfile(SESSION_FILE):
                return

            with open(SESSION_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self._log("セッション復元エラー: データ形式が不正です")
                return

            self.parent_folder_var.set(str(data.get("last_parent_folder", "")))
            common_dir = str(data.get("common_mouth_dir", ""))
            if common_dir:
                if _is_mouth_root(common_dir):
                    self.common_mouth_var.set(common_dir)
                else:
                    self._log(f"警告: 保存された共通mouth_dirが無効です: {common_dir}")

            # Audio device
            audio_device = data.get("audio_device", "")
            if audio_device not in ("", None):
                self.audio_device_var.set(str(audio_device))

            ep = str(data.get("emotion_preset", "standard"))
            if ep not in EMOTION_PRESET_PARAMS:
                ep = "standard"
            self.emotion_preset_var.set(ep)
            self.emotion_hud_var.set(_safe_bool(data.get("emotion_hud", True), default=True))
            self.allow_high_fps_var.set(_safe_bool(data.get("allow_high_fps", False), default=False))
            self.auto_cycle_var.set(_safe_bool(data.get("auto_cycle", False), default=False))
            self.transition_enable_var.set(_safe_bool(data.get("transition_enabled", True), default=True))

            saved_effect = str(data.get("transition_type", "crossfade"))
            self.transition_type_var.set(self._transition_key_to_label(saved_effect))
            transition_sec = _safe_float(
                data.get("transition_sec", DEFAULT_TRANSITION_SEC),
                DEFAULT_TRANSITION_SEC,
                min_v=0.0,
            )
            self.transition_sec_var.set(transition_sec)
            self._on_transition_sec_change(str(self.transition_sec_var.get()))
            self._on_transition_toggle()

            # Load mode (default to full for backwards compatibility)
            saved_load_mode = str(data.get("load_mode", LOAD_MODE_FULL))
            self.load_mode_var.set(self._load_mode_key_to_label(saved_load_mode))

            self.base_fps = _safe_float(data.get("base_fps", 30.0), 30.0, min_v=0.0)
            base_size = data.get("base_size", [1440, 2560])
            if isinstance(base_size, (list, tuple)) and len(base_size) == 2:
                w = _safe_int(base_size[0], 1440, min_v=1)
                h = _safe_int(base_size[1], 2560, min_v=1)
                if w > 0 and h > 0:
                    self.base_size = (w, h)
                else:
                    self.base_size = (1440, 2560)
            else:
                self.base_size = (1440, 2560)

            # Restore sets
            sets_data = data.get("sets", [])
            if isinstance(sets_data, list):
                for set_data in sets_data:
                    if not isinstance(set_data, dict):
                        continue
                    vs = VideoSet.from_dict(set_data)
                    # Validate paths still exist
                    if os.path.isfile(vs.video_path) and os.path.isfile(vs.track_path):
                        vs.fps = probe_video_fps(vs.video_path) or 0.0
                        vs.video_size = probe_video_size(vs.video_path) or (0, 0)
                        vs.frame_count = probe_video_frame_count(vs.video_path) or 0
                        vs.is_valid = bool(vs.mouth_dir and os.path.isdir(vs.mouth_dir))
                        if vs.fps <= 0:
                            vs.warnings.append("FPS取得に失敗（差分チェックをスキップ）")
                        if vs.frame_count <= 0:
                            vs.warnings.append("フレーム数取得に失敗（差分チェックをスキップ）")
                        self.set_manager.sets[vs.id] = vs
            # Apply common mouth dir to missing sets
            self._apply_common_mouth_dir(self.common_mouth_var.get())

            selected = data.get("last_active_set")
            self.selected_set_id = selected if isinstance(selected, str) else None
            if self.selected_set_id not in self.set_manager.sets:
                self.selected_set_id = list(self.set_manager.sets.keys())[0] if self.set_manager.sets else None

            # Generate thumbnails
            self.set_manager.generate_thumbnails_async(
                callback=lambda sid: self.after(0, self._update_sets_display)
            )
            self._update_sets_display()

            self._log(f"セッション復元: {len(self.set_manager.sets)}セット")

        except Exception as e:
            self._log(f"セッション復元エラー: {e}")

    def _save_session(self) -> None:
        """セッション保存"""
        try:
            data = {
                "version": "1.2",
                "last_parent_folder": self.parent_folder_var.get(),
                "common_mouth_dir": self.common_mouth_var.get(),
                "audio_device": self.audio_device_var.get(),
                "emotion_preset": self.emotion_preset_var.get(),
                "emotion_hud": self.emotion_hud_var.get(),
                "allow_high_fps": self.allow_high_fps_var.get(),
                "auto_cycle": self.auto_cycle_var.get(),
                "transition_enabled": self.transition_enable_var.get(),
                "transition_type": self._transition_label_to_key(self.transition_type_var.get()),
                "transition_sec": float(self.transition_sec_var.get()),
                "load_mode": self._load_mode_label_to_key(self.load_mode_var.get()),
                "base_fps": self.base_fps,
                "base_size": list(self.base_size),
                "sets": [vs.to_dict() for vs in self.set_manager.sets.values()],
                "last_active_set": self.selected_set_id,
            }

            with open(SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self._log(f"セッション保存エラー: {e}")

    def _on_close(self) -> None:
        """ウィンドウクローズ"""
        # Stop preview polling first
        self._stop_preview_polling()
        self._frame_queue = None

        if self.live_runner and self.live_runner.running:
            self.live_runner.stop()
        self._save_session()
        self.destroy()


# ============================================================
# Main
# ============================================================

def main():
    print(f"[info] multi_video_live_gui {__VERSION__}")
    app = MultiVideoLiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
