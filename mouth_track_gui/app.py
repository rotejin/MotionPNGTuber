#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouth_track_gui.py  (One-Click HQ)

最高品質・設定最小の「一気通貫」GUI
- 動画を選ぶ
- 解析 (auto_mouth_track_v2.py)  ※自動修復 + early-stop
- 解析後にキャリブ画面を自動表示 (calibrate_mouth_track.py)
- キャリブが終わったら口消しを自動生成 (auto_erase_mouth.py)
- 最後に口消し動画を自動プレビュー

ユーザーが触る設定:
- 口消し範囲 (coverage)

注意:
- サブプロセス出力の文字化け/Unicode問題を避けるため、UTF-8環境変数を付与します。
"""

from __future__ import annotations

import os
import sys
import shutil
import json
import tempfile
from pathlib import Path
import queue
import threading
import subprocess
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from motionpngtuber.platform_open import (
    open_path_with_default_app,
    prefer_native_video_preview,
)

from motionpngtuber.workflow_validation import (
    WorkflowPaths,
    build_workflow_paths,
    format_missing_path_message,
    validate_existing_dir,
    validate_existing_file,
)
from .state import (
    HERE,
    safe_bool,
    safe_int,
    safe_float,
    load_session,
    save_session,
)
from .services import (
    script_contains,
    list_input_devices,
    find_input_device_item,
    ensure_backend_sanity,
    guess_mouth_dir,
    is_emotion_level_mouth_root,
    list_character_dirs,
    resolve_character_dir,
    best_open_sprite_for_character,
)
from .runner import CommandRunner, RunResult
from .actions import (
    ActionPlan,
    PostAction,
    plan_track_and_calib,
    plan_calib_only,
    plan_erase,
    plan_live,
    resolve_runtime_script,
)
from .ui import (
    WidgetRefs,
    UiVars,
    UiCallbacks,
    build_ui,
    STOP_BTN_TEXT_DEFAULT,
)
from .live_ipc import (
    LiveIpcSession,
    cleanup_live_ipc_session,
    create_live_ipc_session,
)
from motionpngtuber.mouth_color_adjust import (
    MouthColorAdjust,
    clamp_mouth_color_adjust,
)


# --- smoothing presets (GUI) ---
SMOOTHING_PRESETS: dict[str, float | None] = {
    "Auto（今のまま）": None,  # pass nothing -> keep current default behavior
    "ゆっくり（1.5）": 1.5,
    "普通（3.0）": 3.0,
    "高速（6.0）": 6.0,
    "追従最優先（0）": 0.0,  # disable smoothing
}
SMOOTHING_LABELS = list(SMOOTHING_PRESETS.keys())




# --- emotion preset (GUI / runtime) ---
EMOTION_PRESETS: dict[str, str] = {
    "安定（配信向け）": "stable",
    "標準": "standard",
    "キビキビ（ゲーム向け）": "snappy",
}
EMOTION_PRESET_LABELS = list(EMOTION_PRESETS.keys())

SOFT_STOP_GRACE_SEC = 3.0
STOP_BTN_TEXT_SOFT = "停止予約中（もう一度で強制停止）"
MAX_LOG_LINES = 200  # ログ表示の上限行数

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mouth Track One-Click (HQ)")
        self.geometry("1180x860")

        self.log_q: "queue.Queue[str]" = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.active_proc: subprocess.Popen | None = None  # legacy; use runner
        self.stop_mode = "none"  # none / soft / force
        self.soft_requested_at: float | None = None
        self._soft_warn_job: str | None = None
        self.runner = CommandRunner(on_log=self.log)
        self._live_ipc_session: LiveIpcSession | None = None
        self._live_ipc_token = ""
        self._live_color_control_active = False
        self._live_color_control_path = ""
        self._live_color_control_job: str | None = None
        self._auto_color_request_path = ""
        self._auto_color_result_path = ""
        self._auto_color_request_pending = False
        self._auto_color_request_id: str | None = None
        self._auto_color_request_deadline = 0.0
        self._auto_color_poll_job: str | None = None
        self._suspend_mouth_color_adjust_events = False

        sess = load_session()
        # GUIでは元動画を表示したいが、runtimeは背景としてmouthlessを使いたいので
        # sessionには video(=背景用) と source_video(=元動画) を分けて保存する
        self.video_var = tk.StringVar(value=str(sess.get("source_video", sess.get("video", "")) or ""))
        self.mouth_dir_var = tk.StringVar(value=str(sess.get("mouth_dir", "")) or "")

        # --- character / emotion-auto (runtime) ---
        self.character_var = tk.StringVar(value=str(sess.get("character", "")))

        _ep = str(sess.get("emotion_preset", "標準"))
        if _ep not in EMOTION_PRESETS:
            _ep = "標準"
        self.emotion_preset_var = tk.StringVar(value=_ep)

        self.emotion_hud_var = tk.BooleanVar(value=safe_bool(sess.get("emotion_hud", False), default=False))
        self.coverage_var = tk.DoubleVar(value=safe_float(sess.get("coverage", 0.60), 0.60, min_v=0.40, max_v=0.90))
        self.mouth_brightness_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_brightness", 0.0), 0.0, min_v=-32.0, max_v=32.0),
        )
        self.mouth_saturation_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_saturation", 1.0), 1.0, min_v=0.70, max_v=1.50),
        )
        self.mouth_warmth_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_warmth", 0.0), 0.0, min_v=-24.0, max_v=24.0),
        )
        self.mouth_color_strength_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_color_strength", 0.75), 0.75, min_v=0.0, max_v=1.0),
        )
        self.mouth_edge_priority_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_edge_priority", 0.85), 0.85, min_v=0.0, max_v=1.0),
        )
        self.mouth_edge_width_ratio_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_edge_width_ratio", 0.10), 0.10, min_v=0.02, max_v=0.20),
        )
        self.mouth_inspect_boost_var = tk.DoubleVar(
            value=safe_float(sess.get("mouth_inspect_boost", 1.0), 1.0, min_v=1.0, max_v=4.0),
        )
        # 解析→キャリブの mouth quad 余白係数。
        # 通常は 2.1 を推奨し、必要な場合だけ詳細設定で微調整できるようにする。
        self.pad_var = tk.DoubleVar(
            value=safe_float(sess.get("pad", 2.10), 2.10, min_v=1.20, max_v=3.20),
        )

        # erase shading preset (GUI only): plane=ON, none=OFF
        _esh = sess.get("erase_shading", sess.get("shading", "plane"))
        _esh_str = str(_esh).lower()
        self.erase_shading_var = tk.BooleanVar(value=(_esh_str != "none"))

        # tracking smoothing preset (GUI only)
        _smooth = sess.get("smoothing", "Auto（今のまま）")
        if _smooth not in SMOOTHING_PRESETS:
            _smooth = "Auto（今のまま）"
        self.smoothing_menu_var = tk.StringVar(value=_smooth)

        # runtime用：オーディオ入力デバイス
        self.audio_device_var = tk.IntVar(value=safe_int(sess.get("audio_device", 31), 31, min_v=0))
        _audio_spec = str(sess.get("audio_device_spec", "") or "").strip()
        if not _audio_spec:
            _audio_spec = f"sd:{int(self.audio_device_var.get())}"
        self.audio_device_spec_var = tk.StringVar(value=_audio_spec)
        self.audio_device_menu_var = tk.StringVar(value="")

        # Progress (step-level)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="待機中")
        self._progress_total = 1

        self._ui = self._build_ui()

        # Refresh character list when mouth root changes
        self._char_refresh_job = None
        self.mouth_dir_var.trace_add("write", lambda *_: self._schedule_refresh_characters())
        self._refresh_audio_devices(init=True)

        if self.video_var.get() and not self.mouth_dir_var.get():
            self._autofill_mouth_dir()
        self._refresh_characters(init=True)

        self.after(0, self._fit_initial_window)
        self.after(100, self._poll_logs)

    # ----- UI -----
    def _build_ui(self) -> WidgetRefs:
        ui_vars = UiVars(
            video=self.video_var,
            mouth_dir=self.mouth_dir_var,
            character=self.character_var,
            pad=self.pad_var,
            coverage=self.coverage_var,
            mouth_brightness=self.mouth_brightness_var,
            mouth_saturation=self.mouth_saturation_var,
            mouth_warmth=self.mouth_warmth_var,
            mouth_color_strength=self.mouth_color_strength_var,
            mouth_edge_priority=self.mouth_edge_priority_var,
            mouth_edge_width_ratio=self.mouth_edge_width_ratio_var,
            mouth_inspect_boost=self.mouth_inspect_boost_var,
            erase_shading=self.erase_shading_var,
            smoothing_menu=self.smoothing_menu_var,
            audio_device_menu=self.audio_device_menu_var,
            emotion_preset=self.emotion_preset_var,
            emotion_hud=self.emotion_hud_var,
            progress=self.progress_var,
            progress_text=self.progress_text_var,
        )
        ui_callbacks = UiCallbacks(
            on_open_sprite_extractor=self.on_open_sprite_extractor,
            on_pick_video=self.on_pick_video,
            on_pick_mouth_dir=self.on_pick_mouth_dir,
            refresh_characters=self._refresh_characters,
            refresh_audio_devices=self._refresh_audio_devices,
            on_track_and_calib=self.on_track_and_calib,
            on_calib_only=self.on_calib_only,
            on_erase_mouthless=self.on_erase_mouthless,
            on_preview_erase_range=self.on_preview_erase_range,
            on_live_run=self.on_live_run,
            on_auto_mouth_color_adjust=self.on_auto_mouth_color_adjust,
            on_mouth_color_adjust_changed=self._on_mouth_color_adjust_changed,
            on_reset_mouth_color_adjust=self._reset_mouth_color_adjust,
            on_stop=self.on_stop,
            clear_log=self._clear_log,
            save_session=self._save_session,
        )
        refs = build_ui(
            self,
            ui_vars,
            ui_callbacks,
            smoothing_labels=SMOOTHING_LABELS,
            emotion_preset_labels=EMOTION_PRESET_LABELS,
        )
        # Map widget refs to self for backward compatibility
        self.cmb_character = refs.cmb_character
        self.cmb_audio = refs.cmb_audio
        self.cmb_smooth = refs.cmb_smooth
        self.cmb_emotion_preset = refs.cmb_emotion_preset
        self.pad_value_label = refs.pad_value_label
        self.cov_label = refs.cov_label
        self.btn_track_calib = refs.btn_track_calib
        self.btn_calib_only = refs.btn_calib_only
        self.btn_erase = refs.btn_erase
        self.btn_erase_range = refs.btn_erase_range
        self.btn_live = refs.btn_live
        self.btn_auto_color = refs.btn_auto_color
        self.btn_stop = refs.btn_stop
        self.progress = refs.progress_bar
        self.txt = refs.txt
        return refs

    def _fit_initial_window(self) -> None:
        try:
            self.update_idletasks()
            screen_w = max(1, int(self.winfo_screenwidth()))
            screen_h = max(1, int(self.winfo_screenheight()))
            avail_w = max(900, screen_w - 80)
            avail_h = max(700, screen_h - 100)

            req_w = max(1180, int(self.winfo_reqwidth()) + 20)
            req_h = max(860, int(self.winfo_reqheight()) + 20)

            self.minsize(min(req_w, avail_w), min(req_h, avail_h))

            needs_zoom = (req_w > avail_w) or (req_h > avail_h)
            if needs_zoom and sys.platform.startswith("win"):
                try:
                    self.state("zoomed")
                    return
                except Exception:
                    pass

            win_w = min(req_w, avail_w)
            win_h = min(req_h, avail_h)
            x = max(0, (screen_w - win_w) // 2)
            y = max(0, (screen_h - win_h) // 2)
            self.geometry(f"{win_w}x{win_h}+{x}+{y}")
        except Exception:
            pass

    def _build_mouth_color_adjust(self) -> MouthColorAdjust:
        return clamp_mouth_color_adjust(
            MouthColorAdjust(
                brightness=float(self.mouth_brightness_var.get()),
                saturation=float(self.mouth_saturation_var.get()),
                warmth=float(self.mouth_warmth_var.get()),
                color_strength=float(self.mouth_color_strength_var.get()),
                edge_priority=float(self.mouth_edge_priority_var.get()),
                edge_width_ratio=float(self.mouth_edge_width_ratio_var.get()),
                inspect_boost=float(self.mouth_inspect_boost_var.get()),
            ),
        )

    def _set_mouth_color_adjust(self, cfg: MouthColorAdjust) -> None:
        self.mouth_brightness_var.set(float(cfg.brightness))
        self.mouth_saturation_var.set(float(cfg.saturation))
        self.mouth_warmth_var.set(float(cfg.warmth))
        self.mouth_color_strength_var.set(float(cfg.color_strength))
        self.mouth_edge_priority_var.set(float(cfg.edge_priority))
        self.mouth_edge_width_ratio_var.set(float(cfg.edge_width_ratio))
        self.mouth_inspect_boost_var.set(float(cfg.inspect_boost))

    def _reset_mouth_color_adjust(self) -> None:
        self._cancel_live_color_control_job()
        self._suspend_mouth_color_adjust_events = True
        try:
            self._set_mouth_color_adjust(MouthColorAdjust())
        finally:
            self._suspend_mouth_color_adjust_events = False
        self._save_session({
            "mouth_brightness": 0.0,
            "mouth_saturation": 1.0,
            "mouth_warmth": 0.0,
            "mouth_color_strength": 0.75,
            "mouth_edge_priority": 0.85,
            "mouth_edge_width_ratio": 0.10,
            "mouth_inspect_boost": 1.0,
        })
        self._schedule_write_live_color_control()

    def _on_mouth_color_adjust_changed(self, *_args) -> None:
        if self._suspend_mouth_color_adjust_events:
            return
        self._schedule_write_live_color_control()

    def _build_live_color_control_payload(
        self,
        cfg: MouthColorAdjust | None = None,
        *,
        updated_at: float | None = None,
    ) -> dict[str, float | str]:
        if cfg is None:
            cfg = self._build_mouth_color_adjust()
        payload: dict[str, float | str] = {
            "updated_at": float(time.time() if updated_at is None else updated_at),
            "mouth_brightness": float(cfg.brightness),
            "mouth_saturation": float(cfg.saturation),
            "mouth_warmth": float(cfg.warmth),
            "mouth_color_strength": float(cfg.color_strength),
            "mouth_edge_priority": float(cfg.edge_priority),
            "mouth_edge_width_ratio": float(cfg.edge_width_ratio),
            "mouth_inspect_boost": float(cfg.inspect_boost),
        }
        if self._live_ipc_token:
            payload["session_token"] = self._live_ipc_token
        return payload

    def _set_live_ipc_session(self, session: LiveIpcSession | None) -> None:
        self._live_ipc_session = session
        if session is None:
            self._live_ipc_token = ""
            self._live_color_control_path = ""
            self._auto_color_request_path = ""
            self._auto_color_result_path = ""
            return
        self._live_ipc_token = session.session_token
        self._live_color_control_path = session.live_color_control_path
        self._auto_color_request_path = session.auto_color_request_path
        self._auto_color_result_path = session.auto_color_result_path

    def _cleanup_live_ipc_session(self) -> None:
        session = self._live_ipc_session
        self._set_live_ipc_session(None)
        cleanup_live_ipc_session(session)

    def _write_json_atomic(self, path: str, payload: dict) -> None:
        out_path = os.path.abspath(path)
        directory = os.path.dirname(out_path) or HERE
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".mouth_color_live_",
            suffix=".json",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, out_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass

    def _load_json_file(self, path: str) -> dict | None:
        if not path or (not os.path.isfile(path)):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    def _write_live_color_control_payload(self, payload: dict[str, float | str]) -> None:
        self._write_json_atomic(self._live_color_control_path, payload)

    def _write_live_color_control(self) -> None:
        if not self._live_color_control_active:
            return
        self._write_live_color_control_payload(self._build_live_color_control_payload())

    def _flush_live_color_control(self) -> None:
        self._live_color_control_job = None
        self._write_live_color_control()

    def _schedule_write_live_color_control(self, *_args) -> None:
        if not self._live_color_control_active:
            return
        if self._live_color_control_job:
            try:
                self.after_cancel(self._live_color_control_job)
            except Exception:
                pass
            self._live_color_control_job = None
        self._live_color_control_job = self.after(80, self._flush_live_color_control)

    def _cancel_live_color_control_job(self) -> None:
        if self._live_color_control_job:
            try:
                self.after_cancel(self._live_color_control_job)
            except Exception:
                pass
            self._live_color_control_job = None

    def _set_auto_color_button_enabled(self, enabled: bool, *, text: str | None = None) -> None:
        def _apply() -> None:
            try:
                self.btn_auto_color.configure(state=("normal" if enabled else "disabled"))
                if text is not None:
                    self.btn_auto_color.configure(text=text)
            except Exception:
                pass
        self.after(0, _apply)

    def _clear_live_color_control(self) -> None:
        try:
            if os.path.isfile(self._live_color_control_path):
                os.unlink(self._live_color_control_path)
        except Exception as e:
            self.log(f"[warn] ライブ調整ファイルの削除に失敗しました: {e}")

    def _clear_auto_color_files(self) -> None:
        for path in (self._auto_color_request_path, self._auto_color_result_path):
            try:
                if os.path.isfile(path):
                    os.unlink(path)
            except Exception as e:
                self.log(f"[warn] 自動補正ファイルの削除に失敗しました: {e}")

    def _cancel_auto_color_poll(self) -> None:
        if self._auto_color_poll_job:
            try:
                self.after_cancel(self._auto_color_poll_job)
            except Exception:
                pass
            self._auto_color_poll_job = None

    def _finish_auto_color_request(self, *, success: bool) -> None:
        self._cancel_auto_color_poll()
        self._auto_color_request_pending = False
        self._auto_color_request_id = None
        self._auto_color_request_deadline = 0.0
        self._set_auto_color_button_enabled(
            self._live_color_control_active,
            text="色なじみ自動補正",
        )
        if success:
            self.log("[auto-color] 自動補正を反映しました")

    def _apply_auto_color_result(self, data: dict) -> None:
        current_cfg = self._build_mouth_color_adjust()
        new_cfg = clamp_mouth_color_adjust(
            MouthColorAdjust(
                brightness=float(data.get("mouth_brightness", current_cfg.brightness)),
                saturation=float(data.get("mouth_saturation", current_cfg.saturation)),
                warmth=float(data.get("mouth_warmth", current_cfg.warmth)),
                color_strength=float(data.get("mouth_color_strength", current_cfg.color_strength)),
                edge_priority=current_cfg.edge_priority,
                edge_width_ratio=current_cfg.edge_width_ratio,
                inspect_boost=current_cfg.inspect_boost,
            ),
        )
        self._cancel_live_color_control_job()
        self._suspend_mouth_color_adjust_events = True
        try:
            self._set_mouth_color_adjust(new_cfg)
        finally:
            self._suspend_mouth_color_adjust_events = False
        self._save_session({
            "mouth_brightness": float(new_cfg.brightness),
            "mouth_saturation": float(new_cfg.saturation),
            "mouth_warmth": float(new_cfg.warmth),
            "mouth_color_strength": float(new_cfg.color_strength),
        })
        if self._live_color_control_active:
            apply_updated_at = float(data.get("apply_updated_at", time.time()) or time.time())
            payload = self._build_live_color_control_payload(new_cfg, updated_at=apply_updated_at)
            self._write_live_color_control_payload(payload)
        bg_n = int(data.get("bg_sample_count", 0) or 0)
        mouth_n = int(data.get("mouth_sample_count", 0) or 0)
        self.log(
            "[auto-color] "
            f"bri={new_cfg.brightness:.0f} sat={new_cfg.saturation:.2f} "
            f"warm={new_cfg.warmth:.0f} strength={new_cfg.color_strength:.2f} "
            f"(bg={bg_n} mouth={mouth_n})"
        )

    def _poll_auto_color_result(self) -> None:
        self._auto_color_poll_job = None
        if (not self._auto_color_request_pending) or (not self._auto_color_request_id):
            return
        if time.time() >= self._auto_color_request_deadline:
            self.log("[auto-color warn] 自動補正がタイムアウトしました")
            self._finish_auto_color_request(success=False)
            return
        try:
            data = self._load_json_file(self._auto_color_result_path)
        except Exception as e:
            self.log(f"[auto-color warn] result 読み込み失敗: {e}")
            data = None
        if (
            not data
            or str(data.get("request_id", "")) != self._auto_color_request_id
            or str(data.get("session_token", "") or "") != self._live_ipc_token
        ):
            self._auto_color_poll_job = self.after(120, self._poll_auto_color_result)
            return
        if data.get("error"):
            self.log(f"[auto-color warn] 自動補正に失敗しました: {data.get('error')}")
            self._clear_auto_color_files()
            self._finish_auto_color_request(success=False)
            return
        self._apply_auto_color_result(data)
        self._clear_auto_color_files()
        self._finish_auto_color_request(success=True)

    def on_auto_mouth_color_adjust(self) -> None:
        if not self._live_color_control_active:
            messagebox.showinfo("自動補正", "ライブ実行中のみ使用できます。")
            return
        if self._auto_color_request_pending:
            self.log("[auto-color] 既に自動補正を実行中です")
            return
        request_id = f"{time.time():.6f}"
        payload = {
            "request_id": request_id,
            "requested_at": float(time.time()),
            "session_token": self._live_ipc_token,
        }
        try:
            self._clear_auto_color_files()
            self._write_json_atomic(self._auto_color_request_path, payload)
        except Exception as e:
            self.log(f"[auto-color warn] request 書き込み失敗: {e}")
            return
        self._auto_color_request_pending = True
        self._auto_color_request_id = request_id
        self._auto_color_request_deadline = time.time() + 5.0
        self._set_auto_color_button_enabled(False, text="自動補正待ち…")
        self.log("[auto-color] 自動補正を要求しました")
        self._auto_color_poll_job = self.after(120, self._poll_auto_color_result)

    # ----- logging (thread-safe) -----
    def log(self, s: str) -> None:
        self.log_q.put(s)

    def _save_session(self, payload: dict) -> bool:
        ok = save_session(payload)
        if not ok:
            self.log(
                "[warn] セッション保存に失敗しました。"
                "次回起動時に設定が復元されない可能性があります。"
            )
        return ok

    def _poll_logs(self) -> None:
        try:
            while True:
                s = self.log_q.get_nowait()
                # Remove null bytes from log text
                s = s.replace("\x00", "")
                self.txt.configure(state="normal")
                self.txt.insert("end", s + "\n")
                # 上限チェック
                line_count = int(self.txt.index("end-1c").split(".")[0])
                if line_count > MAX_LOG_LINES:
                    excess = line_count - MAX_LOG_LINES
                    self.txt.delete("1.0", f"{excess + 1}.0")
                self.txt.see("end")
                self.txt.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(100, self._poll_logs)

    def _clear_log(self) -> None:
        """ログをクリア（キューも空にする）"""
        # キューをドレイン
        try:
            while True:
                self.log_q.get_nowait()
        except queue.Empty:
            pass
        # Textをクリア
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.configure(state="disabled")

    # ----- misc helpers -----
    def on_open_sprite_extractor(self) -> None:
        script_path = os.path.join(HERE, "mouth_sprite_extractor_gui.py")
        if not os.path.isfile(script_path):
            self._show_error("エラー", f"口素材抽出ツールが見つかりません:\n{script_path}")
            return

        python_exe = sys.executable or "python"
        cmd = [python_exe, script_path]
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        popen_kwargs: dict[str, object] = {
            "cwd": HERE,
            "env": env,
        }
        if sys.platform.startswith("win"):
            flags = 0
            flags |= int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
            popen_kwargs["creationflags"] = flags
        try:
            subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603
            self.log("[gui] 口PNG素材作成ツールを起動しました")
            self.log("[gui] 素材作成後は、この画面に戻って『mouthフォルダ（口PNG素材）』を選択してください")
        except Exception as e:
            self._show_error("エラー", f"口素材抽出ツールを起動できませんでした。\n{e}")

    def _guess_mouth_dir(self) -> str:
        v = self.video_var.get().strip()
        if not v:
            return ""
        return guess_mouth_dir(v)

    def _set_mouth_dir(self, mouth_dir: str) -> None:
        self.mouth_dir_var.set(mouth_dir)

    def _post_set_mouth_dir(self, mouth_dir: str) -> None:
        self.after(0, lambda md=mouth_dir: self._set_mouth_dir(md))

    def _autofill_mouth_dir(self) -> str:
        md = self._guess_mouth_dir()
        if md:
            self._set_mouth_dir(md)
        return md

    def _refresh_audio_devices(self, init: bool = False) -> None:
        devices = list_input_devices()
        if not devices:
            if init:
                cur_spec = self.audio_device_spec_var.get().strip()
                self.audio_device_menu_var.set(cur_spec or f"{self.audio_device_var.get()}: (未取得)")
            return

        self._audio_items = devices  # optional stash
        values = [str(item.get("display", "")) for item in devices]
        self.cmb_audio["values"] = values

        cur = int(self.audio_device_var.get())
        cur_spec = self.audio_device_spec_var.get().strip()
        sel_item = find_input_device_item(devices, cur_spec) if cur_spec else None
        if sel_item is None:
            sel_item = find_input_device_item(devices, f"sd:{cur}")
        if sel_item is None:
            sel_item = devices[0]

        idx0 = sel_item.get("index")
        if idx0 is not None:
            self.audio_device_var.set(int(idx0))
        self.audio_device_spec_var.set(str(sel_item.get("spec", "")).strip())
        self.audio_device_menu_var.set(str(sel_item.get("display", "")))

        def _on_select(_evt=None):
            s = self.audio_device_menu_var.get()
            item = find_input_device_item(self._audio_items, s)
            if item is None:
                return
            spec = str(item.get("spec", "")).strip()
            self.audio_device_spec_var.set(spec)
            payload = {"audio_device_spec": spec}
            idx = item.get("index")
            if idx is not None:
                self.audio_device_var.set(int(idx))
                payload["audio_device"] = int(idx)
            self._save_session(payload)

        self.cmb_audio.bind("<<ComboboxSelected>>", _on_select)
        if init:
            _on_select()


    def _schedule_refresh_characters(self) -> None:
        """Debounced refresh for character list."""
        try:
            if getattr(self, "_char_refresh_job", None):
                self.after_cancel(self._char_refresh_job)  # type: ignore[arg-type]
        except Exception:
            pass
        self._char_refresh_job = self.after(150, self._refresh_characters)

    def _refresh_characters(self, init: bool = False) -> None:
        """Populate character combobox from mouth_dir root."""
        mouth_root = self.mouth_dir_var.get().strip()

        # If mouth_root is already emotion-level (no character layer), character selection is not needed.
        if is_emotion_level_mouth_root(mouth_root):
            try:
                self.cmb_character.configure(state="disabled")
                self.cmb_character["values"] = ["(不要：直下が感情フォルダ)"]
            except Exception:
                pass
            self.character_var.set("")
            return

        chars = list_character_dirs(mouth_root)
        if not chars:
            # Keep enabled but show placeholder.
            try:
                self.cmb_character.configure(state="readonly")
                self.cmb_character["values"] = ["(なし)"]
            except Exception:
                pass
            self.character_var.set("")
            return

        try:
            self.cmb_character.configure(state="readonly")
            self.cmb_character["values"] = chars
        except Exception:
            pass

        cur = (self.character_var.get() or "").strip()
        if cur not in chars:
            # Auto-select when there is only one character
            if len(chars) == 1:
                self._set_character(chars[0], persist=True)
            elif init:
                self.character_var.set("")

    def _emotion_preset_key(self) -> str:
        return EMOTION_PRESETS.get(self.emotion_preset_var.get(), "standard")

    def _resolve_character_for_action(self) -> str | None:
        return App._resolve_character_for_action_value(
            self,
            self.mouth_dir_var.get().strip(),
            (self.character_var.get() or "").strip(),
        )

    def _resolve_character_for_action_value(
        self,
        mouth_root: str,
        selected_character: str,
    ) -> str | None:
        """Return effective character name for current mouth_root.
        - ""   : no character layer (mouth_root is emotion-level)
        - name  : selected / auto-selected character
        - None  : error (multiple candidates but not selected)
        """
        if is_emotion_level_mouth_root(mouth_root):
            return ""

        chars = list_character_dirs(mouth_root)
        if not chars:
            return ""

        cur = (selected_character or "").strip()
        if cur in chars:
            return cur

        if len(chars) == 1:
            self._post_set_character(chars[0], persist=True)
            return chars[0]

        self._show_error("エラー", "キャラクターを選択してください（mouth_dir直下のフォルダから選びます）。")
        return None

    def _set_character(self, character: str, *, persist: bool = False) -> None:
        self.character_var.set(character)
        if persist:
            self._save_session({"character": character})

    def _post_set_character(self, character: str, *, persist: bool = False) -> None:
        self.after(0, lambda c=character, p=persist: self._set_character(c, persist=p))

    def _runtime_supports(self, runtime_py: str, flags: list[str]) -> bool:
        return script_contains(runtime_py, flags)

    def _warn_soft_stop(self) -> None:
        if self.stop_mode != "soft":
            return
        self.log("[gui] 停止予約中: 終了待機中。必要ならもう一度で強制停止してください。")

    def _set_stop_mode(self, mode: str) -> None:
        def _apply():
            if self._soft_warn_job:
                try:
                    self.after_cancel(self._soft_warn_job)
                except Exception:
                    pass
                self._soft_warn_job = None

            self.stop_mode = mode
            if mode == "soft":
                self.stop_flag.set()
                self.soft_requested_at = time.monotonic()
                self.btn_stop.configure(text=STOP_BTN_TEXT_SOFT)
                self._soft_warn_job = self.after(
                    int(SOFT_STOP_GRACE_SEC * 1000),
                    self._warn_soft_stop,
                )
            elif mode == "force":
                self.stop_flag.set()
                self.soft_requested_at = None
                self.btn_stop.configure(text=STOP_BTN_TEXT_SOFT)
            else:
                self.stop_flag.clear()
                self.soft_requested_at = None
                self.btn_stop.configure(text=STOP_BTN_TEXT_DEFAULT)

        self.after(0, _apply)

    def _set_running(self, running: bool) -> None:
        def _apply():
            st = "disabled" if running else "normal"
            self.btn_track_calib.configure(state=st)
            self.btn_calib_only.configure(state=st)
            self.btn_erase.configure(state=st)
            self.btn_erase_range.configure(state=st)
            self.btn_live.configure(state=st)
            self.btn_auto_color.configure(state=("normal" if self._live_color_control_active else "disabled"))
            self.btn_stop.configure(state=("normal" if running else "disabled"))
            if not running:
                self._set_stop_mode("none")
                self._progress_reset()
        self.after(0, _apply)

    def _progress_reset(self) -> None:
        def _apply():
            self.progress.configure(mode="determinate", maximum=1.0)
            self.progress_var.set(0.0)
            self.progress_text_var.set("待機中")
        self.after(0, _apply)

    def _progress_begin(self, total_steps: int, text: str) -> None:
        def _apply():
            self._progress_total = max(1, int(total_steps))
            self.progress.configure(mode="determinate", maximum=self._progress_total)
            self.progress_var.set(0.0)
            self.progress_text_var.set(text)
        self.after(0, _apply)

    def _progress_step(self, step: int, text: str) -> None:
        def _apply():
            self.progress.configure(mode="determinate", maximum=self._progress_total)
            val = min(max(0, int(step)), int(self._progress_total))
            self.progress_var.set(val)
            self.progress_text_var.set(text)
        self.after(0, _apply)

    def _show_error(self, title: str, msg: str) -> None:
        self.after(0, lambda: messagebox.showerror(title, msg))

    def _show_warn(self, title: str, msg: str) -> None:
        self.after(0, lambda: messagebox.showwarning(title, msg))

    def _apply_preview_selection(self, pad: float, coverage: float) -> None:
        pad_v = round(float(pad), 2)
        cov_v = round(float(coverage), 2)
        self.pad_var.set(pad_v)
        self.coverage_var.set(cov_v)
        self._save_session({"pad": pad_v, "coverage": cov_v})
        self.log(f"[gui] 見た目確認の設定を反映しました: pad={pad_v:.2f} / 口消し範囲={cov_v:.2f}")

    def _resolve_workflow_paths(
        self,
        *,
        require_track: bool = False,
        require_calibrated: bool = False,
        prefer_calibrated: bool = False,
        source_video: str | None = None,
    ) -> WorkflowPaths | None:
        paths, err = build_workflow_paths(
            source_video if source_video is not None else self.video_var.get().strip(),
            require_track=require_track,
            require_calibrated=require_calibrated,
            prefer_calibrated=prefer_calibrated,
        )
        if paths is None:
            self._show_error("エラー", err)
            return None
        return paths

    def _resolve_mouth_root_value(self, mouth_root: str) -> str | None:
        path, err = validate_existing_dir(
            mouth_root,
            empty_message="mouthフォルダを選択してください。",
            missing_label="mouthフォルダ",
        )
        if path is None:
            self._show_error("エラー", err)
            return None
        return path

    def _resolve_mouth_root(self, *, auto_fill: bool = False) -> str | None:
        mouth_root = self.mouth_dir_var.get().strip()
        if auto_fill and not mouth_root:
            mouth_root = self._guess_mouth_dir()
            if mouth_root:
                self._post_set_mouth_dir(mouth_root)
        return App._resolve_mouth_root_value(self, mouth_root)

    def _resolve_loop_video(self, loop_video: str) -> str | None:
        path, err = validate_existing_file(
            loop_video,
            empty_message="背景動画が未設定です。先に『② 口消し動画生成』を実行してください。",
            missing_label="背景動画",
        )
        if path is None:
            if not loop_video:
                err = "背景動画が見つかりません（先に口消し動画生成を推奨）"
            self._show_error("エラー", err)
            return None
        return path

    def _resolve_open_sprite(self, mouth_root: str, char: str) -> str | None:
        open_sprite = best_open_sprite_for_character(mouth_root, char)
        if open_sprite:
            return open_sprite

        searched_base = resolve_character_dir(mouth_root, char)
        self._show_error(
            "エラー",
            format_missing_path_message(
                "キャリブ用 open.png",
                os.path.join(searched_base, "open.png"),
                "mouthフォルダ直下、または Default / neutral などの感情フォルダ内に open.png を置いてください。",
            ),
        )
        return None

    # ----- file pickers -----
    def on_pick_video(self) -> None:
        if sys.platform == "darwin":  # Mac
            p = filedialog.askopenfilename(title="動画を選択")
        else:  # Windows/Linux
            p = filedialog.askopenfilename(
                title="動画を選択",
                filetypes=[("Video", "*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v"), ("All", "*.*")],
            )
        if not p:
            return
        self.video_var.set(p)
        self._autofill_mouth_dir()
        # 選択直後は video=source_video として保存（まだmouthless未生成のため）
        self._save_session({
            "video": self.video_var.get(),
            "source_video": self.video_var.get(),
            "mouth_dir": self.mouth_dir_var.get(),
            "coverage": float(self.coverage_var.get()),
            "pad": float(self.pad_var.get()),
            "audio_device": int(self.audio_device_var.get()),
            "audio_device_spec": self.audio_device_spec_var.get(),
            "character": self.character_var.get(),
            "emotion_preset": self.emotion_preset_var.get(),
            "emotion_hud": bool(self.emotion_hud_var.get()),
        })

    def on_pick_mouth_dir(self) -> None:
        d = filedialog.askdirectory(title="mouthフォルダを選択")
        if not d:
            return
        self.mouth_dir_var.set(d)
        self._refresh_characters(init=True)
        self._save_session({
            "video": self.video_var.get(),
            "source_video": self.video_var.get(),
            "mouth_dir": self.mouth_dir_var.get(),
            "coverage": float(self.coverage_var.get()),
            "pad": float(self.pad_var.get()),
            "audio_device": int(self.audio_device_var.get()),
            "audio_device_spec": self.audio_device_spec_var.get(),
            "character": self.character_var.get(),
            "emotion_preset": self.emotion_preset_var.get(),
            "emotion_hud": bool(self.emotion_hud_var.get()),
        })

    def on_stop(self) -> None:
        if self.stop_mode == "none":
            self.log("[gui] stop requested. will stop after current step.")
            self._set_stop_mode("soft")
            self.runner.request_soft_stop()
            return
        if self.stop_mode == "soft":
            self.log("[gui] force stop requested. terminating active process.")
            self._set_stop_mode("force")
            self.runner.force_stop()

    # ----- subprocess runner (delegates to CommandRunner) -----
    def _run_cmd_stream(
        self,
        cmd: list[str],
        cwd: str | None = None,
        *,
        allow_soft_interrupt: bool = False,
    ) -> int:
        """Run command via CommandRunner and return exit code."""
        return self._run_cmd_stream_result(
            cmd, cwd=cwd, allow_soft_interrupt=allow_soft_interrupt,
        ).returncode

    def _run_cmd_stream_result(
        self,
        cmd: list[str],
        cwd: str | None = None,
        *,
        allow_soft_interrupt: bool = False,
    ) -> RunResult:
        """Run command via CommandRunner and return the full result."""
        return self.runner.run_stream(
            cmd, cwd=cwd, allow_soft_stop=allow_soft_interrupt,
        )

    def _execute_plan(self, plan: ActionPlan) -> int:
        """Execute an ActionPlan on the worker thread.

        Handles session saves, progress updates, logging, error checks,
        and post-actions.  Returns the return code of the last executed step.
        """
        if plan.session_init:
            self._save_session(plan.session_init)

        self._progress_begin(plan.total_steps, f"{plan.name}準備中…")

        last_rc = 0
        skipped = False
        for i, step in enumerate(plan.steps, 1):
            if step.skip_on_stop and self.runner.soft_requested:
                prog = step.progress_label or step.label
                self.log(f"[info] 停止予約のため、{prog}以降をスキップします。")
                skipped = True
                break

            if plan.total_steps > 1:
                self.log(f"\n=== [{i}/{plan.total_steps}] {step.label} ===")
            else:
                self.log(f"\n=== {step.label} ===")

            if step.pre_log:
                self.log(step.pre_log)

            prog = step.progress_label or step.label
            self._progress_step(i, f"{prog}中… ({i}/{plan.total_steps})")

            result = self.runner.run_stream(
                step.cmd, cwd=step.cwd, allow_soft_stop=step.allow_soft_stop,
            )
            last_rc = result.returncode

            # ``was_stopped`` is reserved for cases where this step's child
            # process was actually interrupted.  A late stop request after the
            # child already exited leaves ``was_stopped`` false and is handled
            # by the next-step / post-action ``soft_requested`` checks below.
            if result.was_stopped:
                skipped = True
                self.log(
                    "[info] 強制停止しました。"
                    if self.stop_mode == "force"
                    else "[info] 停止しました。"
                )
                self._progress_step(i, f"{prog}停止")
                break

            if last_rc != 0 or any(
                not os.path.isfile(p) for p in step.expected_outputs
            ):
                if step.error_msg:
                    self._show_error("失敗", f"{step.error_msg} (rc={last_rc})")
                return last_rc

            self._progress_step(i, f"{prog}完了 ({i}/{plan.total_steps})")

        if not skipped:
            if plan.session_final:
                self._save_session(plan.session_final)

            if self.runner.soft_requested and plan.post_actions:
                self.log("[info] 停止予約のため、ブラウザ用出力とプレビューをスキップします。")
            else:
                for action in plan.post_actions:
                    self._run_post_action(action)

            if plan.completion_msg:
                self.log(f"\n{plan.completion_msg}")

        return last_rc

    def _finalize_live_run_result(self, result: RunResult) -> None:
        rc = int(result.returncode)
        self.log(f"\n[live] finished rc={rc}")
        # ``was_stopped`` is true only if the live child was actually
        # interrupted.  A late stop request after child exit must still report
        # the real exit status below.
        if result.was_stopped:
            self.log(
                "[info] ライブを強制停止しました。"
                if self.stop_mode == "force"
                else "[info] ライブを停止しました。"
            )
            self._progress_step(1, "ライブ停止")
            return
        if rc != 0:
            self._progress_step(1, "ライブ異常終了")
            self._show_error(
                "失敗",
                f"ライブ実行が異常終了しました (rc={rc})\n"
                "ログを確認してください。",
            )
            return
        self._progress_step(1, "ライブ終了")

    def _run_post_action(self, action: PostAction) -> None:
        """Execute a post-plan action."""
        if action.tag == "export_browser" and len(action.args) >= 2:
            self.log("\n=== ブラウザ用データ出力 ===")
            self._export_browser_assets(action.args[0], action.args[1])
        elif action.tag == "open_preview" and len(action.args) >= 1:
            self.log("\nプレビューを起動します…")
            self._open_video_preview(action.args[0])

    # ----- preview -----
    def _open_video_preview(self, video_path: str) -> None:
        if prefer_native_video_preview():
            try:
                open_path_with_default_app(video_path)
                return
            except Exception as e:
                self.log(f"[warn] cannot open preview automatically: {e}")
                self.log(f"[info] output: {video_path}")
                return

        # Try OpenCV playback first (if available)
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                win = "preview (q/ESC=close, space=pause)"
                paused = False
                while True:
                    if not paused:
                        ok, frame = cap.read()
                        if not ok:
                            break
                    cv2.imshow(win, frame)
                    k = cv2.waitKey(15) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord(" "):
                        paused = not paused
                cap.release()
                cv2.destroyWindow(win)
                return
        except Exception:
            pass

        # Fallback to OS open
        try:
            open_path_with_default_app(video_path)
        except Exception as e:
            self.log(f"[warn] cannot open preview automatically: {e}")
            self.log(f"[info] output: {video_path}")

    def _export_browser_assets(self, mouthless_mp4: str, calib_npz: str) -> None:
        if not os.path.isfile(mouthless_mp4):
            self.log("[warn] ブラウザ用出力: 口消し動画が見つかりません。")
            return
        if not os.path.isfile(calib_npz):
            self.log("[warn] ブラウザ用出力: mouth_track_calibrated.npz がありません。")
            return

        fps = None
        try:
            import numpy as np  # type: ignore
            with np.load(calib_npz, allow_pickle=False) as npz:
                if "fps" in npz:
                    fps = float(npz["fps"])
        except Exception as e:
            self.log(f"[warn] ブラウザ用出力: fps取得に失敗しました: {e}")

        if not fps or fps <= 0:
            self.log("[warn] ブラウザ用出力: fpsが不明のためCFR変換をスキップします。")
            fps = None

        out_dir = os.path.dirname(os.path.abspath(mouthless_mp4))

        try:
            from convert_npz_to_json import convert_npz_to_json  # type: ignore
            convert_npz_to_json(Path(calib_npz), Path(out_dir))
            self.log(f"[info] ブラウザ用JSON出力: {os.path.join(out_dir, 'mouth_track.json')}")
        except Exception as e:
            self.log(f"[warn] ブラウザ用JSON出力に失敗しました: {e}")

        if not fps:
            return

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.log("[warn] ブラウザ用出力: ffmpegが見つからないためH.264変換をスキップします。")
            return

        h264_mp4 = os.path.splitext(mouthless_mp4)[0] + "_h264.mp4"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            mouthless_mp4,
            "-vf",
            f"fps={fps}",
            "-r",
            f"{fps}",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            h264_mp4,
        ]
        rc = self._run_cmd_stream(cmd, cwd=HERE)
        if rc != 0 or (not os.path.isfile(h264_mp4)):
            self.log(f"[warn] ブラウザ用H.264変換に失敗しました (rc={rc})")
        else:
            self.log(f"[info] ブラウザ用H.264出力: {h264_mp4}")

    # ----- workflow buttons -----
    def _start_worker(self, target) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_flag.clear()
        self.runner.reset()
        self._set_stop_mode("none")
        self._set_running(True)
        def runner():
            try:
                target()
            finally:
                # ワーカーが何で終わっても UI を戻す
                self._set_running(False)
        self.worker_thread = threading.Thread(target=runner, daemon=True)
        self.worker_thread.start()

    def on_track_and_calib(self) -> None:
        def _worker():
            try:
                base_dir = HERE
                ok, msg = ensure_backend_sanity(base_dir)
                if not ok:
                    self._show_error("エラー", msg)
                    return
                paths = self._resolve_workflow_paths()
                if paths is None:
                    return
                mouth_dir = self._resolve_mouth_root(auto_fill=True)
                if mouth_dir is None:
                    return
                char = self._resolve_character_for_action()
                if char is None:
                    return
                open_sprite = self._resolve_open_sprite(mouth_dir, char)
                if open_sprite is None:
                    return
                color_cfg = self._build_mouth_color_adjust()

                plan = plan_track_and_calib(
                    base_dir=base_dir,
                    video=paths.source_video,
                    track_npz=paths.track_npz,
                    calib_npz=paths.calib_npz,
                    open_sprite=open_sprite,
                    mouth_dir=mouth_dir,
                    pad=float(self.pad_var.get()),
                    coverage=float(self.coverage_var.get()),
                    audio_device=int(self.audio_device_var.get()),
                    audio_device_spec=self.audio_device_spec_var.get(),
                    smoothing_cutoff=SMOOTHING_PRESETS.get(
                        self.smoothing_menu_var.get(),
                    ),
                    smoothing_label=self.smoothing_menu_var.get(),
                    mouth_brightness=color_cfg.brightness,
                    mouth_saturation=color_cfg.saturation,
                    mouth_warmth=color_cfg.warmth,
                    mouth_color_strength=color_cfg.color_strength,
                    mouth_edge_priority=color_cfg.edge_priority,
                    mouth_edge_width_ratio=color_cfg.edge_width_ratio,
                    mouth_inspect_boost=color_cfg.inspect_boost,
                )
                self._execute_plan(plan)
            except Exception as e:
                self._show_error("エラー", str(e))
        self._start_worker(_worker)

    def on_calib_only(self) -> None:
        def _worker():
            try:
                base_dir = HERE
                paths = self._resolve_workflow_paths(require_track=True)
                if paths is None:
                    return
                mouth_dir = self._resolve_mouth_root(auto_fill=False)
                if mouth_dir is None:
                    return
                char = self._resolve_character_for_action()
                if char is None:
                    return
                open_sprite = self._resolve_open_sprite(mouth_dir, char)
                if open_sprite is None:
                    return
                color_cfg = self._build_mouth_color_adjust()

                plan = plan_calib_only(
                    base_dir=base_dir,
                    video=paths.source_video,
                    track_npz=paths.track_npz,
                    calib_npz=paths.calib_npz,
                    open_sprite=open_sprite,
                    mouth_brightness=color_cfg.brightness,
                    mouth_saturation=color_cfg.saturation,
                    mouth_warmth=color_cfg.warmth,
                    mouth_color_strength=color_cfg.color_strength,
                    mouth_edge_priority=color_cfg.edge_priority,
                    mouth_edge_width_ratio=color_cfg.edge_width_ratio,
                    mouth_inspect_boost=color_cfg.inspect_boost,
                )
                self._execute_plan(plan)
            except Exception as e:
                self._show_error("エラー", str(e))
        self._start_worker(_worker)

    def on_erase_mouthless(self) -> None:
        def _worker():
            try:
                base_dir = HERE
                paths = self._resolve_workflow_paths(
                    require_track=True,
                    require_calibrated=True,
                    prefer_calibrated=True,
                )
                if paths is None:
                    return
                video = paths.source_video
                mouth_dir = self._resolve_mouth_root(auto_fill=False)
                if mouth_dir is None:
                    return

                erase_track = paths.preferred_track
                if erase_track is None:
                    self._show_error(
                        "エラー",
                        format_missing_path_message(
                            "口消しに使う track",
                            paths.calib_npz,
                            "『キャリブのみ（やり直し）』または『① 解析→キャリブ』を実行してください。",
                        ),
                    )
                    return

                name = os.path.splitext(os.path.basename(video))[0]
                mouthless_mp4 = os.path.join(paths.out_dir, f"{name}_mouthless.mp4")

                plan = plan_erase(
                    base_dir=base_dir,
                    video=video,
                    mouth_dir=mouth_dir,
                    track_npz=paths.track_npz,
                    calib_npz=paths.calib_npz,
                    erase_track=erase_track,
                    mouthless_mp4=mouthless_mp4,
                    coverage=float(self.coverage_var.get()),
                    pad=float(self.pad_var.get()),
                    audio_device=int(self.audio_device_var.get()),
                    erase_shading=bool(self.erase_shading_var.get()),
                    audio_device_spec=self.audio_device_spec_var.get(),
                )
                self._execute_plan(plan)
            except Exception as e:
                self._show_error("エラー", str(e))
        self._start_worker(_worker)

    def on_preview_erase_range(self) -> None:
        """フル書き出し前に、pad / 口消し範囲を軽量確認するプレビュー。"""
        def _worker():
            try:
                try:
                    import cv2  # type: ignore  # noqa: F401
                    import numpy  # type: ignore  # noqa: F401
                except Exception:
                    self._show_error("エラー", "OpenCV(cv2) と numpy が必要です。")
                    return

                paths = self._resolve_workflow_paths(
                    require_track=False,
                    require_calibrated=False,
                    prefer_calibrated=True,
                )
                if paths is None:
                    return
                track_path = paths.preferred_track
                if track_path is None:
                    self._show_error(
                        "エラー",
                        format_missing_path_message(
                            "mouth_track.npz / mouth_track_calibrated.npz",
                            paths.track_npz,
                            "先に『① 解析→キャリブ』を実行してください。",
                        ),
                    )
                    return

                open_sprite: str | None = None
                mouth_root = self.mouth_dir_var.get().strip()
                cur_char = (self.character_var.get() or "").strip()
                try:
                    if mouth_root and os.path.isdir(mouth_root):
                        if is_emotion_level_mouth_root(mouth_root):
                            open_sprite = best_open_sprite_for_character(mouth_root, "")
                        elif cur_char:
                            open_sprite = best_open_sprite_for_character(mouth_root, cur_char)
                except Exception as e:
                    self.log(f"[warn] 見た目確認用 open.png の解決に失敗しました: {e}")

                from .preview import run_erase_range_preview

                selection = run_erase_range_preview(
                    video=paths.source_video,
                    track_path=track_path,
                    track_npz=paths.track_npz,
                    calib_npz=paths.calib_npz,
                    coverage=float(self.coverage_var.get()),
                    preview_pad=float(self.pad_var.get()),
                    open_sprite=open_sprite,
                    color_adjust=self._build_mouth_color_adjust(),
                    stop_flag=self.stop_flag,
                    log_fn=self.log,
                    show_error=self._show_error,
                )
                if selection.applied:
                    self.after(
                        0,
                        lambda p=selection.pad, c=selection.coverage: self._apply_preview_selection(p, c),
                    )
            except Exception as e:
                self._show_error("エラー", str(e))

        # プレビューは外部プロセスではないが、UIが固まらないようワーカで回す
        self._start_worker(_worker)

    def on_live_run(self) -> None:
        live_video_value = self.video_var.get().strip()
        live_mouth_root_value = self.mouth_dir_var.get().strip()
        live_selected_character = (self.character_var.get() or "").strip()
        live_audio_device = int(self.audio_device_var.get())
        live_audio_device_spec = self.audio_device_spec_var.get().strip()
        live_emotion_preset_label = self.emotion_preset_var.get()
        live_emotion_preset_key = self._emotion_preset_key()
        live_emotion_hud = bool(self.emotion_hud_var.get())
        live_color_cfg = self._build_mouth_color_adjust()

        def _worker():
            try:
                base_dir = HERE
                sess = load_session()

                loop_video = self._resolve_loop_video(
                    str(sess.get("video") or live_video_value),
                )
                if loop_video is None:
                    return

                paths = self._resolve_workflow_paths(
                    require_track=True,
                    require_calibrated=False,
                    prefer_calibrated=False,
                )
                if paths is None:
                    return

                mouth_root = self._resolve_mouth_root_value(live_mouth_root_value)
                if mouth_root is None:
                    return
                char = self._resolve_character_for_action_value(
                    live_mouth_root_value,
                    live_selected_character,
                )
                if char is None:
                    return
                mouth_dir = resolve_character_dir(mouth_root, char)

                runtime_py = resolve_runtime_script(base_dir)

                if not script_contains(runtime_py, ["--emotion-auto"]):
                    self.log("[warn] runtime が感情オートに未対応のため、従来モードで実行します。")
                self._cleanup_live_ipc_session()
                self._set_live_ipc_session(create_live_ipc_session())
                initial_live_color_payload = self._build_live_color_control_payload(live_color_cfg)
                self._live_color_control_active = True
                self._clear_live_color_control()
                self._clear_auto_color_files()
                self._write_live_color_control_payload(initial_live_color_payload)
                self._set_auto_color_button_enabled(True)

                plan = plan_live(
                    base_dir=base_dir,
                    runtime_py=runtime_py,
                    loop_video=loop_video,
                    mouth_dir=mouth_dir,
                    track_npz=paths.track_npz,
                    calib_npz=paths.calib_npz,
                    device_idx=live_audio_device,
                    audio_device_spec=live_audio_device_spec,
                    character=char,
                    emotion_preset_label=live_emotion_preset_label,
                    emotion_preset_key=live_emotion_preset_key,
                    emotion_hud=live_emotion_hud,
                    mouth_brightness=live_color_cfg.brightness,
                    mouth_saturation=live_color_cfg.saturation,
                    mouth_warmth=live_color_cfg.warmth,
                    mouth_color_strength=live_color_cfg.color_strength,
                    mouth_edge_priority=live_color_cfg.edge_priority,
                    mouth_edge_width_ratio=live_color_cfg.edge_width_ratio,
                    mouth_inspect_boost=live_color_cfg.inspect_boost,
                    mouth_ipc_token=self._live_ipc_token,
                    live_color_control_path=self._live_color_control_path,
                    auto_color_request_path=self._auto_color_request_path,
                    auto_color_result_path=self._auto_color_result_path,
                )

                self._save_session(plan.session_init)
                step = plan.steps[0]
                self.log(f"\n=== {step.label} ===")
                self._progress_begin(1, "ライブ準備中…")
                self._progress_step(1, "ライブ実行中…")
                result = self._run_cmd_stream_result(
                    step.cmd, cwd=step.cwd,
                    allow_soft_interrupt=step.allow_soft_stop,
                )
                self._finalize_live_run_result(result)
            except Exception as e:
                self._show_error("エラー", str(e))
            finally:
                self._cancel_auto_color_poll()
                self._auto_color_request_pending = False
                self._auto_color_request_id = None
                self._auto_color_request_deadline = 0.0
                self._live_color_control_active = False
                self._clear_live_color_control()
                self._clear_auto_color_files()
                self._cleanup_live_ipc_session()
                self._set_auto_color_button_enabled(False, text="色なじみ自動補正")
        self._start_worker(_worker)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
