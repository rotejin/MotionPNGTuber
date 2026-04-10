"""UI builder for mouth_track_gui.

Phase 6 of the mouth_track_gui refactoring plan.
Extracts _build_ui widget creation from the App class.
Pure Tk/ttk widget construction — no business logic.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WidgetRefs:
    """References to widgets that App needs after construction."""

    cmb_character: ttk.Combobox
    cmb_audio: ttk.Combobox
    cmb_smooth: ttk.Combobox
    cmb_emotion_preset: ttk.Combobox
    pad_value_label: ttk.Label
    cov_label: ttk.Label
    btn_track_calib: ttk.Button
    btn_calib_only: ttk.Button
    btn_erase: ttk.Button
    btn_erase_range: ttk.Button
    btn_live: ttk.Button
    btn_auto_color: ttk.Button
    btn_stop: ttk.Button
    progress_bar: ttk.Progressbar
    txt: tk.Text


@dataclass
class UiVars:
    """Tk variables provided by App for widget binding."""

    video: tk.StringVar
    mouth_dir: tk.StringVar
    character: tk.StringVar
    pad: tk.DoubleVar
    coverage: tk.DoubleVar
    mouth_brightness: tk.DoubleVar
    mouth_saturation: tk.DoubleVar
    mouth_warmth: tk.DoubleVar
    mouth_color_strength: tk.DoubleVar
    mouth_edge_priority: tk.DoubleVar
    mouth_edge_width_ratio: tk.DoubleVar
    mouth_inspect_boost: tk.DoubleVar
    erase_shading: tk.BooleanVar
    smoothing_menu: tk.StringVar
    audio_device_menu: tk.StringVar
    emotion_preset: tk.StringVar
    emotion_hud: tk.BooleanVar
    progress: tk.DoubleVar
    progress_text: tk.StringVar


@dataclass
class UiCallbacks:
    """Callbacks injected from App for button/bind actions."""

    on_open_sprite_extractor: Callable
    on_pick_video: Callable
    on_pick_mouth_dir: Callable
    refresh_characters: Callable
    refresh_audio_devices: Callable
    on_track_and_calib: Callable
    on_calib_only: Callable
    on_erase_mouthless: Callable
    on_preview_erase_range: Callable
    on_live_run: Callable
    on_auto_mouth_color_adjust: Callable
    on_mouth_color_adjust_changed: Callable
    on_reset_mouth_color_adjust: Callable
    on_stop: Callable
    clear_log: Callable
    save_session: Callable[[dict], object]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_BTN_TEXT_DEFAULT = "中断（現在の処理が終わったら停止）"


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_ui(
    parent: tk.Tk,
    vars: UiVars,
    callbacks: UiCallbacks,
    *,
    smoothing_labels: list[str],
    emotion_preset_labels: list[str],
) -> WidgetRefs:
    """Construct the full UI and return widget references.

    All widget creation and layout is done here.
    Event bindings that trigger save_session or App methods are wired
    via the callbacks parameter.
    """
    pad = 10
    frm = ttk.Frame(parent)
    frm.pack(fill="both", expand=True, padx=pad, pady=pad)

    # --- Quick guide / recommended flow ---
    guide = ttk.LabelFrame(frm, text="おすすめの流れ", padding=8)
    guide.pack(fill="x", pady=(0, 8))
    ttk.Label(
        guide,
        text=(
            "1. 口PNG素材が未作成なら、まず下の『口PNG素材を作る』を開く\n"
            "2. 元動画を選ぶ\n"
            "3. ① 解析→キャリブ → 見た目確認（軽量） → ② 口消し動画生成 → ③ ライブ実行 の順で進める"
        ),
        justify="left",
    ).pack(anchor="w")

    prep = ttk.LabelFrame(frm, text="最初の準備", padding=8)
    prep.pack(fill="x", pady=(0, 8))
    ttk.Button(
        prep,
        text="口PNG素材を作る（おすすめ）",
        command=callbacks.on_open_sprite_extractor,
    ).pack(side="left")
    ttk.Label(
        prep,
        text="mouthフォルダがまだ無い場合は、先にこちらで口PNG素材を作成します。",
    ).pack(side="left", padx=10)

    # --- Video row ---
    row1 = ttk.Frame(frm)
    row1.pack(fill="x", pady=(0, 8))
    ttk.Label(row1, text="動画").pack(side="left")
    ttk.Entry(row1, textvariable=vars.video).pack(side="left", fill="x", expand=True, padx=8)
    ttk.Button(row1, text="選択…", command=callbacks.on_pick_video).pack(side="left")

    # --- Mouth dir row ---
    row2 = ttk.Frame(frm)
    row2.pack(fill="x", pady=(0, 4))
    ttk.Label(row2, text="mouthフォルダ（口PNG素材）").pack(side="left")
    ttk.Entry(row2, textvariable=vars.mouth_dir).pack(side="left", fill="x", expand=True, padx=8)
    ttk.Button(row2, text="選択…", command=callbacks.on_pick_mouth_dir).pack(side="left")
    ttk.Label(
        frm,
        text="口PNG素材を作成した後、このフォルダを選択してください。動画選択後は自動推定も試みます。",
        font=("", 9),
    ).pack(anchor="w", pady=(0, 8))

    # --- Character row ---
    row2a = ttk.Frame(frm)
    row2a.pack(fill="x", pady=(0, 8))
    ttk.Label(row2a, text="キャラクター").pack(side="left")
    cmb_character = ttk.Combobox(row2a, textvariable=vars.character, state="readonly")
    cmb_character.pack(side="left", fill="x", expand=True, padx=8)
    ttk.Button(row2a, text="再読込", command=callbacks.refresh_characters).pack(side="left")
    cmb_character.bind(
        "<<ComboboxSelected>>",
        lambda _evt=None: callbacks.save_session({"character": vars.character.get()}),
    )

    # --- Advanced settings (collapsed by default) ---
    advanced_wrap = ttk.Frame(frm)
    advanced_wrap.pack(fill="x", pady=(0, 8))
    advanced_open = tk.BooleanVar(value=False)
    advanced_btn = ttk.Button(advanced_wrap, text="詳細設定を開く")
    advanced_btn.pack(anchor="w")

    advanced_body = ttk.LabelFrame(
        advanced_wrap,
        text="詳細設定（通常はそのままでOK）",
        padding=8,
    )

    def _update_pad_label(*_args) -> None:
        pad_value_label.config(text=f"{vars.pad.get():.2f}")

    def _save_pad(*_args) -> None:
        callbacks.save_session({"pad": float(vars.pad.get())})
        _update_pad_label()

    def _toggle_advanced() -> None:
        if advanced_open.get():
            advanced_body.pack_forget()
            advanced_open.set(False)
            advanced_btn.config(text="詳細設定を開く")
        else:
            advanced_body.pack(fill="x", pady=(6, 0))
            advanced_open.set(True)
            advanced_btn.config(text="詳細設定を閉じる")

    advanced_btn.config(command=_toggle_advanced)

    row_adv_pad = ttk.Frame(advanced_body)
    row_adv_pad.pack(fill="x", pady=(0, 6))
    ttk.Label(row_adv_pad, text="口配置の余白係数").pack(side="left")
    ttk.Scale(
        row_adv_pad,
        from_=1.20,
        to=3.20,
        variable=vars.pad,
        orient="horizontal",
    ).pack(side="left", fill="x", expand=True, padx=8)
    pad_value_label = ttk.Label(row_adv_pad, text=f"{vars.pad.get():.2f}")
    pad_value_label.pack(side="left")
    vars.pad.trace_add("write", _save_pad)

    ttk.Label(
        advanced_body,
        text=(
            "普段は 2.1 のままでOKです。\n"
            "・口PNGが小さく見える / 口の端が切れる → 少し上げる（例: 2.3〜2.6）\n"
            "・口が大きすぎる / 顎や頬まで拾う → 少し下げる（例: 1.8〜2.0）\n"
            "※ この設定は主に『① 解析→キャリブ』の口配置サイズに効きます。\n"
            "※ 『見た目確認（軽量）』で 1.9 / 2.1 / 2.3 前後を見比べてから決めるのがおすすめです。"
        ),
        justify="left",
        font=("", 9),
    ).pack(anchor="w")

    def _add_float_slider(
        parent: ttk.Frame,
        *,
        label: str,
        var: tk.DoubleVar,
        from_: float,
        to: float,
        fmt: str,
        save_key: str,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 6))
        ttk.Label(row, text=label, width=18).pack(side="left")
        ttk.Scale(
            row,
            from_=from_,
            to=to,
            variable=var,
            orient="horizontal",
        ).pack(side="left", fill="x", expand=True, padx=8)
        value_label = ttk.Label(row, text=fmt.format(var.get()), width=8)
        value_label.pack(side="left")

        def _save(*_args) -> None:
            value_label.config(text=fmt.format(var.get()))
            callbacks.save_session({save_key: float(var.get())})
            callbacks.on_mouth_color_adjust_changed()

        var.trace_add("write", _save)

    # --- Coverage slider ---
    row3 = ttk.Frame(frm)
    row3.pack(fill="x", pady=(0, 8))
    ttk.Label(row3, text="口消し範囲").pack(side="left")
    ttk.Scale(row3, from_=0.40, to=0.90, variable=vars.coverage, orient="horizontal").pack(
        side="left", fill="x", expand=True, padx=8,
    )
    cov_label = ttk.Label(row3, text=f"{vars.coverage.get():.2f}")
    cov_label.pack(side="left")
    vars.coverage.trace_add("write", lambda *_: cov_label.config(text=f"{vars.coverage.get():.2f}"))
    ttk.Label(frm, text="口が少し残るなら上げる / 広すぎるなら下げる", font=("", 9)).pack(anchor="w", pady=(0, 8))

    # --- Erase shading toggle ---
    row3a = ttk.Frame(frm)
    row3a.pack(fill="x", pady=(0, 8))
    ttk.Label(row3a, text="影なじませ（口消し）").pack(side="left")
    ttk.Checkbutton(
        row3a,
        text="有効（plane）",
        variable=vars.erase_shading,
        command=lambda: callbacks.save_session(
            {"erase_shading": "plane" if vars.erase_shading.get() else "none"},
        ),
    ).pack(side="left", padx=8)
    ttk.Label(row3a, text="OFFで顎の黒にじみを軽減").pack(side="left")

    # --- Smoothing preset ---
    row3b = ttk.Frame(frm)
    row3b.pack(fill="x", pady=(0, 8))
    ttk.Label(row3b, text="スムージング（トラック）").pack(side="left")
    cmb_smooth = ttk.Combobox(
        row3b,
        textvariable=vars.smoothing_menu,
        state="readonly",
        values=smoothing_labels,
    )
    cmb_smooth.pack(side="left", fill="x", expand=True, padx=8)
    cmb_smooth.bind(
        "<<ComboboxSelected>>",
        lambda _evt=None: callbacks.save_session({"smoothing": vars.smoothing_menu.get()}),
    )

    # --- Audio device ---
    row4 = ttk.Frame(frm)
    row4.pack(fill="x", pady=(0, 10))
    ttk.Label(row4, text="オーディオ入力デバイス（ライブ用）").pack(side="left")
    cmb_audio = ttk.Combobox(row4, textvariable=vars.audio_device_menu, state="readonly")
    cmb_audio.pack(side="left", fill="x", expand=True, padx=8)
    ttk.Button(row4, text="再読込", command=callbacks.refresh_audio_devices).pack(side="left")

    # --- Emotion auto ---
    row4b = ttk.Frame(frm)
    row4b.pack(fill="x", pady=(0, 10))
    ttk.Label(row4b, text="感情オート（音声）").pack(side="left")
    cmb_emotion_preset = ttk.Combobox(
        row4b,
        textvariable=vars.emotion_preset,
        state="readonly",
        values=emotion_preset_labels,
    )
    cmb_emotion_preset.pack(side="left", fill="x", expand=True, padx=8)
    cmb_emotion_preset.bind(
        "<<ComboboxSelected>>",
        lambda _evt=None: callbacks.save_session({"emotion_preset": vars.emotion_preset.get()}),
    )
    ttk.Checkbutton(
        row4b,
        text="HUD表示",
        variable=vars.emotion_hud,
        command=lambda: callbacks.save_session({"emotion_hud": bool(vars.emotion_hud.get())}),
    ).pack(side="left", padx=8)

    # --- Workflow buttons ---
    ttk.Label(frm, text="実行ステップ（通常は左から順番）", font=("", 9, "bold")).pack(anchor="w", pady=(2, 4))
    workflow_wrap = ttk.Frame(frm)
    workflow_wrap.pack(fill="x", pady=(0, 10))

    workflow_left = ttk.Frame(workflow_wrap)
    workflow_left.pack(side="left", fill="x", expand=True)

    row_btn = ttk.Frame(workflow_left)
    row_btn.pack(fill="x", pady=(0, 6))

    btn_track_calib = ttk.Button(row_btn, text="① 解析→キャリブ", command=callbacks.on_track_and_calib)
    btn_track_calib.pack(side="left")

    btn_calib_only = ttk.Button(row_btn, text="キャリブのみ（やり直し）", command=callbacks.on_calib_only)
    btn_calib_only.pack(side="left", padx=8)

    btn_erase_range = ttk.Button(row_btn, text="見た目確認（軽量）", command=callbacks.on_preview_erase_range)
    btn_erase_range.pack(side="left", padx=8)

    btn_erase = ttk.Button(row_btn, text="② 口消し動画生成", command=callbacks.on_erase_mouthless)
    btn_erase.pack(side="left")

    btn_live = ttk.Button(row_btn, text="③ ライブ実行", command=callbacks.on_live_run)
    btn_live.pack(side="left", padx=8)

    btn_stop = ttk.Button(
        row_btn, text=STOP_BTN_TEXT_DEFAULT, command=callbacks.on_stop, state="disabled",
    )
    btn_stop.pack(side="right")

    ttk.Label(
        workflow_left,
        text="見た目確認はフル動画を書き出さずに、pad と口消し範囲をその場で見比べて反映できます。",
        font=("", 9),
    ).pack(anchor="w", pady=(0, 0))

    final_adjust = ttk.LabelFrame(
        workflow_wrap,
        text="最終調整（キャリブ確認＆ライブ見た目）",
        padding=8,
    )
    final_adjust.pack(side="right", fill="y", padx=(12, 0))

    _add_float_slider(
        final_adjust,
        label="口PNG 明るさ",
        var=vars.mouth_brightness,
        from_=-32.0,
        to=32.0,
        fmt="{:.0f}",
        save_key="mouth_brightness",
    )
    _add_float_slider(
        final_adjust,
        label="口PNG 彩度",
        var=vars.mouth_saturation,
        from_=0.70,
        to=1.50,
        fmt="{:.2f}",
        save_key="mouth_saturation",
    )
    _add_float_slider(
        final_adjust,
        label="口PNG 暖色/寒色",
        var=vars.mouth_warmth,
        from_=-24.0,
        to=24.0,
        fmt="{:.0f}",
        save_key="mouth_warmth",
    )
    _add_float_slider(
        final_adjust,
        label="補正強度",
        var=vars.mouth_color_strength,
        from_=0.00,
        to=1.00,
        fmt="{:.2f}",
        save_key="mouth_color_strength",
    )
    _add_float_slider(
        final_adjust,
        label="外周優先度",
        var=vars.mouth_edge_priority,
        from_=0.00,
        to=1.00,
        fmt="{:.2f}",
        save_key="mouth_edge_priority",
    )
    _add_float_slider(
        final_adjust,
        label="外周補正幅",
        var=vars.mouth_edge_width_ratio,
        from_=0.02,
        to=0.20,
        fmt="{:.2f}",
        save_key="mouth_edge_width_ratio",
    )
    _add_float_slider(
        final_adjust,
        label="確認表示 色差強調",
        var=vars.mouth_inspect_boost,
        from_=1.00,
        to=4.00,
        fmt="{:.2f}",
        save_key="mouth_inspect_boost",
    )

    row_reset = ttk.Frame(final_adjust)
    row_reset.pack(fill="x", pady=(2, 6))
    btn_auto_color = ttk.Button(
        row_reset,
        text="色なじみ自動補正",
        command=callbacks.on_auto_mouth_color_adjust,
        state="disabled",
    )
    btn_auto_color.pack(side="left")
    ttk.Button(
        row_reset,
        text="既定値に戻す",
        command=callbacks.on_reset_mouth_color_adjust,
    ).pack(side="left", padx=(8, 0))

    ttk.Label(
        final_adjust,
        text=(
            "この調整はキャリブ確認とライブ見た目に反映されます。\n"
            "ライブ実行中は変更が数百msで反映されます。変更内容は自動保存されます。\n"
            "『色なじみ自動補正』はライブ実行中の現在フレームを元に推定します。\n"
            "『確認表示 色差強調』は表示専用で、素材ファイルや動画出力は変更しません。"
        ),
        justify="left",
        font=("", 9),
    ).pack(anchor="w")

    # --- Progress ---
    prog = ttk.Frame(frm)
    prog.pack(fill="x", pady=(0, 6))
    ttk.Label(prog, text="進捗").pack(side="left")
    ttk.Label(prog, textvariable=vars.progress_text).pack(side="left", padx=8)
    progress_bar = ttk.Progressbar(
        prog, variable=vars.progress, maximum=1.0, mode="determinate",
    )
    progress_bar.pack(side="left", fill="x", expand=True, padx=8)

    # --- Log ---
    log_header = ttk.Frame(frm)
    log_header.pack(fill="x")
    ttk.Label(log_header, text="ログ").pack(side="left", anchor="w")
    ttk.Button(log_header, text="ログクリア", command=callbacks.clear_log).pack(side="right")

    txt = tk.Text(frm, height=22, wrap="word")
    txt.pack(fill="both", expand=True)
    txt.configure(state="disabled")

    return WidgetRefs(
        cmb_character=cmb_character,
        cmb_audio=cmb_audio,
        cmb_smooth=cmb_smooth,
        cmb_emotion_preset=cmb_emotion_preset,
        pad_value_label=pad_value_label,
        cov_label=cov_label,
        btn_track_calib=btn_track_calib,
        btn_calib_only=btn_calib_only,
        btn_erase=btn_erase,
        btn_erase_range=btn_erase_range,
        btn_live=btn_live,
        btn_auto_color=btn_auto_color,
        btn_stop=btn_stop,
        progress_bar=progress_bar,
        txt=txt,
    )
