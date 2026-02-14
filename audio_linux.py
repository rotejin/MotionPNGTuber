"""
audio_linux.py

Linux (PulseAudio) 固有のオーディオデバイス補助。
PortAudio (ALSA backend) が USB デバイスを個別列挙しない問題を
PulseAudio 経由で補完する。

他の OS では全関数が no-op として安全に呼び出せる。
"""

from __future__ import annotations

import platform
import subprocess


def is_linux() -> bool:
    return platform.system() == "Linux"


# ---------------------------------------------------------------------------
# PulseAudio ソース列挙
# ---------------------------------------------------------------------------

def list_pulse_input_sources() -> list[str]:
    """PulseAudio の入力ソース名一覧を返す (.monitor を除く)。"""
    if not is_linux():
        return []
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"],
            text=True, timeout=5,
        )
    except Exception:
        return []
    sources = []
    for line in out.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            name = parts[1]
            if ".monitor" not in name:
                sources.append(name)
    return sources


def set_pulse_default_source(pa_name: str) -> bool:
    """PulseAudio のデフォルト入力ソースを設定する。成功なら True。"""
    if not is_linux():
        return False
    try:
        subprocess.check_call(
            ["pactl", "set-default-source", pa_name], timeout=5,
        )
        print(f"[audio/linux] set PulseAudio default source -> {pa_name}")
        return True
    except Exception as e:
        print(f"[audio/linux] warning: failed to set default source: {e}")
        return False


# ---------------------------------------------------------------------------
# sounddevice デバイス index ヘルパー
# ---------------------------------------------------------------------------

def _find_sd_device_index(sd, name: str) -> int | None:
    """sounddevice デバイス一覧から名前が一致する入力デバイスの index を返す。"""
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("name") == name and dev.get("max_input_channels", 0) > 0:
                return i
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# デバイス解決 (ランタイム用)
# ---------------------------------------------------------------------------

def prepare_device(device_int: int | None, sd) -> int | None:
    """デバイス index を検証し、Linux で無効なら PulseAudio 経由の index に置き換えて返す。

    - Linux 以外、または index が有効 → そのまま返す
    - Linux で index が無効 → PulseAudio フォールバックを試み、
      成功すれば pulse デバイスの index を返す
    - フォールバックも失敗 → 元の値をそのまま返す
    """
    if not is_linux():
        return device_int
    if device_int is None:
        return device_int

    # デバイス index が有効か確認
    try:
        dev = sd.query_devices(device_int, "input")
        if dev.get("max_input_channels", 0) > 0:
            return device_int
    except Exception:
        pass

    # 無効 → PulseAudio フォールバック
    fallback = _resolve_via_pulse(sd)
    return fallback if fallback is not None else device_int


def _resolve_via_pulse(sd) -> int | None:
    """PulseAudio の最初の入力ソースをデフォルトに設定し、pulse デバイス index を返す。"""
    if not is_linux():
        return None

    sources = list_pulse_input_sources()
    if not sources:
        return None

    # 最初の PulseAudio 入力ソースをデフォルトに設定
    set_pulse_default_source(sources[0])
    pulse_idx = _find_sd_device_index(sd, "pulse")
    if pulse_idx is not None:
        print(f"[audio/linux] fallback -> pulse device #{pulse_idx} ({sources[0]})")
        return pulse_idx

    default_idx = _find_sd_device_index(sd, "default")
    if default_idx is not None:
        print(f"[audio/linux] fallback -> default device #{default_idx} ({sources[0]})")
        return default_idx

    return None


def resolve_device_by_name(name_query: str, sd) -> int | None:
    """デバイス名 (部分一致) で PulseAudio ソースを検索・設定し、
    sounddevice の pulse デバイス index を返す。

    見つからなければ None。
    """
    if not is_linux():
        return None

    # まず sounddevice 側で名前一致を探す
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0:
                if name_query.lower() in dev["name"].lower():
                    print(f"[audio/linux] matched sounddevice #{i}: {dev['name']}")
                    return i
    except Exception:
        pass

    # PulseAudio ソースから検索
    for pa_name in list_pulse_input_sources():
        if name_query.lower() in pa_name.lower():
            print(f"[audio/linux] matched PulseAudio source: {pa_name}")
            set_pulse_default_source(pa_name)
            pulse_idx = _find_sd_device_index(sd, "pulse")
            if pulse_idx is not None:
                return pulse_idx
            return _find_sd_device_index(sd, "default")

    return None


# ---------------------------------------------------------------------------
# GUI デバイスリスト拡張
# ---------------------------------------------------------------------------

def add_device_list_linux(devices: list, sd) -> list:
    """既存のデバイスリストに PulseAudio 入力ソースを追加して返す。

    devices: [(index, display_string), ...] 形式。
    sounddevice に既に列挙されているソースは追加しない。
    """
    if not is_linux():
        return devices

    pulse_idx = _find_sd_device_index(sd, "pulse")
    if pulse_idx is None:
        return devices

    # 既存デバイス名を収集 (重複排除用)
    existing_names = set()
    for _, disp in devices:
        existing_names.add(disp.lower())

    for pa_name in list_pulse_input_sources():
        # 既に sounddevice 側の表示名に含まれている場合はスキップ
        if any(pa_name.lower() in n for n in existing_names):
            continue
        devices.append((pulse_idx, f"pa:{pa_name}  (via pulse)"))

    return devices


def add_audio_device_linux_str(devices: list[str], sd) -> list[str]:
    """文字列リスト版の add_device_list_linux (multi_video_live_gui 用)。"""
    if not is_linux():
        return devices

    pulse_idx = _find_sd_device_index(sd, "pulse")
    if pulse_idx is None:
        return devices

    existing_lower = {d.lower() for d in devices}

    for pa_name in list_pulse_input_sources():
        if any(pa_name.lower() in n for n in existing_lower):
            continue
        devices.append(f"pa:{pa_name}  (via pulse)")

    return devices


# ---------------------------------------------------------------------------
# GUI デバイス選択ハンドラ
# ---------------------------------------------------------------------------

def handle_pa_device_selection(device_str: str, sd) -> int | None:
    """'pa:<source_name>  (via pulse)' 形式のデバイス文字列を処理する。

    PulseAudio のデフォルトソースを設定し、pulse デバイスの index を返す。
    pa: 形式でなければ None を返す (= 通常処理に委譲)。
    """
    if not device_str.startswith("pa:"):
        return None

    pa_name = device_str[3:].split("  (via pulse)")[0].strip()
    set_pulse_default_source(pa_name)
    return _find_sd_device_index(sd, "pulse")
