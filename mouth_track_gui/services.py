"""Service helpers for mouth_track_gui.

Phase 2 of the mouth_track_gui refactoring plan.
Extracts top-level utility functions (file/device helpers, character
resolution, sprite lookup) that do not depend on tk or the App class.
"""
from __future__ import annotations

import os

from .state import HERE

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def script_contains(path: str, needles: list[str]) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            s = f.read()
        return all(n in s for n in needles)
    except Exception:
        return False


def list_input_devices() -> list[dict]:
    """Returns GUI-ready input device items.

    Each item has:
      - spec: "sd:<index>" or "pa:<source>"
      - index: sounddevice input index or None
      - display: user-facing text
    """
    try:
        import sounddevice as sd  # type: ignore
        from motionpngtuber.audio_linux import augment_devices_for_linux

        devices: list[dict] = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                name = str(d.get("name", ""))[:64]
                ch = int(d.get("max_input_channels", 0))
                sr = int(float(d.get("default_samplerate", 0)) or 0)
                devices.append({
                    "spec": f"sd:{i}",
                    "index": i,
                    "display": f"{i}: {name}  (ch={ch}, sr={sr})",
                })
        return augment_devices_for_linux(devices, sd)
    except Exception:
        return []


def find_input_device_item(items: list[dict], spec_or_display: str) -> dict | None:
    target = str(spec_or_display or "").strip()
    if not target:
        return None
    for item in items:
        if str(item.get("spec", "")).strip() == target:
            return item
        if str(item.get("display", "")).strip() == target:
            return item
    return None


def display_to_audio_spec(display: str) -> str | None:
    item = find_input_device_item(list_input_devices(), display)
    if item is None:
        return None
    spec = str(item.get("spec", "")).strip()
    return spec or None


def ensure_backend_sanity(base_dir: str) -> tuple[bool, str]:
    """Prevent the common 'file got swapped/overwritten' situation."""
    track_py = os.path.join(base_dir, "auto_mouth_track_v2.py")
    erase_py = os.path.join(base_dir, "auto_erase_mouth.py")

    if not os.path.isfile(track_py):
        return False, "auto_mouth_track_v2.py が見つかりません。"
    if not os.path.isfile(erase_py):
        return False, "auto_erase_mouth.py が見つかりません。"

    if not script_contains(track_py, ["--pad", "--det-scale", "--min-conf"]):
        return (
            False,
            "auto_mouth_track_v2.py が追跡用スクリプトではないようです（--pad 等が見つかりません）。\n"
            "ファイルが入れ替わっていないか確認してください。",
        )

    if not script_contains(erase_py, ["--track", "--coverage"]):
        return (
            False,
            "auto_erase_mouth.py が口消し用スクリプトではないようです（--track/--coverage が見つかりません）。\n"
            "ファイルが入れ替わっていないか確認してください。",
        )

    return True, ""


# ---------------------------------------------------------------------------
# Mouth directory / character resolution
# ---------------------------------------------------------------------------

def guess_mouth_dir(video_path: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(video_path))
    cand = os.path.join(base_dir, "mouth")
    if os.path.isdir(cand):
        return cand
    cand = os.path.join(HERE, "mouth")
    if os.path.isdir(cand):
        return cand
    return ""


def best_open_sprite(mouth_dir: str) -> str:
    if not mouth_dir or not os.path.isdir(mouth_dir):
        return ""
    p = os.path.join(mouth_dir, "open.png")
    if os.path.isfile(p):
        return p
    try:
        for name in os.listdir(mouth_dir):
            if name.lower() == "open.png":
                p2 = os.path.join(mouth_dir, name)
                if os.path.isfile(p2):
                    return p2
    except Exception:
        pass
    return ""


EMOTION_DIR_NAMES = {"default", "neutral", "happy", "angry", "sad", "excited"}


def is_emotion_level_mouth_root(mouth_root: str) -> bool:
    """Heuristic: mouth_root is already a character directory (no character layer),
    if it contains open.png directly OR contains multiple emotion-named subfolders."""
    if not mouth_root or not os.path.isdir(mouth_root):
        return False
    if os.path.isfile(os.path.join(mouth_root, "open.png")):
        return True
    try:
        subs = [d for d in os.listdir(mouth_root) if os.path.isdir(os.path.join(mouth_root, d))]
        low = {d.lower() for d in subs}
        return len(low & EMOTION_DIR_NAMES) >= 2
    except Exception:
        return False


def list_character_dirs(mouth_root: str) -> list[str]:
    """Return character folder candidates under mouth_root.
    If mouth_root looks like an emotion-level folder, return []."""
    if not mouth_root or not os.path.isdir(mouth_root):
        return []
    if is_emotion_level_mouth_root(mouth_root):
        return []
    try:
        subs = [d for d in os.listdir(mouth_root) if os.path.isdir(os.path.join(mouth_root, d))]
        chars = [d for d in subs if d.lower() not in EMOTION_DIR_NAMES]
        chars.sort(key=lambda x: x.lower())
        return chars
    except Exception:
        return []


def resolve_character_dir(mouth_root: str, character: str) -> str:
    """Resolve mouth directory passed to runtime / used for sprite search.
    If character is valid, use mouth_root/character, else use mouth_root."""
    if not mouth_root:
        return ""
    if character:
        cand = os.path.join(mouth_root, character)
        if os.path.isdir(cand):
            return cand
    return mouth_root


def best_open_sprite_for_character(mouth_root: str, character: str) -> str:
    """Find open.png for calibration.
    Priority:
      1) <mouth_dir>/open.png (backward compat)
      2) <mouth_dir>/(Default|neutral|...)/open.png
      3) first found in immediate subfolders
    where <mouth_dir> is mouth_root or mouth_root/character.
    """
    base = resolve_character_dir(mouth_root, character)
    if not base or not os.path.isdir(base):
        return ""

    # 1) direct
    p = os.path.join(base, "open.png")
    if os.path.isfile(p):
        return p
    try:
        for name in os.listdir(base):
            if name.lower() == "open.png":
                p2 = os.path.join(base, name)
                if os.path.isfile(p2):
                    return p2
    except Exception:
        pass

    # 2) preferred emotion folders
    preferred = ["Default", "default", "neutral", "Neutral", "Normal", "normal"]
    for em in preferred:
        d = os.path.join(base, em)
        if not os.path.isdir(d):
            continue
        p = os.path.join(d, "open.png")
        if os.path.isfile(p):
            return p
        try:
            for name in os.listdir(d):
                if name.lower() == "open.png":
                    p2 = os.path.join(d, name)
                    if os.path.isfile(p2):
                        return p2
        except Exception:
            pass

    # 3) any immediate subfolder
    try:
        for sub in os.listdir(base):
            d = os.path.join(base, sub)
            if not os.path.isdir(d):
                continue
            p = os.path.join(d, "open.png")
            if os.path.isfile(p):
                return p
            for name in os.listdir(d):
                if name.lower() == "open.png":
                    p2 = os.path.join(d, name)
                    if os.path.isfile(p2):
                        return p2
    except Exception:
        pass

    return ""
