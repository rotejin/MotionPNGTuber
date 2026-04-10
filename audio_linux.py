"""Linux audio helper utilities for MotionPNGTuber.

This module isolates PulseAudio / PipeWire specific behavior.
On non-Linux platforms, all functions are safe no-ops.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def has_pactl() -> bool:
    return is_linux() and (shutil.which("pactl") is not None)


def list_pulse_input_sources() -> list[str]:
    if not has_pactl():
        return []
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"],
            text=True,
            timeout=5,
        )
    except Exception:
        return []

    sources: list[str] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1].strip()
        if not name or ".monitor" in name:
            continue
        sources.append(name)
    return sources


def get_pulse_default_source() -> str | None:
    if not has_pactl():
        return None
    try:
        out = subprocess.check_output(
            ["pactl", "get-default-source"],
            text=True,
            timeout=5,
        ).strip()
    except Exception:
        return None
    return out or None


def set_pulse_default_source(name: str) -> bool:
    if not has_pactl() or not name:
        return False
    try:
        subprocess.check_call(
            ["pactl", "set-default-source", name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return True
    except Exception:
        return False


def _query_devices(sd: Any) -> list[dict]:
    try:
        return list(sd.query_devices())
    except Exception:
        return []


def _find_sd_input_index_by_exact_name(sd: Any, name: str) -> int | None:
    target = (name or "").strip().lower()
    if not target:
        return None
    for i, dev in enumerate(_query_devices(sd)):
        try:
            if int(dev.get("max_input_channels", 0) or 0) <= 0:
                continue
            if str(dev.get("name", "")).strip().lower() == target:
                return i
        except Exception:
            continue
    return None


def _find_sd_input_index_by_contains(sd: Any, keyword: str) -> int | None:
    target = (keyword or "").strip().lower()
    if not target:
        return None
    for i, dev in enumerate(_query_devices(sd)):
        try:
            if int(dev.get("max_input_channels", 0) or 0) <= 0:
                continue
            if target in str(dev.get("name", "")).lower():
                return i
        except Exception:
            continue
    return None


def _validate_sd_input_index(sd: Any, index: int | None) -> int | None:
    if index is None:
        return None
    try:
        dev = sd.query_devices(index, "input")
    except Exception:
        return None
    try:
        if int(dev.get("max_input_channels", 0) or 0) <= 0:
            return None
    except Exception:
        return None
    return int(index)


def normalize_audio_device_spec(spec_or_display: str | int | None) -> str | None:
    if spec_or_display is None:
        return None
    if isinstance(spec_or_display, int):
        return f"sd:{int(spec_or_display)}"

    s = str(spec_or_display).strip()
    if not s:
        return None

    if s.startswith("sd:"):
        body = s[3:].strip()
        return f"sd:{int(body)}" if body.isdigit() else None
    if s.startswith("pa:"):
        body = s[3:].strip()
        return f"pa:{body}" if body else None

    if s.isdigit():
        return f"sd:{int(s)}"

    head, sep, _tail = s.partition(":")
    if sep and head.strip().isdigit():
        return f"sd:{int(head.strip())}"

    return None


def _display_name_from_index(sd: Any, index: int | None) -> str:
    if index is None:
        return ""
    try:
        dev = sd.query_devices(index, "input")
        return str(dev.get("name", "") or "")
    except Exception:
        return ""


def resolve_audio_device_spec(
    spec: str | None,
    sd: Any,
    fallback_index: int | None = None,
    *,
    prefer_default_source: bool = False,
    allow_default_source_switch: bool = False,
) -> dict[str, Any]:
    normalized = normalize_audio_device_spec(spec)
    fallback_valid = _validate_sd_input_index(sd, fallback_index)
    result: dict[str, Any] = {
        "resolved_index": None,
        "effective_spec": normalized,
        "display_name": "",
        "strategy": "none",
        "pulse_source": None,
        "needs_env_apply": False,
        "needs_default_source_switch": False,
        "error": "",
    }

    if normalized is None:
        result["resolved_index"] = fallback_valid
        result["display_name"] = _display_name_from_index(sd, fallback_valid)
        if fallback_valid is None:
            result["error"] = "no audio device specified"
        return result

    if normalized.startswith("sd:"):
        idx = _validate_sd_input_index(sd, int(normalized[3:]))
        result["resolved_index"] = idx if idx is not None else fallback_valid
        result["display_name"] = _display_name_from_index(sd, result["resolved_index"])
        if result["resolved_index"] is None:
            result["error"] = f"sounddevice input not found: {normalized}"
        return result

    if not normalized.startswith("pa:"):
        result["resolved_index"] = fallback_valid
        result["display_name"] = _display_name_from_index(sd, fallback_valid)
        if fallback_valid is None:
            result["error"] = f"unsupported audio spec: {normalized}"
        return result

    source_name = normalized[3:].strip()
    result["pulse_source"] = source_name
    result["display_name"] = source_name

    if not is_linux():
        result["resolved_index"] = fallback_valid
        if fallback_valid is None:
            result["error"] = f"pulse spec is unsupported on this platform: {normalized}"
        return result

    direct_idx = _find_sd_input_index_by_contains(sd, source_name)
    if direct_idx is not None:
        result["resolved_index"] = direct_idx
        result["display_name"] = _display_name_from_index(sd, direct_idx) or source_name
        return result

    pulse_idx = _find_sd_input_index_by_exact_name(sd, "pulse")
    if pulse_idx is None:
        pulse_idx = _find_sd_input_index_by_contains(sd, "pulse")

    default_idx = _find_sd_input_index_by_exact_name(sd, "default")
    if default_idx is None:
        default_idx = _find_sd_input_index_by_contains(sd, "default")

    can_switch_default_source = bool(prefer_default_source and allow_default_source_switch and has_pactl())
    idx_candidates = [default_idx, pulse_idx] if can_switch_default_source else [pulse_idx]
    chosen_idx = next((idx for idx in idx_candidates if idx is not None), None)

    if chosen_idx is not None:
        result["resolved_index"] = chosen_idx
        result["display_name"] = _display_name_from_index(sd, chosen_idx) or source_name
        if can_switch_default_source and chosen_idx == default_idx:
            result["strategy"] = "set_default_source"
            result["needs_default_source_switch"] = True
        else:
            result["strategy"] = "pulse_env"
            result["needs_env_apply"] = True
        return result

    if fallback_valid is not None:
        result["resolved_index"] = fallback_valid
        result["display_name"] = _display_name_from_index(sd, fallback_valid)
        result["error"] = f"pulse source fallback used for {normalized}"
        return result

    result["error"] = f"failed to resolve pulse source: {normalized}"
    return result


def apply_audio_resolution_for_current_process(resolution: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if not is_linux() or not resolution:
        return state

    pulse_source = str(resolution.get("pulse_source") or "").strip()

    if resolution.get("needs_env_apply") and pulse_source:
        state["original_pulse_source"] = os.environ.get("PULSE_SOURCE")
        os.environ["PULSE_SOURCE"] = pulse_source

    if resolution.get("needs_default_source_switch") and pulse_source and has_pactl():
        original_default = get_pulse_default_source()
        state["original_default_source"] = original_default
        state["default_source_switched"] = set_pulse_default_source(pulse_source)

    return state


def cleanup_audio_device_resolution(resolution: dict[str, Any], state: dict[str, Any] | None) -> None:
    if not is_linux() or not state:
        return

    if "original_pulse_source" in state:
        original = state.get("original_pulse_source")
        if original is None:
            os.environ.pop("PULSE_SOURCE", None)
        else:
            os.environ["PULSE_SOURCE"] = str(original)

    if state.get("default_source_switched"):
        original_default = state.get("original_default_source")
        if isinstance(original_default, str) and original_default:
            set_pulse_default_source(original_default)


def augment_devices_for_linux(base_items: list[dict[str, Any]], sd: Any) -> list[dict[str, Any]]:
    if not is_linux():
        return base_items

    devices = list(base_items)
    existing_specs = {str(item.get("spec", "")).strip().lower() for item in devices}
    for source_name in list_pulse_input_sources():
        spec = f"pa:{source_name}"
        if spec.lower() in existing_specs:
            continue
        devices.append({
            "spec": spec,
            "index": None,
            "display": f"{spec}  (via pulse)",
        })
    return devices
