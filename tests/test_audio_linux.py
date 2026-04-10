"""Tests for Linux audio helper utilities."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import motionpngtuber.audio_linux as audio_linux


class NormalizeAudioDeviceSpecTests(unittest.TestCase):
    def test_normalize_variants(self):
        self.assertIsNone(audio_linux.normalize_audio_device_spec(None))
        self.assertIsNone(audio_linux.normalize_audio_device_spec(""))
        self.assertEqual(audio_linux.normalize_audio_device_spec(3), "sd:3")
        self.assertEqual(audio_linux.normalize_audio_device_spec("3"), "sd:3")
        self.assertEqual(audio_linux.normalize_audio_device_spec("sd:3"), "sd:3")
        self.assertEqual(audio_linux.normalize_audio_device_spec("pa:foo"), "pa:foo")
        self.assertEqual(
            audio_linux.normalize_audio_device_spec("3: USB Mic (ch=1, sr=48000)"),
            "sd:3",
        )


class NonLinuxNoOpTests(unittest.TestCase):
    def test_windows_noop(self):
        with mock.patch("motionpngtuber.audio_linux.is_linux", return_value=False):
            self.assertEqual(audio_linux.list_pulse_input_sources(), [])
            self.assertEqual(audio_linux.augment_devices_for_linux([{"spec": "sd:0"}], object()), [{"spec": "sd:0"}])

    def test_macos_noop(self):
        with mock.patch("motionpngtuber.audio_linux.is_linux", return_value=False):
            resolution = audio_linux.resolve_audio_device_spec("pa:test", mock.Mock(), fallback_index=None)
            self.assertIsNone(resolution["resolved_index"])
            self.assertEqual(resolution["strategy"], "none")

    def test_apply_cleanup_noop(self):
        with mock.patch("motionpngtuber.audio_linux.is_linux", return_value=False):
            state = audio_linux.apply_audio_resolution_for_current_process({"strategy": "pulse_env"})
            self.assertEqual(state, {})
            audio_linux.cleanup_audio_device_resolution({}, state)


class PactlFallbackTests(unittest.TestCase):
    def test_pactl_missing(self):
        with mock.patch("motionpngtuber.audio_linux.has_pactl", return_value=False):
            self.assertEqual(audio_linux.list_pulse_input_sources(), [])
            self.assertIsNone(audio_linux.get_pulse_default_source())

    def test_build_like_env_and_restore(self):
        with mock.patch("motionpngtuber.audio_linux.is_linux", return_value=True):
            old = os.environ.get("PULSE_SOURCE")
            try:
                resolution = {
                    "needs_env_apply": True,
                    "needs_default_source_switch": False,
                    "pulse_source": "alsa_input.test",
                }
                state = audio_linux.apply_audio_resolution_for_current_process(resolution)
                self.assertEqual(os.environ.get("PULSE_SOURCE"), "alsa_input.test")
                audio_linux.cleanup_audio_device_resolution(resolution, state)
            finally:
                if old is None:
                    os.environ.pop("PULSE_SOURCE", None)
                else:
                    os.environ["PULSE_SOURCE"] = old


class _FakeSoundDevice:
    def __init__(self, devices):
        self._devices = [dict(d) for d in devices]

    def query_devices(self, index=None, kind=None):
        if index is None:
            return list(self._devices)
        dev = dict(self._devices[int(index)])
        if kind == "input" and int(dev.get("max_input_channels", 0) or 0) <= 0:
            raise ValueError("not an input device")
        return dev


class LinuxResolutionTests(unittest.TestCase):
    def test_resolve_pa_prefers_pulse_env_first(self):
        fake_sd = _FakeSoundDevice([
            {"name": "default", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "pulse", "max_input_channels": 2, "default_samplerate": 48000},
        ])
        with (
            mock.patch("motionpngtuber.audio_linux.is_linux", return_value=True),
            mock.patch("motionpngtuber.audio_linux.has_pactl", return_value=True),
        ):
            resolution = audio_linux.resolve_audio_device_spec(
                "pa:alsa_input.test",
                fake_sd,
                fallback_index=None,
                prefer_default_source=False,
            )
        self.assertEqual(resolution["resolved_index"], 1)
        self.assertEqual(resolution["strategy"], "pulse_env")
        self.assertTrue(resolution["needs_env_apply"])
        self.assertFalse(resolution["needs_default_source_switch"])

    def test_resolve_pa_can_switch_default_source_as_fallback(self):
        fake_sd = _FakeSoundDevice([
            {"name": "default", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "pulse", "max_input_channels": 2, "default_samplerate": 48000},
        ])
        with (
            mock.patch("motionpngtuber.audio_linux.is_linux", return_value=True),
            mock.patch("motionpngtuber.audio_linux.has_pactl", return_value=True),
        ):
            resolution = audio_linux.resolve_audio_device_spec(
                "pa:alsa_input.test",
                fake_sd,
                fallback_index=None,
                prefer_default_source=True,
                allow_default_source_switch=True,
            )
        self.assertEqual(resolution["resolved_index"], 0)
        self.assertEqual(resolution["strategy"], "set_default_source")
        self.assertFalse(resolution["needs_env_apply"])
        self.assertTrue(resolution["needs_default_source_switch"])

    def test_resolve_pa_does_not_switch_default_source_without_opt_in(self):
        fake_sd = _FakeSoundDevice([
            {"name": "default", "max_input_channels": 2, "default_samplerate": 48000},
            {"name": "pulse", "max_input_channels": 2, "default_samplerate": 48000},
        ])
        with (
            mock.patch("motionpngtuber.audio_linux.is_linux", return_value=True),
            mock.patch("motionpngtuber.audio_linux.has_pactl", return_value=True),
        ):
            resolution = audio_linux.resolve_audio_device_spec(
                "pa:alsa_input.test",
                fake_sd,
                fallback_index=None,
                prefer_default_source=True,
                allow_default_source_switch=False,
            )
        self.assertEqual(resolution["resolved_index"], 1)
        self.assertEqual(resolution["strategy"], "pulse_env")
        self.assertTrue(resolution["needs_env_apply"])
        self.assertFalse(resolution["needs_default_source_switch"])

    def test_augment_devices_for_linux_adds_pulse_sources(self):
        base = [{"spec": "sd:0", "index": 0, "display": "0: Mic"}]
        with (
            mock.patch("motionpngtuber.audio_linux.is_linux", return_value=True),
            mock.patch("motionpngtuber.audio_linux.list_pulse_input_sources", return_value=["alsa_input.test"]),
        ):
            devices = audio_linux.augment_devices_for_linux(base, object())
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[1]["spec"], "pa:alsa_input.test")
        self.assertIn("(via pulse)", devices[1]["display"])


if __name__ == "__main__":
    unittest.main()
