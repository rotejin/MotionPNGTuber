"""Tests for platform_open helpers."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import motionpngtuber.platform_open as platform_open


class OpenPathWithDefaultAppTests(unittest.TestCase):
    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            platform_open.open_path_with_default_app("")

    def test_windows_uses_startfile(self):
        with patch.object(platform_open.sys, "platform", "win32"):
            with patch.object(platform_open.os, "startfile", create=True) as startfile:
                platform_open.open_path_with_default_app("demo.txt")
                startfile.assert_called_once()

    def test_macos_uses_open(self):
        with patch.object(platform_open.sys, "platform", "darwin"):
            with patch.object(platform_open.subprocess, "Popen") as popen:
                platform_open.open_path_with_default_app("demo.txt")
                popen.assert_called_once()
                self.assertEqual(popen.call_args.args[0][0], "open")

    def test_linux_uses_xdg_open(self):
        with patch.object(platform_open.sys, "platform", "linux"):
            with patch.object(platform_open.subprocess, "Popen") as popen:
                platform_open.open_path_with_default_app("demo.txt")
                popen.assert_called_once()
                self.assertEqual(popen.call_args.args[0][0], "xdg-open")


class PreferNativeVideoPreviewTests(unittest.TestCase):
    def test_true_on_macos(self):
        with patch.object(platform_open.sys, "platform", "darwin"):
            self.assertTrue(platform_open.prefer_native_video_preview())

    def test_false_on_windows(self):
        with patch.object(platform_open.sys, "platform", "win32"):
            self.assertFalse(platform_open.prefer_native_video_preview())


if __name__ == "__main__":
    unittest.main()
