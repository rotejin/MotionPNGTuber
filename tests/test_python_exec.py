import ntpath
import types
import unittest
from unittest import mock

from motionpngtuber.python_exec import resolve_python_subprocess_executable


def windows_os(isfile):
    """A stand-in for the os module on Windows, so that these tests also run on
    Linux and macOS (where os.path cannot split Windows paths)."""
    path = types.SimpleNamespace(
        basename=ntpath.basename, dirname=ntpath.dirname, join=ntpath.join, isfile=isfile,
    )
    return types.SimpleNamespace(name="nt", path=path)


class ResolvePythonSubprocessExecutableTests(unittest.TestCase):
    def test_non_windows_keeps_original_executable(self):
        with mock.patch("motionpngtuber.python_exec.os.name", "posix"):
            self.assertEqual(
                resolve_python_subprocess_executable("/usr/bin/python3"),
                "/usr/bin/python3",
            )

    def test_windows_pythonw_switches_to_python_when_sibling_exists(self):
        exe = r"C:\proj\.venv\Scripts\pythonw.exe"
        with mock.patch("motionpngtuber.python_exec.os", windows_os(lambda _p: True)):
            self.assertEqual(
                resolve_python_subprocess_executable(exe),
                r"C:\proj\.venv\Scripts\python.exe",
            )

    def test_windows_pythonw_stays_when_sibling_missing(self):
        exe = r"C:\proj\.venv\Scripts\pythonw.exe"
        with mock.patch("motionpngtuber.python_exec.os", windows_os(lambda _p: False)):
            self.assertEqual(resolve_python_subprocess_executable(exe), exe)

    def test_windows_python_exe_is_kept(self):
        exe = r"C:\proj\.venv\Scripts\python.exe"
        with mock.patch("motionpngtuber.python_exec.os", windows_os(lambda _p: True)):
            self.assertEqual(resolve_python_subprocess_executable(exe), exe)
