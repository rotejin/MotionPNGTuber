"""Tests for mouth_track_gui.runner module (Phase 3)."""
import sys
import threading
import time
import unittest
from unittest import mock

from mouth_track_gui.runner import CommandRunner, DrainResult, RunResult


class RunResultTests(unittest.TestCase):
    def test_fields(self):
        r = RunResult(returncode=0, was_stopped=False)
        self.assertEqual(r.returncode, 0)
        self.assertFalse(r.was_stopped)
        self.assertFalse(r.stop_requested)

    def test_stopped_result(self):
        r = RunResult(returncode=1, was_stopped=True)
        self.assertTrue(r.was_stopped)


class CommandRunnerBasicTests(unittest.TestCase):
    def test_simple_echo(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        result = runner.run_stream(
            [sys.executable, "-c", "print('hello world')"]
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.was_stopped)
        # Should contain at least the [cmd] line and "hello world"
        full = "\n".join(logs)
        self.assertIn("hello world", full)

    def test_nonzero_exit(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        result = runner.run_stream(
            [sys.executable, "-c", "import sys; sys.exit(42)"]
        )
        self.assertEqual(result.returncode, 42)
        self.assertFalse(result.was_stopped)

    def test_invalid_command(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        result = runner.run_stream(
            ["__nonexistent_command_12345__"]
        )
        self.assertEqual(result.returncode, 999)
        full = "\n".join(logs)
        self.assertIn("[error]", full)

    def test_is_running_property(self):
        runner = CommandRunner(on_log=lambda _: None)
        self.assertFalse(runner.is_running)

    def test_multiline_output(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        code = "for i in range(5): print(f'line{i}')"
        result = runner.run_stream([sys.executable, "-c", code])
        self.assertEqual(result.returncode, 0)
        output_lines = [l for l in logs if l.startswith("line")]
        self.assertEqual(len(output_lines), 5)

    def test_env_passthrough(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        result = runner.run_stream(
            [sys.executable, "-c", "import os; print(os.environ.get('TEST_VAR_XYZ', ''))"],
            env={"TEST_VAR_XYZ": "hello123"},
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello123", "\n".join(logs))


class CommandRunnerForceStopTests(unittest.TestCase):
    def test_force_stop(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)
        code = "import time\nfor i in range(100):\n    print(f'tick{i}', flush=True)\n    time.sleep(0.1)"

        def stop_later():
            time.sleep(0.3)
            runner.force_stop()

        threading.Thread(target=stop_later, daemon=True).start()
        result = runner.run_stream([sys.executable, "-c", code])
        self.assertTrue(result.was_stopped)
        self.assertFalse(runner.is_running)


class CommandRunnerSoftStopTests(unittest.TestCase):
    def test_soft_then_force_reports_stopped(self):
        """Soft stop followed by force stop should report was_stopped."""
        logs = []
        runner = CommandRunner(on_log=logs.append)
        # Use a script that ignores SIGINT and runs long enough
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "for i in range(200):\n"
            "    print(f'tick{i}', flush=True)\n"
            "    time.sleep(0.05)\n"
        )

        def stop_later():
            time.sleep(0.3)
            runner.request_soft_stop()
            time.sleep(0.3)
            runner.force_stop()

        threading.Thread(target=stop_later, daemon=True).start()
        result = runner.run_stream(
            [sys.executable, "-c", code], allow_soft_stop=True
        )
        # Either soft or force stop should mark was_stopped
        self.assertTrue(result.was_stopped)

    def test_normal_completion_is_not_stopped(self):
        """A process that exits normally should have was_stopped=False."""
        logs = []
        runner = CommandRunner(on_log=logs.append)
        result = runner.run_stream(
            [sys.executable, "-c", "print('ok')"], allow_soft_stop=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.was_stopped)


class CommandRunnerSoftStopRegressionTests(unittest.TestCase):
    """Regression tests for stop semantics (Codex review issues)."""

    def test_soft_stop_does_not_kill_allow_false_process(self):
        """request_soft_stop must NOT interrupt allow_soft_stop=False commands."""
        logs = []
        runner = CommandRunner(on_log=logs.append)
        code = "import time\nfor i in range(5):\n    print(f'tick{i}', flush=True)\n    time.sleep(0.05)"

        def soft_later():
            time.sleep(0.05)
            runner.request_soft_stop()

        threading.Thread(target=soft_later, daemon=True).start()
        result = runner.run_stream(
            [sys.executable, "-c", code], allow_soft_stop=False
        )
        # Process must complete normally — soft stop is just a reservation
        self.assertEqual(result.returncode, 0)
        ticks = [l for l in logs if l.startswith("tick")]
        self.assertEqual(len(ticks), 5)

    def test_pending_soft_stop_observed_by_next_run(self):
        """A soft stop requested before run_stream must be observable."""
        logs = []
        runner = CommandRunner(on_log=logs.append)
        runner.request_soft_stop()
        # soft_requested should be true going into run_stream
        self.assertTrue(runner.soft_requested)
        # A short-lived process with allow_soft_stop=True
        result = runner.run_stream(
            [sys.executable, "-c", "print('hi')"], allow_soft_stop=True
        )
        self.assertTrue(result.stop_requested)

    def test_reset_clears_flags(self):
        """reset() should clear pending stop flags."""
        runner = CommandRunner(on_log=lambda _: None)
        runner.request_soft_stop()
        self.assertTrue(runner.soft_requested)
        runner.reset()
        self.assertFalse(runner.soft_requested)

    def test_drain_output_waits_for_sentinel_before_exit(self):
        """Process exit alone must not drop trailing output lines."""
        logs = []
        runner = CommandRunner(on_log=logs.append)

        class DelayedStdout:
            def __init__(self):
                self._items = [("first\n", 0.0), ("second\n", 0.2)]
                self._idx = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._idx >= len(self._items):
                    raise StopIteration
                line, delay = self._items[self._idx]
                self._idx += 1
                if delay > 0:
                    time.sleep(delay)
                return line

        class FakePopen:
            def __init__(self):
                self.stdout = DelayedStdout()

            def poll(self):
                return 0

        drain = runner._drain_output(FakePopen(), allow_soft_stop=False)
        self.assertFalse(drain.was_stopped)
        self.assertFalse(drain.stop_requested)
        self.assertEqual(logs, ["first", "second"])

    def test_drain_output_late_soft_stop_after_exit_sets_request_only(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)

        class DelayedSentinelStdout:
            def __init__(self):
                self._stage = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._stage == 0:
                    self._stage = 1
                    return "first\n"
                if self._stage == 1:
                    self._stage = 2
                    time.sleep(0.2)
                raise StopIteration

        class FakePopen:
            def __init__(self):
                self.stdout = DelayedSentinelStdout()

            def poll(self):
                return 0

        def request_later():
            time.sleep(0.05)
            runner.request_soft_stop()

        threading.Thread(target=request_later, daemon=True).start()
        drain = runner._drain_output(FakePopen(), allow_soft_stop=True)
        self.assertFalse(drain.was_stopped)
        self.assertTrue(drain.stop_requested)
        self.assertEqual(logs, ["first"])

    def test_drain_output_late_force_stop_after_exit_sets_request_only(self):
        logs = []
        runner = CommandRunner(on_log=logs.append)

        class DelayedSentinelStdout:
            def __init__(self):
                self._stage = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._stage == 0:
                    self._stage = 1
                    return "first\n"
                if self._stage == 1:
                    self._stage = 2
                    time.sleep(0.2)
                raise StopIteration

        class FakePopen:
            def __init__(self):
                self.stdout = DelayedSentinelStdout()

            def poll(self):
                return 0

        def request_later():
            time.sleep(0.05)
            runner.force_stop()

        threading.Thread(target=request_later, daemon=True).start()
        drain = runner._drain_output(FakePopen(), allow_soft_stop=True)
        self.assertFalse(drain.was_stopped)
        self.assertTrue(drain.stop_requested)
        self.assertEqual(logs, ["first"])

    def test_run_stream_uses_start_new_session_on_unix(self):
        """Unix launch should use start_new_session instead of preexec_fn."""
        logs = []
        runner = CommandRunner(on_log=logs.append)

        fake_proc = mock.Mock()
        fake_proc.wait.return_value = 0
        fake_proc.poll.return_value = 0
        fake_proc.stdout.close.return_value = None

        with mock.patch("mouth_track_gui.runner.sys.platform", "linux"):
            with mock.patch.object(
                runner,
                "_drain_output",
                return_value=DrainResult(
                    was_stopped=False, stop_requested=False,
                ),
            ):
                with mock.patch("mouth_track_gui.runner.subprocess.Popen", return_value=fake_proc) as popen_mock:
                    result = runner.run_stream([sys.executable, "-c", "print('ok')"])

        self.assertEqual(result.returncode, 0)
        self.assertFalse(result.was_stopped)
        kwargs = popen_mock.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("preexec_fn", kwargs)


if __name__ == "__main__":
    unittest.main()
