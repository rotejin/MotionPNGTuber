import os
import tempfile
import unittest

from workflow_validation import (
    build_workflow_paths,
    format_missing_path_message,
    format_missing_paths_message,
    summarize_named_issues,
    validate_existing_dir,
    validate_existing_file,
)


class WorkflowValidationTests(unittest.TestCase):
    def test_validate_existing_file_returns_helpful_error_for_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, "missing.mp4")
            path, err = validate_existing_file(
                missing,
                empty_message="動画を選択してください。",
                missing_label="動画ファイル",
            )
            self.assertIsNone(path)
            self.assertIn("動画ファイル が見つかりません。", err)
            self.assertIn(missing, err)

    def test_validate_existing_dir_returns_helpful_error_for_empty_input(self):
        path, err = validate_existing_dir(
            "",
            empty_message="mouthフォルダを選択してください。",
            missing_label="mouthフォルダ",
        )
        self.assertIsNone(path)
        self.assertEqual(err, "mouthフォルダを選択してください。")

    def test_build_workflow_paths_prefers_calibrated_when_requested(self):
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "loop.mp4")
            track = os.path.join(td, "mouth_track.npz")
            calibrated = os.path.join(td, "mouth_track_calibrated.npz")
            open(video, "wb").close()
            open(track, "wb").close()
            open(calibrated, "wb").close()

            paths, err = build_workflow_paths(
                video,
                require_track=True,
                require_calibrated=True,
                prefer_calibrated=True,
            )
            self.assertEqual(err, "")
            self.assertIsNotNone(paths)
            assert paths is not None
            self.assertEqual(paths.preferred_track, calibrated)

    def test_build_workflow_paths_reports_missing_track_with_next_step(self):
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "loop.mp4")
            open(video, "wb").close()

            paths, err = build_workflow_paths(video, require_track=True)
            self.assertIsNone(paths)
            self.assertIn("mouth_track.npz が見つかりません。", err)
            self.assertIn("① 解析→キャリブ", err)

    def test_format_missing_path_message_appends_next_step(self):
        msg = format_missing_path_message("mouth_track.npz", "C:/tmp/mouth_track.npz", "次は解析を実行")
        self.assertIn("mouth_track.npz が見つかりません。", msg)
        self.assertIn("C:/tmp/mouth_track.npz", msg)
        self.assertIn("次は解析を実行", msg)

    def test_format_missing_paths_message_lists_all_candidates(self):
        msg = format_missing_paths_message(
            "動画ファイル",
            ["C:/a/loop_mouthless.mp4", "C:/a/loop.mp4"],
            "動画を書き出してください。",
        )
        self.assertIn("動画ファイル が見つかりません。", msg)
        self.assertIn("C:/a/loop_mouthless.mp4", msg)
        self.assertIn("C:/a/loop.mp4", msg)
        self.assertIn("動画を書き出してください。", msg)

    def test_summarize_named_issues_limits_and_appends_hint(self):
        msg = summarize_named_issues(
            "有効な動画セットがありません。",
            [("A", "理由1"), ("B", "理由2"), ("C", "理由3"), ("D", "理由4")],
            tail_hint="ログを確認してください。",
            limit=3,
        )
        self.assertIn("A: 理由1", msg)
        self.assertIn("C: 理由3", msg)
        self.assertIn("…", msg)
        self.assertIn("ログを確認してください。", msg)


if __name__ == "__main__":
    unittest.main()
