from __future__ import annotations

import unittest

from loop_lipsync_runtime_patched_emotion_auto import (
    classify_mouth_level_with_hysteresis,
    resolve_emotion_auto_target,
)


class MouthLevelHysteresisTests(unittest.TestCase):
    def test_closed_state_waits_for_deadband_before_opening(self):
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.33, 0.30, 0.52, "closed"),
            "closed",
        )
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.34, 0.30, 0.52, "closed"),
            "half",
        )

    def test_half_state_waits_for_open_deadband(self):
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.55, 0.30, 0.52, "half"),
            "half",
        )
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.56, 0.30, 0.52, "half"),
            "open",
        )

    def test_open_state_needs_margin_before_falling_back(self):
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.49, 0.30, 0.52, "open"),
            "open",
        )
        self.assertEqual(
            classify_mouth_level_with_hysteresis(0.47, 0.30, 0.52, "open"),
            "half",
        )


class EmotionAutoTargetResolutionTests(unittest.TestCase):
    def test_silence_has_priority_over_confidence_and_voicing(self):
        label, target, reason = resolve_emotion_auto_target(
            "happy",
            {"rms_db": -80.0, "confidence": 0.99, "voiced": 1.0},
            ["Neutral", "Happy"],
            "Neutral",
            silence_db=-65.0,
            min_conf=0.45,
        )
        self.assertEqual((label, target, reason), ("neutral", "Neutral", "silence"))

    def test_unvoiced_holds_current_even_when_label_exists(self):
        label, target, reason = resolve_emotion_auto_target(
            "happy",
            {"rms_db": -20.0, "confidence": 0.99, "voiced": 0.0},
            ["Neutral", "Happy"],
            "Neutral",
            silence_db=-65.0,
            min_conf=0.45,
        )
        self.assertEqual((label, target, reason), (None, None, "unvoiced"))

    def test_low_confidence_holds_current(self):
        label, target, reason = resolve_emotion_auto_target(
            "happy",
            {"rms_db": -20.0, "confidence": 0.20, "voiced": 1.0},
            ["Neutral", "Happy"],
            "Neutral",
            silence_db=-65.0,
            min_conf=0.45,
        )
        self.assertEqual((label, target, reason), (None, None, "low_conf"))

    def test_confident_voiced_label_maps_to_matching_set(self):
        label, target, reason = resolve_emotion_auto_target(
            "happy",
            {"rms_db": -20.0, "confidence": 0.80, "voiced": 1.0},
            ["Neutral", "Happy"],
            "Neutral",
            silence_db=-65.0,
            min_conf=0.45,
        )
        self.assertEqual((label, target, reason), ("happy", "Happy", "label"))


if __name__ == "__main__":
    unittest.main()
