#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_emotion_audio.py

Lightweight, real-time (low-latency) audio-only emotion estimator for live avatar / lip-sync systems.

- Dependencies: numpy only
- Output emotions (5): neutral / happy / angry / sad / excited
- Designed to be called repeatedly with short audio chunks from a microphone stream.
- Includes smoothing + hysteresis to avoid flicker.
- Includes optional "hint" bias injection for hybrid (audio + external correction) setups.

This is NOT a scientific emotion classifier. It's a pragmatic heuristic tuned for:
- voiced/unvoiced gating
- arousal proxy from loudness/brightness
- rough valence proxy from pitch/brightness patterns
- stability/hold to avoid rapid toggling
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


# -----------------------------
# Helpers
# -----------------------------
def _db(x: float, floor: float = 1e-12) -> float:
    x = max(float(x), floor)
    return 20.0 * float(np.log10(x))


def _clamp(x: float, a: float, b: float) -> float:
    return float(max(a, min(b, x)))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    s = np.sum(e)
    if s <= 0:
        return np.ones_like(z) / float(len(z))
    return e / s


def _one_pole(prev: float, x: float, alpha: float) -> float:
    return float(prev + alpha * (x - prev))


def _acf_pitch_hz(x: np.ndarray, sr: int, fmin: float = 70.0, fmax: float = 400.0) -> float:
    """
    Very simple pitch estimator using autocorrelation peak.
    Returns 0 if unreliable.
    """
    if x.size < 256:
        return 0.0
    x = x.astype(np.float32)
    x = x - float(np.mean(x))
    # energy gate
    e = float(np.sqrt(np.mean(x * x) + 1e-12))
    if e < 1e-4:
        return 0.0

    # compute acf
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    acf0 = float(acf[0] + 1e-12)
    acf = acf / acf0

    # lag bounds
    lag_min = int(sr / float(fmax))
    lag_max = int(sr / float(fmin))
    if lag_max <= lag_min + 2:
        return 0.0

    seg = acf[lag_min:lag_max]
    if seg.size < 3:
        return 0.0

    idx = int(np.argmax(seg))
    peak = float(seg[idx])
    lag = lag_min + idx
    if peak < 0.25:
        return 0.0
    return float(sr / lag)


def _spectral_centroid(x: np.ndarray, sr: int) -> float:
    """
    centroid in Hz
    """
    if x.size < 256:
        return 0.0
    x = x.astype(np.float32)
    w = np.hanning(x.size).astype(np.float32)
    X = np.fft.rfft(x * w)
    mag = np.abs(X) + 1e-12
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    return float((freqs * mag).sum() / mag.sum())


def _zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = x.astype(np.float32)
    s = np.sign(x)
    s[s == 0] = 1
    return float(np.mean(s[1:] != s[:-1]))


# -----------------------------
# Public API
# -----------------------------
EMOTIONS = ("neutral", "happy", "angry", "sad", "excited")


@dataclass
class Hint:
    label: str
    strength: float  # 0..1
    ttl_sec: float


class RealtimeEmotionAnalyzer:
    """
    Realtime emotion analyzer (audio-only).

    Call update(x) repeatedly with short chunks (e.g. 10-50ms to 100ms).
    It maintains internal smoothing + hysteresis state.

    Parameters (tunable):
      sr: sample rate
      smooth_alpha: smoothing for features (0..1). Larger => faster response.
      min_hold_sec: minimum time to keep a decided label
      cand_stable_sec: candidate must persist this long before switching
      switch_margin: minimum probability margin to switch labels
    """

    def __init__(
        self,
        sr: int = 48000,
        smooth_alpha: float = 0.25,
        min_hold_sec: float = 0.45,
        cand_stable_sec: float = 0.22,
        switch_margin: float = 0.10,
    ):
        self.sr = int(sr)
        self.smooth_alpha = float(smooth_alpha)
        self.min_hold_sec = float(min_hold_sec)
        self.cand_stable_sec = float(cand_stable_sec)
        self.switch_margin = float(switch_margin)

        # internal time counter (sec), advanced by update chunk length
        self._t = 0.0

        # state: current label + when it started
        self.current = "neutral"
        self._current_since = 0.0

        # candidate label + when it started being candidate
        self._candidate: Optional[Tuple[str, float]] = None  # (label, since)

        # hint bias
        self._hint: Optional[Hint] = None
        self._hint_until = 0.0

        # smoothed features
        self._rms_db = -80.0
        self._centroid_hz = 0.0
        self._pitch_hz = 0.0
        self._zcr = 0.0

        # noise floor tracking
        self._noise_db = -80.0
        self._voiced_lp = 0.0

        # configuration thresholds
        self._silence_db = -55.0  # below => unvoiced
        self._voiced_db_margin = 8.0  # above noise by this => voiced
        self._zcr_unvoiced = 0.18

    def set_hint(self, label: str, strength: float = 0.6, ttl_sec: float = 1.0) -> None:
        """
        Optional bias injection (e.g. from face/pose or manual controls).
        """
        label = str(label).strip().lower()
        if label not in EMOTIONS:
            return
        self._hint = Hint(label=label, strength=float(_clamp(strength, 0.0, 1.0)), ttl_sec=float(max(0.0, ttl_sec)))
        self._hint_until = self._t + self._hint.ttl_sec

    def _update_features(self, x: np.ndarray) -> Dict[str, float]:
        x = np.asarray(x, dtype=np.float32).flatten()
        n = int(x.size)
        if n <= 0:
            return {"rms_db": self._rms_db, "centroid_hz": self._centroid_hz, "pitch_hz": self._pitch_hz, "zcr": self._zcr}

        # time advance
        dt = float(n) / float(self.sr)
        self._t += dt

        # raw features
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        rms_db = _db(rms)
        centroid_hz = _spectral_centroid(x, self.sr)
        pitch_hz = _acf_pitch_hz(x, self.sr)
        zcr = _zcr(x)

        # noise tracking (slow)
        # if quiet, track downwards more; if loud, track very slowly upwards
        if rms_db < self._noise_db + 3.0:
            self._noise_db = _one_pole(self._noise_db, rms_db, 0.10)
        else:
            self._noise_db = _one_pole(self._noise_db, rms_db, 0.01)

        # voiced estimate
        voiced_raw = 1.0
        if rms_db < self._silence_db:
            voiced_raw = 0.0
        if (rms_db - self._noise_db) < self._voiced_db_margin:
            voiced_raw = min(voiced_raw, 0.25)
        if zcr > self._zcr_unvoiced:
            voiced_raw = min(voiced_raw, 0.25)
        if pitch_hz <= 0.0:
            voiced_raw = min(voiced_raw, 0.35)

        self._voiced_lp = _one_pole(self._voiced_lp, voiced_raw, 0.35)

        # smooth features (one-pole)
        a = self.smooth_alpha
        self._rms_db = _one_pole(self._rms_db, rms_db, a)
        self._centroid_hz = _one_pole(self._centroid_hz, centroid_hz, a)
        if pitch_hz > 0:
            self._pitch_hz = _one_pole(self._pitch_hz, pitch_hz, a)
        else:
            # decay pitch slowly when absent
            self._pitch_hz = _one_pole(self._pitch_hz, 0.0, 0.08)
        self._zcr = _one_pole(self._zcr, zcr, a)

        return {"rms_db": self._rms_db, "centroid_hz": self._centroid_hz, "pitch_hz": self._pitch_hz, "zcr": self._zcr}

    def _score_emotions(self, f: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Heuristic scoring for 5 emotions.

        - arousal proxy: loudness + brightness (centroid) + voiced
        - anger proxy: high arousal + higher zcr/brightness + lower pitch
        - happiness proxy: medium-high arousal + higher pitch + brightness
        - sadness proxy: low arousal + lower pitch + lower brightness
        - neutral proxy: moderate low arousal / unvoiced
        """
        rms_db = float(f["rms_db"])
        centroid_hz = float(f["centroid_hz"])
        pitch_hz = float(f["pitch_hz"])
        zcr = float(f["zcr"])

        voiced = float(self._voiced_lp >= 0.5)
        # normalize proxies
        loud = _clamp((rms_db + 60.0) / 30.0, 0.0, 1.0)  # -60..-30 -> 0..1
        bright = _clamp((centroid_hz - 600.0) / 2000.0, 0.0, 1.0)  # 600..2600Hz
        pitch_n = _clamp((pitch_hz - 90.0) / 210.0, 0.0, 1.0)  # 90..300Hz
        zcr_n = _clamp((zcr - 0.03) / 0.22, 0.0, 1.0)

        arousal = _clamp(0.55 * loud + 0.30 * bright + 0.15 * (1.0 if voiced else 0.0), 0.0, 1.0)

        # base scores
        s_neutral = 0.55 * (1.0 - arousal) + 0.45 * (0.0 if voiced else 1.0)
        s_excited = 0.70 * arousal + 0.30 * pitch_n
        s_happy = 0.40 * arousal + 0.35 * pitch_n + 0.25 * bright
        s_angry = 0.55 * arousal + 0.25 * bright + 0.20 * zcr_n - 0.20 * pitch_n
        s_sad = 0.65 * (1.0 - arousal) + 0.20 * (1.0 - pitch_n) + 0.15 * (1.0 - bright)

        # unvoiced strongly pushes to neutral
        if not voiced:
            s_neutral += 0.35
            s_sad += 0.10
            s_excited -= 0.15
            s_angry -= 0.15
            s_happy -= 0.10

        scores = np.array([s_neutral, s_happy, s_angry, s_sad, s_excited], dtype=np.float32)

        info = {
            "voiced": float(voiced),
            "arousal": float(arousal),
            "loud": float(loud),
            "bright": float(bright),
            "pitch_n": float(pitch_n),
            "zcr_n": float(zcr_n),
        }
        return scores, info

    def update(self, x: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Update internal state using audio chunk x.
        Returns (label, info).

        info includes:
          rms_db, centroid_hz, pitch_hz, zcr,
          voiced (0/1), arousal, confidence (0..1), probs for each label.
        """
        feats = self._update_features(x)
        scores, extra = self._score_emotions(feats)

        # convert to probabilities
        probs = _softmax(scores)
        idx = int(np.argmax(probs))
        label = EMOTIONS[idx]
        top = float(probs[idx])

        # confidence = top - second
        p_sorted = np.sort(probs)
        second = float(p_sorted[-2]) if p_sorted.size >= 2 else 0.0
        confidence = _clamp(top - second, 0.0, 1.0)

        # apply hint (if active)
        if self._hint is not None and self._t <= self._hint_until:
            try:
                h = self._hint
                # blend by adding bias in logit space
                bias = np.zeros_like(probs, dtype=np.float32)
                bias_idx = EMOTIONS.index(h.label)
                bias[bias_idx] = float(h.strength) * 1.2
                probs2 = _softmax(np.log(probs + 1e-6).astype(np.float32) + bias)
                probs = probs2
                idx = int(np.argmax(probs))
                label = EMOTIONS[idx]
                top = float(probs[idx])
                p_sorted = np.sort(probs)
                second = float(p_sorted[-2]) if p_sorted.size >= 2 else 0.0
                confidence = _clamp(top - second, 0.0, 1.0)
            except Exception:
                pass
        else:
            self._hint = None

        info: Dict[str, float] = {}
        info.update(feats)
        info.update(extra)
        info["confidence"] = float(confidence)
        # also expose probs
        for i, lab in enumerate(EMOTIONS):
            info[f"p_{lab}"] = float(probs[i])

        # -----------------------------
        # Hysteresis / hold
        # -----------------------------
        # If unvoiced, we do NOT switch away from neutral aggressively.
        # (Your runtime will additionally enforce "silence => neutral")
        now = self._t
        if (now - self._current_since) < self.min_hold_sec:
            return self.current, info

        # If current already equals predicted, keep
        if label == self.current:
            self._candidate = None
            return self.current, info

        # Require margin vs current probability
        try:
            cur_idx = EMOTIONS.index(self.current)
            cur_p = float(probs[cur_idx])
        except Exception:
            cur_p = 0.0

        if (top - cur_p) < self.switch_margin:
            # not strong enough to switch
            self._candidate = None
            return self.current, info

        # Candidate stability
        if self._candidate is None or self._candidate[0] != label:
            self._candidate = (label, now)
            return self.current, info

        cand_label, cand_since = self._candidate
        if (now - cand_since) < self.cand_stable_sec:
            return self.current, info

        # Switch!
        self.current = cand_label
        self._current_since = now
        self._candidate = None
        return self.current, info


if __name__ == "__main__":
    # Tiny smoke test: generate synthetic voiced segments
    sr = 48000
    emo = RealtimeEmotionAnalyzer(sr=sr)

    def tone(freq, sec, amp=0.1):
        t = np.arange(int(sr * sec), dtype=np.float32) / sr
        return amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    tests = [
        ("neutral-ish", tone(180, 0.4, 0.04)),
        ("happy-ish", tone(240, 0.4, 0.06)),
        ("excited-ish", tone(320, 0.4, 0.12)),
        ("sad-ish", tone(130, 0.4, 0.03)),
    ]
    for name, sig in tests:
        lab, info = emo.update(sig)
        print(name, "->", lab, "conf", round(info["confidence"], 3))
