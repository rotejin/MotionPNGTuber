#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_mouth_track.py

face_track_anime_detector.py を複数パラメータでリトライし、
トラック品質（valid率 / confidence / jitter）を採点して最良の npz を採用するラッパー。

追加（超実用拡張）:
- bad segment（invalid/低confidenceが連続する区間）だけを部分再解析して "パッチ当て"（任意、デフォルトON）
  * 全体再解析より速く、長尺動画でも安心して回せる
  * 部分再解析が失敗/悪化した場合は元の結果を保持

注意:
- スコアは万能ではないので、閾値はやや緩め推奨。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from image_io import write_image_file
from python_exec import resolve_python_subprocess_executable


@dataclass
class Metrics:
    valid_rate: float
    mean_conf: float
    p10_conf: float
    jitter_p95: float  # normalized by diag (0..)
    n_frames: int
    n_valid: int

    def passes(self, min_valid_rate: float, min_mean_conf: float, max_jitter_p95: float) -> bool:
        if self.n_frames <= 0:
            return False
        if self.valid_rate < min_valid_rate:
            return False
        if self.mean_conf < min_mean_conf:
            return False
        if self.jitter_p95 > max_jitter_p95:
            return False
        return True


def _quad_centers(quads: np.ndarray) -> np.ndarray:
    # (N,4,2) -> (N,2)
    return quads.mean(axis=1)


def compute_metrics_npz(npz_path: str) -> Metrics:
    npz = np.load(npz_path, allow_pickle=False)
    quads = np.asarray(npz["quad"], dtype=np.float32)
    N = int(quads.shape[0])
    valid = np.asarray(npz.get("valid", np.ones((N,), np.uint8)), dtype=np.uint8) > 0
    conf = np.asarray(npz.get("confidence", np.ones((N,), np.float32)), dtype=np.float32)
    conf = conf[:N]
    w = float(npz.get("w", 0))
    h = float(npz.get("h", 0))
    diag = float(np.hypot(w, h)) if (w and h) else 1.0

    n_valid = int(valid.sum())
    valid_rate = float(n_valid / max(1, N))
    mean_conf = float(conf[valid].mean()) if n_valid > 0 else 0.0
    p10_conf = float(np.percentile(conf[valid], 10)) if n_valid > 0 else 0.0

    # Compute jitter only between consecutive VALID frames to avoid
    # artifacts from hold-filled invalid frames.
    centers = _quad_centers(quads)
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= 2:
        # Compute diff only between consecutive valid frames
        consecutive_mask = np.diff(valid_indices) == 1
        if np.any(consecutive_mask):
            # Get pairs of consecutive valid frames
            pairs_start = valid_indices[:-1][consecutive_mask]
            pairs_end = valid_indices[1:][consecutive_mask]
            diffs = centers[pairs_end] - centers[pairs_start]
            d = np.linalg.norm(diffs, axis=1) / max(1e-6, diag)
            jitter_p95 = float(np.percentile(d, 95))
        else:
            jitter_p95 = 0.0
    else:
        jitter_p95 = 0.0

    return Metrics(
        valid_rate=valid_rate,
        mean_conf=mean_conf,
        p10_conf=p10_conf,
        jitter_p95=jitter_p95,
        n_frames=N,
        n_valid=n_valid,
    )


def score(metrics: Metrics) -> float:
    """
    Larger is better.
    - valid_rate and mean_conf up
    - jitter down
    """
    # clamp
    v = float(np.clip(metrics.valid_rate, 0.0, 1.0))
    mc = float(np.clip(metrics.mean_conf, 0.0, 1.0))
    p10 = float(np.clip(metrics.p10_conf, 0.0, 1.0))
    j = float(max(0.0, metrics.jitter_p95))
    # weights: valid is most important, then conf. jitter acts as penalty.
    return (2.2 * v + 1.0 * mc + 0.6 * p10) - (1.8 * j)


def backup_copy_if_exists(path: str) -> str | None:
    """Best-effort backup (copy) for small outputs like .npz."""
    try:
        if os.path.isfile(path):
            ts = time.strftime("%Y%m%d-%H%M%S")
            bak = f"{path}.bak.{ts}"
            shutil.copy2(path, bak)
            return bak
    except Exception:
        return None
    return None


def save_best_debug_outputs(debug_dir: str, best_npz: str, report: dict) -> None:
    """Write debug JSON and a simple PNG plot for the chosen track."""
    os.makedirs(debug_dir, exist_ok=True)
    json_path = os.path.join(debug_dir, "auto_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # plot from npz (confidence/valid/jitter) without matplotlib
    z = np.load(best_npz, allow_pickle=False)
    conf = np.asarray(z.get("confidence", []), dtype=np.float32)
    valid = np.asarray(z.get("valid", []), dtype=np.uint8)
    quad = np.asarray(z.get("quad", []), dtype=np.float32)
    if quad.ndim == 3 and quad.shape[1:] == (4, 2):
        centers = quad.mean(axis=1)
        d = np.linalg.norm(np.diff(centers, axis=0), axis=1).astype(np.float32)
        jitter = np.concatenate([np.zeros((1,), np.float32), d])
    else:
        jitter = np.zeros((int(max(conf.size, valid.size)),), np.float32)

    W, H = 1200, 360
    M = 40
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.rectangle(img, (M, M), (W - M, H - M), (220, 220, 220), 1)
    cv2.putText(img, "auto best metrics", (M, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    series = [
        ("confidence", conf),
        ("valid", valid.astype(np.float32)),
        ("jitter_px", jitter),
    ]
    colors = [(30, 30, 30), (0, 128, 255), (255, 0, 128)]

    def norm_y(name: str, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, np.float32)
        if y.size == 0:
            return y
        if name == "valid":
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.quantile(y, 0.05))
            hi = float(np.quantile(y, 0.95))
            if abs(hi - lo) < 1e-6:
                hi = lo + 1.0
        y = (y - lo) / (hi - lo)
        return np.clip(y, 0.0, 1.0)

    for k, (name, y) in enumerate(series):
        if y.size <= 1:
            continue
        y = norm_y(name, y)
        xs = np.linspace(M, W - M, num=y.size).astype(np.int32)
        ys = (H - M - (y * (H - 2 * M))).astype(np.int32)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, colors[k % len(colors)], 2, cv2.LINE_AA)
        cv2.putText(img, name, (M + 10 + 180 * k, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colors[k % len(colors)], 2, cv2.LINE_AA)

    png_path = os.path.join(debug_dir, "auto_best.png")
    if not write_image_file(png_path, img):
        raise RuntimeError(f"Failed to save debug image: {png_path}")



def run_detector(detector_py: str, args_map: Dict[str, Optional[str]]) -> int:
    cmd: List[str] = [resolve_python_subprocess_executable(), detector_py]
    for k, v in args_map.items():
        if v is None:
            continue
        cmd += [k, str(v)]
    return subprocess.call(cmd)


def _extract_subvideo(video_path: str, start_f: int, end_f: int, out_path: str) -> bool:
    """Extract [start_f, end_f] inclusive frames to out_path (mp4).

    NOTE: We use sequential read-and-discard instead of cv2.CAP_PROP_POS_FRAMES seek
    because seek is unreliable for non-keyframes in H.264/H.265 encoded videos.
    This is slower but guarantees frame accuracy for segment repair.
    """
    start_f = int(max(0, start_f))
    end_f = int(max(start_f, end_f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    if w <= 0 or h <= 0:
        cap.release()
        return False

    # mp4v is widely available (not perfect but pragmatic)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        cap.release()
        return False

    # Skip frames up to start_f using grab() (faster than read() as it skips decoding)
    for _ in range(start_f):
        if not cap.grab():
            vw.release()
            cap.release()
            return False

    # Read and write the target frames
    ok_any = False
    for _ in range(end_f - start_f + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        vw.write(frame)
        ok_any = True

    vw.release()
    cap.release()
    return ok_any and os.path.isfile(out_path)


def _find_bad_segments(valid: np.ndarray, conf: Optional[np.ndarray], bad_conf_thr: float,
                       max_len: int, min_len: int = 2) -> List[Tuple[int, int]]:
    """
    Return list of [s,e] inclusive segments where:
    - invalid OR conf < thr
    - contiguous
    - length between min_len..max_len
    """
    N = len(valid)
    bad = ~valid
    if conf is not None:
        bad = bad | (conf < float(bad_conf_thr))

    segs: List[Tuple[int, int]] = []
    i = 0
    while i < N:
        if not bad[i]:
            i += 1
            continue
        s = i
        while i < N and bad[i]:
            i += 1
        e = i - 1
        L = e - s + 1
        if L >= min_len and L <= max_len:
            segs.append((s, e))
    return segs


def _stitch_segment(base_npz: str, seg_npz: str, dst_npz: str, offset: int) -> None:
    """Replace [offset:offset+len(seg)) in base with seg arrays. Keep other keys from base."""
    b = np.load(base_npz, allow_pickle=False)
    s = np.load(seg_npz, allow_pickle=False)
    b_quad = np.asarray(b["quad"], dtype=np.float32).copy()
    b_valid = np.asarray(b.get("valid", np.ones((b_quad.shape[0],), np.uint8)), dtype=np.uint8).copy()
    b_conf = np.asarray(b.get("confidence", np.ones((b_quad.shape[0],), np.float32)), dtype=np.float32).copy()

    s_quad = np.asarray(s["quad"], dtype=np.float32)
    s_valid = np.asarray(s.get("valid", np.ones((s_quad.shape[0],), np.uint8)), dtype=np.uint8)
    s_conf = np.asarray(s.get("confidence", np.ones((s_quad.shape[0],), np.float32)), dtype=np.float32)

    N = b_quad.shape[0]
    M = s_quad.shape[0]
    off = int(max(0, min(offset, N)))
    m2 = int(min(M, N - off))
    if m2 <= 0:
        # nothing to do
        np.savez_compressed(dst_npz, **{k: b[k] for k in b.files})
        return

    b_quad[off:off + m2] = s_quad[:m2]
    b_valid[off:off + m2] = s_valid[:m2]
    b_conf[off:off + m2] = s_conf[:m2]

    out = {}
    for k in b.files:
        if k == "quad":
            out[k] = b_quad.astype(np.float32)
        elif k == "valid":
            out[k] = b_valid.astype(np.uint8)
        elif k == "confidence":
            out[k] = b_conf.astype(np.float32)
        else:
            out[k] = b[k]
    np.savez_compressed(dst_npz, **out)


def _segment_repair(video: str, detector_py: str, best_npz: str,
                    base_args: Dict[str, Optional[str]],
                    bad_conf_thr: float, bad_max_len: int, pad_frames: int,
                    max_segments: int = 6) -> Optional[str]:
    """
    Try to patch bad segments by re-running detector on extracted subvideos.
    Return path to repaired npz if improved, else None.
    """
    npz = np.load(best_npz, allow_pickle=False)
    quads = np.asarray(npz["quad"], dtype=np.float32)
    N = int(quads.shape[0])
    valid = np.asarray(npz.get("valid", np.ones((N,), np.uint8)), dtype=np.uint8) > 0
    conf = np.asarray(npz.get("confidence", np.ones((N,), np.float32)), dtype=np.float32)

    segs = _find_bad_segments(valid, conf, bad_conf_thr=bad_conf_thr, max_len=bad_max_len, min_len=2)
    if not segs:
        return None

    # prioritize longer / earlier segments (rough)
    segs = sorted(segs, key=lambda t: (-(t[1]-t[0]+1), t[0]))[:max_segments]

    tmpdir = tempfile.mkdtemp(prefix="auto_track_seg_")
    work_npz = os.path.join(tmpdir, "work.npz")
    shutil.copy2(best_npz, work_npz)

    try:
        for (s, e) in segs:
            ss = max(0, s - pad_frames)
            ee = min(N - 1, e + pad_frames)

            sub_mp4 = os.path.join(tmpdir, f"seg_{ss}_{ee}.mp4")
            if not _extract_subvideo(video, ss, ee, sub_mp4):
                continue

            seg_out = os.path.join(tmpdir, f"seg_{ss}_{ee}.npz")

            # aggressive args for segment
            seg_args = dict(base_args)
            seg_args["--video"] = sub_mp4
            seg_args["--out"] = seg_out
            seg_args["--stride"] = "1"
            # relax smoothing skip to avoid "smooth skipped"
            seg_args["--valid-min-ratio"] = "0.0"
            seg_args["--valid-min-count"] = "0"
            # slightly lower min_conf if present
            if "--min-conf" in seg_args:
                try:
                    seg_args["--min-conf"] = str(max(0.15, float(seg_args["--min-conf"]) - 0.15))
                except Exception:
                    seg_args["--min-conf"] = str(0.35)

            rc = run_detector(detector_py, seg_args)
            if rc != 0 or (not os.path.isfile(seg_out)):
                continue

            # stitch only the center (original bad region), not including padding
            off = ss
            # We'll replace entire extracted range to benefit smoothing; safer than only [s:e]
            dst_npz = os.path.join(tmpdir, "work2.npz")
            _stitch_segment(work_npz, seg_out, dst_npz, offset=off)
            os.replace(dst_npz, work_npz)

        # compare metrics
        before = compute_metrics_npz(best_npz)
        after = compute_metrics_npz(work_npz)
        if score(after) >= score(before) + 0.02:  # require some improvement
            fd, repaired = tempfile.mkstemp(prefix="auto_track_seg_repaired_", suffix=".npz")
            os.close(fd)
            try:
                shutil.copy2(work_npz, repaired)
            except Exception:
                try:
                    os.unlink(repaired)
                except OSError:
                    pass
                raise
            return repaired
        return None
    finally:
        # keep tmpdir only if improved? we already copied out if improved.
        shutil.rmtree(tmpdir, ignore_errors=True)




def get_video_info(video_path: str) -> Tuple[float, int]:
    """Return (fps, frame_count). Best-effort (can be 0 on some containers)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 30.0, 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, n_frames


def _extract_probe_video(video_path: str, start_f: int, count_f: int, out_path: str) -> bool:
    """Fast-ish probe extractor. Frame accuracy is not critical (ranking only)."""
    start_f = int(max(0, start_f))
    count_f = int(max(1, count_f))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if w <= 0 or h <= 0:
        cap.release()
        return False

    # Try seek (best-effort). If it fails, fall back to grab().
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_f))
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        cap.release()
        return False

    ok_any = False
    for _ in range(count_f):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        vw.write(frame)
        ok_any = True

    vw.release()
    cap.release()
    return ok_any and os.path.isfile(out_path)


def _make_probe_video(video_path: str, tmpdir: str, fps: float, n_frames: int) -> Optional[str]:
    """Create a short probe clip for ranking candidates. Returns path or None."""
    if n_frames <= 0:
        return None

    # 6 seconds around the middle (clamped)
    probe_sec = 6.0
    count_f = int(max(90, min(int(fps * probe_sec), max(90, n_frames // 5))))
    start_f = int(max(0, min(n_frames - count_f, int(n_frames * 0.50 - count_f * 0.50))))

    out_path = os.path.join(tmpdir, f"probe_{start_f}_{start_f+count_f-1}.mp4")

    # If ffmpeg exists, prefer it (much faster on long GOP videos)
    ff = shutil.which("ffmpeg")
    if ff:
        try:
            start_t = float(start_f) / float(max(1e-6, fps))
            dur_t = float(count_f) / float(max(1e-6, fps))
            cmd = [
                ff, "-y",
                "-ss", f"{start_t:.3f}",
                "-t", f"{dur_t:.3f}",
                "-i", video_path,
                "-an",
                # Re-encode quickly for compatibility
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                out_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            if os.path.isfile(out_path) and os.path.getsize(out_path) > 1024:
                return out_path
        except Exception:
            pass

    # Fallback to OpenCV
    if _extract_probe_video(video_path, start_f, count_f, out_path):
        return out_path
    return None


def _build_candidates_fixed(args, base: Dict[str, Optional[str]]) -> List[Tuple[str, Dict[str, Optional[str]]]]:
    """Original-ish candidate list (trimmed to max_tries)."""
    candidates: List[Tuple[str, Dict[str, Optional[str]]]] = []

    def add(tag: str, overrides: Dict[str, Optional[str]]) -> None:
        m = dict(base)
        m.update(overrides)
        candidates.append((tag, m))

    add("base", {})

    if int(args.stride) != 1:
        add("stride1", {"--stride": "1"})

    # lower min_conf
    add("minconf-0.40", {"--min-conf": str(min(0.40, float(args.min_conf)))})
    add("minconf-0.30", {"--min-conf": "0.30"})

    # det_scale tweaks
    ds = float(args.det_scale)
    add("detscale-0.90", {"--det-scale": f"{max(0.6, ds * 0.90):.3f}"})
    add("detscale-1.10", {"--det-scale": f"{min(1.8, ds * 1.10):.3f}"})

    # force smoothing + stride1 + minconf
    add("forceSmooth", {
        "--stride": "1",
        "--min-conf": "0.30",
        "--valid-min-ratio": "0.0",
        "--valid-min-count": "0",
    })

    return candidates[: max(1, int(args.max_tries))]


def _build_candidates_adaptive(
    args,
    base: Dict[str, Optional[str]],
    probe_base: Metrics,
) -> List[Tuple[str, Dict[str, Optional[str]]]]:
    """Adaptive candidate plan based on a cheap probe of the base settings.

    Goal:
    - If probe suggests detection is weak -> try more detection-friendly params earlier
    - If probe suggests jitter is main issue -> try stronger smoothing / stricter conf
    - If probe already looks good -> keep tries small (avoid waste)
    """
    candidates: List[Tuple[str, Dict[str, Optional[str]]]] = []
    seen: set[str] = set()

    def add(tag: str, overrides: Dict[str, Optional[str]]) -> None:
        if tag in seen:
            return
        m = dict(base)
        m.update(overrides)
        candidates.append((tag, m))
        seen.add(tag)

    ds = float(args.det_scale)
    stride = int(args.stride)

    add("base", {})

    # If already good enough on probe, don't explode tries.
    good_enough = probe_base.passes(
        min_valid_rate=float(args.min_valid_rate),
        min_mean_conf=float(args.min_mean_conf),
        max_jitter_p95=float(args.max_jitter_p95),
    )

    if good_enough:
        if stride != 1:
            add("stride1", {"--stride": "1"})
        # Tiny det_scale nudge as a cheap alternative
        add("detscale-1.10", {"--det-scale": f"{min(1.8, ds * 1.10):.3f}"})
        return candidates[: max(1, int(args.max_tries))]

    # --- Detection rescue path ---
    det_weak = (probe_base.valid_rate < float(args.min_valid_rate)) or (probe_base.mean_conf < float(args.min_mean_conf))
    jittery = (probe_base.jitter_p95 > float(args.max_jitter_p95))

    if det_weak:
        if stride != 1:
            add("stride1", {"--stride": "1"})
        add("detscale-1.10", {"--det-scale": f"{min(2.0, ds * 1.10):.3f}"})
        add("minconf-0.40", {"--min-conf": str(min(0.40, float(args.min_conf)))})
        add("minconf-0.30", {"--min-conf": "0.30"})
        add("detscale-1.25", {"--det-scale": f"{min(2.0, ds * 1.25):.3f}"})
        add("forceSmooth", {
            "--stride": "1",
            "--min-conf": "0.30",
            "--valid-min-ratio": "0.0",
            "--valid-min-count": "0",
        })
        # Last-resort rescue (only if room)
        add("rescue-strong", {
            "--stride": "1",
            "--det-scale": f"{min(2.0, ds * 1.40):.3f}",
            "--min-conf": "0.20",
            "--smooth-cutoff": "2.0",
            "--valid-min-ratio": "0.0",
            "--valid-min-count": "0",
        })

    # --- Jitter rescue path ---
    if jittery and not det_weak:
        # Stronger smoothing (lower cutoff) + optionally stricter confidence
        add("smooth-2.0", {"--smooth-cutoff": "2.0", "--valid-min-ratio": "0.0", "--valid-min-count": "0"})
        add("smooth-1.5", {"--smooth-cutoff": "1.5", "--valid-min-ratio": "0.0", "--valid-min-count": "0"})
        # Sometimes jitter is false positives; try stricter min-conf
        try:
            mc_up = min(0.75, float(args.min_conf) + 0.10)
        except Exception:
            mc_up = 0.60
        add("minconf-up", {"--min-conf": f"{mc_up:.2f}"})

    # Add one det_scale down as a sanity check (can help on over-zoom)
    add("detscale-0.90", {"--det-scale": f"{max(0.6, ds * 0.90):.3f}"})

    return candidates[: max(1, int(args.max_tries))]


def main() -> int:
    ap = argparse.ArgumentParser(description="Auto-retry mouth tracking with quality scoring.")
    ap.add_argument("--video", required=True, help="input video")
    ap.add_argument("--out", required=True, help="final output npz")
    ap.add_argument("--detector", default="face_track_anime_detector.py", help="path to detector script")
    # base detector args
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--quality", default="custom")
    ap.add_argument("--det-scale", dest="det_scale", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pad", type=float, default=2.1)
    ap.add_argument("--sprite-aspect", type=float, default=1.0)
    ap.add_argument("--quad-mode", default="hybrid")
    ap.add_argument("--min-mouth-w-ratio", type=float, default=0.12)
    ap.add_argument("--min-mouth-w-px", type=float, default=16.0)
    ap.add_argument("--min-conf", dest="min_conf", type=float, default=0.5)
    ap.add_argument("--smooth-cutoff", type=float, default=3.0)
    ap.add_argument("--max-angle-change", type=float, default=15.0)
    ap.add_argument("--valid-min-ratio", type=float, default=0.05)
    ap.add_argument("--valid-min-count", type=int, default=5)
    ap.add_argument("--ref-sprite-w", type=int, default=128)
    ap.add_argument("--ref-sprite-h", type=int, default=85)
    # NOTE: auto script uses this as a DEBUG DIRECTORY (not detector's mp4 path)
    ap.add_argument("--debug", default="")

    # scoring thresholds
    ap.add_argument("--min-valid-rate", type=float, default=0.70)
    ap.add_argument("--min-mean-conf", type=float, default=0.50)
    ap.add_argument("--max-jitter-p95", type=float, default=0.030)

    # candidate generation
    ap.add_argument("--max-tries", type=int, default=8, help="max candidate runs")
    ap.add_argument("--early-stop", action="store_true", help="stop early if thresholds pass")

    # segment repair
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--segment-repair", action="store_true", help="enable segment repair (default)")
    g.add_argument("--no-segment-repair", action="store_true", help="disable segment repair")
    ap.set_defaults(segment_repair=True)

    ap.add_argument("--bad-conf-thr", type=float, default=0.35, help="frames with conf<thr treated as bad for segment repair")
    ap.add_argument("--bad-max-frac", type=float, default=0.25, help="if bad frames fraction > this, skip segment repair")
    ap.add_argument("--bad-max-len", type=int, default=45, help="max length (frames) per bad segment to repair")
    ap.add_argument("--seg-pad", type=int, default=10, help="padding frames around bad segment")
    ap.add_argument("--max-segments", type=int, default=6, help="max segments to repair")
    args = ap.parse_args()

    candidate_reports: List[Dict[str, object]] = []

    # handle mutually exclusive default
    if args.no_segment_repair:
        args.segment_repair = False

    if not os.path.isfile(args.video):
        print(f"[error] video not found: {args.video}", file=sys.stderr)
        return 2

    detector_py = args.detector
    if not os.path.isfile(detector_py):
        here = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(here, detector_py)
        if os.path.isfile(p):
            detector_py = p
    if not os.path.isfile(detector_py):
        print(f"[error] detector not found: {args.detector}", file=sys.stderr)
        return 2

    # Base args to detector (do NOT forward --debug; auto uses it as a directory)
    base: Dict[str, Optional[str]] = {
        "--video": args.video,
        "--out": None,  # filled per run
        "--device": args.device,
        "--quality": args.quality,
        "--det-scale": f"{args.det_scale}",
        "--stride": f"{args.stride}",
        "--pad": f"{args.pad}",
        "--sprite-aspect": f"{args.sprite_aspect}",
        "--quad-mode": f"{args.quad_mode}",
        "--min-mouth-w-ratio": f"{args.min_mouth_w_ratio}",
        "--min-mouth-w-px": f"{args.min_mouth_w_px}",
        "--min-conf": f"{args.min_conf}",
        "--smooth-cutoff": f"{args.smooth_cutoff}",
        "--max-angle-change": f"{args.max_angle_change}",
        "--valid-min-ratio": f"{args.valid_min_ratio}",
        "--valid-min-count": f"{args.valid_min_count}",
        "--ref-sprite-w": f"{args.ref_sprite_w}",
        "--ref-sprite-h": f"{args.ref_sprite_h}",
    }

    tmpdir = tempfile.mkdtemp(prefix="auto_mouth_track_")
    best_score = -1e9
    best_path: Optional[str] = None
    best_metrics: Optional[Metrics] = None
    best_tag: Optional[str] = None
    segment_repair_tmp: Optional[str] = None

    probe_enabled = False
    probe_video: Optional[str] = None
    probe_base_metrics: Optional[Metrics] = None
    probe_by_tag: Dict[str, Dict[str, object]] = {}

    try:
        # Decide whether to enable probe mode (no new CLI; only for long videos + multiple tries)
        fps, n_frames = get_video_info(args.video)
        long_enough = (n_frames >= int(max(600, fps * 20.0)))  # ~20s+
        probe_enabled = bool(long_enough and int(args.max_tries) >= 4)

        if probe_enabled:
            probe_video = _make_probe_video(args.video, tmpdir, fps=fps, n_frames=n_frames)
            if probe_video:
                # Probe base once to decide the candidate plan
                base_probe_out = os.path.join(tmpdir, "probe_base.npz")
                base_probe_args = dict(base)
                base_probe_args["--video"] = probe_video
                base_probe_args["--out"] = base_probe_out
                print(f"[auto] probe enabled (clip={os.path.basename(probe_video)})", flush=True)
                rc = run_detector(detector_py, base_probe_args)
                if rc == 0 and os.path.isfile(base_probe_out):
                    probe_base_metrics = compute_metrics_npz(base_probe_out)
                    print(f"[auto] probe(base): valid_rate={probe_base_metrics.valid_rate:.3f}, "
                          f"mean_conf={probe_base_metrics.mean_conf:.3f}, jitter_p95={probe_base_metrics.jitter_p95:.4f}", flush=True)
                else:
                    probe_enabled = False
            else:
                probe_enabled = False

        # Build candidate plan
        if probe_enabled and probe_base_metrics is not None:
            candidates = _build_candidates_adaptive(args, base, probe_base_metrics)
        else:
            candidates = _build_candidates_fixed(args, base)

        # Probe-rank candidates (cheap) to reduce wasted full runs on long videos
        candidates_ranked = list(candidates)
        if probe_enabled and probe_video:
            probe_scores: List[Tuple[float, int]] = []
            for i, (tag, argmap) in enumerate(candidates):
                out_probe = os.path.join(tmpdir, f"probe_{i:02d}_{tag}.npz")
                pmap = dict(argmap)
                pmap["--video"] = probe_video
                pmap["--out"] = out_probe
                rc = run_detector(detector_py, pmap)
                if rc != 0 or (not os.path.isfile(out_probe)):
                    probe_scores.append((-1e9, i))
                    probe_by_tag[tag] = {"ok": False}
                    continue
                met = compute_metrics_npz(out_probe)
                sc = score(met)
                probe_scores.append((float(sc), i))
                probe_by_tag[tag] = {
                    "ok": True,
                    "score": float(sc),
                    "metrics": {
                        "n_frames": int(met.n_frames),
                        "n_valid": int(met.n_valid),
                        "valid_rate": float(met.valid_rate),
                        "mean_conf": float(met.mean_conf),
                        "p10_conf": float(met.p10_conf),
                        "jitter_p95": float(met.jitter_p95),
                    },
                }

            probe_scores.sort(key=lambda t: t[0], reverse=True)
            candidates_ranked = [candidates[i] for (_, i) in probe_scores]
            print("[auto] probe ranking:", flush=True)
            for rank, (tag, _) in enumerate(candidates_ranked[: min(5, len(candidates_ranked))], start=1):
                info = probe_by_tag.get(tag, {})
                if info.get("ok"):
                    print(f"[auto]   {rank}) {tag}  score={info.get('score'):.4f}", flush=True)
                else:
                    print(f"[auto]   {rank}) {tag}  (probe failed)", flush=True)

        # Progress (TTY only; no new CLI flags)
        use_tqdm = sys.stderr.isatty()
        try:
            from tqdm import tqdm  # type: ignore
        except Exception:
            tqdm = None  # type: ignore
        it_cand = list(enumerate(candidates_ranked))
        if use_tqdm and tqdm is not None and len(it_cand) > 1:
            it_cand = tqdm(it_cand, total=len(it_cand), desc="AutoTrack", unit="try", dynamic_ncols=True)

        for i, (tag, argmap) in it_cand:
            out_tmp = os.path.join(tmpdir, f"cand_{i:02d}_{tag}.npz")
            argmap = dict(argmap)
            argmap["--video"] = args.video
            argmap["--out"] = out_tmp

            print(f"[auto] run {i+1}/{len(candidates_ranked)} tag={tag}", flush=True)
            rc = run_detector(detector_py, argmap)
            if rc != 0:
                print(f"[auto]   -> detector failed rc={rc}", flush=True)
                continue
            if not os.path.isfile(out_tmp):
                print(f"[auto]   -> npz not found: {out_tmp}", flush=True)
                continue

            met = compute_metrics_npz(out_tmp)
            sc = score(met)
            print(f"[auto] metrics: valid={met.n_valid}/{met.n_frames} ({met.valid_rate:.3f}), "
                  f"mean_conf={met.mean_conf:.3f}, p10_conf={met.p10_conf:.3f}, jitter_p95={met.jitter_p95:.4f}", flush=True)
            print(f"[auto] score: {sc:.4f}", flush=True)

            entry: Dict[str, object] = {
                "tag": tag,
                "score": float(sc),
                "metrics": {
                    "n_frames": int(met.n_frames),
                    "n_valid": int(met.n_valid),
                    "valid_rate": float(met.valid_rate),
                    "mean_conf": float(met.mean_conf),
                    "p10_conf": float(met.p10_conf),
                    "jitter_p95": float(met.jitter_p95),
                },
                "npz": out_tmp,
            }
            if probe_enabled and tag in probe_by_tag:
                entry["probe"] = probe_by_tag[tag]
            candidate_reports.append(entry)

            if sc > best_score:
                best_score = sc
                best_path = out_tmp
                best_metrics = met
                best_tag = tag

            passes = met.passes(args.min_valid_rate, args.min_mean_conf, args.max_jitter_p95)
            if passes and (args.early_stop or probe_enabled):
                # In probe mode we stop as soon as "good enough" to avoid waste on long videos.
                why = "early-stop" if args.early_stop else "probe-mode auto-stop"
                print(f"[auto] [OK] passes thresholds on tag={tag} -> stop ({why})", flush=True)
                best_path = out_tmp
                best_metrics = met
                best_tag = tag
                break

        if best_path is None:
            print("[error] no candidate succeeded", file=sys.stderr)
            return 3

        # Optional: segment repair (patch bad runs without full rerun)
        if args.segment_repair:
            # quick bad-frame fraction check
            npz = np.load(best_path, allow_pickle=False)
            valid = np.asarray(npz.get("valid", np.ones((npz["quad"].shape[0],), np.uint8)), dtype=np.uint8) > 0
            conf = np.asarray(npz.get("confidence", np.ones((npz["quad"].shape[0],), np.float32)), dtype=np.float32)
            bad = (~valid) | (conf < float(args.bad_conf_thr))
            frac = float(bad.mean()) if len(bad) else 1.0
            if frac <= float(args.bad_max_frac):
                repaired = _segment_repair(
                    video=args.video,
                    detector_py=detector_py,
                    best_npz=best_path,
                    base_args=base,
                    bad_conf_thr=float(args.bad_conf_thr),
                    bad_max_len=int(args.bad_max_len),
                    pad_frames=int(args.seg_pad),
                    max_segments=int(args.max_segments),
                )
                if repaired and os.path.isfile(repaired):
                    segment_repair_tmp = repaired
                    met2 = compute_metrics_npz(repaired)
                    sc2 = score(met2)
                    if sc2 > best_score:
                        print(f"[auto] [OK] segment repair improved: {best_score:.4f} -> {sc2:.4f}", flush=True)
                        best_score = sc2
                        best_metrics = met2
                        best_tag = f"{best_tag}+segRepair"
                        best_path = repaired
                    else:
                        print(f"[auto] segment repair did not improve (kept original).", flush=True)
            else:
                print(f"[auto] segment repair skipped (bad_frac={frac:.3f} > {args.bad_max_frac}).", flush=True)

        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        backup_copy_if_exists(args.out)
        shutil.copy2(best_path, args.out)
        print(f"[auto] [OK] best tag={best_tag} score={best_score:.4f} -> saved: {args.out}", flush=True)

        # Minimal warning only when something looks off (avoid noisy UX)
        if best_metrics and best_metrics.valid_rate < 0.80:
            print(f"[warn] tracking quality still low after auto selection: valid_rate={best_metrics.valid_rate:.2f}. "
                  f"Consider det-scale↑ or stride↓ for this video.", flush=True)

        # Debug outputs (only when --debug is set)
        if args.debug:
            try:
                report = {
                    "video": os.path.abspath(args.video),
                    "out": os.path.abspath(args.out),
                    "probe_enabled": bool(probe_enabled),
                    "probe_clip": os.path.basename(probe_video) if probe_video else None,
                    "best": {
                        "tag": best_tag,
                        "score": float(best_score),
                        "metrics": {
                            "n_frames": int(best_metrics.n_frames) if best_metrics else None,
                            "n_valid": int(best_metrics.n_valid) if best_metrics else None,
                            "valid_rate": float(best_metrics.valid_rate) if best_metrics else None,
                            "mean_conf": float(best_metrics.mean_conf) if best_metrics else None,
                            "p10_conf": float(best_metrics.p10_conf) if best_metrics else None,
                            "jitter_p95": float(best_metrics.jitter_p95) if best_metrics else None,
                        },
                    },
                    "candidates": candidate_reports,
                }
                save_best_debug_outputs(args.debug, best_path, report)
                print(f"[debug] auto metrics: {os.path.join(args.debug, 'auto_metrics.json')}", flush=True)
            except Exception as e:
                print(f"[warn] failed to write auto debug outputs: {e}", flush=True)

        if best_metrics:
            print(f"[auto] best metrics: valid_rate={best_metrics.valid_rate:.3f}, mean_conf={best_metrics.mean_conf:.3f}, "
                  f"p10_conf={best_metrics.p10_conf:.3f}, jitter_p95={best_metrics.jitter_p95:.4f}", flush=True)
        return 0

    finally:
        if segment_repair_tmp:
            try:
                if os.path.isfile(segment_repair_tmp):
                    os.unlink(segment_repair_tmp)
            except OSError:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)



if __name__ == "__main__":
    raise SystemExit(main())
