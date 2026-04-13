[Japanese version](README.ja.md)

# MotionPNGTuber

> **Developer Preview --- 2026/04/10 Experimental Build**
> This is a development snapshot. Features and APIs may change without notice.

**Beyond PNGTuber, before Live2D** --- Video-based real-time lip sync system

By using looping video, you can achieve rich expressions like **hair swaying** and **clothing fluttering** that traditional PNGTubers can't deliver. No specialized knowledge like Live2D is required --- just an MP4 video and mouth sprites to get started.

## Updates

| Date | Details |
|------|---------|
| 2026/04/10 | **Developer snapshot release**. HUD display now defaults to OFF. Live IPC isolation, mouth-color regression fixes. Cleaned up unused assets and review files |
| 2026/04/05 | **Ubuntu 22.04 experimental support** (Linux audio input via PulseAudio/PipeWire, `audio_device_spec` for `sd:`/`pa:` selection). **Added mouth PNG color correction & live auto-correction** (brightness/saturation/color temperature sliders, edge-priority correction, auto color-blending button). Repository cleanup, `mouth_track_gui` stability fixes |
| 2026/04/04 | Reorganized `mouth_track_gui` for current workflow. Fixed Mac open handling and Japanese path issues. Added mouth placement margin factor to advanced settings. Added lightweight preview to check pad/mouth-erase range before full export |
| 2026/04/03 | Revised mouth PNG extraction GUI, stable checkpoint cleanup |
| 2026/01/09 | Migrated package management to **uv** |

## Features

| Feature | Description |
|---------|-------------|
| Real-time lip sync | Character's mouth moves in sync with microphone input |
| Auto emotion detection | Estimates emotion from voice and auto-switches expressions |
| Hair & physics motion | Looping video enables natural hair and clothing movement |
| Mouth PNG color blending | Adjust brightness, saturation, color temperature via sliders + auto-correction |
| macOS support | Runs on Apple Silicon (M1/M2/M3/M4) (experimental) |
| Ubuntu support | Experimental support for Ubuntu 22.04 x86_64 |

---

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
  - [Windows](#windows)
  - [macOS (Experimental)](#macos-experimental)
  - [Ubuntu 22.04 (Experimental)](#ubuntu-2204-experimental)
- [Usage](#usage)
  - [Main GUI](#main-gui)
  - [Mouth PNG Creation GUI](#mouth-png-creation-gui)
  - [Visual Check (Lightweight)](#visual-check-lightweight)
  - [Advanced Settings](#advanced-settings)
  - [Mouth PNG Color Correction](#mouth-png-color-correction)
- [Detailed Reference](#detailed-reference)
- [Tests](#tests)

---

## Quick Start

### Requirements

- Python 3.10
- uv (package manager)

### 3 Steps to Try

```bash
# 1. Install
uv sync

# 2. Launch GUI
uv run python mouth_track_gui.py

# 3. Try with samples
#    Video: assets/asmr_tomari/asmr_loop.mp4
#    Mouth: assets/asmr_tomari/mouth
#    → (1) Analyze → Calibrate → (2) Generate mouthless video → (3) Live run
```

> Note: some older articles still refer to `assets01` / `assets03`. The current sample set on `main` is `assets/asmr_tomari/`.

---

## Installation

### Windows

<details open>
<summary><b>Click to expand</b></summary>

#### 1. Prerequisites

- [Python 3.10](https://www.python.org/downloads/) (check "Add Python to PATH" during installation)
- uv:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### 2. Install

```bash
# Run in the project directory
uv sync
```

#### 3. Verify

```bash
uv run python -c "import cv2; import torch; print('OK')"
```

</details>

### macOS (Experimental)

<details>
<summary><b>Click to expand (Apple Silicon: M1/M2/M3/M4)</b></summary>

#### 1. Prepare pyproject.toml

```sh
cp pyproject.toml pyproject.win.toml
cp pyproject.macos.toml pyproject.toml
```

#### 2. Base packages

```sh
uv venv .venv && uv sync
uv pip install pip setuptools wheel torch==2.0.1 torchvision==0.15.2
```

#### 3. Build xtcocotools from source

```sh
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && ../../.venv/bin/python -m pip install -e . && cd ../..
```

#### 4. Build mmcv-full from source (~5 minutes)

```sh
cd deps
curl -L https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.7.0.tar.gz -o mmcv-1.7.0.tar.gz
tar xzf mmcv-1.7.0.tar.gz && cd mmcv-1.7.0
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py develop
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py build_ext --inplace
cd ../..
```

#### 5. Remaining packages

```sh
uv pip install --no-build-isolation anime-face-detector
uv pip install mmdet==2.28.0 mmpose==0.29.0
```

#### 6. Launch

```sh
.venv/bin/python mouth_track_gui.py
```

#### Notes

- Do not delete the `deps/` directory
- Use `+`/`-` keys for calibration zoom (scroll wheel may not work)

</details>

### Ubuntu 22.04 (Experimental)

<details>
<summary><b>Click to expand (x86_64 / NVIDIA)</b></summary>

#### 1. Prerequisites

- Ubuntu 22.04 LTS
- Python 3.10
- `uv`
- NVIDIA GPU / CUDA 11.7 compatible environment recommended

#### 2. System packages

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 pulseaudio-utils
```

#### 3. Install

```bash
uv sync
```

#### 4. Verify

```bash
uv run python -c "import cv2; import torch; print('OK')"
```

#### 5. Launch

```bash
uv run python mouth_track_gui.py
```

#### Notes

- Linux audio input enumeration can differ from Windows/macOS.
- If your USB mic does not appear as a normal `sd:` device, try a `pa:... (via pulse)` item in the audio list.
- If `pactl` is unavailable, Linux audio fallback support is limited.

</details>

---

## Usage

### Main GUI

```bash
uv run python mouth_track_gui.py
```

On macOS, install dependencies from `pyproject.macos.toml` and use the same entry point.

Note: the default `pyproject.toml` is intended for Windows / Linux dependencies.
macOS uses `pyproject.macos.toml`, and the default `pyproject.toml` is now scoped so `uv` won't try to re-resolve `mmcv-full` for darwin.

#### Workflow

1. **Create mouth PNG sprites** --- only if the mouth folder doesn't exist yet
2. **Select video** → choose a loop video
3. **Select mouth folder** → choose the folder containing mouth sprites
4. **(1) Analyze → Calibrate** → adjust mouth position and press Space to confirm ([controls](#calibration-controls))
5. **Visual check (lightweight)** → preview pad / mouth-erase range without full export
6. **(2) Generate mouthless video** → generates video with mouth erased
7. **(3) Live run** → speak into the mic and the mouth moves!
8. **If the mouth color looks off** → press auto color-blending or adjust sliders manually ([details](#mouth-png-color-correction))

#### Default Settings (changed 2026/04/10)

| Setting | Default |
|---------|---------|
| Shadow blending (mouth erase) | ON |
| HUD display | OFF |

Both can be toggled via checkboxes in the GUI. If a value is already saved in the session, the saved value takes precedence.

---

### Mouth PNG Creation GUI

```bash
uv run python mouth_sprite_extractor_gui.py
```

If you don't have a mouth folder yet, it's recommended to create one here first.
You can also launch it from the "Create mouth PNG sprites (recommended)" button in the main GUI.
During analysis after selecting a video, the **Processing Status** area at the top shows the current step with a progress bar.

---

### Visual Check (Lightweight)

A preview for quickly checking **pad** and **mouth-erase range** before exporting the full mouthless video.

- Uses existing `mouth_track.npz` / `mouth_track_calibrated.npz` as-is, so you can skip heavy processing
- If `open.png` is found, you can also see how the mouth PNG overlays
- Press `Enter` to apply the selected settings to the GUI

#### Key Controls

| Key | Function |
|-----|----------|
| Top buttons / `1` `2` `3` | Select pad candidate |
| `r` / `f` | Increase / decrease mouth-erase range |
| `a` / `d` | Previous / next frame |
| `[` / `]` | Jump 10 frames back / forward |
| `Space` | Play / pause |
| Apply to GUI button / `Enter` | Apply pad / mouth-erase range to GUI |
| `Esc` / `q` | Close without applying |

---

### Advanced Settings

**Advanced settings can usually be left closed.**

The main setting is the **mouth placement margin factor**, which affects mouth placement size during **(1) Analyze → Calibrate**.

| Symptom | Fix |
|---------|-----|
| Mouth PNG looks too small / edges are clipped | Increase slightly (e.g., `2.3` ~ `2.6`) |
| Mouth is too large / picks up jaw or cheeks | Decrease slightly (e.g., `1.8` ~ `2.0`) |

The default value of **`2.1`** usually works fine.

Recommended workflow:

1. Run **(1) Analyze → Calibrate** with `2.1`
2. **Visual check (lightweight)**
3. Compare values around `1.9 / 2.1 / 2.3`
4. Apply the best value, then run **(2) Generate mouthless video**

---

### Mouth PNG Color Correction

Adjusts color blending between mouth PNGs and the base video.
Moving the sliders during live run **reflects changes in real time (~hundreds of ms)** without restarting the runtime.

#### Adjustable Parameters

| Parameter | Description |
|-----------|-------------|
| Mouth PNG brightness | Overall mouth brightness |
| Mouth PNG saturation | Color vividness |
| Mouth PNG warm/cool | Color temperature |
| Correction intensity | Overall correction strength |
| Edge priority | How strongly correction applies to mouth edges |
| Edge correction width | Range of edge correction effect |
| Preview: color diff highlight | Makes subtle color mismatches easier to spot (preview only; does not modify assets) |

> **How edge priority works**: Correction is not applied uniformly across the entire mouth PNG --- it applies more strongly to the outer edges.
> This helps reduce issues like "only the mouth edges look out of place" or "the skin boundary doesn't blend."

#### Auto Color Blending

Press the **"Auto color blending"** button during live run to automatically compare the background video's mouth area with the mouth PNG's edge colors and suggest correction values.
You can still fine-tune with sliders after auto-correction.

- Transparent areas of the mouth PNG are ignored; only the **colored edge areas** are analyzed
- The auto-correction button is **only available during live run**
- Changes are auto-saved

#### Recommended Workflow

1. Start **(3) Live run**
2. Optionally increase **color diff highlight** to check for color mismatches
3. Press **Auto color blending**
4. Fine-tune with sliders

---

## Detailed Reference

<details>
<summary><b>What You Need</b></summary>

### Video (.mp4)

- A short looping video (a few seconds)
- Face should not be obscured

### Mouth Sprites (.png x 5)

| File | Description |
|------|-------------|
| `open.png` | Mouth open |
| `closed.png` | Mouth closed |
| `half.png` | Half open |
| `e.png` | Custom shape |
| `u.png` | Custom shape |

- Format: PNG (with transparency)
- Recommended size: ~128px width
- Having just `open.png` can sometimes compensate for missing sprites, but having all 5 is more stable

</details>

<details>
<summary><b>Calibration Controls</b></summary>

### Mouse

| Action | Function |
|--------|----------|
| Left drag | Move |
| Scroll wheel | Zoom in/out |
| Right drag | Rotate |

### Keyboard

| Key | Function |
|-----|----------|
| Arrow keys | Fine movement |
| `W`/`A`/`S`/`D` | Fine movement (alternative when arrow keys don't work on Mac) |
| `+`/`-` | Zoom in/out |
| `z`/`x` | Rotate |
| `Space` / `Enter` | Confirm |
| `Esc` | Cancel |

</details>

<details>
<summary><b>Emotion Detection</b></summary>

| Emotion | Detection Criteria |
|---------|-------------------|
| neutral | Default state |
| happy | High pitch, bright tone |
| angry | Strong voice, high energy |
| sad | Low voice, quiet tone |
| excited | Very high energy |

### Presets

| Preset | Characteristics |
|--------|----------------|
| Stable | Slow emotion changes (for streaming) |
| Standard | Balanced |
| Responsive | Fast reactions (for gaming) |

</details>

<details>
<summary><b>Browser Output</b></summary>

After completing **(2) Generate mouthless video**, the following files are exported to the same folder:

- `mouth_track.json`
- `*_mouthless_h264.mp4` (if `ffmpeg` is found)

Use these when passing mouth track data to a browser implementation.

- `mouth_track.json` is the main track export
- `*_mouthless_h264.mp4` is recommended for browser / player use when `ffmpeg` is available
- If `ffmpeg` is missing, the GUI still exports `mouth_track.json` and logs that H.264 export was skipped

For **MotionPNGTuber_Player** and other browser implementations that use **AudioWorklet**, opening files via `file://` may block mic / worklet loading. In that case, serve the folder locally:

```bash
python -m http.server 8000
# then open http://localhost:8000
```

</details>

<details>
<summary><b>Command Line Usage</b></summary>

```bash
# Face tracking
uv run python auto_mouth_track_v2.py --video loop.mp4 --out mouth_track.npz

# Calibration
uv run python calibrate_mouth_track.py --video loop.mp4 --track mouth_track.npz --sprite open.png

# Generate mouthless video
uv run python auto_erase_mouth.py --video loop.mp4 --track mouth_track_calibrated.npz --out loop_mouthless.mp4

# Real-time execution
uv run python loop_lipsync_runtime_patched_emotion_auto.py \
  --loop-video loop_mouthless.mp4 \
  --mouth-dir mouth_dir/Char \
  --track mouth_track_calibrated.npz
```

</details>

<details>
<summary><b>Directory Structure</b></summary>

```text
MotionPNGTuber/
├── mouth_track_gui.py                              # Main GUI (entry point)
├── mouth_track_gui/                                # Main GUI package
│   ├── __init__.py
│   ├── __main__.py
│   ├── _paths.py                                   #   Path definitions
│   ├── app.py                                      #   Application core
│   ├── ui.py                                       #   UI construction
│   ├── state.py                                    #   Session management
│   ├── services.py                                 #   File/device helpers
│   ├── actions.py                                  #   Command assembly
│   ├── runner.py                                   #   Subprocess execution
│   ├── preview.py                                  #   Lightweight preview
│   └── live_ipc.py                                 #   Live runtime IPC
├── mouth_sprite_extractor_gui.py                   # Mouth PNG creation GUI
├── mouth_sprite_extractor.py                       # CLI wrapper for mouth sprite extraction
├── auto_mouth_track_v2.py                          # Mouth position tracking
├── calibrate_mouth_track.py                        # Calibration
├── auto_erase_mouth.py                             # Mouthless video generation
├── erase_mouth_offline.py                          # Mouth erase core processing
├── loop_lipsync_runtime_patched_emotion_auto.py    # Live runtime
├── face_track_anime_detector.py                    # Anime face detection
├── motionpngtuber/                                 # Shared library package
│   ├── __init__.py
│   ├── mouth_sprite_extractor.py                   #   Core mouth sprite extraction logic
│   ├── python_exec.py                              #   Python executable resolver
│   ├── audio_linux.py                              #   Linux audio helper (PulseAudio/PipeWire)
│   ├── mouth_color_adjust.py                       #   Mouth PNG color correction logic
│   ├── lipsync_core.py                             #   Shared module
│   ├── image_io.py                                 #   Image I/O (Unicode path support)
│   ├── platform_open.py                            #   Platform-specific open handling
│   ├── auto_crop_estimator.py                      #   Auto crop estimation
│   ├── mouth_auto_classifier.py                    #   Mouth shape auto classification
│   ├── mouth_feature_analyzer.py                   #   Mouth feature analysis
│   ├── realtime_emotion_audio.py                   #   Real-time emotion audio analysis
│   └── workflow_validation.py                      #   Workflow validation
├── convert_npz_to_json.py                          # npz → JSON conversion
├── tests/                                          # Test suite
├── assets/                                         # Sample assets
├── mouth_dir/                                      # Mouth sprites (sample)
├── pyproject.toml                                  # Dependencies (Windows / Linux)
└── pyproject.macos.toml                            # Dependencies (macOS)
```

</details>

<details>
<summary><b>Troubleshooting</b></summary>

### Mouth position is misaligned

- First try **Recalibrate only**
- If still too small / too large, use **Visual check (lightweight)** to compare `pad` values
- Then adjust the **margin factor in advanced settings** slightly if needed

### Black smudge in mouth erase

- Try turning **Shadow blending** OFF

### `uv sync` fails

```bash
uv cache clean
uv sync
```

### CUDA not recognized

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Analysis stops on RTX 50-series / newer GPUs

- Root cause: the current Torch / CUDA build may not fully match your GPU architecture
- Prevention: the GUI now starts analysis with `--device auto`, which tries CUDA first and then CPU fallback
- If analysis still fails, rerun from CLI with `--device cpu` and check the log for CUDA compatibility messages

### Analysis looks frozen

- Initial analysis can take time, especially on long videos
- If GPU fallback switches to CPU, processing becomes much slower but is still expected
- Check the progress area and log before assuming the app has hung

</details>

---

## Tests

```bash
# All tests
uv run python -m unittest discover -s tests -v

# E2E smoke only
uv run python -m unittest discover -s tests -p "test_e2e_smoke.py" -v
```

Note: keep `-s tests`. Plain `python -m unittest discover` may collect **0 tests**
in this repository layout depending on the launch directory.

As of 2026-04-10: verified with **257 tests / 3 skip** passing.

---

## License

MIT License

## Acknowledgements

- [anime-face-detector](https://github.com/hysts/anime-face-detector)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
