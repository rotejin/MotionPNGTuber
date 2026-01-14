# MotionPNGTuber

**PNGTuber以上、Live2D未満** — 動画ベースのリアルタイム口パク（リップシンク）システム

ループ動画を使うことで、従来のPNGTuberでは表現できなかった**髪の毛の揺れ**や**衣装のなびき**をリッチに表現できます。Live2Dのような専門知識は不要で、MP4動画と口スプライトさえあれば始められます。

📖 **[詳細な使い方はこちら（note）](https://note.com/rotejin/n/n2b12c9be0b81)**

## 📢 更新情報

| 日付 | 内容 |
|------|------|
| 2026/01/13 | **ブラウザ版プレイヤー**追加、**複数動画切り替えGUI**追加、macOS対応 |
| 2026/01/09 | パッケージ管理を **uv** に移行 |

## ✨ 特徴

| 機能 | 説明 |
|------|------|
| 🎤 リアルタイム口パク | マイク入力に合わせてキャラクターの口が動く |
| 🎭 感情自動判定 | 音声から感情を推定し、表情を自動切替 |
| 💨 髪・揺れ物の動き | ループ動画なので髪や衣装が自然に揺れる |
| 🌐 ブラウザ版 | Python不要の軽量版、OBSで直接使用可能 |
| 🎬 複数動画切り替え | 複数モーションをボタンで瞬時に切替 |
| 🍎 macOS対応 | Apple Silicon (M1/M2/M3/M4) で動作 |

---

## 📋 目次

- [クイックスタート](#-クイックスタート)
- [インストール](#-インストール)
  - [Windows](#windows)
  - [macOS (実験的)](#macos-実験的)
- [使い方](#-使い方)
  - [メインGUI](#メインgui)
  - [ブラウザ版プレイヤー](#-ブラウザ版プレイヤー)
  - [複数動画切り替えGUI](#-複数動画切り替えgui)
- [詳細リファレンス](#-詳細リファレンス)

---

## 🚀 クイックスタート

### 必要なもの

- Python 3.10
- uv（パッケージマネージャー）

### 3ステップで試す

```bash
# 1. インストール
uv sync

# 2. GUI起動
uv run python mouth_track_gui.py

# 3. サンプルで試す
#    動画: assets/assets03/loop.mp4
#    mouth: assets/assets03/mouth
#    → ① 解析→キャリブ → ② 口消し動画生成 → ③ ライブ実行
```

---

## 🔧 インストール

### Windows

<details open>
<summary><b>クリックして展開</b></summary>

#### 1. 前提条件

- [Python 3.10](https://www.python.org/downloads/)（インストール時に「Add Python to PATH」にチェック）
- uv:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

#### 2. インストール

```bash
# プロジェクトディレクトリで実行
uv sync
```

#### 3. 確認

```bash
uv run python -c "import cv2; import torch; print('OK')"
```

</details>

### macOS (実験的)

<details>
<summary><b>クリックして展開（Apple Silicon: M1/M2/M3/M4）</b></summary>

#### 1. pyproject.tomlの準備

```sh
cp pyproject.toml pyproject.win.toml
cp pyproject.macos.toml pyproject.toml
```

#### 2. 基本パッケージ

```sh
uv venv .venv && uv sync
uv pip install pip setuptools wheel torch==2.0.1 torchvision==0.15.2
```

#### 3. xtcocotoolsをソースからビルド

```sh
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && ../../.venv/bin/python -m pip install -e . && cd ../..
```

#### 4. mmcv-fullをソースからビルド（約5分）

```sh
cd deps
curl -L https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.7.0.tar.gz -o mmcv-1.7.0.tar.gz
tar xzf mmcv-1.7.0.tar.gz && cd mmcv-1.7.0
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py develop
cd ../..
```

#### 5. 残りのパッケージ

```sh
uv pip install --no-build-isolation anime-face-detector
uv pip install mmdet==2.28.0 mmpose==0.29.0
```

#### 6. 起動

```sh
.venv/bin/python mouth_track_gui.py
```

#### 注意事項

- `deps/` ディレクトリは削除しないこと
- キャリブレーションの拡大縮小は `+`/`-` キーで行う（ホイール不可）

</details>

---

## 🎮 使い方

### メインGUI

```bash
uv run python mouth_track_gui.py
```

#### ワークフロー

1. **動画を選択** → ループ動画を選ぶ
2. **mouthフォルダを選択** → 口スプライトがあるフォルダを選ぶ
3. **① 解析→キャリブ** → 口の位置を調整してSpaceで確定（[操作方法](#-キャリブレーション操作)）
4. **② 口消し動画生成** → 口を消した動画を生成
5. **③ ライブ実行** → マイクに話すと口が動く！

---

### 🎬 複数動画切り替えGUI (実験的)

複数モーションを自動orクリックで切り替え。

```bash
uv run python multi_video_live_gui.py
```

#### 使い方

1. **親フォルダを選択** → 「自動検出」で複数フォルダを一括登録
   - または「+追加」で個別フォルダを登録
2. **設定を調整**（オプション）
   - 感情オート: stable / standard / snappy
   - 切替演出: クロスフェードの有効/秒数
   - 自動巡回: ループ終端で次のセットへ自動切替
3. **▶ ライブ開始** → 動画セットボタンで切り替え

## 📚 詳細リファレンス

<details>
<summary><b>📦 準備するもの</b></summary>

### 動画（.mp4）

- ループ再生できる短い動画（数秒程度）
- 顔が隠れていないもの

### 口スプライト（.png × 5枚）

| ファイル | 説明 |
|----------|------|
| `open.png` | 口を開けた状態 |
| `closed.png` | 口を閉じた状態 |
| `half.png` | 半開き |
| `e.png` | 任意の形状 |
| `u.png` | 任意の形状 |

- 画像形式: PNG（透過対応）
- 推奨サイズ: 幅128px程度

</details>

<details>
<summary><b>🎯 キャリブレーション操作</b></summary>

### マウス操作

| 操作 | 機能 |
|------|------|
| 左ドラッグ | 移動 |
| ホイール | 拡大・縮小 |
| 右ドラッグ | 回転 |

### キーボード操作

| キー | 機能 |
|------|------|
| 矢印キー | 微移動 |
| `+`/`-` | 拡大・縮小 |
| `z`/`x` | 回転 |
| `Space` | 確定 |
| `Esc` | キャンセル |

</details>

<details>
<summary><b>🎭 感情判定について</b></summary>

| 感情 | 判定基準 |
|------|----------|
| neutral | 標準状態 |
| happy | 高い声、明るいトーン |
| angry | 強い声、高エネルギー |
| sad | 低い声、静かなトーン |
| excited | 非常に高いエネルギー |

### プリセット

| プリセット | 特徴 |
|------------|------|
| 安定 | 感情変化がゆっくり（配信向け） |
| 標準 | バランス重視 |
| キビキビ | 反応が素早い（ゲーム向け） |

</details>

<details>
<summary><b>📁 フォルダ構成</b></summary>

```
MotionPNGTuber/
├── mouth_track_gui.py              # メインGUI
├── multi_video_live_gui.py         # 複数動画切り替えGUI
├── convert_npz_to_json.py          # npz→JSON変換
├── lipsync_core.py                 # 共通モジュール
├── MotionPNGTuber_Player/          # ブラウザ版
├── pyproject.toml                  # 依存関係（Windows）
├── pyproject.macos.toml            # 依存関係（macOS）
└── mouth_dir/                      # 口スプライト
```

</details>

<details>
<summary><b>⌨️ コマンドライン使用</b></summary>

```bash
# 顔トラッキング
uv run python face_track_anime_detector.py --video loop.mp4 --out mouth_track.npz

# キャリブレーション
uv run python calibrate_mouth_track.py --video loop.mp4 --track mouth_track.npz --sprite open.png

# 口消し動画生成
uv run python auto_erase_mouth.py --video loop.mp4 --track mouth_track_calibrated.npz --out loop_mouthless.mp4

# リアルタイム実行
uv run python loop_lipsync_runtime_patched_emotion_auto.py --loop-video loop_mouthless.mp4 --mouth-dir mouth_dir/Char --track mouth_track_calibrated.npz
```

</details>

<details>
<summary><b>❓ トラブルシューティング</b></summary>

### uv sync が失敗する

```bash
uv cache clean
uv sync
```

### CUDA が認識されない

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### 口の位置がズレる

「キャリブのみ（やり直し）」でキャリブレーションをやり直す

### 口消しに黒いにじみが出る

「影なじませ」をOFFにする

</details>

<details>
<summary><b>🎁 おまけツール</b></summary>

### 口消しチューナーGUI

```bash
uv run python mouth_erase_tuner_gui.py
```

画像の口部分を削除できる単体ツール。

### 口スプライト抽出GUI

```bash
uv run python mouth_sprite_extractor_gui.py
```

動画から口スプライト（5種類のPNG）を自動抽出。

</details>

---

## 📄 ライセンス

MIT License

## 🙏 謝辞

- [anime-face-detector](https://github.com/hysts/anime-face-detector)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
