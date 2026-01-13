# MotionPNGTuber

**PNGTuber以上、Live2D未満** — 動画ベースのリアルタイム口パク（リップシンク）システム

## 📢 更新情報

| 日付 | 内容 |
|------|------|
| 2026/01/09 | パッケージ管理を **uv** に移行。`uv sync` 1コマンドでインストール完了に。従来の複雑なpip/mim手順が不要になりました。 |

ループ動画を使うことで、従来のPNGTuberでは表現できなかった**髪の毛の揺れ**や**衣装のなびき**をリッチに表現できます。Live2Dのような専門知識は不要で、MP4動画と口スプライトさえあれば始められます。

## ✨ 特徴

- 🎤 **リアルタイム口パク**: マイク入力に合わせてキャラクターの口が動く
- 🤖 **AITuber対応**: 仮想オーディオデバイスを使えばAITuberのアバターとしても利用可能
- 🎭 **感情自動判定**: 音声から感情（喜び・怒り・悲しみ等）を推定し、表情を自動切替
- 💨 **髪・揺れ物の動き**: ループ動画なので髪や衣装が自然に揺れる
- 🖼️ **高精度口消し**: 元動画から口を自然に消去し、口スプライトを合成
- 🎯 **簡単キャリブレーション**: マウス操作で口の位置・サイズ・回転を調整
- 🖥️ **GUIツール**: ワンクリックで解析から実行まで

## 🎯 こんな人におすすめ

| ユースケース | 説明 |
|--------------|------|
| **PNGTuberからのステップアップ** | 静止画では物足りないけどLive2Dは難しい |
| **VTuber配信** | マイク入力でリアルタイムにアバターを動かしたい |
| **AITuber** | AI音声出力を仮想オーディオ経由でアバターに反映したい |

---

## 🚀 クイックスタート

サンプルアセットを使って、すぐに試すことができます。

### 0. 前提条件（未インストールの場合）

- **Python 3.10**: [ダウンロード](https://www.python.org/downloads/)（インストール時に「Add Python to PATH」にチェック）
- **uv**: Pythonパッケージマネージャー（[インストール方法](https://docs.astral.sh/uv/getting-started/installation/)）
  ```bash
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 1. インストール（初回のみ）

```bash
# プロジェクトディレクトリで実行
uv sync
```

これだけで仮想環境の作成と全ての依存パッケージのインストールが完了します。

### 2. GUIを起動

```bash
uv run python mouth_track_gui.py
```

### 3. サンプルで試す

1. **動画を選択**: `assets/assets03/loop.mp4` を選択
2. **mouthフォルダを選択**: `assets/assets03/mouth` を選択
3. **① 解析→キャリブ**: クリックして口の位置を調整 → Space で確定
4. **② 口消し動画生成**: クリックして待つ
5. **③ ライブ実行**: マイクに向かって話すと口が動く！

> 💡 各アセットフォルダには `パラメーター参考例XX.png` が入っています。設定の参考にしてください。

---

## 📦 準備するもの

### 1. アニメキャラクターの動画（.mp4）

- ループ再生できる短い動画（数秒程度）
- **顔が隠れていないもの**（顔が隠れているとリップシンクが正しく動作しません）
- 髪の揺れや衣装のなびきがあるとリッチな表現に

### 2. 口スプライト画像（.png × 5枚）

キャラクターに合った口の画像を5種類用意します：

| ファイル名 | 説明 |
|------------|------|
| `open.png` | 口を開けた状態 |
| `closed.png` | 口を閉じた状態 |
| `half.png` | 半開き |
| `e.png` | 上記3つとは異なる形状（任意） |
| `u.png` | 上記4つとは異なる形状（任意） |

- 画像形式: PNG（透過対応・RGBA）
- 推奨サイズ: 幅128px程度

### 3. 感情別口スプライト（オプション）

感情表現を使いたい場合は、感情ごとに口スプライトを用意します（Default / Happy / Angry / Sad / Excited）。

**感情別スプライトがなくても1種類の口スプライトだけで動作します。** 表現の幅を広げたい場合に追加してください。

---

## 💻 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10 / 11 |
| Python | 3.10（3.10.x推奨） |
| GPU | NVIDIA GPU + CUDA 11.7（推奨）または CPU のみ |
| RAM | 8GB以上推奨 |

### GPUとCPUについて

- **GPU（CUDA）使用時**: 顔トラッキングが高速（約30FPS）
- **CPUのみ**: トラッキングは低速（約2-5FPS）だが動作可能。リアルタイム再生自体はCPUで十分

---

## 🔧 インストール手順（Windows）

### 1. 前提条件

#### Python 3.10

1. [Python 3.10.x](https://www.python.org/downloads/) をダウンロード
2. インストール時に「**Add Python to PATH**」にチェック

#### uv（Pythonパッケージマネージャー）

```powershell
# PowerShellで実行
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

詳細は [uv公式ドキュメント](https://docs.astral.sh/uv/getting-started/installation/) を参照。

### 2. 依存パッケージのインストール

```bash
# プロジェクトディレクトリに移動
cd MotionPNGTuber

# 仮想環境の作成と依存パッケージのインストール（1コマンドで完了）
uv sync
```

`uv sync` を実行すると：
- `.venv` ディレクトリに仮想環境が自動作成される
- `pyproject.toml` と `uv.lock` に基づいて全ての依存パッケージがインストールされる
- PyTorch（CUDA 11.7対応）、mmcv-full、anime-face-detector など全て自動でインストールされる

### インストール確認

```bash
uv run python -c "import cv2; import numpy; import sounddevice; from anime_face_detector import create_detector; print('OK')"
```

---

## 📁 フォルダ構成

```
MotionPNGTuber/
├── mouth_track_gui.py              # メインGUI
├── mouth_erase_tuner_gui.py        # 口消しチューナーGUI（おまけツール）
├── mouth_sprite_extractor_gui.py   # 口スプライト抽出GUI（おまけツール）
├── mouth_sprite_extractor.py       # 口スプライト抽出コアモジュール
├── loop_lipsync_runtime_patched_emotion_auto.py  # リアルタイム実行（感情対応）
├── face_track_anime_detector.py    # 顔トラッキング
├── calibrate_mouth_track.py        # キャリブレーション
├── erase_mouth_offline.py          # 口消し処理
├── auto_mouth_track_v2.py          # 自動トラッキング
├── auto_erase_mouth.py             # 自動口消し
├── preview_mouth_track.py          # トラッキングプレビュー
├── realtime_emotion_audio.py       # 感情解析
├── pyproject.toml                  # uv依存関係定義
├── uv.lock                         # uvロックファイル
├── assets/                         # 動画アセット（サンプル）
│   ├── assets01/
│   │   ├── loop.mp4
│   │   └── パラメーター参考例01.png
│   ├── assets02/
│   │   ├── loop.mp4
│   │   └── パラメーター参考例02.png
│   └── assets03/
│       ├── loop.mp4
│       ├── mouth/                  # 個別の口スプライト
│       └── パラメーター参考例03.png
└── mouth_dir/                      # 感情別口スプライト
    └── Tomari/
        ├── Default/
        ├── Happy/
        ├── Angry/
        ├── Sad/
        └── Excited/
```

### 口スプライト（Mouthフォルダ）の構成

感情表現を使用するには、以下の構造で口スプライトを配置します：

```
mouth_dir/
└── Tomari/              # キャラクター名（任意）
    ├── Default/         # デフォルト表情
    │   ├── open.png     # 口を開けた状態
    │   ├── closed.png   # 口を閉じた状態
    │   ├── half.png     # 半開き
    │   ├── e.png        # 「え」の口（open/closed/halfとは異なる形状なら何でも可）
    │   └── u.png        # 「う」の口（open/closed/halfとは異なる形状なら何でも可）
    ├── Happy/           # 嬉しい
    │   ├── open.png
    │   ├── closed.png
    │   ├── half.png
    │   ├── e.png
    │   └── u.png
    ├── Angry/           # 怒り
    ├── Sad/             # 悲しみ
    └── Excited/         # 興奮
```

**重要事項**:
- 各フォルダに必ず `open.png`, `closed.png`, `half.png`, `e.png`, `u.png` の5ファイルを配置
- `e.png` と `u.png` は、`open.png` / `closed.png` / `half.png` とは**異なる形状**であれば何でも構いません（「え」「う」の口に限らず、任意の口の形状でOK）
- 画像形式: PNG（透過対応・RGBA）
- 推奨サイズ: 幅128px程度（アスペクト比 約1.5）
- 感情フォルダ名: `Default`, `Happy`, `Angry`, `Sad`, `Excited`（大文字小文字どちらでも可）

---

## 🎮 使い方

### GUI（推奨）

```bash
uv run python mouth_track_gui.py
```

#### ワークフロー

1. **動画を選択**: 「選択…」ボタンで元動画（loop.mp4など）を選ぶ
2. **mouthフォルダを選択**: 口スプライトがあるフォルダを選ぶ
3. **キャラクターを選択**: 複数キャラがある場合はドロップダウンで選択
4. **① 解析→キャリブ**: クリックすると顔トラッキング→キャリブレーション画面へ
5. **② 口消し動画生成**: 口を消した動画を生成
6. **③ ライブ実行**: マイク入力でリアルタイム口パク開始

#### 設定項目

| 項目 | 説明 |
|------|------|
| pad（追跡余白） | トラッキング領域の大きさ（1.0〜3.0） |
| 口消し強さ | 口を消す範囲の大きさ（0.4〜0.9） |
| 影なじませ | ONで陰影を自然になじませる（顎の黒にじみが出る場合はOFF） |
| スムージング | 口の動きの滑らかさ（ゆっくり〜追従最優先） |
| 感情オート | 安定（配信向け）/ 標準 / キビキビ（ゲーム向け） |
| HUD | 現在の感情を画面に表示 |

---

## 🎯 キャリブレーション操作

「① 解析→キャリブ」実行後、キャリブレーション画面が表示されます。

### マウス操作

| 操作 | 機能 |
|------|------|
| 左ドラッグ | 口スプライトを移動 |
| ホイール | 拡大・縮小 |
| Ctrl + ホイール | 微調整 |
| 右ドラッグ | 回転 |

### キーボード操作

| キー | 機能 |
|------|------|
| 矢印キー | 微移動 |
| `+` / `-` | 拡大・縮小 |
| `z` / `x` | 回転（左回り / 右回り） |
| `[` / `]` | フレーム移動（プレビュー用） |
| `r` | リセット（初期値に戻す） |
| `Space` / `Enter` | 確定して保存 |
| `q` / `Esc` | キャンセル |

---

## 👀 口消し範囲プレビュー操作

「口消し範囲プレビュー」ボタンで、口消しの範囲を事前確認できます。

### 表示

- **赤色**: 実際に消去される領域（中心マスク）
- **黄色**: 陰影推定に使う外周領域（ring）
- **緑枠**: トラッキングされた口の四角形

### キーボード操作

| キー | 機能 |
|------|------|
| `Space` | 再生 / 一時停止 |
| `a` / `d` | 1フレーム戻る / 進む |
| `[` / `]` | 10フレーム戻る / 進む |
| `q` / `Esc` | 終了 |

---

## 🎁 おまけツール: 口消しチューナーGUI

```bash
uv run python mouth_erase_tuner_gui.py
```

画像の口部分を削除できる単体ツールです。MotionPNGTuber本体とは独立して使用できます。

---

## 🎨 おまけツール: 口スプライト抽出GUI

```bash
uv run python mouth_sprite_extractor_gui.py
```

動画から口スプライト（5種類のPNG）を自動抽出するツールです。口スプライトを自分で描く必要がなくなります。

### 使い方

1. **動画を選択**: 元動画（loop.mp4など）を選ぶ
2. **解析**: 10枚の候補フレームが表示される（バリエーション別）
3. **割り当て**: 各候補の下のテキストボックスに1-5を入力
   - 1=open, 2=closed, 3=half, 4=e, 5=u
4. **調整**: 切り取り範囲とフェザー幅を設定
5. **プレビュー更新**: 設定を確認
6. **出力**: 動画と同じフォルダに `mouth/` が作成される

### 候補フレームの選び方

10枚の候補は以下からバリエーション豊かに選ばれます：
- 開いた口（高さ最大）: 2枚
- 閉じた口（高さ最小）: 2枚
- 半開き（高さ中央）: 2枚
- 横長の口（e候補）: 2枚
- すぼめた口（u候補）: 2枚

---

## ⌨️ コマンドライン使用

### 1. 顔トラッキング

```bash
uv run python face_track_anime_detector.py \
    --video assets/assets01/loop.mp4 \
    --out assets/assets01/mouth_track.npz \
    --device auto \
    --pad 2.1
```

### 2. キャリブレーション

```bash
uv run python calibrate_mouth_track.py \
    --video assets/assets01/loop.mp4 \
    --track assets/assets01/mouth_track.npz \
    --sprite mouth_dir/Tomari/Default/open.png \
    --out assets/assets01/mouth_track_calibrated.npz
```

### 3. 口消し動画生成

```bash
uv run python auto_erase_mouth.py \
    --video assets/assets01/loop.mp4 \
    --track assets/assets01/mouth_track_calibrated.npz \
    --out assets/assets01/loop_mouthless.mp4 \
    --coverage 0.6
```

### 4. リアルタイム実行

```bash
uv run python loop_lipsync_runtime_patched_emotion_auto.py \
    --loop-video assets/assets01/loop_mouthless.mp4 \
    --mouth-dir mouth_dir/Tomari \
    --track assets/assets01/mouth_track_calibrated.npz \
    --emotion-auto \
    --emotion-preset snappy
```

### デバイス指定

```bash
# GPU使用（自動選択）
--device auto

# 特定のGPU
--device cuda:0

# CPUのみ
--device cpu
```

---

## ❓ トラブルシューティング

### uv sync が失敗する

1. Python 3.10 がインストールされているか確認
2. uv が正しくインストールされているか確認（`uv --version`）
3. キャッシュをクリアして再試行：
   ```bash
   uv cache clean
   uv sync
   ```

### CUDA が認識されない

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```
False の場合：
- NVIDIA ドライバが最新か確認
- CUDA Toolkit 11.7 がインストールされているか確認

### sounddevice でエラー

オーディオデバイスが認識されない場合、Windowsのサウンド設定でマイクが有効か確認。

### 口の位置がズレる

1. 「キャリブのみ（やり直し）」でキャリブレーションをやり直す
2. `pad` の値を調整（大きくすると追跡領域が広がる）

### 口消しに黒いにじみが出る

「影なじませ」をOFFにする（GUIのチェックボックス）

---

## 🎭 感情判定について

音声から以下の5つの感情を自動判定します：

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
| 安定（配信向け） | 感情変化がゆっくり、チラつきにくい |
| 標準 | バランス重視 |
| キビキビ（ゲーム向け） | 感情変化が素早い、反応が良い |

---


## 🍎 インストール手順（Mac） **Experimental**

Apple Silicon (M1/M2/M3/M4) Mac用のインストール手順です。

### 1. pyproject.tomlの準備

Windows用のpyproject.tomlをバックアップして、Mac用に置き換えます。

```sh
cp pyproject.toml pyproject.win.toml
cp pyproject.macos.toml pyproject.toml
```

### 2. 仮想環境作成と基本パッケージ

```sh
uv venv .venv && uv sync
```

### 3. PyTorchとビルドツール

```sh
uv pip install pip setuptools wheel torch==2.0.1 torchvision==0.15.2
```

### 4. xtcocotoolsをソースからビルド

PyPI版はビルドエラーになるため、GitHubからソースビルドします。

```sh
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi
../../.venv/bin/python -m pip install -e .
cd ../..
```

### 5. mmcv-fullをソースからビルド

mmcv-full 1.7.0 はビルド済みバイナリが提供されていないため、ソースからビルドします。5分程度かかります（M3 MacBook Air）

```sh
cd deps
curl -L https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.7.0.tar.gz -o mmcv-1.7.0.tar.gz
tar xzf mmcv-1.7.0.tar.gz
cd mmcv-1.7.0
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py develop
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py build_ext --inplace
cd ../..
```

### 6. anime-face-detectorをインストール

xtcocotools/mmcvは既にあるのでスキップされます。`--no-build-isolation`はchumpy等のビルドに仮想環境のpipが必要なためです。

```sh
uv pip install --no-build-isolation anime-face-detector
```

### 7. mmdet/mmposeを正しいバージョンに修正

anime-face-detectorが最新版を入れてしまうため、正しいバージョンに修正します。

```sh
uv pip install mmdet==2.28.0 mmpose==0.29.0
```

### 8. 起動

```sh
.venv/bin/python mouth_track_gui.py
```

### 注意事項

- `deps/` ディレクトリは削除しないこと（editable installのため参照される）
- PyTorchは2.0.1を使用（最新版はmmcv 1.7.0と互換性がない）
- キャリブレーション時、口画像の大きさの調整は+/-キーで行う。マウスホイールによる拡大・縮小は不可


## 📄 ライセンス

MIT License

---

## 🙏 謝辞

- [anime-face-detector](https://github.com/hysts/anime-face-detector)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
