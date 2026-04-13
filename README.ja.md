[English version](README.md)

# MotionPNGTuber

> **Developer Preview --- 2026/04/10 実験版**
> このバージョンは開発途中のスナップショットです。機能・APIは予告なく変更される場合があります。

**PNGTuber以上、Live2D未満** --- 動画ベースのリアルタイム口パク（リップシンク）システム

ループ動画を使うことで、従来のPNGTuberでは表現できなかった**髪の毛の揺れ**や**衣装のなびき**をリッチに表現できます。Live2Dのような専門知識は不要で、MP4動画と口スプライトさえあれば始められます。

📖 **[詳細な使い方はこちら（YouTube）](https://www.youtube.com/watch?v=mxZHzZ_eAkY)**

## 📢 更新情報

| 日付 | 内容 |
|------|------|
| 2026/04/10 | **開発者版スナップショット公開**。画面情報表示（HUD）のデフォルトをOFFに変更。内部構成の整理とバグ修正 |
| 2026/04/05 | **Ubuntu 22.04 実験対応**（Linuxでもマイク入力が使えるようになりました）。**口PNG色補正機能を追加**（明るさ・彩度・色温度をスライダーで調整＋自動補正ボタン）。GUIの安定性を改善 |
| 2026/04/04 | GUIの使い勝手を改善。Macでのファイルオープンと日本語パスの問題を修正。詳細設定に口配置の余白係数を追加。フル書き出し前に見た目を確認できる**軽量プレビュー**を追加 |
| 2026/04/03 | 口PNG抽出GUIの使い勝手を改善 |
| 2026/01/09 | パッケージ管理を **uv** に移行 |

## ✨ 特徴

| 機能 | 説明 |
|------|------|
| 🎤 リアルタイム口パク | マイク入力に合わせてキャラクターの口が動く |
| 🎭 感情自動判定 | 音声から感情を推定し、表情を自動切替 |
| 💨 髪・揺れ物の動き | ループ動画なので髪や衣装が自然に揺れる |
| 🎨 口PNG色なじみ補正 | 明るさ・彩度・色温度をスライダーで調整＋自動補正 |
| 🍎 macOS対応 | Apple Silicon (M1/M2/M3/M4) で動作（実験的） |
| 🐧 Ubuntu対応 | Ubuntu 22.04 x86_64 を実験対応 |

---

## 📋 目次

- [クイックスタート](#-クイックスタート)
- [インストール](#-インストール)
  - [Windows](#windows)
  - [macOS (実験的)](#macos-実験的)
  - [Ubuntu 22.04 (実験的)](#ubuntu-2204-実験的)
- [使い方](#-使い方)
  - [メインGUI](#メインgui)
  - [口PNG素材作成GUI](#-口png素材作成gui)
  - [見た目確認（軽量）](#-見た目確認軽量)
  - [詳細設定について](#-詳細設定について)
  - [口PNG色味補正](#-口png色味補正)
- [詳細リファレンス](#-詳細リファレンス)
- [テスト](#-テスト)

---

## 🚀 クイックスタート

### 必要なもの

- Python 3.10
- uv（パッケージマネージャー）
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 3ステップで試す

```bash
# 1. インストール
uv sync

# 2. GUI起動
uv run python mouth_track_gui.py

# 3. サンプルで試す
#    動画: assets/asmr_tomari/asmr_loop.mp4
#    mouth: assets/asmr_tomari/mouth
#    → ① 解析→キャリブ → ② 口消し動画生成 → ③ ライブ実行
```

> 📝 古い記事では `assets01` / `assets03` が使われていることがありますが、現在の `main` のサンプルは `assets/asmr_tomari/` です。

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

#### 1. uvのインストール

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 2. pyproject.tomlの準備

macOS では専用の依存定義ファイルを使います（Windows/Linux用の `pyproject.toml` には macOS非対応の依存が含まれるため）。

```sh
cp pyproject.toml pyproject.win.toml
cp pyproject.macos.toml pyproject.toml
```

#### 3. 基本パッケージ

```sh
uv venv .venv && uv sync
uv pip install pip setuptools wheel torch==2.0.1 torchvision==0.15.2
```

#### 4. xtcocotoolsをソースからビルド

```sh
mkdir -p deps && cd deps
git clone https://github.com/jin-s13/xtcocoapi.git
cd xtcocoapi && ../../.venv/bin/python -m pip install -e . && cd ../..
```

#### 5. mmcv-fullをソースからビルド（約5分）

```sh
cd deps
curl -L https://github.com/open-mmlab/mmcv/archive/refs/tags/v1.7.0.tar.gz -o mmcv-1.7.0.tar.gz
tar xzf mmcv-1.7.0.tar.gz && cd mmcv-1.7.0
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py develop
MMCV_WITH_OPS=1 FORCE_CUDA=0 ../../.venv/bin/python setup.py build_ext --inplace
cd ../..
```

#### 6. 残りのパッケージ

```sh
uv pip install --no-build-isolation anime-face-detector
uv pip install mmdet==2.28.0 mmpose==0.29.0
```

#### 7. 起動

```sh
.venv/bin/python mouth_track_gui.py
```

#### 注意事項

- `deps/` ディレクトリは削除しないこと
- キャリブレーションの拡大縮小は `+`/`-` キーで行う（ホイール不可の場合あり）

</details>

### Ubuntu 22.04 (実験的)

<details>
<summary><b>クリックして展開（x86_64 / NVIDIA）</b></summary>

#### 1. 前提条件

- Ubuntu 22.04 LTS
- Python 3.10
- NVIDIA GPU / CUDA 11.7 系推奨

#### 2. uvのインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. システムパッケージ

```bash
sudo apt-get update
sudo apt-get install -y libportaudio2 pulseaudio-utils
```

#### 4. インストール

```bash
uv sync
```

#### 5. 確認

```bash
uv run python -c "import cv2; import torch; print('OK')"
```

#### 6. 起動

```bash
uv run python mouth_track_gui.py
```

#### 注意

- Linux の音声入力列挙は Windows / macOS と挙動が異なる場合があります。
- USB マイクが通常の `sd:` デバイスとして見えない場合は、音声一覧の `pa:... (via pulse)` を試してください。
- `pactl` が使えない環境では Linux 向け音声フォールバックは限定的です。

</details>

---

## 🎮 使い方

### メインGUI

```bash
uv run python mouth_track_gui.py
```

> 📝 macOS をお使いの場合は、先に[macOSインストール手順](#macos-実験的)で依存関係をセットアップしてから起動してください。

#### ワークフロー

1. **口PNG素材を作る** --- mouthフォルダがまだ無い場合だけ
2. **動画を選択** → ループ動画を選ぶ
3. **mouthフォルダを選択** → 口スプライトがあるフォルダを選ぶ
4. **① 解析→キャリブ** → 口の位置を調整してSpaceで確定（[操作方法](#-キャリブレーション操作)）
5. **見た目確認（軽量）** → 余白係数 / 口消し範囲をフル書き出しなしでプレビュー
6. **② 口消し動画生成** → 口を消した動画を生成
7. **③ ライブ実行** → マイクに話すと口が動く！
8. **口の色味が気になったら** → 色なじみ自動補正を押すか、スライダーで手動調整（[詳細](#-口png色味補正)）

#### デフォルト設定（2026/04/10 変更）

| 設定 | 説明 | デフォルト |
|------|------|-----------|
| 影なじませ（口消し） | 口を消した跡と周囲の境界を自然にぼかす機能。ONだと継ぎ目が目立ちにくくなります | ON |
| HUD表示 | ライブ実行中に画面上にFPSや口の開閉状態などの情報を表示する機能 | OFF |

いずれもGUI上のチェックボックスで切り替えられます。セッションに保存済みの値がある場合はそちらが優先されます。

---

### 🖌️ 口PNG素材作成GUI

```bash
uv run python mouth_sprite_extractor_gui.py
```

mouthフォルダがまだ無いなら、先にこっちで作るのがおすすめです。
メインGUI内の「口PNG素材を作る（おすすめ）」ボタンからも起動できます。
動画選択後の解析中は、GUI上部の **「処理状態」** と進行バーに現在の処理内容が表示されます。

---

### 🔍 見た目確認（軽量）

フルの口消し動画を書き出す前に、**余白係数** と **口消し範囲** を軽く確認するためのプレビューです。

- 今ある `mouth_track.npz` / `mouth_track_calibrated.npz` をそのまま使うので、毎回重い処理を回さずに済みます
- `open.png` が見つかれば、口PNGの重なり方も一緒に見られます
- `Enter` で、その場で選んだ設定をGUIへ反映できます
- ⚠️ 余白係数を変更した場合は **① 解析→キャリブ** からやり直しになります

#### キー操作

| キー | 機能 |
|------|------|
| 上のボタン / `1` `2` `3` | 表示中の余白係数候補を選ぶ |
| `r` / `f` | 口消し範囲を広げる / 狭める |
| `a` / `d` | 1フレーム戻る / 進む |
| `[` / `]` | 10フレーム戻る / 進む |
| `Space` | 再生 / 停止 |
| GUIへ反映ボタン / `Enter` | 余白係数 / 口消し範囲をGUIに反映 |
| `Esc` / `q` | 反映せず閉じる |

---

### ⚙️ 詳細設定について

普段は **詳細設定は閉じたままでOK** です。

今入っているのは主に **口配置の余白係数** です。
これは **① 解析→キャリブ** の時の口配置サイズに効きます。

| 症状 | 対処 |
|------|------|
| 口PNGが小さく見える / 口の端が切れる | 少し上げる（例: `2.3` ～ `2.6`） |
| 口が大きすぎる / 顎や頬まで拾う | 少し下げる（例: `1.8` ～ `2.0`） |

通常は **`2.1` のまま** で大丈夫です。

おすすめの確認手順:

1. `2.1` のまま **① 解析→キャリブ**
2. **見た目確認（軽量）** で `1.9 / 2.1 / 2.3` 付近を見比べる
3. 良さそうな値を反映 → **① 解析→キャリブ** をやり直す
4. **② 口消し動画生成** へ進む

> ⚠️ 余白係数は解析時に使われるため、値を変えたら **① 解析→キャリブ** からやり直す必要があります。

---

### 🎨 口PNG色味補正

口PNGとベース動画の色なじみを調整する機能です。
ライブ実行中に右側のスライダーを動かすと **数百ms程度でリアルタイムに反映** されます（runtimeの再起動は不要）。

#### 調整できる項目

| 項目 | 説明 |
|------|------|
| 口PNG 明るさ | 口全体の明暗 |
| 口PNG 彩度 | 色の鮮やかさ |
| 口PNG 暖色/寒色 | 色温度 |
| 補正強度 | 全体の補正の効き具合 |
| 外周優先度 | 口の縁にどれだけ強く効かせるか |
| 外周補正幅 | 外周補正が効く範囲 |
| 確認表示 色差強調 | 微妙な色ズレを見つけやすくする（表示確認用。素材は書き換えません） |

> **外周優先の仕組み**: 補正は口PNG全体に均一にかかるのではなく、口の外周により強く効きます。
> これにより「口の縁だけ浮く」「肌との境界が馴染まない」といった問題を軽減できます。

#### 色なじみ自動補正

ライブ実行中に **「色なじみ自動補正」** ボタンを押すと、背景動画の口周辺と口PNGの外周色を比較して補正値を自動提案します。
自動補正後もそのままスライダーで手動微調整できます。

- 口PNGの透過部分は無視し、**色が付いている外周部分**を見て判断します
- 自動補正ボタンは **ライブ実行中のみ有効** です
- 変更内容は自動保存されます

#### おすすめの調整手順

1. **③ ライブ実行** を開始
2. 必要なら **色差強調** を少し上げて色ズレを確認
3. **色なじみ自動補正** を押す
4. スライダーで微調整して仕上げる

---

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
- `open.png` さえあれば不足分を補えるケースもありますが、基本は5枚揃っている方が安定します

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
| `W`/`A`/`S`/`D` | 微移動（Macで矢印が効きにくい時の代替） |
| `+`/`-` | 拡大・縮小 |
| `z`/`x` | 回転 |
| `Space` / `Enter` | 確定 |
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
<summary><b>🌐 ブラウザ向け出力</b></summary>

`② 口消し動画生成` 完了後、同じフォルダに次のファイルを書き出します。

- `mouth_track.json`
- `*_mouthless_h264.mp4`（`ffmpeg` が見つかった場合）

ブラウザ実装に mouth track を渡したい時はこれを使います。

- 基本となるトラック出力は `mouth_track.json` です
- `*_mouthless_h264.mp4` は `ffmpeg` がある環境でのブラウザ / Player 利用に向いています
- `ffmpeg` が無い場合でも GUI は `mouth_track.json` を出力し、H.264 変換をスキップしたことをログに出します

**MotionPNGTuber_Player** のように **AudioWorklet** を使う実装では、`file://` で直接開くとマイクや worklet 読み込みが制限される場合があります。その場合はローカルサーバー経由で開いてください。

```bash
python -m http.server 8000
# その後 http://localhost:8000 を開く
```

</details>

<details>
<summary><b>⌨️ コマンドライン使用</b></summary>

```bash
# 顔トラッキング
uv run python auto_mouth_track_v2.py --video loop.mp4 --out mouth_track.npz

# キャリブレーション
uv run python calibrate_mouth_track.py --video loop.mp4 --track mouth_track.npz --sprite open.png

# 口消し動画生成
uv run python auto_erase_mouth.py --video loop.mp4 --track mouth_track_calibrated.npz --out loop_mouthless.mp4

# リアルタイム実行
uv run python loop_lipsync_runtime_patched_emotion_auto.py \
  --loop-video loop_mouthless.mp4 \
  --mouth-dir mouth_dir/Char \
  --track mouth_track_calibrated.npz
```

</details>

<details>
<summary><b>📁 フォルダ構成</b></summary>

```text
MotionPNGTuber/
├── mouth_track_gui.py                              # メインGUI（エントリーポイント）
├── mouth_track_gui/                                # メインGUIパッケージ
│   ├── __init__.py
│   ├── __main__.py
│   ├── _paths.py                                   #   パス定義
│   ├── app.py                                      #   アプリケーション本体
│   ├── ui.py                                       #   UI構築
│   ├── state.py                                    #   セッション管理
│   ├── services.py                                 #   ファイル/デバイスヘルパー
│   ├── actions.py                                  #   コマンド組み立て
│   ├── runner.py                                   #   サブプロセス実行
│   ├── preview.py                                  #   軽量見た目確認
│   └── live_ipc.py                                 #   ライブ実行時IPC
├── mouth_sprite_extractor_gui.py                   # 口PNG素材作成GUI
├── mouth_sprite_extractor.py                       # 口PNG素材抽出CLIラッパー
├── auto_mouth_track_v2.py                          # 口位置トラッキング
├── calibrate_mouth_track.py                        # キャリブレーション
├── auto_erase_mouth.py                             # 口消し動画生成
├── erase_mouth_offline.py                          # 口消しコア処理
├── loop_lipsync_runtime_patched_emotion_auto.py    # ライブ実行ランタイム
├── face_track_anime_detector.py                    # アニメ顔検出
├── motionpngtuber/                                 # 共有ライブラリパッケージ
│   ├── __init__.py
│   ├── mouth_sprite_extractor.py                   #   口PNG素材抽出の本体ロジック
│   ├── python_exec.py                              #   Python実行パス解決
│   ├── audio_linux.py                              #   Linux音声ヘルパー（PulseAudio/PipeWire）
│   ├── mouth_color_adjust.py                       #   口PNG色補正ロジック
│   ├── lipsync_core.py                             #   共通モジュール
│   ├── image_io.py                                 #   画像I/O（Unicodeパス対応）
│   ├── platform_open.py                            #   プラットフォーム別open処理
│   ├── auto_crop_estimator.py                      #   自動クロップ推定
│   ├── mouth_auto_classifier.py                    #   口形状自動分類
│   ├── mouth_feature_analyzer.py                   #   口特徴量分析
│   ├── realtime_emotion_audio.py                   #   リアルタイム感情音声解析
│   └── workflow_validation.py                      #   ワークフロー検証
├── convert_npz_to_json.py                          # npz→JSON変換
├── tests/                                          # テストスイート
├── assets/                                         # サンプル素材
├── mouth_dir/                                      # 口スプライト（サンプル）
├── pyproject.toml                                  # 依存関係（Windows / Linux）
└── pyproject.macos.toml                            # 依存関係（macOS）
```

</details>

<details>
<summary><b>❓ トラブルシューティング</b></summary>

### 口の位置がズレる

- まずは **キャリブのみ（やり直し）**
- それでも小さい / 大きいなら、先に **見た目確認（軽量）** で余白係数を見比べる
- その後で必要なら **詳細設定の余白係数** を少しだけ調整

### 口消しに黒いにじみが出る

- **影なじませ** を OFF にしてみる

### `uv sync` が失敗する

```bash
uv cache clean
uv sync
```

### CUDA が認識されない

```bash
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### RTX 50系 / 新しめのGPUで解析が止まる

- 原因: 現在の Torch / CUDA ビルドと GPU アーキテクチャが完全に一致していない可能性があります
- 予防: GUI は `--device auto` 相当で解析を開始し、まず CUDA、失敗時に CPU フォールバックを試みます
- それでも失敗する場合は、CLI から `--device cpu` を明示し、ログ内の CUDA 互換メッセージを確認してください

### 解析がフリーズしたように見える

- 初回解析は動画長や環境によって時間がかかります
- GPU から CPU にフォールバックした場合は特に遅くなります
- 進捗表示とログが動いていれば、処理継続中である可能性が高いです

</details>

<details>
<summary><b>🎁 おまけツール</b></summary>

### 口スプライト抽出CLI

```bash
uv run python mouth_sprite_extractor.py --video loop.mp4 --out mouth/
```

GUIを使わず、コマンドラインから口スプライト（5種類のPNG）を直接抽出できます。

</details>

---

## 🧪 テスト

```bash
# 全テスト
uv run python -m unittest discover -s tests -v

# E2Eスモークだけ
uv run python -m unittest discover -s tests -p "test_e2e_smoke.py" -v
```

補足: このリポジトリ構成では、起動ディレクトリによっては
`python -m unittest discover` だけだと **0 tests** になることがあります。
`-s tests` を付けたコマンドを使ってください。

2026-04-13 時点: **261 tests / 3 skip** で通る状態を確認しています。

---

## 📄 ライセンス

MIT License

## 🙏 謝辞

- [anime-face-detector](https://github.com/hysts/anime-face-detector)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [MMPose](https://github.com/open-mmlab/mmpose)
