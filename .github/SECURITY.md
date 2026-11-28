# セキュリティポリシー / Security Policy

## 対象のバージョン

`main` ブランチを対象に修正します。

## 脆弱性の報告

**脆弱性の詳しい内容を、公開の Issue に書かないでください。**

このリポジトリの **Security** タブ → **Report a vulnerability** から、非公開で報告してください（GitHub の Private vulnerability reporting）。

この方法が使えない場合は、作者の X（[@rotejin](https://x.com/rotejin)）に DM で連絡してください。

書いてほしいこと:

- 影響のあるファイルや機能
- 再現の手順
- 期待した動作と、実際の動作
- OS と、Python のバージョン
- 安全に共有できるなら、再現に使う動画や画像

## このツールの仕組み（安全面）

- **手元の PC の中だけで動きます。** 通信するのは次の 2 つの時だけです。
  - インストールの時（`uv sync`）: PyPI・PyTorch（`download.pytorch.org`）・OpenMMLab（`download.openmmlab.com`）から、`uv.lock` に書いてあるハッシュと同じファイルだけを入れます。
  - 初めて顔検出を使う時: anime-face-detector が、検出の重み（`yolov3`・`hrnetv2`）を、作者の GitHub のリリース（[hysts/anime-face-detector v0.0.1](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1)）から取ってきます。重みは `torch.load`（pickle）で読まれ、中身次第でプログラムが動いてしまうので、**読む前に SHA-256 を確かめ、公開されている物と違えば止めます**（`face_track_anime_detector.py`）。
- 読むのは、自分で選んだ動画・口の画像・口の位置のデータだけです。口の位置のデータ（`.npz`）は pickle を使わずに読み（`allow_pickle=False`）、口の画像は、ライブ実行・キャリブレーション・見た目確認のどれでも **PNG としてだけ** 読みます。
- 書くのは、選んだ動画と同じフォルダへの書き出し（`mouth_track.npz`・口消し動画・`mouth_track.json`・口の PNG 素材の `mouth/` など）と、前回の設定（リポジトリ直下の `.mouth_track_last_session.json`）です。作業用の一時フォルダは、終わったら消します。ライブ実行中の画面とのやり取りは、実行のたびに推測できない名前で作る一時フォルダのファイルで行います。
- ほかに起動するのは、このリポジトリの Python スクリプト（解析・口消しなど）、`ffmpeg`（PATH にあれば、動画の変換に）、Linux の `pactl`（マイクの選択）、Windows の `taskkill`（「停止」で処理を止める時）だけです。どれもシェルを通さずに起動します。口消し動画ができた後の確認の再生は OpenCV の窓で行い、開けない時（と macOS）は、その動画を OS の既定のアプリで開きます。
- サーバーとして待ち受けることはしません。README の、ブラウザで開くためのローカルサーバーは、`--bind 127.0.0.1` で自分の PC からだけ開けるようにしています。

## 直せない既知の脆弱性

顔検出（anime-face-detector 0.0.9）は、mmcv-full 1.7.0・mmdet 2.28.0・mmpose 0.29.0・torch 1.13.1 の組み合わせで動いています。mmcv-full 1.7.0 のビルド済みは torch 1.13 + CUDA 11.7 + Python 3.10 用しか無いため、次の 3 つは、脆弱性が直った版を入れられません。直せる物は直してあり、残っているのは この 3 つの分です（2026-11-29 時点。`uv audit` が出す 70 件を 1 件ずつ読み、下の使い方では届かないことを確かめました）。

| パッケージ | 版 | 直った版を入れられない理由 | このツールでの影響 |
| --- | --- | --- | --- |
| torch | 1.13.1 | 上の組み合わせに固定している | 外から来るデータで問題になるのは `torch.load`（重みの読み込み）で、読むのは SHA-256 を確かめた公開の重みだけです。ほかは、特定の関数（TorchScript・LSTM・CTC 損失など）に細工した引数を渡すと起きる種類の物です |
| Pillow | 12.3.0 | Pillow 13 以上は Python 3.11 以上が要る | Pillow が読むのは口の PNG だけで、PNG としてだけ読むので、TIFF・JPEG・PDF などの読み込みの穴には届きません。PNG の穴は、処理が遅くなる・メモリを多く使う種類の物です |
| setuptools | 69.5.1 | 70 以上では、torch 1.13.1 が起動時に使う `pkg_resources.packaging` が無い | どれも、パッケージを作る・入れる時の機能の穴で、ツールを動かしている間は使いません |

それでも、**知らない人から受け取った動画や口の画像は開かない**ことをおすすめします。

- GitHub Actions（毎週と、変更のたび）で、`tools/audit.py`（`uv audit`）が依存の既知の脆弱性を調べています。除くのは、上の 3 つの、影響を確かめた報告（ID を `tools/audit.py` に書いてある）だけです。上の 3 つでも、新しい報告が出れば、影響を確かめて ID を書き足すまで失敗します。
- `opencv-python` は、NumPy 2 が要る 4.12 以上に上げられないため 4.11.0.86 のままです（今は既知の脆弱性はありません）。OpenCV や Pillow が中に持っている画像・動画のライブラリ（FFmpeg・libpng・libtiff など）の脆弱性は、`uv audit` では見つけられません。
- `pyproject.toml` か `uv.lock` が変わった時は、全部の依存を入れて、顔検出が実際に動くかも確かめています。
- macOS（実験的）は手順に沿って手で入れるため、`uv.lock` と、この検査の対象外です。

## English

Please do not post vulnerability details in public issues. Use **Security → Report a vulnerability** (GitHub private vulnerability reporting) in this repository. If that is not possible, send a direct message to [@rotejin](https://x.com/rotejin) on X.

The tools run on your own PC. They connect to the network only (1) when you install them with `uv sync`, which installs only the files whose hashes are in `uv.lock`, and (2) the first time the face detector runs, when anime-face-detector downloads its weights from [hysts/anime-face-detector v0.0.1](https://github.com/hysts/anime-face-detector/releases/tag/v0.0.1). The weights are read with `torch.load` (pickle), so their SHA-256 is checked first and anything other than the published files is refused. The tools read only the videos, mouth images and tracking data you choose; `.npz` files are read with `allow_pickle=False` and mouth images with the PNG decoder only (in the live runtime, the calibration and the preview alike). They write the exports (tracking data, mouthless videos, `mouth_track.json`, the `mouth/` PNGs) next to the chosen video and the last settings in `.mouth_track_last_session.json`; temporary work folders are removed afterwards. Other programs started are the scripts in this repository, `ffmpeg` if it is on PATH, `pactl` on Linux and `taskkill` on Windows (to stop a running step), all without a shell; the finished mouthless video is played in an OpenCV window, or opened with the default app of the OS when that fails (and on macOS). Nothing listens on the network; the README starts its local web server with `--bind 127.0.0.1`.

Known vulnerabilities that cannot be fixed here: the face detector needs mmcv-full 1.7.0, whose prebuilt wheels exist only for torch 1.13 + CUDA 11.7 + Python 3.10. So torch stays at 1.13.1 (the weights it loads are checked by SHA-256), Pillow at 12.3.0 (Pillow 13 needs Python 3.11; only PNG mouth images are decoded), and setuptools below 70 (torch 1.13.1 needs `pkg_resources.packaging`; the remaining issues are in packaging and install features not used at run time). Even so, do not open videos or mouth images from people you do not know. CI runs `tools/audit.py` (`uv audit`) every week and on every change; it leaves out only the advisories for these three packages whose impact has been checked (their ids are listed in `tools/audit.py`), so a new advisory fails the check until it has been reviewed. opencv-python stays at 4.11.0.86 because 4.12 and later need NumPy 2 (it has no known vulnerabilities at the moment); vulnerabilities in the image and video libraries bundled inside OpenCV and Pillow (FFmpeg, libpng, libtiff, ...) are not visible to `uv audit`. CI also installs everything to check that the face detector still runs whenever `pyproject.toml` or `uv.lock` changes. The experimental macOS setup is installed by hand and is not covered by `uv.lock` or these checks.
