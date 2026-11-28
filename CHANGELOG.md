# 変更履歴 / Changelog

このリポジトリの主な変更を記録します。それより前の細かい更新は、README の「更新情報」を見てください。
Notable changes are recorded here. For earlier updates, see the update table at the top of the README.

## Unreleased

### セキュリティ / Security

- キャリブレーションと「見た目確認（軽量）」も、口の画像を PNG としてだけ読むようにした。今までは OpenCV が中身から形式を選んでいて、`.png` という名前の TIFF・JPEG・WebP などもそのまま読んでいた（ライブ実行は v0.1.2 から PNG だけ）。グレースケール・パレット・16-bit の PNG は、どの画面でも同じ 8-bit の色で読む（16-bit のグレースケールは、今まで真っ白になっていた）。
- [SECURITY.md](.github/SECURITY.md) の「ほかに起動する物」に、Windows の `taskkill`（停止の時）と、口消し動画の確認に OS の既定のアプリで開くことを書き足した。

The calibration and the lightweight preview now read mouth images with the PNG decoder only, as the live runtime does since v0.1.2; before, OpenCV picked the decoder from the contents and also read TIFF, JPEG, WebP and other files named `.png`. Grayscale, palette and 16-bit PNGs are read with the same 8-bit colours everywhere (16-bit grayscale used to become white). SECURITY.md now also lists `taskkill` on Windows and opening the finished video with the default app.

### 修正 / Fixed

- 「② 口消し動画生成」の後に「mouthフォルダを選択」で口の素材を選び直すと、背景の動画が口消し動画から元の動画に戻り、「③ ライブ実行」で口が二重になっていた。今は、口のフォルダを選んでも背景の口消し動画はそのまま残る（動画の欄を別の動画に書き換えていた時だけ、新しい動画として扱う）。

Choosing the mouth folder again after "② 口消し動画生成" switched the background back from the mouthless video to the original one, so "③ ライブ実行" showed two mouths. Choosing the mouth folder now keeps the mouthless background (unless the video field was changed to a different video).

## v0.1.2 - 2026-11-28

### セキュリティ / Security

- 依存の脆弱性を直した: Pillow 12.3.0・urllib3 2.8.0・requests 2.34.2・certifi・idna・pip・Pygments。
- 口の画像は PNG としてだけ読むようにした（`.png` を装った別の形式のファイルで、Pillow のほかの読み込み処理の穴に届かないように）。
- 顔検出の重み（anime-face-detector が初回に取ってくる物）を、`torch.load` で読む前に SHA-256 で確かめるようにした。公開されている物と違えば止める。
- 直せない既知の脆弱性（torch 1.13.1・Pillow 12.3.0・setuptools 69.5.1）と、その理由・影響を [SECURITY.md](.github/SECURITY.md) に書いた。
- README の、ブラウザで開くためのローカルサーバーを `--bind 127.0.0.1` で起動するようにした。

### そのほか / Other

- 同梱のサンプル（妹尾トマリ）が MIT License の対象外であることと、その利用条件を [ASSETS.md](ASSETS.md) に書いた。
- e2e のテストが、リポジトリに無いサンプルを見ていて、ずっと飛ばされていたのを直した。
- 改行を LF にそろえた（`.gitattributes`・`.editorconfig`）。
- セキュリティポリシー・サポートの案内・開発への参加の案内・Issue と Pull Request のテンプレート・CI（Windows と Ubuntu でのテスト、依存の脆弱性の検査、依存が変わった時の全部入りの検査。テストと脆弱性の検査は毎週も走る）・Dependabot（GitHub Actions の更新）を足した。Python の依存は、Dependabot が mmcv-full の配布物を取りに行けず更新できないため、毎週の CI で調べる。

Fixed the vulnerable dependencies (Pillow 12.3.0, urllib3 2.8.0, requests 2.34.2, certifi, idna, pip, Pygments). Mouth images are decoded as PNG only, and the face detector weights are checked by SHA-256 before `torch.load` reads them. The known vulnerabilities that cannot be fixed here (torch 1.13.1, Pillow 12.3.0, setuptools 69.5.1) are explained in SECURITY.md. The README starts its local web server with `--bind 127.0.0.1`. The bundled sample character (Tomari) is not covered by the MIT License; see ASSETS.md. Fixed the e2e tests that were always skipped, normalized line endings to LF, and added the security policy, support and contributing guides, issue and pull request templates, CI (also run every week) and Dependabot for GitHub Actions. Dependabot cannot update the Python dependencies here (it cannot fetch the mmcv-full wheel), so the weekly CI checks them instead.

## v0.1.1 - 2026-04-10

- サンプルを `assets/asmr_tomari/` にまとめた（動画の差し替え、口の画像・口の位置のデータの追加）。古いサンプル（`assets01`・`assets03`）を消した。 / Moved the sample to `assets/asmr_tomari/` and removed the old samples.

## v0.1.0 - 2026-04-10

- 最初のリリース（開発者版のスナップショット）。 / First release (developer preview snapshot).
