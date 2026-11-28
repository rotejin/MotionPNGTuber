# サポート / Support

個人で作っているプロジェクトのため、返事や修正は できる範囲で行います。

## 先に見てほしいもの

- [README の「トラブルシューティング」](../README.ja.md#-詳細リファレンス)（「❓ トラブルシューティング」を開いてください）
- [README の「インストール」](../README.ja.md#-インストール)
- [使い方の動画（YouTube）](https://www.youtube.com/watch?v=mxZHzZ_eAkY)

## 質問・不具合の報告

[Issue](https://github.com/rotejin/MotionPNGTuber/issues/new/choose) から、内容に合うテンプレートを選んでください。

- **不具合の報告**: 入らない（`uv sync`）、起動しない、解析が止まる、口の位置がずれる、口消しがおかしい、ライブ実行で口が動かない
- **要望**: こんな機能がほしい

ほかのリポジトリのことは、そちらへお願いします。

- ブラウザのプレイヤーのこと: [MotionPNGTuber_Player の Issue](https://github.com/rotejin/MotionPNGTuber_Player/issues)
- アバター素材集のこと: [MotionPNGTuber_Avatars の Issue](https://github.com/rotejin/MotionPNGTuber_Avatars/issues)

脆弱性は Issue に書かず、[セキュリティポリシー](SECURITY.md) の方法で報告してください。

## 作者への連絡

Issue に合わない連絡は、作者の X（[@rotejin](https://x.com/rotejin)）へどうぞ。不具合や質問は、同じことで困った人も見られるように、なるべく Issue にお願いします。

## 書いてほしいこと

- OS（例: Windows 11 24H2）
- Python のバージョン（`uv run python --version`）
- GPU の型番と、CUDA が使えるか（`uv run python -c "import torch; print(torch.cuda.is_available())"`）
- どの画面か（メイン GUI・口 PNG 素材作成 GUI・コマンド）
- 起きたことと、その手順。画面のログ欄やターミナルに出たメッセージ
- 動画の大きさと長さ（例: 1920×1080、25fps、8 秒）

## お答えできないこと

- macOS（実験的）での不具合は、作者の手元で確かめられないため、直せないことがあります。
- 動画の作り方や、画像生成 AI の使い方そのもの。

Issue は誰でも見られます。スクリーンショットや動画を貼る前に、個人情報や、使ってよいか分からない画像が写っていないか確かめてください。同梱のサンプル（妹尾トマリ）のスクリーンショットは貼ってかまいません（[ASSETS.md](../ASSETS.md)）。

## English

Support is best-effort. Please check Installation and Troubleshooting in the [README](../README.md) and the [video guide (Japanese)](https://www.youtube.com/watch?v=mxZHzZ_eAkY) first, then open an [issue](https://github.com/rotejin/MotionPNGTuber/issues/new/choose) with your operating system, Python version, GPU and whether CUDA is available, the screen you used, the steps to reproduce, the log messages, and the size and length of the video. For the browser player use [MotionPNGTuber_Player](https://github.com/rotejin/MotionPNGTuber_Player/issues); for the avatar pack use [MotionPNGTuber_Avatars](https://github.com/rotejin/MotionPNGTuber_Avatars/issues). The experimental macOS setup cannot be tested by the author, so macOS problems may stay unfixed. Questions in English are welcome. Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md). For anything else, contact the author on X: [@rotejin](https://x.com/rotejin).
