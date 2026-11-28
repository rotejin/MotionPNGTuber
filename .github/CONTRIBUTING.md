# 開発への参加 / Contributing

> English follows the Japanese.

不具合の報告や改善の提案を歓迎します。個人で作っているプロジェクトのため、返事や取り込みは できる範囲で行います。

## Issue

- 不具合・要望は、[Issue のテンプレート](https://github.com/rotejin/MotionPNGTuber/issues/new/choose) から書いてください（書いてほしいことは [SUPPORT.md](SUPPORT.md)）。
- 脆弱性は公開の Issue に書かず、[SECURITY.md](SECURITY.md) の方法で報告してください。

## Pull Request

大きな変更は、先に Issue で相談してください。小さな修正（誤字、文書の改善、明らかな不具合の修正）はそのまま送ってかまいません。

送る前に確かめてほしいこと:

- `uv sync` のあと、`uv run python -m unittest discover -s tests -v` が全部 ok になる。
- 画面や動きを変えたら、変えた画面を実際に起動して、同梱のサンプル（`assets/asmr_tomari/`）で確かめる。作者は **Windows** で確かめます。
- 依存パッケージを変えたら、`pyproject.toml` と `uv.lock` の両方を送り（`uv lock`）、`python tools/audit.py` が OK になることを確かめる。
- **macOS の手順で `pyproject.toml` を `pyproject.macos.toml` に差し替えた場合は、その `pyproject.toml` と `uv.lock` を送らないでください。** Windows・Linux で torch などが入らなくなります。
- macOS 向けの修正は、作者が確かめられないため、Windows・Linux の動きを変えない形でお願いします。
- 調べるために足した `print` やファイルへの書き出し（例: `/tmp/debug.txt`）は、消してから送る。
- 改行は LF にする（`.gitattributes`・`.editorconfig` のとおり）。
- 文書を変えたら、`README.ja.md` と `README.md` の両方をそろえる。英語が難しければ、日本語だけで送ってもかまいません。

## 画像・動画について

- 不具合の説明のために、画像やスクリーンショットを添えるのはかまいません。他人のキャラクターや、使ってよいか分からない画像は含めないでください。
- 同梱のサンプル（`assets/asmr_tomari/`・`mouth_dir/Tomari/`）を変える Pull Request は、受け付けていません。サンプルは MIT License の対象外です（[ASSETS.md](../ASSETS.md)）。

## ライセンス

送ってもらったコード・文書は、このリポジトリと同じ [MIT License](../LICENSE) で公開されます。

---

## English

Bug reports and suggestions are welcome. This is a personal project, so replies and merges are best-effort.

### Issues

- Use the [issue templates](https://github.com/rotejin/MotionPNGTuber/issues/new/choose) for bugs and feature requests ([SUPPORT.md](SUPPORT.md) lists what to include).
- Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in public issues.

### Pull requests

Please open an issue first for larger changes. Small fixes (typos, documentation, obvious bugs) can be sent directly.

Before sending:

- After `uv sync`, `uv run python -m unittest discover -s tests -v` reports all ok.
- If you changed the UI or behaviour, start the screen you changed and try it with the bundled sample (`assets/asmr_tomari/`). The author tests on **Windows**.
- If you changed dependencies, send both `pyproject.toml` and `uv.lock` (`uv lock`) and check that `python tools/audit.py` reports OK.
- **If you replaced `pyproject.toml` with `pyproject.macos.toml` for the macOS setup, do not send that `pyproject.toml` or `uv.lock`.** It stops torch and the other packages from installing on Windows and Linux.
- The author cannot test on macOS, so macOS fixes must not change how the tools behave on Windows and Linux.
- Remove any `print` calls or debug files (such as `/tmp/debug.txt`) you added while investigating.
- Use LF line endings (see `.gitattributes` and `.editorconfig`).
- Documentation changes keep `README.ja.md` and `README.md` in step. If you cannot write Japanese, English only is fine.

### Images and videos

- Images and screenshots that explain a bug are fine. Do not include other people's characters or images you may not share.
- Pull requests that change the bundled samples (`assets/asmr_tomari/`, `mouth_dir/Tomari/`) are not accepted. The samples are not covered by the MIT License ([ASSETS.md](../ASSETS.md)).

### License

Code and documents you contribute are published under the same [MIT License](../LICENSE) as this repository.
