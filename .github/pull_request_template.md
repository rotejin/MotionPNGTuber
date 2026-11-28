## 変更の内容 / Summary

<!-- 何を、なぜ変えたか / What changed and why -->

関連する Issue / Related issue: #

## 確認したこと / Checklist

- [ ] `uv run python -m unittest discover -s tests -v` が全部 ok だった / The tests are all ok
- [ ] 画面や動きを変えた場合、変えた画面を起動してサンプルで確かめた（OS: ） / For UI or behaviour changes, started the screen and tried it with the sample (OS: )
- [ ] 依存パッケージを変えた場合、`pyproject.toml` と `uv.lock` の両方を含め、`python tools/audit.py` が OK だった / For dependency changes, both `pyproject.toml` and `uv.lock` are included and `python tools/audit.py` is OK
- [ ] macOS 用に差し替えた `pyproject.toml`・`uv.lock` を含めていない / Does not include a `pyproject.toml` or `uv.lock` replaced for macOS
- [ ] 調べるために足した `print` やデバッグ用のファイル出力を消した / Removed debug prints and debug files
- [ ] 文書を変えた場合、日本語版と英語版をそろえた（どちらか一方でも可） / Kept Japanese and English docs in step (one language is fine)
- [ ] 使ってよいか分からない画像・動画を含まない / No images or videos I may not share
- [ ] 送った内容が MIT License で公開されることに同意する / I agree that this contribution is published under the MIT License
