"""依存の既知の脆弱性を調べる（uv audit）。標準ライブラリだけで動きます。

    python tools/audit.py

uv audit が見つけた脆弱性のうち、「直る版が、このアプリの制約では入れられない」物
だけを除き、それ以外が 1 つでもあれば失敗します。除く理由は .github/SECURITY.md の
「直せない既知の脆弱性」に書いてあります（.github/dependabot.yml のコメントと同じ理由）。

Checks the locked dependencies with uv audit. Only vulnerabilities whose fixed
versions this project cannot install are left out (see .github/SECURITY.md);
anything else makes the check fail.
"""
import json
import re
import shutil
import subprocess
import sys

# パッケージ名: 直る版のメジャー版がこの数以上なら、今は入れられないので除く。
# None は、どの版で直っても除く（その版に固定している）。
HELD_BACK = {
    # 顔検出（anime-face-detector 0.0.9）が使う mmcv-full 1.7.0 のビルド済みが、
    # torch 1.13 + CUDA 11.7 + Python 3.10 用しか無いので、torch 1.13.1 に固定している。
    "torch": None,
    # Pillow 13 以上は Python 3.11 以上が要る（このアプリは、上の理由で Python 3.10）。
    "pillow": 13,
    # setuptools 70 以上では、torch 1.13.1 が起動時に使う pkg_resources.packaging が無い。
    "setuptools": 70,
}


def major(version):
    match = re.match(r"\d+", version)
    return int(match.group()) if match else 0


def cannot_install_fix(package, fix_versions):
    if package not in HELD_BACK:
        return False
    floor = HELD_BACK[package]
    if floor is None or not fix_versions:
        return True
    return all(major(v) >= floor for v in fix_versions)


def main():
    uv = shutil.which("uv")
    if uv is None:
        print("NG: uv が見つかりません / uv is not installed")
        return 1
    result = subprocess.run(
        [uv, "audit", "--locked", "--preview-features", "audit-command,json-output",
         "--output-format", "json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    try:
        report = json.loads(result.stdout)
        vulnerabilities = report["vulnerabilities"]
        adverse = report.get("adverse_statuses", [])
    except (ValueError, KeyError, TypeError):
        print("NG: uv audit の結果を読めません / Could not read the uv audit report")
        print(result.stderr)
        return 1

    held = {}
    problems = []
    for v in vulnerabilities:
        package = v["dependency"]["name"]
        if cannot_install_fix(package, v.get("fix_versions") or []):
            held[package] = held.get(package, 0) + 1
        else:
            problems.append(v)

    for package, count in sorted(held.items()):
        print("除外 / held back: %s %d 件（%s）" % (package, count, describe(package)))

    if problems or adverse:
        print("NG: 直せる脆弱性 %d 件・注意が要る状態 %d 件" % (len(problems), len(adverse)))
        for v in problems:
            dep = v["dependency"]
            fixes = ", ".join(v.get("fix_versions") or []) or "なし / none"
            print("  - %s %s: %s %s" % (dep["name"], dep["version"], v["id"], v.get("summary") or ""))
            print("    直る版 / fixed in: %s  %s" % (fixes, v.get("link", "")))
        for status in adverse:
            print("  - %s" % json.dumps(status, ensure_ascii=False))
        return 1
    print("OK: 直せる脆弱性はありません / No fixable vulnerabilities (%d packages)"
          % report.get("summary", {}).get("audited_packages", 0))
    return 0


def describe(package):
    floor = HELD_BACK[package]
    if floor is None:
        return "固定している版 / pinned"
    return "直る版が %d 以上 / fixed only in %d or later" % (floor, floor)


if __name__ == "__main__":
    # Windows のコンソールでも、日本語がそのまま出るようにする
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
