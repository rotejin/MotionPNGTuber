"""依存の既知の脆弱性を調べる（uv audit）。標準ライブラリだけで動きます。

    python tools/audit.py

uv audit が見つけた脆弱性のうち、次の 2 つを両方満たす物だけを除き、それ以外が
1 つでもあれば失敗します。
- 直る版が、このアプリの制約では入れられない（HELD_BACK。理由は .github/SECURITY.md の
  「直せない既知の脆弱性」と .github/dependabot.yml のコメントと同じ）
- このアプリへの影響を確かめてある（REVIEWED に、その ID か別名の ID がある）
除いているパッケージでも、新しい報告は、影響を確かめて REVIEWED に ID を書き足すまで
失敗します。直る版を入れられるようになった報告も失敗します。

Checks the locked dependencies with uv audit. A vulnerability is left out only if its
fixed versions cannot be installed here (HELD_BACK) and its impact on this app has been
checked (its id or one of its aliases is in REVIEWED). Anything else makes the check fail,
including a new advisory for a held-back package, until it is checked and added here.
"""
import json
import re
import shutil
import subprocess
import sys

# パッケージ名: 直る版のメジャー版がこの数以上なら、今は入れられない。
# None は、どの版で直っても入れられない（その版に固定している）。
HELD_BACK = {
    # 顔検出（anime-face-detector 0.0.9）が使う mmcv-full 1.7.0 のビルド済みが、
    # torch 1.13 + CUDA 11.7 + Python 3.10 用しか無いので、torch 1.13.1 に固定している。
    "torch": None,
    # Pillow 13 以上は Python 3.11 以上が要る（このアプリは、上の理由で Python 3.10）。
    "pillow": 13,
    # setuptools 70 以上では、torch 1.13.1 が起動時に使う pkg_resources.packaging が無い。
    "setuptools": 70,
}

# このアプリへの影響を確かめた報告（2026-11-29、uv audit の 70 件・106 の ID）。
# 同じ報告の GHSA と PYSEC の ID を両方書く（uv は、どちらかを id に、残りを aliases に出す）。
# 新しい報告が出たら: 内容を読み、届く使い方をしていないか確かめてから、ここに足す。
# 影響が .github/SECURITY.md の説明と違う時は、SECURITY.md も直す。
REVIEWED = {
    "torch": frozenset({
        # 外から来たモデルのファイルを読むと起きる物。読むのは、SHA-256 を確かめた
        # 公開の重みだけ（face_track_anime_detector.py）。
        "GHSA-53q9-r3pm-6pq6", "PYSEC-2025-41",  # torch.load(weights_only=True) RCE
        "GHSA-63cw-57p8-fm3p", "PYSEC-2026-1856", "PYSEC-2026-2286",  # untrusted checkpoint RCE
        "PYSEC-2026-139",  # pt2 loading handler deserialization
        "PYSEC-2024-259",  # RemoteModule deserialization (distributed; not used)
        # 特定の関数・機能に細工した引数を渡すと起きる物（JIT・LSTM・量子化・Inductor など）。
        # 顔検出は、決まったモデルに動画のフレームを渡すだけ。
        "GHSA-3749-ghw9-m3mg", "PYSEC-2025-191",  # mkldnn_max_pool2d DoS
        "GHSA-5pcm-hx3q-hm94", "PYSEC-2024-252", "PYSEC-2024-328",  # vararg_functions heap overflow
        "GHSA-887c-mr87-cxwp", "PYSEC-2026-1970",  # improper resource shutdown
        "GHSA-c678-jfcj-6jmf", "PYSEC-2025-189",  # tuple handler, None argument
        "GHSA-f4hp-rmr7-r7v8", "PYSEC-2025-192",  # pad_packed_sequence memory use
        "GHSA-pg7h-5qx3-wjr3", "PYSEC-2024-251",  # use-after-free (torch.nn.functional)
        "GHSA-qfhq-4f3w-5fph", "PYSEC-2025-195",  # torch.lstm_cell
        "GHSA-rrmf-rvhw-rf47", "PYSEC-2025-194",  # torch.jit.script
        "GHSA-vgrw-7cvw-pwgx", "PYSEC-2025-193",  # unpack_sequence
        "GHSA-x3gm-94wq-g975", "PYSEC-2025-190",  # quantized sigmoid scale/zero_point
        "PYSEC-2024-250",  # mobile flatbuffer loader OOB read
        "PYSEC-2025-198",  # nn.PairwiseDistance wrong results
        "PYSEC-2025-203",  # torch.linalg.lu DoS
        "PYSEC-2025-204",  # torch.rot90 + randn_like
        "PYSEC-2025-205",  # proxy_tensor.py syntax error (Inductor)
        "PYSEC-2025-206",  # torch.nan_to_num integer overflow
        "PYSEC-2025-207",  # torch.cummin + Inductor
        "PYSEC-2025-208",  # Conv2d + hardshrink + Inductor
        "PYSEC-2025-209",  # to_sparse / to_dense + Inductor
    }),
    "pillow": frozenset({
        # PNG の読み込みの物。口の画像は PNG としてだけ、最初のフレームだけを読み、
        # フレームを進める（seek）ことも EXIF を解析することもしない。起きても、
        # 遅くなる・メモリを多く使う種類。
        "GHSA-5xm2-6676-qcp9",  # APNG frame seek allocates before the bomb check
        "GHSA-g4qv-2jxx-769h", "PYSEC-2026-4342",  # PNG eXIf parsing quadratic time
        "GHSA-v9pv-5gc8-w2m6", "PYSEC-2026-4339",  # multi-frame seek reuses storage
        # ほかの形式の読み込みの物。PNG 以外の読み込み処理には届かない。
        "GHSA-22hg-r758-rrw9",  # MIC (TIFF inside) n_frames loop
        "GHSA-24rc-8hj2-v98c",  # BMP RLE memory
        "GHSA-254f-3936-288h",  # PDF parser recursion
        "GHSA-4558-3h8r-v8xp", "PYSEC-2026-4327",  # raw P;2L / P;4L decoders
        "GHSA-462x-pgg6-55fp",  # tiled TIFF decoder
        "GHSA-58r4-6rv3-gfh8", "PYSEC-2026-4329",  # ImageMorph on zero-width PPM/PGM
        "GHSA-5x8w-8jj5-q52h", "PYSEC-2026-4330",  # FlashPix JPEG tiles
        "GHSA-65qm-jvmh-6pqg",  # GIF seek after DecompressionBombError
        "GHSA-74r2-7ff9-mjj2",  # GIF comments in Image.open()
        "GHSA-78g4-xqwv-q3g8",  # bundled libopenjp2 (JPEG 2000)
        "GHSA-8g5g-8vmr-9782", "PYSEC-2026-4332",  # JPEG decoder with draft()
        "GHSA-9v9v-9qvf-h2hj", "PYSEC-2026-4343",  # WMF / EMF rendering
        "GHSA-c6hw-5grf-ff96",  # TIFF tile / RGBA paths
        "GHSA-cp9v-662q-vvx9", "PYSEC-2026-4333",  # Image.merge() via IPTC
        "GHSA-f2g6-542v-9j58",  # PDF page tree loop
        "GHSA-g842-9rc4-69pw", "PYSEC-2026-4338",  # SPIDER seek bomb check
        "GHSA-gg2r-wvfm-q9rx",  # MSP RLE memory
        "GHSA-mrfp-p49p-3hf2",  # PDF append page tree
        "GHSA-p4wg-p73j-c528",  # PDF page tree recursion
        "GHSA-p8hx-v2p6-rp84", "PYSEC-2026-4337",  # MPO / DCX seek bomb check
        "GHSA-q45j-58j2-45rg",  # TIFF short JPEG strips
        "GHSA-vx6w-vv9f-6gvg",  # EPS (Ghostscript) bomb limit
        "GHSA-w889-685w-c8fh",  # EPS Ghostscript argument injection
        "GHSA-wqgw-9xfh-qf2g",  # TIFF / EXIF IFD memory
        "GHSA-xgh8-jp6q-p5x8", "PYSEC-2026-4335",  # XBM header regex
        "GHSA-3j72-p6qg-6xp7", "PYSEC-2026-4341",  # TarIO negative size
        # 保存（書き出し）の物。Pillow で画像を保存しない（口の PNG は OpenCV で書く）。
        "GHSA-332p-7hrc-92wf",  # TIFF save to BytesIO
        "GHSA-4rr5-89j8-mwwm", "PYSEC-2026-4328",  # TIFF encode to memory
        "GHSA-fc9g-5qg7-mf45",  # GIF comments when saving
        "GHSA-g7cr-3p24-7wf7",  # JPEG comment after ICC profile
        "GHSA-jg86-v3jv-hrpm",  # PDF save_all
        "GHSA-pchr-8j38-3x4p", "PYSEC-2026-4334",  # TIFF ColorMap when saving
        "GHSA-xcvg-w56q-r89m",  # TIFF save with libtiff
        "GHSA-xvh5-g6gv-559v", "PYSEC-2026-4363",  # libtiff encoder tags
        # 使っていない機能の物（文字の描画・Arrow・減色・ランクフィルタ・アフィン変換）。
        # 使うのは open・convert・fromarray・resize（LANCZOS）・ImageTk だけ。
        "GHSA-27h8-gf7j-v5hc", "PYSEC-2026-4326",  # rank filters on a zero-size image
        "GHSA-4rm8-3hmm-39fg", "PYSEC-2026-4340",  # PILfont text width
        "GHSA-654x-cwwv-5j9c", "PYSEC-2026-4336",  # Arrow export
        "GHSA-7mwm-ghpf-9vrm", "PYSEC-2026-4331",  # ScaleAffine (transform, NEAREST)
        "GHSA-j4m9-6w36-pw47",  # FreeType stroker
        "GHSA-rfgq-q3w2-pmv2",  # quantize() with an empty palette
    }),
    "setuptools": frozenset({
        # パッケージを作る・入れる時の機能の物。ツールを動かしている間は使わない。
        "GHSA-5rjg-fvgr-3xxf", "PYSEC-2025-49",  # PackageIndex path traversal
        "GHSA-cx63-2mw6-8hw5", "PYSEC-2026-1918",  # package URL command injection
        "GHSA-h35f-9h28-mq5c", "PYSEC-2026-3447",  # MANIFEST.in exclusion in sdist
        "GHSA-xmj3-9gf7-h52w",  # namespace_packages.txt in legacy wheel install
    }),
}


def major(version):
    match = re.match(r"\d+", version)
    return int(match.group()) if match else 0


def cannot_install_fix(package, fix_versions):
    """True if no fixed version can be installed here (or no fix exists yet)."""
    if package not in HELD_BACK:
        return False
    floor = HELD_BACK[package]
    if floor is None or not fix_versions:
        return True
    return all(major(v) >= floor for v in fix_versions)


def is_reviewed(package, vulnerability):
    ids = {vulnerability["id"], *(vulnerability.get("aliases") or [])}
    return bool(ids & REVIEWED.get(package, frozenset()))


def run_uv_audit():
    """Return (report, stderr), or (None, message) if uv cannot be run."""
    uv = shutil.which("uv")
    if uv is None:
        return None, "uv が見つかりません / uv is not installed"
    result = subprocess.run(
        [uv, "audit", "--locked", "--preview-features", "audit-command,json-output",
         "--output-format", "json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    try:
        return json.loads(result.stdout), result.stderr
    except ValueError:
        return None, "uv audit の結果を読めません / Could not read the uv audit report\n" + result.stderr


def main():
    report, error = run_uv_audit()
    try:
        vulnerabilities = report["vulnerabilities"]
        adverse = report.get("adverse_statuses", [])
    except (KeyError, TypeError):
        print("NG: " + (error or "uv audit の結果を読めません / Could not read the uv audit report"))
        return 1

    held = {}
    new = []
    fixable = []
    for v in vulnerabilities:
        package = v["dependency"]["name"].lower()
        if not cannot_install_fix(package, v.get("fix_versions") or []):
            fixable.append(v)
        elif is_reviewed(package, v):
            held[package] = held.get(package, 0) + 1
        else:
            new.append(v)

    for package, count in sorted(held.items()):
        print("除外 / held back: %s %d 件（影響を確かめた報告。%s）" % (package, count, describe(package)))

    if new or fixable or adverse:
        print("NG: 影響を確かめていない新しい報告 %d 件・直せる脆弱性 %d 件・注意が要る状態 %d 件"
              % (len(new), len(fixable), len(adverse)))
        print("    / %d new advisories not reviewed yet, %d fixable, %d adverse statuses"
              % (len(new), len(fixable), len(adverse)))
        for title, items in (("新しい報告 / new", new), ("直せる / fixable", fixable)):
            for v in items:
                dep = v["dependency"]
                fixes = ", ".join(v.get("fix_versions") or []) or "なし / none"
                aliases = ", ".join(v.get("aliases") or [])
                print("  - [%s] %s %s: %s %s" % (title, dep["name"], dep["version"], v["id"], v.get("summary") or ""))
                print("    別名 / aliases: %s" % (aliases or "なし / none"))
                print("    直る版 / fixed in: %s  %s" % (fixes, v.get("link", "")))
        for status in adverse:
            print("  - %s" % json.dumps(status, ensure_ascii=False))
        if new:
            print("新しい報告は、内容を読み、このアプリで届かないことを確かめてから、tools/audit.py の")
            print("REVIEWED に ID を足してください（影響が SECURITY.md の説明と違えば、そちらも直す）。")
            print("Read each new advisory, check that this app cannot reach it, then add its id to")
            print("REVIEWED in tools/audit.py (and update SECURITY.md if the impact is different).")
        return 1
    print("OK: 影響を確かめていない報告・直せる脆弱性はありません / No unreviewed or fixable vulnerabilities (%d packages)"
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
