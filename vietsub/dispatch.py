#!/usr/bin/env python3
"""dispatch.py — quyết định file giao + có tạo GitHub Release không.

Trả JSON trên stdout, workflow đọc để:
  - chọn đường dẫn file (đã shrink hay nguyên bản)
  - quyết định `gh release create` có cần chạy không
  - biết file size để log/đặt tên

Env đọc:
  OUTPUT_MODE   : auto | artifact | release | both   (mặc định 'auto')
  MAX_MB        : số MB tối đa (mặc định '500')
  VIDEO_PATH    : đường dẫn file MP4 từ render.py

Decision matrix:
  auto    : ≤ MAX_MB → upload_artifact ; > MAX_MB → shrink về ≤ MAX_MB rồi
            upload_artifact. Nếu shrink fail → tạo release (giữ file lớn hơn).
  artifact: ép ≤ MAX_MB, đẩy artifact. Fail thì báo lỗi (không tự đẩy release).
  release : ép ≤ RELEASE_MAX_MB (2000), tạo release.
  both    : shrink về ≤ MAX_MB, upload artifact + tạo release.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ARTIFACT_MAX_MB = 500.0
RELEASE_MAX_MB = 2000.0


def size_mb(p: str) -> float:
    return os.path.getsize(p) / (1024 * 1024)


def shrink_to(in_path: str, max_mb: float) -> tuple[str, dict]:
    """Ép file về ≤ max_mb. Nếu file đã nhỏ hơn thì trả về luôn (skip)."""
    if size_mb(in_path) <= max_mb:
        return in_path, {"ok": True, "mb": round(size_mb(in_path), 2),
                         "scale_h": None, "skipped": True}
    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.splitext(in_path)[0] + f"_shrunk_{int(max_mb)}mb.mp4"
    cp = subprocess.run([sys.executable, os.path.join(here, "shrink.py"),
                         "--in", in_path, "--out", out_path,
                         "--max-mb", str(max_mb)],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"shrink thất bại:\n{cp.stderr[-1500:]}\n{cp.stdout}")
    # shrink.py print JSON pretty-printed nhiều dòng; lấy NGUYÊN stdout rồi parse
    # (trước đây dùng splitlines()[-1] chỉ bắt được '}' cuối → JSONDecodeError).
    info = json.loads(cp.stdout.strip())
    if not info.get("ok"):
        raise RuntimeError(f"file vẫn {info.get('mb')}MB sau khi thử shrink")
    return out_path, info


def main() -> int:
    try:
        mode = (os.environ.get("OUTPUT_MODE", "auto") or "auto").lower()
        max_mb_raw = os.environ.get("MAX_MB", "500")
        max_mb = float(max_mb_raw)
        in_path = os.environ["VIDEO_PATH"]
    except (KeyError, ValueError) as e:
        print(json.dumps({"error": f"env lỗi: {e}"}))
        return 2

    if not os.path.isfile(in_path):
        print(json.dumps({"error": f"không thấy {in_path}"}))
        return 2

    mb = size_mb(in_path)
    out_path = in_path
    do_artifact = False
    do_release = False
    note = ""

    # Caps: 'artifact' giữ margin 50MB dưới GitHub hard limit 500.
    # Khi user đặt max_mb < 50 thì tăng floor lên 5MB để cap không âm.
    ARTIFACT_CAP = max(min(max_mb, ARTIFACT_MAX_MB) - 50, 5)
    # Release cap: cho phép tăng max_mb > 500 (user muốn chất lượng cao)
    RELEASE_CAP = min(RELEASE_MAX_MB, max(max_mb, 500)) - 50

    try:
        if mode == "auto":
            if mb <= ARTIFACT_CAP:
                do_artifact = True
                note = f"{mb:.1f}MB ≤ {max_mb}MB → artifact"
            else:
                # cố shrink artifact trước (ưu tiên artifact vì nhanh)
                try:
                    out_path, _ = shrink_to(in_path, ARTIFACT_CAP)
                    note = f"shrank {mb:.1f}MB → {size_mb(out_path):.1f}MB ≤ 500MB → artifact"
                    do_artifact = True
                except Exception:
                    # không nổi → đẩy release
                    if mb > RELEASE_CAP:
                        out_path, _ = shrink_to(in_path, RELEASE_CAP)
                        note = f"shrank {mb:.1f}MB → {size_mb(out_path):.1f}MB → release"
                    else:
                        out_path = in_path
                        note = f"{mb:.1f}MB > {max_mb}MB nhưng ≤ 2GB → release"
                    do_release = True

        elif mode == "artifact":
            out_path, _ = shrink_to(in_path, ARTIFACT_CAP)
            do_artifact = True
            note = f"shrank to {size_mb(out_path):.1f}MB ≤ 500MB → artifact"

        elif mode == "release":
            out_path, _ = shrink_to(in_path, RELEASE_CAP)
            do_release = True
            note = f"shrank to {size_mb(out_path):.1f}MB → release"

        elif mode == "both":
            # ép nhỏ để cả artifact + release đều nhận
            out_path, _ = shrink_to(in_path, ARTIFACT_CAP)
            do_artifact = True
            do_release = True
            note = f"shrank to {size_mb(out_path):.1f}MB → both"

        else:
            note = f"unknown mode={mode}; auto fallback"
            do_artifact = mb <= ARTIFACT_CAP
            do_release = not do_artifact

    except Exception as e:
        print(json.dumps({"error": str(e), "mb": mb, "path": in_path}, ensure_ascii=False))
        return 3

    result = {
        "mode": mode,
        "max_mb_input": max_mb,
        "original_mb": round(mb, 2),
        "final_mb": round(size_mb(out_path), 2),
        "path": out_path,
        "do_artifact": do_artifact,
        "do_release": do_release,
        "note": note,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
