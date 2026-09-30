#!/usr/bin/env python3
"""shrink.py — nén MP4 để file ≤ max_mb mà giữ chất lượng tối đa.

Chiến lược (2 phase):
  1) 2-pass encode với bitrate mục tiêu:
       total_kbps = max_mb * 8192 / duration
       video_kbps = max(total_kbps - audio_kbps,  200)
  2) Nếu vẫn > max_mb: giảm scale (lần lượt 1080p→720p→540p→480p), lặp 2-pass.

Audio: AAC 128kbps (tiết kiệm bitrate mà vẫn rõ). Nếu video gốc là copy được
thì re-encode audio để tiết kiệm; nếu audio < 128kbps thì giữ nguyên (copy).

Dùng ngay sau render.py khi file gốc vượt ngưỡng artifact (500MB) hoặc
release (2GB) — workflow tự route theo output_mode.

Usage:
  shrink.py --in <video> --out <video> --max-mb <MB> [--audio <kbps>]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys


def run(cmd) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n"
                           f"--- stderr ---\n{cp.stderr[-2000:]}")
    return cp


def probe_duration(video: str) -> float:
    cp = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", video])
    return float(cp.stdout.strip() or 0)


def probe_resolution(video: str) -> tuple[int, int]:
    cp = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video])
    w, h = cp.stdout.strip().split("x")[:2]
    return int(w), int(h)


def size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def two_pass(src: str, dst: str, video_kbps: int, audio_kbps: int,
             scale_h: int | None, pass_log: str) -> tuple[int, int]:
    """2-pass encode; trả về (width, height) của output."""
    vf = f"scale=-2:{scale_h}" if scale_h else None

    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", src, "-c:v", "libx264", "-preset", "veryfast",
            "-b:v", f"{video_kbps}k", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart"]
    if vf:
        base += ["-vf", vf]

    null_dev = "NUL" if os.name == "nt" else "/dev/null"
    run(base + ["-pass", "1", "-passlogfile", pass_log, "-f", "null", null_dev])
    run(base + ["-pass", "2", "-passlogfile", pass_log, dst])

    for ext in ("-0.log", "-0.log.mbtree", "-0.log.temp"):
        try: os.remove(pass_log + ext)
        except FileNotFoundError: pass

    fw, fh = probe_resolution(dst)
    return fw, fh


def fit(input_path: str, output_path: str, max_mb: float,
        audio_kbps: int = 128) -> dict:
    """Nén video cho vừa max_mb. Trả về thông tin để log."""
    dur = probe_duration(input_path)
    if dur <= 0:
        raise RuntimeError("không đọc được duration của video")

    # danh sách chiều cao thử theo thứ tự; None = giữ nguyên
    # scale phải chia hết cho 2 cho yuv420p (libx264 yêu cầu even dim)
    src_w, src_h = probe_resolution(input_path)
    original_h = src_h
    even_h = lambda h: h if h % 2 == 0 else h - 1

    tried = []
    # scale ladder cho video ≤16:9 (chia hết cho 2)
    ladder = []
    for h in [original_h, 1080, 900, 810, 720, 640, 540, 480, 360]:
        eh = even_h(h)
        if eh < 360 or eh > original_h:
            continue
        if eh not in ladder:
            ladder.append(eh)
    ladder.sort(reverse=True)  # thử cao trước (chất lượng tốt hơn)

    pass_log = os.path.join(os.path.dirname(output_path), "ffpass2")
    for scale_h in ladder:
        # bitrate mục tiêu (kb/s) cho tổng video+audio
        total_kbps = int((max_mb * 8192) / dur)
        # video = total - audio - 8% overhead cho muxing + faststart
        video_kbps = max(int(total_kbps - audio_kbps - total_kbps * 0.08), 200)
        # khi scale thấp hơn, bớt bitrate để giữ chất lượng tương đương
        if scale_h and scale_h < original_h:
            video_kbps = int(video_kbps * (scale_h / original_h) ** 1.6)

        tried.append({"scale_h": scale_h, "video_kbps": video_kbps,
                      "audio_kbps": audio_kbps})

        if os.path.exists(output_path):
            os.remove(output_path)
        actual_h, actual_w = two_pass(input_path, output_path, video_kbps,
                                      audio_kbps, scale_h, pass_log)[::-1]
        out_mb = size_mb(output_path)
        sys.stderr.write(
            f"[shrink] scale={actual_h}p video={video_kbps}k audio={audio_kbps}k "
            f"-> {out_mb:.1f}MB\n")

        if out_mb <= max_mb:
            return {"ok": True, "mb": round(out_mb, 2),
                    "scale_h": actual_h, "video_kbps": video_kbps,
                    "audio_kbps": audio_kbps, "attempts": tried}

    return {"ok": False, "mb": round(size_mb(output_path), 2),
            "scale_h": scale_h,
            "attempts": tried}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-mb", type=float, required=True)
    ap.add_argument("--audio", type=int, default=128,
                    help="audio bitrate kbps (mặc định 128)")
    args = ap.parse_args()

    if not os.path.isfile(args.inp):
        print(f"không thấy {args.inp}", file=sys.stderr)
        return 2

    info = fit(args.inp, args.out, args.max_mb, args.audio)
    info["in"] = args.inp
    info["out"] = args.out
    info["max_mb"] = args.max_mb
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
