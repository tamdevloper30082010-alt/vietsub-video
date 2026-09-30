#!/usr/bin/env python3
"""render.py — render video cuối: che chữ gốc (khung đen) + burn chữ Việt đè lên khung.

Mode cover:
  ocr   : OCR dò vùng sub cứng -> khung đen đúng vùng đó (mode audio_detect=false).
  fixed : dải đen cố định ở đáy khung hình, suốt video (mode audio_detect=true).
  off   : không che (video sạch, sub mềm...).

Cả khung đen và chữ Việt nằm chung 1 file ASS → ffmpeg burn 1 lượt, không lệch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import build_ass as B
import cover_text as C

KEEP_EXT = {"mp4", "mov", "mkv", "m4v"}


def log(m):
    print(f"[render] {m}", file=sys.stderr, flush=True)


def probe(video):
    cp = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height:format=duration",
                         "-of", "json", video], capture_output=True, text=True, check=True)
    d = json.loads(cp.stdout)
    st = d["streams"][0]
    return (int(st["width"]), int(st["height"]),
            float(d.get("format", {}).get("duration", 0) or 0))


def write_srt(segs, path):
    def t(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{t(s['start'])} --> {t(s['end'])}\n"
                    f"{s['text'].replace(chr(92) + 'N', chr(10))}\n\n")


def run_ffmpeg(video, ass_name, cwd, out, crf, copy_audio):
    audio = ["-c:a", "copy"] if copy_audio else ["-c:a", "aac", "-b:a", "192k"]
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", video, "-vf", f"ass={ass_name}",
           "-map", "0:v:0", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", *audio, "-movflags", "+faststart", out]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", required=True, help="JSON {start,end,text}")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--font", default="Noto Sans")
    ap.add_argument("--font-size", type=int, default=0)
    ap.add_argument("--cover", choices=["ocr", "fixed", "off"], default="ocr")
    ap.add_argument("--ocr-langs", default="chi_tra+chi_sim+eng")
    ap.add_argument("--interval", type=float, default=0.4,
                    help="đồng bộ với ocr_segments (mặc định 0.4). Khi render OCR fallback "
                         "(không có boxes từ transcript), dùng cùng interval với ocr_segments.")
    ap.add_argument("--fixed-top", type=float, default=0.80)
    ap.add_argument("--fixed-height", type=float, default=0.16)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--transcript", default=None,
                    help="transcript.json có sẵn — dùng boxes (skip OCR lần 2)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    with open(a.segments, encoding="utf-8") as f:
        segs = json.load(f)
    w, h, duration = probe(a.video)

    fs = a.font_size or max(36, round(min(w, h) / 30))
    fs = max(16, min(fs, int((w - 80) / (42 * 0.62))))
    log(f"{w}x{h}, {duration:.0f}s, font {a.font} {fs}px, cover={a.cover}")

    band, windows, extras, mode = None, [], [], a.cover
    if mode == "ocr":
        # Ưu tiên dùng boxes đã có sẵn (transcript.json / ocr cache).
        # Tiết kiệm ~2 phút cho video 30 phút so với gọi C.detect() lại.
        try:
            data = C.load_or_run(a.video, transcript_path=a.transcript,
                                 interval=a.interval, lang=a.ocr_langs)
            info = C.analyze(data["ranges"], h)
        except Exception as e:
            log(f"OCR lỗi ({e}) -> chuyển sang dải đen cố định")
            info = None
        if info:
            band, windows, extras = info["band"], info["windows"], info["outliers"]
            src = data.get("_source", "ocr")
            log(f"OCR [{src}]: {len(data['ranges'])} khoảng; band chính {band}; lạc: {len(extras)}")
        else:
            log("OCR không thấy chữ gốc -> chuyển sang dải đen cố định")
            mode = "fixed"
    if mode == "fixed":
        band = {"x": int(w * 0.04), "y": int(h * a.fixed_top),
                "w": int(w * 0.92), "h": int(h * a.fixed_height)}
        windows = [(0.0, duration + 1)]

    stem, ext = os.path.splitext(os.path.basename(a.video))
    ext = ext.lstrip(".").lower()
    keep_container = ext in KEEP_EXT
    out_ext = ext if keep_container else "mp4"
    out = os.path.abspath(os.path.join(a.outdir, f"{stem}_vietsub.{out_ext}"))
    ass_name = "subtitle.ass"
    ass_path = os.path.join(a.outdir, ass_name)

    n = B.build_ass(segs, w, h, a.font, fs, ass_path, band=band,
                    box_windows=windows, extra_boxes=extras,
                    margin_bottom=max(40, int(h * 0.05)))
    write_srt(segs, os.path.join(a.outdir, f"{stem}_vietsub.srt"))
    log(f"ASS: {n} cue" + (" + khung đen" if band else ""))

    r = run_ffmpeg(os.path.abspath(a.video), ass_name, os.path.abspath(a.outdir),
                   out, a.crf, keep_container)
    if r.returncode != 0 and keep_container:
        log("copy audio thất bại -> AAC")
        r = run_ffmpeg(os.path.abspath(a.video), ass_name,
                       os.path.abspath(a.outdir), out, a.crf, False)
    if r.returncode != 0:
        log("ffmpeg lỗi:\n" + r.stderr[-3000:])
        return 2
    os.replace(ass_path, os.path.join(a.outdir, f"{stem}_vietsub.ass"))
    log(f"xong -> {out}")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
