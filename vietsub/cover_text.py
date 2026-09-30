#!/usr/bin/env python3
"""cover_text.py — dò vùng chữ gốc (sub cứng) bằng tesseract OCR.

Khi mode 'audio_detect=false': đầu vào là video có sub cứng tiếng Trung.
Bước này phát hiện 2 thứ:
  - 'band': dải phụ đề chính (theo thời lượng xuất hiện) -> đặt khung đen + Việt.
  - 'extra_boxes': các box phụ (sub ngoài dải chính, nhận nhầm...) -> che riêng.

Khi mode 'audio_detect=true': KHÔNG chạy script này; render dùng cover_mode='fixed'.

CACHE: dùng chung cơ chế cache với ocr_segments.py:
  ► Nếu transcript.json đã có trường `boxes` (per cue) → dùng luôn để detect band,
    KHÔNG chạy OCR lại.
  ► Nếu không, thử load cache ở ~/.cache/vietsub/ocr_<hash>.json trước khi chạy thật.
  ► detect() cũ vẫn chạy được khi cần (back-compat).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = os.path.expanduser("~/.cache/vietsub")


def run(cmd, env=None):
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr}")
    return cp


def probe_resolution(video):
    cp = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video])
    w, h = cp.stdout.strip().split("x")[:2]
    return int(w), int(h)


def sample_frames(video, interval, workdir, crop_h, frame_h):
    vf = f"fps={1.0 / interval},crop=iw:{crop_h}:0:{frame_h - crop_h}"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-vf", vf, "-compression_level", "1",
         os.path.join(workdir, "frame_%06d.png")])
    files = sorted(f for f in os.listdir(workdir) if f.startswith("frame_"))
    return [(round(i * interval, 3), os.path.join(workdir, f)) for i, f in enumerate(files)]


def ocr_boxes(png, lang, conf_threshold, y_offset, max_w, max_h):
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    cp = run(["tesseract", png, "stdout", "-l", lang, "tsv"], env=env)
    boxes = []
    for line in cp.stdout.splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 12:
            continue
        try:
            level, left, top, w, h, conf = (int(c[0]), int(c[6]), int(c[7]),
                                            int(c[8]), int(c[9]), float(c[10]))
        except ValueError:
            continue
        if level != 5 or conf < conf_threshold or w < 8 or h < 8:
            continue
        if w >= max_w or h >= max_h:
            continue
        boxes.append((left, top + y_offset, w, h))
    return boxes


def boxes_to_band(boxes, frame_w, frame_h, min_band_height):
    if not boxes:
        return None
    x = max(0, min(b[0] for b in boxes) - 10)
    y = max(0, min(b[1] for b in boxes) - 6)
    w = min(frame_w - x, max(b[0] + b[2] for b in boxes) - x + 20)
    h = max(min_band_height, max(b[1] + b[3] for b in boxes) - y + 12)
    return (x, y, w, h)


def union(a, b):
    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2]); y1 = max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def group_into_ranges(samples, gap_merge, pad):
    ranges = []
    start = last = None
    band = None

    def close():
        ranges.append({"t_start": round(max(0.0, start - pad), 3),
                       "t_end": round(last + pad, 3),
                       "x": band[0], "y": band[1], "w": band[2], "h": band[3]})

    for t, b in samples:
        if b is None:
            continue
        if start is not None and t - last > gap_merge:
            close()
            start = None; band = None
        if start is None:
            start, band = t, b
        else:
            band = union(band, b)
        last = t
    if start is not None:
        close()
    return ranges


def analyze(ranges, frame_h):
    """Tách dải chính vs band lạc chỗ."""
    if not ranges:
        return None

    def cy(r):
        return r["y"] + r["h"] / 2
    def dur(r):
        return max(0.1, r["t_end"] - r["t_start"])

    items = sorted(ranges, key=cy)
    half = sum(dur(r) for r in items) / 2
    acc = 0.0; med = items[-1]
    for r in items:
        acc += dur(r)
        if acc >= half:
            med = r
            break
    tol = max(med["h"], 0.06 * frame_h)
    main = [r for r in ranges if abs(cy(r) - (med["y"] + med["h"] / 2)) <= tol]
    outliers = [r for r in ranges if r not in main]
    if not main:
        return None
    x0, y0 = min(r["x"] for r in main), min(r["y"] for r in main)
    x1, y1 = max(r["x"] + r["w"] for r in main), max(r["y"] + r["h"] for r in main)
    return {"band": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
            "windows": [(r["t_start"], r["t_end"]) for r in main],
            "outliers": outliers}


def detect(video, interval=0.4, lang="chi_tra+chi_sim+eng",
           conf=15.0, bottom_ratio=0.40, workers=None):
    """Chạy OCR thật để dò band. (cover_text cũ — không dùng cache ở đây.)"""
    if not shutil.which("tesseract"):
        raise RuntimeError("tesseract not on PATH")
    fw, fh = probe_resolution(video)
    crop_h = min(fh, int(round(fh * (bottom_ratio + 0.05))))
    y_off = fh - crop_h
    zone_y = fh * (1.0 - bottom_ratio)
    min_band_h = int(round(0.055 * fh))
    workers = workers or min(8, os.cpu_count() or 4)

    with tempfile.TemporaryDirectory(prefix="cover_text_") as work:
        frames = sample_frames(video, interval, work, crop_h, fh)
        print(f"[ocr] {len(frames)} khung, {workers} luồng, lang={lang}",
              file=sys.stderr, flush=True)

        def process(item):
            t, png = item
            boxes = [b for b in ocr_boxes(png, lang, conf, y_off, 0.9 * fw, 0.6 * crop_h)
                     if b[1] + b[3] / 2 >= zone_y]
            return (t, boxes_to_band(boxes, fw, fh, min_band_h))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(process, frames))

    samples.sort(key=lambda s: s[0])
    ranges = group_into_ranges(samples, gap_merge=interval * 2.5, pad=interval)
    for r in ranges:
        r["x"], r["y"] = max(0, r["x"]), max(0, r["y"])
        r["w"], r["h"] = min(r["w"], fw - r["x"]), min(r["h"], fh - r["y"])
    return {"frame_w": fw, "frame_h": fh, "ranges": ranges}


def _ranges_from_transcript_boxes(transcript: dict, frame_w: int, frame_h: int,
                                  gap_merge: float = 1.5) -> list[dict]:
    """Tạo ranges (band theo thời gian) từ cue.boxes của transcript.json — không cần OCR."""
    ranges: list[dict] = []
    for s in transcript.get("segments", []):
        boxes = s.get("boxes") or []
        if not boxes:
            continue
        # union tất cả box của cue
        xs = [b["x"] for b in boxes]
        ys = [b["y"] for b in boxes]
        xe = [b["x"] + b["w"] for b in boxes]
        ye = [b["y"] + b["h"] for b in boxes]
        x0, y0 = min(xs), min(ys)
        x1, y1 = max(xe), max(ye)
        # kẹp biên
        x0 = max(0, x0); y0 = max(0, y0)
        w_ = min(frame_w - x0, x1 - x0)
        h_ = min(frame_h - y0, y1 - y0)
        ranges.append({
            "t_start": float(s["start"]),
            "t_end": float(s["end"]),
            "x": x0, "y": y0, "w": w_, "h": h_,
        })
    # merge các cue gần nhau (cùng band)
    ranges.sort(key=lambda r: r["t_start"])
    merged: list[dict] = []
    for r in ranges:
        if merged and r["t_start"] - merged[-1]["t_end"] < gap_merge:
            prev = merged[-1]
            # union band
            x0 = min(prev["x"], r["x"])
            y0 = min(prev["y"], r["y"])
            x1 = max(prev["x"] + prev["w"], r["x"] + r["w"])
            y1 = max(prev["y"] + prev["h"], r["y"] + r["h"])
            prev["x"], prev["y"], prev["w"], prev["h"] = x0, y0, x1 - x0, y1 - y0
            prev["t_end"] = r["t_end"]
        else:
            merged.append(dict(r))
    return merged


def load_or_run(video, transcript_path=None,
                interval=0.4, lang="chi_tra+chi_sim+eng",
                conf=15.0, bottom_ratio=0.40, workers=None):
    """Cache-first + reuse transcript.boxes nếu có.

    Thứ tự ưu tiên:
      1. Nếu transcript_path trỏ tới JSON có `segments[].boxes` → dùng luôn.
      2. Thử load cache OCR (do ocr_segments.py ghi).
      3. Fallback: chạy detect() thật (sẽ OCR lại).
    """
    fw, fh = probe_resolution(video)

    # 1) transcript.boxes có sẵn → dùng luôn, không OCR
    if transcript_path and os.path.isfile(transcript_path):
        try:
            with open(transcript_path, encoding="utf-8") as f:
                tr = json.load(f)
            has_boxes = any(s.get("boxes") for s in tr.get("segments", []))
            if has_boxes:
                print("[cover] dùng boxes từ transcript.json (skip OCR)",
                      file=sys.stderr, flush=True)
                ranges = _ranges_from_transcript_boxes(tr, fw, fh,
                                                       gap_merge=interval * 2.5)
                return {"frame_w": fw, "frame_h": fh, "ranges": ranges,
                        "_source": "transcript_boxes"}
        except Exception as e:
            print(f"[cover] transcript load fail: {e}", file=sys.stderr)

    # 2) cache OCR (cùng cache dir với ocr_segments.py)
    try:
        from ocr_segments import load_cache as _load_ocr_cache
        cached = _load_ocr_cache(video, interval, lang, conf, bottom_ratio)
        if cached and any(s.get("boxes") for s in cached.get("segments", [])):
            print("[cover] dùng cache OCR từ ocr_segments (skip OCR)",
                  file=sys.stderr, flush=True)
            ranges = _ranges_from_transcript_boxes(cached, fw, fh,
                                                   gap_merge=interval * 2.5)
            return {"frame_w": fw, "frame_h": fh, "ranges": ranges,
                    "_source": "ocr_cache"}
    except Exception as e:
        print(f"[cover] cache load fail: {e}", file=sys.stderr)

    # 3) fallback: OCR thật
    print("[cover] không có cache → chạy OCR thật",
          file=sys.stderr, flush=True)
    return detect(video, interval=interval, lang=lang, conf=conf,
                  bottom_ratio=bottom_ratio, workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.4)
    ap.add_argument("--lang", default="chi_tra+chi_sim+eng")
    ap.add_argument("--conf", type=float, default=20.0)
    ap.add_argument("--bottom-ratio", type=float, default=0.32)
    ap.add_argument("--transcript", default=None,
                    help="transcript.json có sẵn — sẽ dùng boxes nếu có (skip OCR)")
    a = ap.parse_args()
    data = load_or_run(a.video, transcript_path=a.transcript,
                       interval=a.interval, lang=a.lang,
                       conf=a.conf, bottom_ratio=a.bottom_ratio)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[ocr] {len(data['ranges'])} vùng -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
