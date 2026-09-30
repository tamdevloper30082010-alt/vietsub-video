#!/usr/bin/env python3
"""ocr_segments.py — mode 'audio_detect=false': đọc phụ đề cứng tiếng Trung trong video.

Pipeline:
  1. ffmpeg tách dải đáy khung hình theo lưới thời gian (~0.6s/ảnh).
  2. tesseract OCR từng ảnh (chi_tra+chi_sim+eng), trả về word-boxes có confidence.
  3. Gom các box theo cụm (cluster) trong cùng frame, rồi nối các cluster liên tiếp
     thành 'cue' có [start, end] và text_zh.
  4. Xuất segments.json theo schema dùng chung:
        {"language": "zh", "duration": <s>,
         "segments": [{"start": <s>, "end": <s>, "text": <zh>, "boxes":[...]}],
         "_ocr_cache_key": "..."}    ← để cover_text/render dùng lại.

CACHE: kết quả OCR (đã gom cue + bbox) được cache theo SHA256
        (video_path + duration + interval + lang + conf + bottom-ratio).
        Cache tại ~/.cache/vietsub/ocr_<hash>.json.
        ► Nếu cache hit → skip frame extraction + tesseract, tiết kiệm ~2 phút
          cho video 30 phút.
        ► use_or_run(video, ...): tiện cho các caller (cover_text, render).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

CACHE_DIR = os.path.expanduser("~/.cache/vietsub")
SENT_END = tuple("。？！?!….")
SOFT_END = tuple("，、,;；:：")


# ---------- Cache helpers ----------

def _cache_key(video: str, duration: float, interval: float, lang: str,
               conf: float, bottom_ratio: float) -> str:
    """SHA256 của input params. Đổi param → cache miss → chạy lại OCR."""
    h = hashlib.sha256()
    h.update(f"video={os.path.abspath(video)}|size={os.path.getsize(video)}|".encode())
    h.update(f"dur={duration:.3f}|int={interval:.3f}|".encode())
    h.update(f"lang={lang}|conf={conf:.1f}|bot={bottom_ratio:.3f}|".encode())
    return h.hexdigest()[:16]


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"ocr_{key}.json")


def load_cache(video: str, interval: float, lang: str, conf: float,
               bottom_ratio: float, duration: float | None = None) -> dict | None:
    if duration is None:
        try:
            duration = probe_duration(video)
        except Exception:
            return None
    key = _cache_key(video, duration, interval, lang, conf, bottom_ratio)
    p = _cache_path(key)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            # sanity check: phải có fields mình viết
            if data.get("language") == "zh" and isinstance(data.get("segments"), list):
                data["_ocr_cache_key"] = key
                return data
        except Exception:
            return None
    return None


def save_cache(payload: dict, key: str) -> None:
    p = _cache_path(key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ocr-seg] cache save fail: {e}", file=sys.stderr)


# ---------- Helpers ----------

def run(cmd: list[str], env: dict | None = None) -> subprocess.CompletedProcess:
    cp = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if cp.returncode != 0:
        raise RuntimeError(f"command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr}")
    return cp


def probe_resolution(video: str) -> tuple[int, int]:
    cp = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video])
    w, h = cp.stdout.strip().split("x")[:2]
    return int(w), int(h)


def probe_duration(video: str) -> float:
    cp = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "csv=p=0", video])
    return float(cp.stdout.strip() or 0)


def sample_bottom_strip(video: str, interval: float, workdir: str,
                        crop_h: int, frame_h: int) -> list[tuple[float, str]]:
    vf = f"fps={1.0 / interval},crop=iw:{crop_h}:0:{frame_h - crop_h}"
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", video, "-vf", vf, "-compression_level", "1",
         os.path.join(workdir, "frame_%06d.png")])
    files = sorted(f for f in os.listdir(workdir) if f.startswith("frame_"))
    return [(round(i * interval, 3), os.path.join(workdir, f)) for i, f in enumerate(files)]


def ocr_words(png: str, lang: str, conf_threshold: float,
              frame_w: int | None = None, frame_h: int | None = None,
              max_box_w_ratio: float = 0.6,
              max_box_h_ratio: float = 0.4) -> list[dict]:
    """OCR 1 frame PNG. Trả list word-boxes.

    FIX #3 (nhiều bug ẩn):
      - conf_threshold: filter noise (mặc định 15 → quá thấp, tăng lên 25)
      - max_box_w_ratio: loại box có w > 60% frame width (watermark, intro, logo)
      - max_box_h_ratio: loại box có h > 40% frame height (graphic overlay)
      - clip y + h vào trong frame (trước đây boxes có y=999 trong frame 720 → render ra ngoài)
    """
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    cp = run(["tesseract", png, "stdout", "-l", lang, "tsv"], env=env)
    out = []
    for line in cp.stdout.splitlines()[1:]:
        c = line.split("\t")
        if len(c) < 12:
            continue
        try:
            level = int(c[0]); left = int(c[6]); top = int(c[7])
            w = int(c[8]); h = int(c[9]); conf = float(c[10])
        except ValueError:
            continue
        if level != 5 or conf < conf_threshold or w < 8 or h < 8:
            continue

        # FIX #3b: filter watermark/intro logo (box quá lớn so với frame)
        if frame_w and w > max_box_w_ratio * frame_w:
            continue
        if frame_h and h > max_box_h_ratio * frame_h:
            continue

        text = (c[11] or "").strip()
        if not text:
            continue

        # FIX #3c: clip box vào trong frame (đề phòng OCR trả y+h > frame_h)
        if frame_h:
            top = max(0, min(top, frame_h - 1))
            h = min(h, frame_h - top)
        if frame_w:
            left = max(0, min(left, frame_w - 1))
            w = min(w, frame_w - left)

        out.append({"x": left, "y": top, "w": w, "h": h, "text": text, "conf": conf})
    return out


def cluster_lines(words: list[dict], y_tol: int) -> list[dict]:
    """Gom các word theo đường baseline (cùng dòng) trong 1 frame."""
    if not words:
        return []
    words.sort(key=lambda w: (w["y"], w["x"]))
    lines = []
    cur = [words[0]]
    for w in words[1:]:
        ref = cur[-1]
        if abs(w["y"] + w["h"] / 2 - (ref["y"] + ref["h"] / 2)) <= y_tol:
            cur.append(w)
        else:
            lines.append(cur); cur = [w]
    lines.append(cur)
    return [{
        "x0": min(w["x"] for w in ln),
        "x1": max(w["x"] + w["w"] for w in ln),
        "y0": min(w["y"] for w in ln),
        "y1": max(w["y"] + w["h"] for w in ln),
        "text": "".join(w["text"] for w in sorted(ln, key=lambda x: x["x"])),
    } for ln in lines]


def group_frames_into_cues(samples: list[tuple[float, list[dict]]],
                           frame_w: int, frame_h: int,
                           bottom_ratio: float, y_tol_ratio: float,
                           same_line_dx_ratio: float,
                           gap_merge: float) -> list[dict]:
    """Gom các dòng giống nhau ở các frame liên tiếp thành 1 cue [start, end]."""
    zone_y = frame_h * (1.0 - bottom_ratio)
    y_tol = max(6, int(frame_h * y_tol_ratio))
    same_dx = max(20, int(frame_w * same_line_dx_ratio))

    cues: list[dict] = []

    def flush(end_t: float):
        if "lines" not in cur or not cur["lines"]:
            return
        cur["lines"].sort(key=lambda l: l["y0"])
        text = "\n".join(l["text"] for l in cur["lines"]).strip()
        if not text:
            return
        cues.append({
            "start": round(cur["t_start"], 3),
            "end": round(end_t, 3),
            "text": text,
            "lines": [{
                "x": l["x0"], "y": l["y0"] + y_off_correct,
                "w": l["x1"] - l["x0"], "h": l["y1"] - l["y0"],
            } for l in cur["lines"]],
        })

    y_off_correct = frame_h - (frame_h * (1.0 - bottom_ratio))

    cur: dict = {}

    for t, lines in samples:
        kept = [ln for ln in lines if ln["y1"] >= zone_y]
        if not kept:
            if cur:
                flush(t)
                cur = {}
            continue

        if not cur:
            cur = {"t_start": t, "lines": kept}
            continue

        prev = cur["lines"]
        matched = 0
        for ln in kept:
            if any(abs(ln["x0"] - p["x0"]) <= same_dx and abs(ln["y0"] - p["y0"]) <= y_tol
                   for p in prev):
                matched += 1
        if matched == 0 or t - cur.get("last_t", cur["t_start"]) > gap_merge:
            flush(t)
            cur = {"t_start": t, "lines": kept}
        else:
            cur["lines"] = kept
            cur["last_t"] = t

    if cur:
        flush(samples[-1][0] + 0.6)

    merged = []
    for c in cues:
        if merged and c["start"] - merged[-1]["end"] < gap_merge and c["lines"] and merged[-1]["lines"]:
            prev = merged[-1]
            top_y = prev["lines"][0]["y"]
            new_y = c["lines"][0]["y"]
            if abs(top_y - new_y) <= y_tol:
                prev["end"] = c["end"]
                prev["text"] = prev["text"] + "\n" + c["text"]
                prev["lines"].extend(c["lines"])
                continue
        merged.append(c)
    return merged


def run_ocr(video: str, out_path: str,
            interval: float = 0.4, lang: str = "chi_tra+chi_sim+eng",
            conf: float = 15.0, bottom_ratio: float = 0.40,
            workers: int = 0) -> dict:
    """Chạy OCR thật (frame extract + tesseract), ghi cache, trả payload."""
    if not shutil.which("tesseract"):
        print("cần cài tesseract (apt: tesseract-ocr + gói ngôn ngữ)", file=sys.stderr)
        raise SystemExit(2)

    fw, fh = probe_resolution(video)
    duration = probe_duration(video)
    crop_h = min(fh, int(round(fh * (bottom_ratio + 0.05))))
    y_off = fh - crop_h
    workers = workers or min(8, os.cpu_count() or 4)
    cache_key = _cache_key(video, duration, interval, lang, conf, bottom_ratio)

    print(f"[ocr-seg] {fw}x{fh}, dur={duration:.1f}s, crop={crop_h}px, lang={lang}",
          file=sys.stderr, flush=True)

    def process(item):
        t, png = item
        # FIX #3: truyền frame_w/frame_h để filter box quá lớn (watermark) + clip bounds
        ws = ocr_words(png, lang, conf, frame_w=fw, frame_h=fh)
        for w in ws:
            w["y"] += y_off
        y_tol = max(6, int(fh * 0.012))
        lines = cluster_lines(ws, y_tol)
        return t, lines

    with tempfile.TemporaryDirectory(prefix="ocr_seg_") as work:
        frames = sample_bottom_strip(video, interval, work, crop_h, fh)
        print(f"[ocr-seg] {len(frames)} frame, {workers} luồng", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(process, frames))

    cues = group_frames_into_cues(
        samples, fw, fh, bottom_ratio,
        y_tol_ratio=0.012, same_line_dx_ratio=0.04, gap_merge=interval * 2.5)

    segs = []
    for c in cues:
        segs.append({"start": c["start"], "end": c["end"], "text": c["text"].replace("\n", "\\N"),
                     "boxes": c["lines"]})

    payload = {"language": "zh", "duration": duration, "segments": segs,
               "_ocr_cache_key": cache_key}
    save_cache(payload, cache_key)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ocr-seg] {len(segs)} cue -> {out_path} (cached {cache_key})", file=sys.stderr)

    # FIX #5: phát hiện trường hợp user bật sai mode (video không có sub cứng Trung
    # nhưng user chọn audio_detect=false). Nếu OCR trả về 0 cue HOẶC text thuần ASCII
    # không có ký tự Trung → gợi ý chuyển sang audio_detect=true.
    if not segs:
        print("[ocr-seg][WARN] OCR không tìm thấy cue nào — có thể video KHÔNG có sub "
              "cứng tiếng Trung. Hãy thử chạy lại với audio_detect=true để dịch từ audio.",
              file=sys.stderr)
    else:
        joined = "".join(s["text"] for s in segs)
        has_cjk = any(0x4E00 <= ord(c) <= 0x9FFF or 0x3400 <= ord(c) <= 0x4DBF
                      for c in joined)
        if not has_cjk and len(joined) > 100:
            print(f"[ocr-seg][WARN] OCR text không có ký tự Trung (chỉ ASCII/{len(joined)} "
                  f"ký tự). Có thể video không có sub cứng tiếng Trung — hãy thử "
                  f"audio_detect=true để dịch từ audio.", file=sys.stderr)

    return payload


def load_or_run(video: str, out_path: str,
                interval: float = 0.4, lang: str = "chi_tra+chi_sim+eng",
                conf: float = 15.0, bottom_ratio: float = 0.40,
                workers: int = 0, use_cache: bool = True) -> dict:
    """Cache-first: thử cache trước, fail mới chạy thật. Trả payload đầy đủ."""
    if use_cache:
        cached = load_cache(video, interval, lang, conf, bottom_ratio)
        if cached:
            print(f"[ocr-seg] cache hit, skip OCR", file=sys.stderr)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)
            return cached
    return run_ocr(video, out_path, interval=interval, lang=lang,
                   conf=conf, bottom_ratio=bottom_ratio, workers=workers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.4,
                    help="khoảng cách giữa 2 frame OCR (giây). Giảm xuống 0.3 nếu sub thoáng qua; "
                         "tăng lên 0.6-0.8 nếu video có sub ổn định (chạy nhanh hơn).")
    ap.add_argument("--lang", default="chi_tra+chi_sim+eng")
    ap.add_argument("--conf", type=float, default=15.0,
                    help="ngưỡng confidence tesseract (0-100). Mặc định 15 để giữ text nhỏ/mờ; "
                         "tăng lên 25-30 nếu thấy quá nhiều text nhiễu.")
    ap.add_argument("--bottom-ratio", type=float, default=0.40,
                    help="tỷ lệ chiều cao khung hình dùng để OCR dải đáy (0.1-0.6). "
                         "Mặc định 0.40 cover được phần lớn phim tu tiên; tăng lên 0.5-0.6 nếu sub "
                         "nằm giữa khung (cảnh chiến đấu/zoom).")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true",
                    help="tắt cache (force OCR lại)")
    a = ap.parse_args()
    if a.no_cache:
        run_ocr(a.video, a.out, interval=a.interval, lang=a.lang,
                conf=a.conf, bottom_ratio=a.bottom_ratio, workers=a.workers)
    else:
        load_or_run(a.video, a.out, interval=a.interval, lang=a.lang,
                    conf=a.conf, bottom_ratio=a.bottom_ratio, workers=a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
