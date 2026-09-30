#!/usr/bin/env python3
"""srt_prepare.py — đọc file SRT tiếng Trung (user upload) → transcript.json theo schema chuẩn.

Mục đích: thay thế bước OCR phụ đề cứng (chất lượng kém với phim tu tiên/cổ đại).
User upload file SRT mềm (soft sub) đã được tách sẵn từ video, hoặc tải từ URL.

Input:
  --in <file.srt>         đường dẫn file SRT
  --out <transcript.json> output transcript.json (schema dùng chung với translate_srt)
  --lang <zh|auto>        ngôn ngữ nguồn (mặc định 'auto' = detect từ text)

Schema output (giống output của transcribe.py / ocr_segments.py):
  {
    "language": "zh",
    "duration": <float>,
    "source": "srt_upload",
    "segments": [
      {"start": <s>, "end": <s>, "text": <str>},
      ...
    ]
  }

Robust:
  - Chịu BOM (\ufeff) ở đầu file
  - Chịu CRLF / LF / CR line endings
  - Timing có dấu ',' (chuẩn SRT) hoặc '.' (một số tool)
  - Index line có thể bị thiếu/không liên tục
  - Text có thể có nhiều dòng, có <i>/<b>/font tags (giữ nguyên cho Gemini hiểu)
  - Dòng rỗng giữa cues có thể là \n hoặc \r\n\r\n
"""
from __future__ import annotations

import argparse
import json
import re
import sys

# Pattern: 00:00:00,000 --> 00:00:00,000  (cả ',' và '.')
TIME_RE = re.compile(
    r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*'
    r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})'
)


def parse_time(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, '0')[:3]) / 1000.0


def detect_lang(text: str) -> str:
    """Phát hiện ngôn ngữ đơn giản dựa trên Unicode ranges."""
    has_cjk = 0   # Trung / Nhật / Hàn
    has_hira = 0  # Hiragana (Nhật)
    has_kata = 0  # Katakana (Nhật)
    has_hangul = 0
    has_latin = 0
    total = 0
    for ch in text:
        code = ord(ch)
        if not ch.isalpha():
            continue
        total += 1
        if 0x3040 <= code <= 0x309F:
            has_hira += 1
        elif 0x30A0 <= code <= 0x30FF:
            has_kata += 1
        elif 0xAC00 <= code <= 0xD7AF:
            has_hangul += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
            has_cjk += 1
        elif 0x41 <= code <= 0x5A or 0x61 <= code <= 0x7A:
            has_latin += 1
    if total == 0:
        return "unknown"
    # Quyết định
    if has_hangul > total * 0.1:
        return "ko"
    if has_hira + has_kata > total * 0.1:
        return "ja"
    if has_cjk > total * 0.05:
        return "zh"
    if has_latin > total * 0.5:
        return "en"
    return "unknown"


def parse_srt(content: str) -> list[dict]:
    """Parse nội dung SRT thành list cue {start, end, text}.

    Bỏ qua cue rỗng text. Tự sửa text cue:
      - gộp các dòng text trong cùng cue thành 1 (dùng \\N để giữ line break ASS)
      - bỏ HTML tag đơn giản <i>, <b>, <font ...> (vẫn giữ text bên trong)
    """
    # bỏ BOM
    if content.startswith("\ufeff"):
        content = content[1:]
    # chuẩn hoá line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    cues: list[dict] = []
    # Tách blocks theo dòng trống (một hoặc nhiều \n liên tiếp)
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = [l for l in block.split("\n") if l.strip() != ""]
        if len(lines) < 2:
            continue
        # Dòng đầu có thể là index (số) hoặc là timing
        timing_line_idx = None
        if TIME_RE.search(lines[0]):
            timing_line_idx = 0
        elif len(lines) >= 2 and TIME_RE.search(lines[1]):
            timing_line_idx = 1
        else:
            continue
        m = TIME_RE.search(lines[timing_line_idx])
        if not m:
            continue
        try:
            start = parse_time(*m.groups()[:4])
            end = parse_time(*m.groups()[4:])
        except Exception:
            continue
        if end <= start:
            continue
        text_lines = lines[timing_line_idx + 1:]
        if not text_lines:
            continue
        text = "\n".join(text_lines).strip()
        # bỏ tag đơn giản: <i>, </i>, <b>, </b>, <font color="...">, </font>
        # (giữ text bên trong)
        text = re.sub(r"</?(i|b|u)(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?font(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
        # SRT dùng \n cho line break trong cue — convert sang \\N (ASS literal)
        text = text.replace("\n", "\\N").strip()
        if not text:
            continue
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    # Sort theo start (một số SRT có thể lệch)
    cues.sort(key=lambda c: c["start"])
    return cues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="file SRT đầu vào")
    ap.add_argument("--out", required=True, help="file transcript.json đầu ra")
    ap.add_argument("--lang", default="auto",
                    help="ngôn ngữ nguồn: zh | ja | ko | en | auto (mặc định auto)")
    a = ap.parse_args()

    try:
        with open(a.inp, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # Thử GBK (một số SRT Trung encoding)
        with open(a.inp, encoding="gbk") as f:
            content = f.read()

    cues = parse_srt(content)
    if not cues:
        print(f"[srt-prepare] không parse được cue nào từ {a.inp}", file=sys.stderr)
        return 2

    duration = cues[-1]["end"] if cues else 0.0
    if a.lang == "auto":
        lang = detect_lang("".join(c["text"] for c in cues))
    else:
        lang = a.lang

    out = {
        "language": lang,
        "duration": round(duration, 3),
        "source": "srt_upload",
        "segments": cues,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[srt-prepare] {len(cues)} cue · lang={lang} · dur={duration:.1f}s → {a.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
