#!/usr/bin/env python3
"""transcribe.py — mode 'audio_detect=true': nghe audio bằng faster-whisper.

Xuất:
    {"language": "zh", "duration": <s>,
     "segments": [{"start": <s>, "end": <s>, "text": <zh>}, ...]}

Whisper segment có thể dài tới ~30s — tách lại theo word-timestamp thành
cue ngắn (theo dấu câu / khoảng lặng / độ dài) để dịch & hiển thị dễ đọc.
Khi cover_mode='fixed' (vì không có sub cứng), chỉ cần timing chuẩn là đủ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

SENT_END = tuple("。？！?!….")
SOFT_END = tuple("，、,;；:：")


def split_words(words, max_dur: float = 6.0, soft_dur: float = 3.5, gap: float = 1.0):
    cues, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w.word for w in cur).strip()
        if text:
            cues.append({
                "start": round(cur[0].start, 3),
                "end": round(cur[-1].end, 3),
                "text": text,
            })
        cur.clear()

    for w in words:
        if cur and (w.start - cur[-1].end) > gap:
            flush()
        cur.append(w)
        dur = cur[-1].end - cur[0].start
        tok = (w.word or "").strip()
        if tok.endswith(SENT_END) or dur >= max_dur or (dur >= soft_dur and tok.endswith(SOFT_END)):
            flush()
    flush()
    return cues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="zh", help="zh / auto / en / ja ...")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    lang = None if args.language in ("", "auto") else args.language

    seg_iter, info = model.transcribe(
        args.audio, language=lang, beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True, condition_on_previous_text=False,
    )
    print(f"[asr] lang={info.language} p={info.language_probability:.2f} dur={info.duration:.1f}s",
          file=sys.stderr, flush=True)

    cues, last = [], 0.0
    for seg in seg_iter:
        if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
            continue
        if seg.words:
            cues.extend(split_words(seg.words))
        elif seg.text.strip():
            cues.append({"start": round(seg.start, 3), "end": round(seg.end, 3),
                         "text": seg.text.strip()})
        if seg.end - last >= 60:
            last = seg.end
            print(f"[asr] {seg.end:.0f}/{info.duration:.0f}s", file=sys.stderr, flush=True)

    if not cues:
        print("[asr][ERROR] không nhận dạng được lời thoại nào", file=sys.stderr)
        return 3

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"language": info.language, "duration": info.duration,
                   "segments": cues}, f, ensure_ascii=False, indent=2)
    print(f"[asr] {len(cues)} cue -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
