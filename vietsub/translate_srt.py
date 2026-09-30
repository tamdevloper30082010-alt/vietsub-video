#!/usr/bin/env python3
"""translate_srt.py — Dịch SRT sang tiếng Việt bằng ĐÚNG 1 Gemini call.

Pipeline cũ (translate.py) dùng 5-10 calls:
  1) analyze → glossary
  2-N) batch translate
  N+1) review nhất quán

Mỗi call là 1 điểm fail (model có thể cắt output, rate-limit, schema sai…).
auto-split batch giúp được phần nào nhưng model "lite" còn trả array rỗng khi bị áp lực.

Pipeline mới (file này):
  ► GỘP analyze + dịch + glossary vào 1 prompt duy nhất.
  ► 1 call Gemini → output JSON {glossary, translations}.
  ► Validate: phải có translation cho MỌI cue (không skip).
  ► Nếu thiếu → retry 1 lần với prompt nhấn mạnh.
  ► Vẫn thiếu → fallback dùng text gốc (đỡ fail cả workflow).
  ► Cache theo SHA256(transcript text) → re-run cùng video = 0 calls.

Output:
  - <out>            : file .srt Việt (cho user xem)
  - <out>.segments.json : [{start, end, text}] (cho render.py)
  - <out>.usage.json : {calls, in_tokens, out_tokens, model}
  - glossary.json    : {glossary, model} (cho debug)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from typing import Any

logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("google.generativeai").setLevel(logging.ERROR)

CACHE_DIR = os.path.expanduser("~/.cache/vietsub")

# Bảng model Gemini Free Tier (cập nhật T9/2026, https://ai.google.dev/gemini-api/docs/models):
#
#   Model                          RPM   RPD       Quality   Notes
#   ───────────────────────────────────────────────────────────────────
#   gemini-3.8-flash               5–10  20–1500  ★★★★★     ← DEFAULT, mới nhất Google khuyến nghị
#   gemini-3.7-flash               5–10  20–1500  ★★★★★
#   gemini-3.5-flash               5     20       ★★★★      chất lượng cao, RPD thấp
#   gemini-3.5-flash-lite          15    500      ★★★       quota rộng, dịch OK
#   gemini-3.1-flash-lite          15    500–1000 ★★★       quota rộng, stable
#   gemini-2.5-flash               10–15 ~1.500   ★★★★      DEPRECATED cho user mới (404 NOT_FOUND)
#   gemini-2.5-pro                 5     50       ★★★★★     trial-only
#   gemini-2.5-flash-lite          10–15 20       ★★        RETIRE 20/10/2026 ❌
#   gemini-3.1-pro-preview                            ★★★★★  PAID ONLY 💸
#
# CẬP NHẬT QUAN TRỌNG (T9/2026): Google đã RETIRE gemini-2.5-flash cho user mới,
# trả 404 NOT_FOUND với message "Please update your code to use models/gemini-3.8-flash".
# Bug đã gặp trong log workflow: 6 retry 404 → fail cả workflow. Default đã đổi
# sang gemini-3.8-flash (model Google khuyến nghị cho new users).
#
# Lưu ý:
#   - RPD reset lúc 00:00 PT (Pacific Time) → 07:00 sáng UTC+7 VN.
#   - Quota tính theo PROJECT, không theo API key (tạo key mới cùng project không thêm quota).
#   - Bật billing → MẤT free tier hoàn toàn, mọi call đều tính tiền.
#   - 3.x series chia sẻ 5,000 search-grounding prompts/tháng.
#
# Recommendation cho vietsub-video:
#   - 1 video ~ 1 Gemini call → quota không phải vấn đề với Flash thường.
#   - Chất lượng dịch quan trọng hơn tốc độ → CHỌN gemini-3.8-flash.
#   - Nếu cần quota rộng (test nhiều video/ngày) → CHỌN gemini-3.5-flash-lite (500 RPD).
DEFAULT_MODEL = "gemini-3.8-flash"

# Max output tokens. 16000 đủ cho ~500 cue × 30 chars Việt ≈ 15000 chars.
# Nếu video >500 cue thì tăng lên 24000.
DEFAULT_MAX_TOKENS = 16000

SYSTEM_PROMPT = """Bạn là dịch giả phụ đề tiếng Việt chuyên phim hoạt hình Trung Quốc (tu tiên, cổ đại, võ hiệp, huyền huyễn, cung đình).

NHIỆM VỤ: Dịch TOÀN BỘ file SRT bên dưới sang tiếng Việt + TRÍCH GLOSSARY từ transcript.

OUTPUT: JSON đúng schema {glossary, translations}.

═══════════════════════════════════════════
PHẦN 1 — GLOSSARY (trích trước khi dịch):
═══════════════════════════════════════════
glossary là mảng, có thể rỗng [] nếu không có thuật ngữ đáng nhớ.
Mỗi mục {src, vi, asr_variants?}:
  • CHỈ liệt kê: TÊN RIÊNG / MÔN PHÁI / PHÁP BẢO / CẢNH GIỚI / DANH XƯNG CUNG ĐÌNH
    xuất hiện ≥ 2 lần HOẶC quan trọng cho mạch phim (đạo hiệu chính, môn phái chính, pháp bảo chính).
  • Phim tu tiên / cổ trang / võ hiệp: BẮT BUỘC dùng âm Hán Việt thông dụng.
    Ví dụ:
      '云岚宗' → vi: 'Vân Lam Tông' (môn phái PHẢI giữ hậu tố 'Tông')
      '萧炎' → vi: 'Tiêu Viêm'
      '林修崖' → vi: 'Lâm Tu Nhai'
      '小医仙' → vi: 'Tiểu Y Tiên'
      '青云门' → vi: 'Thanh Vân Môn'
  • Tên có cấu trúc <Họ + Tên> (vd 'Thanh Triều Lý'): phiên âm Hán Việt cả họ và tên,
    KHÔNG đảo vị trí.
  • KHÔNG dịch nghĩa thường cho tên riêng.
  • asr_variants (optional): các biến thể OCR/ASR hay nhầm của tên đó. Để mảng rỗng [] nếu không có.

═══════════════════════════════════════════
PHẦN 2 — TRANSLATIONS (dịch TOÀN BỘ cue):
═══════════════════════════════════════════
translations là mảng có CHÍNH XÁC {N} mục (với {N} = số cue trong SRT).
Mỗi mục {id: integer, vi: string}:
  • id khớp với số thứ tự cue trong SRT (bắt đầu từ 1).
  • vi là dịch Tiếng Việt tự nhiên, văn phong phù hợp thể loại.
  • BẮT BUỘC trả đủ {N} translations, KHÔNG ĐƯỢC bỏ sót cue nào, KHÔNG ĐƯỢC gộp cue.
  • Nếu cue chỉ có nhạc/SFX (vd '♪ nhạc ♪') → trả vi rỗng "" vẫn giữ id đó.
  • Câu chỉ có filler Trung (嗯/啊/呃/哦) → filler Việt tương đương:
      嗯 → 'ừm' (hoặc 'ờ' nếu gọn)
      啊 → 'à'
      呃 → 'ờ'
      哦 → 'ồ'
    KHÔNG để filler Trung nguyên xi trong phụ đề Việt.

═══════════════════════════════════════════
QUY TẮC PHONG CÁCH:
═══════════════════════════════════════════
• GIỮ văn hoá Trung Hoa (đạo hiệu, môn phái, pháp bảo, tu vi, cảnh giới, danh xưng cung đình)
  — không Việt hoá kiểu phim Việt.
• HẬU TỐ môn phái / tổ chức PHẢI GIỮ NGUYÊN, KHÔNG lược bỏ:
  Các · Tông · Giáo · Phái · Môn · Đường · Sơn · Bang · Hội · Tự.
• XƯNG HÔ CUNG ĐÌNH giữ nguyên Hán Việt:
  Hoàng Thượng · Hoàng hậu · Quý phi · Tần phi · Công chúa · Hoàng tử · Thái hậu · Nữ Triều.
  KHÔNG thay bằng 'Ngài' / 'Bệ hạ' đại thể.
• Sửa lỗi ASR/OCR rõ ràng dựa vào ngữ cảnh trước/sau + glossary.
• KHÔNG chú thích, KHÔNG dài dòng, KHÔNG thêm lời giải thích.
• Mỗi vi KHÔNG chứa ký tự xuống dòng thật (\\n). Nếu cần ngắt dòng dùng '\\N' (literal).

═══════════════════════════════════════════
ĐỊNH DẠNG BẮT BUỘC:
═══════════════════════════════════════════
• Output JSON đúng schema, không kèm markdown, không giải thích ngoài JSON.
"""

RETRY_HINT = """
LƯU Ý QUAN TRỌNG: lần trước bạn chỉ trả {got}/{expected} translations. Bạn PHẢI trả
CHÍNH XÁC {expected} translations (mỗi cue 1 entry), KHÔNG được bỏ sót cue nào, KHÔNG
được gộp cue. Hãy dịch lại đầy đủ."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "glossary": {"type": "array", "items": {"type": "object", "properties": {
            "src": {"type": "string"},
            "vi": {"type": "string"},
            "asr_variants": {"type": "array", "items": {"type": "string"}},
        }, "required": ["src", "vi"]}},
        "translations": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"},
            "vi": {"type": "string"},
        }, "required": ["id", "vi"]}},
    },
    "required": ["glossary", "translations"],
}


def log(msg: str) -> None:
    print(f"[translate-srt] {msg}", file=sys.stderr, flush=True)


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("thiếu GEMINI_API_KEY (Settings → Secrets → Actions)")
    try:
        from google import genai
        return ("new", genai.Client(api_key=api_key))
    except Exception:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return ("old", genai)


def classify_error(err: Exception) -> str:
    msg = (str(err) or "").upper()
    if "503" in msg or "529" in msg or "OVERLOADED" in msg or "UNAVAILABLE" in msg:
        return "503"
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE" in msg or "QUOTA" in msg:
        return "429"
    return "other"


def retry_sleep(attempt: int, err: Exception) -> None:
    kind = classify_error(err)
    if kind == "503":
        wait = min(30.0 * (2 ** attempt), 240.0) + random.uniform(0, 5.0)
        log(f"retry {attempt + 1}: server overloaded → chờ {wait:.1f}s")
    elif kind == "429":
        wait = 60.0 + random.uniform(0, 5.0)
        log(f"retry {attempt + 1}: quota/rate-limit → chờ {wait:.1f}s")
    else:
        wait = 1.0 * (2 ** attempt) + random.uniform(0, 1.0)
        log(f"retry {attempt + 1}: chờ {wait:.1f}s ({type(err).__name__}: {str(err)[:120]})")
    time.sleep(wait)


def call_gemini(client, model: str, system: str, user: str,
                max_tokens: int, max_retries: int):
    """Gọi Gemini 1 lần, có retry. Trả (dict parsed JSON, usage)."""
    backend_tag, obj = client
    last = None
    for attempt in range(max_retries):
        try:
            if backend_tag == "new":
                from google.genai import types
                resp = obj.models.generate_content(
                    model=model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        response_schema=OUTPUT_SCHEMA,
                        max_output_tokens=max_tokens,
                        temperature=0.2,
                    ),
                )
                txt = (resp.text or "").strip()
                usage = getattr(resp, "usage_metadata", None)
            else:
                m = obj.GenerativeModel(model, system_instruction=system,
                                        generation_config={
                                            "response_mime_type": "application/json",
                                            "response_schema": OUTPUT_SCHEMA,
                                            "max_output_tokens": max_tokens,
                                            "temperature": 0.2,
                                        })
                resp = m.generate_content(user)
                txt = (resp.text or "").strip()
                usage = getattr(resp, "usage_metadata", None)
            in_t = int(getattr(usage, "prompt_token_count", 0) or 0)
            out_t = int(getattr(usage, "candidates_token_count", 0) or 0)

            s = txt
            m_json = re.search(r"```(?:json)?\s*([\s\S]+?)```", s)
            if m_json:
                s = m_json.group(1).strip()
            result = json.loads(s)
            return result, {"in_tokens": in_t, "out_tokens": out_t, "calls": 1}
        except Exception as e:
            last = e
            log(f"call attempt {attempt + 1} fail: {type(e).__name__}: {str(e)[:160]}")
            if attempt < max_retries - 1:
                retry_sleep(attempt, e)
    raise RuntimeError(f"Gemini call failed (đã retry {max_retries} lần): {last}")


def transcript_hash(segs) -> str:
    """SHA256 của (id + text) các cue. KHÔNG phụ thuộc start/end."""
    h = hashlib.sha256()
    for s in segs:
        h.update(f"{s.get('id', '')}|{s.get('text', '')}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"translate_srt_{key}.json")


def load_cache(key: str):
    p = cache_path(key)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_cache(key: str, payload) -> None:
    p = cache_path(key)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"cache save fail: {e}")


def build_user_prompt(segs, language: str) -> str:
    """Build prompt với toàn bộ SRT content."""
    lines = [
        f"Ngôn ngữ nguồn: {language}",
        f"Tổng cộng {len(segs)} cue (id bắt đầu từ 1).",
        "",
        "SRT content (cue_id|duration_seconds|text Trung):",
        "",
    ]
    for i, s in enumerate(segs):
        dur = s.get("end", 0) - s.get("start", 0)
        lines.append(f"{i + 1}| [{dur:.1f}s] {s.get('text', '')}")
    return "\n".join(lines)


def validate_output(out: dict, expected_ids: list[int]) -> dict:
    """Trích {id: vi} từ output.

    FIX #1 (bug ẩn nghiêm trọng): model Gemini đôi khi trả id lệch (id=0, id=n+1,
    id âm, hoặc thiếu hàng loạt). Trước đây code `if sid in expected_ids` filter hết
    → 54/81 cue bị mất, fallback text Trung gốc. Bây giờ:
      - id hợp lệ (1..n) → lấy bình thường.
      - id lệch (0, âm, >n) → remap theo thứ tự xuất hiện trong output về id tiếp theo
        còn thiếu trong expected_ids.
      - Nếu vẫn thừa (output có nhiều hơn n cue) → bỏ qua (giữ id đầu n).
    """
    result: dict[int, str] = {}
    # Pass 1: id trực tiếp hợp lệ
    for item in out.get("translations", []):
        sid = item.get("id")
        vi = (item.get("vi") or "").strip()
        if isinstance(sid, int) and sid in expected_ids and sid not in result:
            result[sid] = vi

    # Pass 2: remap id lệch theo vị trí xuất hiện
    missing = [i for i in expected_ids if i not in result]
    if missing:
        idx = 0
        for item in out.get("translations", []):
            sid = item.get("id")
            if isinstance(sid, int) and sid in expected_ids:
                continue  # đã xử lý ở pass 1
            vi = (item.get("vi") or "").strip()
            if not vi:
                continue
            if idx >= len(missing):
                break
            result[missing[idx]] = vi
            idx += 1
    return result


def write_srt_file(vi_dict: dict, segs, out_path: str) -> None:
    """Viết file .srt Việt từ {id: vi} + list segs (để lấy timing).

    FIX #2c: cue thiếu (không có trong vi_dict) sẽ BỊ BỎ QUA — không ghi text Trung gốc
    ra file .srt, tránh sub rác / sub lẫn Việt-Trung trong cùng video.
    """
    def fmt_time(t):
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3600000)
        m, rem = divmod(rem, 60000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs):
            cue_id = i + 1
            if cue_id not in vi_dict:
                continue  # FIX #2c: bỏ cue thiếu
            vi = vi_dict[cue_id]
            # SRT dùng \n thật để xuống dòng trong cue
            vi_lines = vi.replace("\\N", "\n")
            written += 1
            f.write(f"{written}\n")
            f.write(f"{fmt_time(s['start'])} --> {fmt_time(s['end'])}\n")
            f.write(f"{vi_lines}\n")
            f.write("\n")


def wrap_for_render(vi: str) -> str:
    """Wrap text ≤ 42 chars/dòng, ≤ 2 dòng cho ASS burn.
    Giữ \\N (literal, không xuống dòng thật) — build_ass.py sẽ hiểu.

    FIX #6 (bug ẩn): text rác từ OCR có thể chứa nhiều "\\N" literal rải rác
    (như "2\\Ni\\N|Pe]\\N|"). Nếu giữ nguyên, build_ass.py sẽ coi là line-break
    ASS → render ra nhiều dòng rác / hoặc fail parse. Tốt nhất là LỌC bỏ tất cả
    \\N rồi chỉ chèn 1 \\N ở vị trí wrap đúng.
    """
    # FIX #6: chuẩn hoá — gộp tất cả \n/\\N thành space trước khi wrap
    text = vi.replace("\n", " ").replace("\\N", " ").strip()
    # Bỏ luôn các ký tự ASS đặc biệt có thể gây parse lỗi
    text = text.replace("\r", " ").replace("\t", " ")
    # Gộp nhiều space liên tiếp
    text = " ".join(text.split())
    if len(text) <= 42:
        return text
    if len(text) <= 84:
        mid = len(text) // 2
        space = text.rfind(" ", max(0, mid - 10), mid + 10)
        if space <= 0:
            space = text.find(" ", mid)
        if space > 0:
            return text[:space] + "\\N" + text[space + 1:]
        return text[:42] + "\\N" + text[42:]
    truncated = text[:80]
    last_space = truncated.rfind(" ")
    if last_space > 40:
        truncated = truncated[:last_space]
    return truncated + "\\N" + "..."


def write_segments_json(vi_dict: dict, segs, out_path: str) -> int:
    """Viết file segments_vi.json cho render.py. Trả số cue thực sự viết.

    FIX #2c: cue thiếu trong vi_dict sẽ BỊ BỎ QUA — không render phụ đề rác.
    """
    out = []
    for i, s in enumerate(segs):
        cue_id = i + 1
        if cue_id not in vi_dict:
            continue  # FIX #2c: bỏ cue thiếu
        vi = vi_dict[cue_id]
        out.append({"start": s["start"], "end": s["end"], "text": wrap_for_render(vi)})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dịch SRT sang tiếng Việt bằng 1 Gemini call duy nhất.")
    ap.add_argument("--in", dest="inp", required=True, help="transcript.json input")
    ap.add_argument("--out", required=True, help="file .srt output")
    ap.add_argument("--segments-out", default=None,
                    help="file segments_vi.json output (cho render.py). "
                         "Mặc định: thay .srt bằng .segments.json cùng tên")
    ap.add_argument("--glossary-out", default=None, help="file glossary.json output")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"model Gemini (mặc định {DEFAULT_MODEL} — 20 RPD, ổn định)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"max_output_tokens (mặc định {DEFAULT_MAX_TOKENS} - đủ cho ~500 cue)")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--no-cache", action="store_true",
                    help="tắt cache (force gọi Gemini lại)")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        data = json.load(f)
    language = data.get("language", "unknown")
    segs_raw = data.get("segments", [])
    segs = [{"id": i, "start": float(s["start"]), "end": float(s["end"]),
             "text": (s.get("text") or "").strip()}
            for i, s in enumerate(segs_raw)]
    n = len(segs)
    log(f"input: {n} cue, lang={language}, model={args.model}")

    if n == 0:
        log("⚠ input rỗng — ghi file output rỗng và thoát")
        open(args.out, "w").close()
        if args.segments_out or True:
            segs_out = args.segments_out or args.out.rsplit(".", 1)[0] + ".segments.json"
            with open(segs_out, "w", encoding="utf-8") as f:
                json.dump([], f)
        return 0

    expected_ids = list(range(1, n + 1))

    if language == "vi":
        log("nguồn đã là tiếng Việt — bỏ qua dịch")
        vi_dict = {i + 1: s["text"] for i, s in enumerate(segs)}
        glossary: list = []
        usage = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "model": args.model,
                 "cache_hit": False}
    else:
        key = transcript_hash(segs)
        cached = None if args.no_cache else load_cache(key)
        if cached and "vi_dict" in cached:
            log(f"cache HIT (key={key}) — skip Gemini, dùng {len(cached['vi_dict'])} translation cached")
            vi_dict = {int(k): v for k, v in cached["vi_dict"].items()}
            glossary = cached.get("glossary", [])
            usage = {"calls": 0, "in_tokens": 0, "out_tokens": 0, "model": args.model,
                     "cache_hit": True, "cache_key": key}
        else:
            log(f"cache miss — gọi Gemini 1 lần (key={key}, max_tokens={args.max_tokens})")
            client = get_client()
            user_prompt = build_user_prompt(segs, language)

            out, u = call_gemini(client, args.model, SYSTEM_PROMPT, user_prompt,
                                 args.max_tokens, args.max_retries)
            vi_dict = validate_output(out, expected_ids)
            log(f"  → {len(vi_dict)}/{n} translations từ response")

            # Nếu thiếu quá nhiều → retry 1 lần với hint
            missing = [i for i in expected_ids if i not in vi_dict]
            if missing:
                log(f"⚠ thiếu {len(missing)}/{n} cue — retry với hint nhấn mạnh")
                retry_prompt = user_prompt + RETRY_HINT.format(
                    got=len(vi_dict), expected=n)
                out2, u2 = call_gemini(client, args.model, SYSTEM_PROMPT, retry_prompt,
                                       args.max_tokens, args.max_retries)
                vi_dict2 = validate_output(out2, expected_ids)
                # Merge: ưu tiên bản retry nếu có content (không rỗng)
                for k, v in vi_dict2.items():
                    if v and (k not in vi_dict or not vi_dict[k]):
                        vi_dict[k] = v
                # Sum usage
                u["in_tokens"] += u2["in_tokens"]
                u["out_tokens"] += u2["out_tokens"]
                u["calls"] += u2["calls"]
                log(f"  → sau retry: {len(vi_dict)}/{n} translations")

            # FIX #2: nếu vẫn thiếu NHIỀU (>20% cue), retry theo lô nhỏ với prompt chỉ
            # chứa các cue thiếu. Mục đích: cứu được các cue bị model cắt output.
            # Trước đây fallback thẳng text Trung gốc (rác OCR) → render phụ đề rác.
            still_missing = [i for i in expected_ids
                             if i not in vi_dict or not vi_dict.get(i, "").strip()]
            if len(still_missing) > max(2, n // 5):
                log(f"⚠ vẫn thiếu {len(still_missing)}/{n} cue — retry theo lô nhỏ "
                    f"(≤15 cue/lần, dễ ép model trả đủ)")
                # Chia thành lô 15 cue, kèm context 1 cue trước/sau
                BATCH = 15
                for chunk_start in range(0, len(still_missing), BATCH):
                    chunk = still_missing[chunk_start:chunk_start + BATCH]
                    sub_prompt = build_user_prompt(
                        [segs[i - 1] for i in chunk], language)
                    sub_prompt += RETRY_HINT.format(got=0, expected=len(chunk))
                    try:
                        out3, u3 = call_gemini(client, args.model, SYSTEM_PROMPT,
                                               sub_prompt, args.max_tokens,
                                               args.max_retries)
                        vi_dict3 = validate_output(out3, chunk)
                        for k, v in vi_dict3.items():
                            if v and (k not in vi_dict or not vi_dict.get(k, "").strip()):
                                vi_dict[k] = v
                        u["in_tokens"] += u3["in_tokens"]
                        u["out_tokens"] += u3["out_tokens"]
                        u["calls"] += u3["calls"]
                    except Exception as e:
                        log(f"  lô {chunk[0]}–{chunk[-1]} retry fail: {e}")
                # tính lại still_missing
                still_missing = [i for i in expected_ids
                                 if i not in vi_dict or not vi_dict.get(i, "").strip()]
                log(f"  → sau batch-retry: {n - len(still_missing)}/{n} OK")

            if still_missing:
                # FIX #2b: TUYỆT ĐỐI KHÔNG ghi text Trung rác lên video Vietsub.
                # Vì OCR text có thể là rác (watermark/logo/sub sai), fallback sẽ khiến
                # user thấy phụ đề tiếng Trung lẫn tiếng Việt = "không đầy đủ hoàn chỉnh".
                # → BỎ QUA các cue thiếu (không ghi vào vi_dict → render sẽ skip cue này).
                # → Ghi log cảnh báo để user debug.
                log(f"⚠ vẫn thiếu {len(still_missing)}/{n} cue — BỎ QUA (không ghi "
                    f"text Trung rác lên video Vietsub)")
                for i in still_missing:
                    seg = segs[i - 1]
                    log(f"  cue {i} ({seg['start']:.1f}s) bị bỏ: {seg['text'][:60]!r}")
                    # KHÔNG gán seg['text'] nữa - để cue này KHÔNG render
                    vi_dict.pop(i, None)

            glossary = out.get("glossary", []) or []
            usage = {"calls": u["calls"], "in_tokens": u["in_tokens"],
                     "out_tokens": u["out_tokens"], "model": args.model,
                     "cache_hit": False, "cache_key": key}

            # Save cache (chỉ khi ≥95% cue đầy đủ)
            if len(still_missing) <= max(1, n // 20):
                save_cache(key, {"vi_dict": {str(k): v for k, v in vi_dict.items()},
                                 "glossary": glossary})
                log(f"đã cache (key={key})")

    # Output .srt (cho user)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_srt_file(vi_dict, segs, args.out)
    log(f"đã viết .srt: {args.out} ({n} cue)")

    # Output segments.json (cho render.py)
    segs_out_path = args.segments_out or args.out.rsplit(".", 1)[0] + ".segments.json"
    n_written = write_segments_json(vi_dict, segs, segs_out_path)
    log(f"đã viết segments.json: {segs_out_path} ({n_written} cue)")

    # Glossary
    if args.glossary_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.glossary_out)) or ".", exist_ok=True)
        with open(args.glossary_out, "w", encoding="utf-8") as f:
            json.dump({"glossary": glossary, "model": args.model,
                       "cue_count": n, "translation_count": len(vi_dict)},
                      f, ensure_ascii=False, indent=2)
        log(f"đã viết glossary: {args.glossary_out} ({len(glossary)} mục)")

    # Usage
    usage_path = args.out + ".usage.json"
    with open(usage_path, "w") as f:
        json.dump(usage, f)
    log(f"xong: {n} cue | calls={usage['calls']} "
        f"in={usage['in_tokens']:,} out={usage['out_tokens']:,} "
        f"cache_hit={usage.get('cache_hit', False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
