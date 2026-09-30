#!/usr/bin/env python3
"""build_ass.py — sinh file ASS có 2 layer:
  - Layer 0 (style 'Box'): vẽ hình chữ nhật đen đặc đúng vùng chữ gốc (đã được OCR
    dò sẵn) hoặc dải đen cố định ở đáy. Cùng timing với cue Việt + nới 1 nhịp 2
    đầu để không hở chữ gốc.
  - Layer 1 (style 'Default'): chữ Việt căn giữa khung đen.

Cả 2 layer nằm chung 1 file .ass → ffmpeg burn 1 lượt, không bao giờ lệch giữa
khung và text.
"""

DRAW = r"{\an7\pos(0,0)\p1}m %d %d l %d %d l %d %d l %d %d{\p0}"


def ass_time(t: float) -> str:
    t = max(0.0, float(t))
    h, m = int(t // 3600), int((t % 3600) // 60)
    return f"{h:d}:{m:02d}:{t - h * 3600 - m * 60:05.2f}"


def ass_escape(text: str) -> str:
    text = text.replace("\\N", "\x00")
    text = text.replace("\\", "\\\\").replace("\n", "\x00")
    text = text.replace("{", r"\{").replace("}", r"\}")
    return text.replace("\x00", r"\N")


def merge_intervals(intervals, gap: float = 0.4):
    out = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def layout_box(band: dict, segs, res_w: int, res_h: int, font_size: int):
    """Tính hình chữ nhật đen + vị trí text, vừa đủ chứa text.

    Bug đã sửa (so với bản trước):
      - Padding tăng từ 6-9px lên 14-18px (text có khoảng thở, không sát mép).
      - Nếu band rộng (cover=fixed, band_w > 70% res_w) → box thu nhỏ vừa text,
        KHÔNG full-width (trước đây box rộng toàn frame → trông khủng khiếp).
      - Nếu band hẹp (cover=ocr, band_w < 50% res_w) → box rộng = max(band_w,
        text_w) để vừa che sub cứng gốc.
      - text_y luôn nằm trong box, không clip ra ngoài frame ảo giác "tràn".
    """
    fs = font_size
    flat = [l for s in segs for l in s["text"].split("\\N")]
    max_chars = max((len(l) for l in flat), default=20)
    max_lines = max((s["text"].count("\\N") + 1 for s in segs), default=1)

    # Padding đủ — text có khoảng thở trong khung, không sát mép
    pad_w = max(24, int(fs * 0.7))   # đủ cho 2 bên
    pad_h = max(14, int(fs * 0.4))   # đủ cho trên dưới

    text_w = max_chars * fs * 0.6 + fs * 1.2   # width text ước tính
    need_h = max_lines * fs * 1.25 + fs * 0.8   # height text ước tính

    # Quyết định box width/height
    band_is_wide = band["w"] / res_w > 0.7   # cover=fixed thường wide
    band_is_narrow = band["w"] / res_w < 0.5  # OCR có thể narrow

    if band_is_wide:
        # Cover=fixed: box thu gọn vừa text + pad, căn giữa frame
        box_w = min(text_w + 2 * pad_w, band["w"])
    elif band_is_narrow:
        # OCR band hẹp: box rộng = band + pad để che trọn sub gốc
        box_w = band["w"] + 2 * pad_w
    else:
        # Trường hợp trung gian: max(band, text) + pad
        box_w = max(band["w"], text_w) + 2 * pad_w

    # Box height: max(band_h, need_h + 2*pad_h)
    h = max(band["h"], need_h + 2 * pad_h)

    # Căn box theo chiều ngang: giữa frame
    cx = res_w / 2
    x0 = cx - box_w / 2
    x0 = max(0.0, min(x0, res_w - box_w))
    x1 = x0 + box_w

    # Căn box theo chiều dọc: GIỮA band y gốc, clip vào frame
    cy = band["y"] + band["h"] / 2
    y0 = cy - h / 2
    y0 = max(0.0, min(y0, res_h - h))
    y1 = y0 + h

    # Text Y = giữa box (luôn trong box)
    text_y = int(y0 + h / 2)
    return (int(x0), int(y0), int(x1), int(y1)), text_y


def build_ass(segs, res_w, res_h, font, font_size, out_path,
              band=None, box_windows=None, extra_boxes=None, margin_bottom=60):
    fs = font_size
    lines = [
        "[Script Info]", "Title: Vietsub burn-in", "ScriptType: v4.00+", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", f"PlayResX: {res_w}", f"PlayResY: {res_h}", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Style Default:
        #   BorderStyle=1 (Outline+Shadow) → giữ để text có viền rõ trên nền đen
        #   Outline=2, Shadow=0 (đã bỏ shadow) — trước đây Shadow=1 khiến text bị đổ bóng
        #     ra ngoài khung đen → trông như text tràn ra ngoài.
        #   Alignment=2 (bottom-center) để mặc định nếu không có \pos.
        f"Style: Default,{font},{fs},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,40,40,{margin_bottom},1",
        "Style: Box,Arial,20,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    text_pos = ""
    if band:
        (x0, y0, x1, y1), text_y = layout_box(band, segs, res_w, res_h, fs)
        # cộng dồn từng window OCR + mở rộng 1 nhịp 2 đầu mỗi cue để khung LUÔN phủ xuyên suốt
        wins = list(box_windows or [])
        wins += [(s["start"] - 0.05, s["end"] + 0.05) for s in segs]
        for s, e in merge_intervals(wins):
            lines.append(
                f"Dialogue: 0,{ass_time(s)},{ass_time(e)},Box,,0,0,0,,"
                f"{DRAW % (x0, y0, x1, y0, x1, y1, x0, y1)}"
            )
        text_pos = r"{\an5\pos(%d,%d)}" % (res_w // 2, text_y)

    for r in (extra_boxes or []):
        p = 4
        bx0, by0 = max(0, r["x"] - p), max(0, r["y"] - p)
        bx1, by1 = min(res_w, r["x"] + r["w"] + p), min(res_h, r["y"] + r["h"] + p)
        lines.append(
            f"Dialogue: 0,{ass_time(r['t_start'])},{ass_time(r['t_end'])},Box,,0,0,0,,"
            f"{DRAW % (bx0, by0, bx1, by0, bx1, by1, bx0, by1)}"
        )

    kept = 0
    for s in segs:
        if float(s["end"]) <= float(s["start"]) or not s.get("text"):
            continue
        kept += 1
        lines.append(
            f"Dialogue: 1,{ass_time(s['start'])},{ass_time(s['end'])},Default,,0,0,0,,"
            f"{text_pos}{ass_escape(s['text'])}"
        )
    if not kept:
        raise ValueError("không có cue nào để burn")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return kept
