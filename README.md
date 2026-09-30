# Vietsub video (Google Gemini) — GitHub Actions

Workflow chạy trên **GitHub Actions**, dịch phim hoạt hình Trung Quốc (tu tiên, cổ đại, võ hiệp, cung đình…) sang tiếng Việt, **giữ phong vị Hán Việt** (tên riêng phiên âm Hán Việt, môn phái, pháp bảo, cảnh giới, danh xưng cung đình).

**Điểm đặc biệt của pipeline này:** Dịch bằng **đúng 1 Gemini call duy nhất** (gộp analyze + dịch + glossary). Đơn giản, ít fail, dễ debug.

| `Audio detect` | Cách lấy lời thoại | Cách che chữ gốc |
|---|---|---|
| `false` *(mặc định)* | OCR phụ đề cứng (tesseract `chi_tra+chi_sim+eng`) | Khung đen **đúng vùng** sub gốc |
| `true` | Nghe audio bằng faster-whisper | Dải đen cố định ở đáy khung hình |

Cả 2 mode: **chữ Việt đè lên khung đen** (cùng 1 file ASS, ffmpeg burn 1 lượt) → không bao giờ lệch timing.

---

## 1. Cài đặt (một lần)

1. Push repo lên GitHub.
2. **Settings → Secrets and variables → Actions → New repository secret**:
   - `GEMINI_API_KEY` — bắt buộc. Lấy từ [Google AI Studio](https://aistudio.google.com/apikey) (free, không cần thẻ).
   - `YT_COOKIES` — tuỳ chọn. Nội dung file `cookies.txt` nếu YouTube chặn IP runner.

## 2. Chạy workflow

**Actions → "Vietsub video (Gemini)" → Run workflow**, điền panel:

| Input | Mặc định | Ý nghĩa |
|---|---|---|
| `video_url` | — | URL video (bắt buộc: YouTube/TikTok/Facebook/direct mp4...) |
| `audio_detect` | `false` | `false` = OCR sub cứng · `true` = nghe audio |
| `output_mode` | `auto` | `auto` / `artifact` / `release` / `both` |
| `max_video_mb` | `500` | Giới hạn MB output |
| `target_lang` | `vi` | Ngôn ngữ đích |
| `force_no_cache` | `false` | Bỏ qua cache khi nghi cache lỗi |
| `gemini_model` | `gemini-3.6-flash` | Xem bảng model bên dưới |
| `ocr_bottom_ratio` | `0.40` | Tỷ lệ khung hình để OCR (0.2-0.6) |
| `ocr_interval` | `0.4` | Khoảng cách giữa 2 frame OCR (giây) |
| `ocr_conf` | `15` | Ngưỡng confidence tesseract |
| `batch_size` | `30` | Số cue mỗi call Gemini (không dùng nữa ở pipeline 1-call, giữ để tương thích) |

Kết quả: Artifacts và/hoặc Release tuỳ `output_mode`. File: `<tên>_vietsub.mp4`, `.srt`, `.ass`, kèm `debug/transcript.json`, `glossary.json`.

---

## Pipeline (đơn giản — 1 Gemini call duy nhất)

```
[1] Nhận URL từ panel
[2] yt-dlp tải video
[3] ffmpeg tách audio 16k mono
[4] Phát hiện sub:
      audio_detect=true  → faster-whisper large-v3-turbo → transcript.json
      audio_detect=false → tesseract OCR → transcript.json (kèm bbox per cue)
[5] Dịch bằng 1 GEMINI CALL duy nhất (translate_srt.py):
      Input  : toàn bộ transcript.json
      Prompt : "dịch + trích glossary, output JSON {glossary, translations}"
      Output : 1 JSON response → tách thành .srt Việt + segments_vi.json
      ▸ Cache: SHA256(transcript text) → re-run cùng video = 0 calls
      ▸ Nếu thiếu translation → retry 1 lần với prompt hint nhấn mạnh
      ▸ Vẫn thiếu → fallback dùng text gốc (đỡ fail cả workflow)
[6] Render (cover + burn):
      Mode OCR  → khung đen đúng bbox (tái dùng transcript.json.boxes)
      Mode fixed → dải đen cố định đáy khung
                          → <tên>_vietsub.mp4
```

**Tại sao 1 call đủ?**
- Video 30 phút ≈ 400 cue × 50 chars Trung ≈ 20K chars input ≈ 5K tokens → dễ dàng trong context window 1M của Gemini.
- Output ~10-15K tokens (400 cue × ~30 chars Việt) → `max_output_tokens=16000` đủ.
- Model đọc cả kịch bản trong 1 lần → **tự đảm bảo nhất quán** (glossary, xưng hô, tên riêng) mà không cần review riêng.

---

## Model Gemini nào phù hợp?

Pipeline này gọi **đúng 1 call / video**, nên quota quan trọng hơn throughput:

| Model | Free RPD | Ghi chú |
|---|---|---|
| **`gemini-3.6-flash`** *(mặc định)* | 20 | Ổn định, đủ ~15 video/ngày. |
| `gemini-3.7-flash` | 20 | Mới hơn 3.6. |
| `gemini-3.8-flash` | 20 | Mới nhất (Sep 2026), quota có thể chưa mature. |
| `gemini-flash-latest` | 20 | Alias Google. |
| `gemini-3.5-flash-lite` | 500 | Quota rộng NHƯNG hay trả output rỗng. Không khuyến nghị. |
| `gemini-3.1-pro-preview` | ❌ | Pro tier, cần billing. |

**Đổi model:** panel input `gemini_model` hoặc sửa `DEFAULT_MODEL` trong `vietsub/translate_srt.py`.

---

## Tại sao giữ được văn hoá Trung Hoa?

Prompt trong `vietsub/translate_srt.py` ép Gemini:

- **Glossary** do model tự trích từ transcript → mỗi tên riêng / thuật ngữ **1 dạng xuyên suốt**, ưu tiên Hán Việt (Lý Thanh Triều, Vân Lam Tông, Thanh Vân Các, Tiêu Viêm, Lâm Tu Nhai…).
- **Hậu tố môn phái** (Các · Tông · Giáo · Phái · Môn · Đường · Sơn · Bang · Hội · Tự) **không được lược bỏ**.
- **Xưng hô cung đình** (Hoàng Thượng · Hoàng hậu · Quý phi · Công chúa · Hoàng tử · Thái hậu · Nữ Triều) giữ nguyên Hán Việt.
- **Filler Trung** (嗯/啊/呃/哦) → Việt tự nhiên (ừm / à / ờ / ồ).
- **Dịch giả được nhắc** "KHÔNG Việt hoá kiếm hiệp/tu tiên sang phong cách phim Việt" — giữ đạo hiệu, môn phái, pháp bảo, cảnh giới.

---

## Tại sao khung đen không bỏ sót?

- **Mode OCR**: OCR dò cả frame, gộp frame có cùng dòng chữ thành cue có `[start, end]`. Build ASS mở rộng mỗi cue thêm `±0.05s` rồi `merge_intervals` → khung **luôn phủ xuyên suốt**.
- **Mode fixed**: dải đen cố định `[(0.04W, 0.80H) → (0.96W, 0.96H)]` suốt video.
- Khung và text cùng nằm trong **1 file ASS** (layer 0 + layer 1), `ffmpeg -vf ass=…` render 1 lượt.

---

## Cache

| Cache | Khi nào dùng | Vị trí |
|---|---|---|
| **Translate cache** | Re-run cùng video → 0 Gemini call | `~/.cache/vietsub/translate_srt_<sha256>.json` |
| **OCR cache** | Re-run cùng video → 0 OCR call (~2-3 phút tiết kiệm) | `~/.cache/vietsub/ocr_<sha256>.json` |
| **actions/cache** | Cache giữa các lần run trên Actions | Workflow step "Cache glossary + OCR" |

Cache key = SHA256(transcript text) hoặc SHA256(video path + size + params). Đổi params → cache miss.

---

## Cách giao kết quả

| `output_mode` | File ≤ cap | File > cap | Tạo Release |
|---|---|---|---|
| `auto` *(mặc định)* | → artifact | cố shrink artifact, fail thì → release | nếu artifact thất bại |
| `artifact` | → artifact (≤500MB) | shrink xuống ≤450MB → artifact | **không** |
| `release` | → release (giữ chất lượng) | shrink xuống ≤1.95GB → release | **luôn** |
| `both` | → cả 2 | shrink về ≤450MB → cả 2 | **luôn** |

### Auto-shrink (`shrink.py`)

Khi file output > cap:
- 2-pass encode với bitrate mục tiêu
- Ladder resolution: 1080p → 720p → 540p → 480p → 360p
- Audio AAC 128k

### Lưu ý quota
- **Artifact**: ≤500MB/file, miễn phí (public repo).
- **Release**: ≤2GB/file, tổng 10GB/release, vĩnh viễn.

---

## Cấu trúc repo

```
.github/workflows/vietsub.yml      workflow chính
vietsub/
  transcribe.py                    ASR faster-whisper (mode audio_detect=true)
  ocr_segments.py                  OCR + cache (mode audio_detect=false)
  translate_srt.py                 DỊCH BẰNG 1 GEMINI CALL (gộp analyze + dịch + glossary)
  cover_text.py                    dò vùng sub gốc + load_or_run (cache)
  build_ass.py                     sinh file ASS 2 layer (khung + text)
  render.py                        ffmpeg burn-in
  shrink.py                        2-pass encode
  dispatch.py                      đo file + chọn artifact/release/both
  requirements.txt                 faster-whisper, yt-dlp, google-genai
.gitignore
```

---

## Lưu ý vận hành

- **Thời gian chạy** (ước lượng trên runner free, video 10 phút):
  - Tải video: ~30s
  - Whisper large-v3-turbo: ~5-8 phút CPU
  - OCR dải đáy (~0.4s/frame, song song 8 luồng): ~3 phút (cache hit = 0s)
  - Gemini 1 call (analyze + dịch): ~10-30s
  - ffmpeg burn-in: ~1-3 phút
- **Video > 30 phút**: có thể chạm `timeout-minutes: 480`. Bump lên nếu cần.
- **429 / 503**: retry 6x với exponential backoff riêng (503/529 → 30-240s, 429 → 60s).
- **Cache hygiene**: workflow tự xoá `ocr_*.json > 30 ngày` để tránh phình cache.
- **Tesseract tiếng Trung**: dùng cả `chi_tra` (phồn thể) + `chi_sim` (giản thể) + `eng`.
