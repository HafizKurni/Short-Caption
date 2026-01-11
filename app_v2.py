import os
import sys
import tempfile
from typing import List, Dict, Optional, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

# MoviePy v2
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, ColorClip
from moviepy.video.fx import Resize, FadeIn, FadeOut  # v2 effects


# -----------------------------
# Config
# -----------------------------
DEFAULT_FONT_PATH = "fonts/Inter-SemiBold.ttf"

FALLBACK_FONT_CANDIDATES = [
    DEFAULT_FONT_PATH,          # you said you have this
    "fonts/Montserrat-Bold.ttf",
    "fonts/Noto Sans CJK Regular.otf",    # you said you have this
    "fonts/NotoSansCJK-Regular.otf",
    "fonts/Inter-Regular.ttf",
    "fonts/Roboto-Bold.ttf",
    "fonts/Roboto-Regular.ttf",
    "DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "arial.ttf",
]

LANG_OPTIONS = {
    "English": "en",
    "Bahasa Indonesia": "id",
    "Bahasa Melayu": "ms",
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Italian": "it",
    "Dutch": "nl",
    "Russian": "ru",
    "Turkish": "tr",
    "Thai": "th",
    "Vietnamese": "vi",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Urdu": "ur",
}


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


# --- Language-aware font selection (NO slider/param changes) ---
def _is_cjk_lang(lang_code: str) -> bool:
    return lang_code in ("zh-CN", "zh-TW", "ja", "ko")


def _font_candidates_for_lang(lang_code: str) -> List[str]:
    # You said you will have:
    # - Montserrat-Bold.ttf
    # - Noto Sans CJK Regular.otf
    cjk_first = [
        "fonts/Noto Sans CJK Regular.otf",
        "fonts/NotoSansCJK-Regular.otf",
        "fonts/Noto Sans CJK Regular.otf",
        "fonts/NotoSansCJK-Regular.otf",
    ]
    latin_first = [
        "font/Montserrat-Bold.ttf",
        "fonts/Montserrat-Bold.ttf",
        DEFAULT_FONT_PATH,
    ]
    if _is_cjk_lang(lang_code):
        return cjk_first + latin_first + FALLBACK_FONT_CANDIDATES
    else:
        return latin_first + cjk_first + FALLBACK_FONT_CANDIDATES


# Keep same function name/signature; internally it picks by target_lang in session_state
def load_font(font_size: int) -> ImageFont.ImageFont:
    lang_code = st.session_state.get("target_lang", "en")
    for fp in _font_candidates_for_lang(lang_code):
        try:
            # If it's a path-like string, require it exists
            if fp and (os.path.isabs(fp) or "/" in fp or "\\" in fp):
                if os.path.exists(fp):
                    return ImageFont.truetype(fp, font_size)
                continue
            # Otherwise try as font name
            return ImageFont.truetype(fp, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_with_stroke(
    draw: ImageDraw.ImageDraw,
    pos: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int, int],
    stroke_fill: Tuple[int, int, int, int],
    stroke_width: int,
):
    x, y = pos
    sw = int(stroke_width)
    if sw > 0:
        for dx in range(-sw, sw + 1):
            for dy in range(-sw, sw + 1):
                if dx * dx + dy * dy <= sw * sw:
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# -----------------------------
# Frame + overlay (preview)
# -----------------------------
def get_frame_at(video_path: str, t: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = int(max(0, t) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def overlay_rgba_on_rgb(rgb_frame: np.ndarray, rgba_overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = rgba_overlay.shape[:2]
    H, W = rgb_frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return rgb_frame
    overlay_crop = rgba_overlay[y1 - y : y2 - y, x1 - x : x2 - x]
    base_crop = rgb_frame[y1:y2, x1:x2]
    alpha = overlay_crop[..., 3:4].astype(np.float32) / 255.0
    out = base_crop.astype(np.float32) * (1 - alpha) + overlay_crop[..., :3].astype(np.float32) * alpha
    rgb_frame[y1:y2, x1:x2] = out.astype(np.uint8)
    return rgb_frame


# -----------------------------
# Whisper word timestamps -> chunks
# -----------------------------
@st.cache_resource
def load_whisper(model_size: str) -> WhisperModel:
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_words(
    video_path: str,
    whisper_model: WhisperModel,
    language_hint: Optional[str] = None,
) -> Tuple[List[Dict], str]:
    segments, _ = whisper_model.transcribe(
        video_path,
        language=language_hint,
        vad_filter=True,
        word_timestamps=True,
    )
    words = []
    full_parts = []
    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            full_parts.append(seg_text)
        if getattr(seg, "words", None):
            for w in seg.words:
                wtxt = (w.word or "").strip()
                if not wtxt:
                    continue
                words.append({"start": float(w.start), "end": float(w.end), "word": wtxt})
    return words, " ".join(full_parts).strip()


def build_chunks_from_words(words: List[Dict], n: int = 4) -> List[Dict]:
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) == n:
            chunks.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "words": cur,
                "text": " ".join([x["word"] for x in cur]),
            })
            cur = []
    if cur:
        chunks.append({
            "start": cur[0]["start"],
            "end": cur[-1]["end"],
            "words": cur,
            "text": " ".join([x["word"] for x in cur]),
        })
    return chunks


def translate_chunks_keep_timing(chunks: List[Dict], target_lang: str) -> List[Dict]:
    translator = GoogleTranslator(source="auto", target=target_lang)
    out = []
    for c in chunks:
        start, end = c["start"], c["end"]
        dur = max(0.10, end - start)
        try:
            t = (translator.translate(c["text"]) or "").strip()
        except Exception:
            t = c["text"]

        t_words = t.split()
        if not t_words:
            t_words = c["text"].split()

        step = dur / len(t_words)
        tw = []
        for i, w in enumerate(t_words):
            ws = start + i * step
            we = min(end, ws + step)
            tw.append({"start": ws, "end": we, "word": w})

        out.append({
            "start": start,
            "end": end,
            "text": " ".join([x["word"] for x in tw]),
            "words": tw,
        })
    return out


# -----------------------------
# Karaoke layout + render (NO background)
# -----------------------------
def compute_word_positions_centered(
    words: List[Dict],
    font: ImageFont.ImageFont,
    video_width: int,
) -> List[Dict]:
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    space_w, _ = measure_text(d, " ", font)

    widths = []
    for w in words:
        ww, _ = measure_text(d, w["word"], font)
        widths.append(ww)

    total_w = sum(widths) + space_w * max(0, len(words) - 1)
    start_x = (video_width - total_w) // 2

    x = int(start_x)
    out = []
    for i, w in enumerate(words):
        out.append({**w, "x": x, "w": widths[i]})
        x += widths[i] + space_w
    return out


def render_full_line_rgba(
    words_with_pos: List[Dict],
    video_width: int,
    font: ImageFont.ImageFont,
    base_color_hex: str,
    stroke_width: int,
    stroke_color_hex: str,
    extra_pad_y: int = 12,
) -> Tuple[np.ndarray, int, int]:
    ascent, descent = font.getmetrics()
    line_h = ascent + descent

    sw = int(stroke_width)
    pad = sw + int(extra_pad_y)

    img_h = line_h + 2 * pad
    img = Image.new("RGBA", (video_width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill = _hex_to_rgba(base_color_hex, 255)
    stroke_fill = _hex_to_rgba(stroke_color_hex, 255)

    y_text = pad
    for w in words_with_pos:
        draw_text_with_stroke(draw, (int(w["x"]), int(y_text)), w["word"], font, fill, stroke_fill, sw)

    return np.array(img), img_h, y_text


def render_word_rgba(
    word: str,
    font: ImageFont.ImageFont,
    color_hex: str,
    stroke_width: int,
    stroke_color_hex: str,
    extra_pad_y: int,
) -> Tuple[np.ndarray, int, int, int]:
    ascent, descent = font.getmetrics()
    line_h = ascent + descent

    sw = int(stroke_width)
    pad = sw + int(extra_pad_y)

    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    w, _ = measure_text(d, word, font)

    img_w = w + 2 * pad
    img_h = line_h + 2 * pad

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill = _hex_to_rgba(color_hex, 255)
    stroke_fill = _hex_to_rgba(stroke_color_hex, 255)

    draw_text_with_stroke(draw, (pad, pad), word, font, fill, stroke_fill, sw)

    arr = np.array(img)
    return arr, img_w, img_h, pad


def pop_scale(t_local: float, duration: float, base: float = 1.0, peak: float = 1.22) -> float:
    if duration <= 0:
        return base
    x = max(0.0, min(1.0, t_local / duration))
    tri = 1.0 - abs(2 * x - 1.0)
    return base + (peak - base) * tri


# def chunk_scale(t_local: float, dur: float, in_time: float = 0.14, out_time: float = 0.14,
#                 base: float = 1.0, peak: float = 1.06) -> float:
#     if dur <= 0:
#         return base
#     if t_local < in_time:
#         x = max(0.0, min(1.0, t_local / in_time))
#         return base + (peak - base) * x
#     if t_local > dur - out_time:
#         x = max(0.0, min(1.0, (dur - t_local) / out_time))
#         return base + (peak - base) * x
#     return base


def find_chunk_at_time(chunks: List[Dict], t: float) -> Optional[Dict]:
    for c in chunks:
        if c["start"] <= t < c["end"]:
            return c
    return None


def find_word_at_time(words: List[Dict], t: float) -> Optional[Dict]:
    for w in words:
        if w["start"] <= t < w["end"]:
            return w
    return None


# -----------------------------
# Vertical output (MoviePy v2)
# -----------------------------
def make_vertical_base_clip(base: VideoFileClip, OUT_W: int, OUT_H: int, mode: str) -> VideoFileClip:
    w, h = base.w, base.h
    target_aspect = OUT_W / OUT_H
    src_aspect = w / h

    if mode == "Crop to fill (center)":
        if src_aspect >= target_aspect:
            scaled = base.resized(height=OUT_H)
        else:
            scaled = base.resized(width=OUT_W)
        return scaled.cropped(
            x_center=scaled.w / 2,
            y_center=scaled.h / 2,
            width=OUT_W,
            height=OUT_H,
        )

    if mode == "Fit with black bars":
        fitted = base.resized(height=OUT_H) if src_aspect < target_aspect else base.resized(width=OUT_W)
        if fitted.w > OUT_W:
            fitted = fitted.resized(width=OUT_W)
        if fitted.h > OUT_H:
            fitted = fitted.resized(height=OUT_H)
        bg = ColorClip((OUT_W, OUT_H), color=(0, 0, 0)).with_duration(base.duration)
        comp = CompositeVideoClip([bg, fitted.with_position(("center", "center"))])
        return comp.with_audio(base.audio)

    if mode == "Fit with blur background":
        if src_aspect >= target_aspect:
            bg_fill = base.resized(height=OUT_H)
        else:
            bg_fill = base.resized(width=OUT_W)
        bg_fill = bg_fill.cropped(
            x_center=bg_fill.w / 2,
            y_center=bg_fill.h / 2,
            width=OUT_W,
            height=OUT_H,
        )

        def blur_frame(get_frame, t):
            frame = get_frame(t)
            return cv2.GaussianBlur(frame, (51, 51), 0)

        bg_blur = bg_fill.transform(blur_frame)

        fg = base.resized(height=OUT_H) if src_aspect < target_aspect else base.resized(width=OUT_W)
        if fg.w > OUT_W:
            fg = fg.resized(width=OUT_W)
        if fg.h > OUT_H:
            fg = fg.resized(height=OUT_H)

        comp = CompositeVideoClip([bg_blur, fg.with_position(("center", "center"))])
        return comp.with_audio(base.audio)

    return base


def make_vertical_frame_preview(frame_rgb: np.ndarray, OUT_W: int, OUT_H: int, mode: str) -> np.ndarray:
    H, W = frame_rgb.shape[:2]
    target_aspect = OUT_W / OUT_H
    src_aspect = W / H

    if mode == "Crop to fill (center)":
        # scale to fill then center crop
        if src_aspect >= target_aspect:
            new_h = OUT_H
            new_w = int(W * (new_h / H))
        else:
            new_w = OUT_W
            new_h = int(H * (new_w / W))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = max(0, (new_w - OUT_W) // 2)
        y0 = max(0, (new_h - OUT_H) // 2)
        return resized[y0:y0 + OUT_H, x0:x0 + OUT_W]

    if mode == "Fit with black bars":
        canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        # fit inside
        if src_aspect >= target_aspect:
            new_w = OUT_W
            new_h = int(H * (new_w / W))
        else:
            new_h = OUT_H
            new_w = int(W * (new_h / H))
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (OUT_W - new_w) // 2
        y0 = (OUT_H - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    if mode == "Fit with blur background":
        # background fill + blur
        bg = make_vertical_frame_preview(frame_rgb, OUT_W, OUT_H, "Crop to fill (center)")
        bg = cv2.GaussianBlur(bg, (51, 51), 0)
        # foreground fit
        if src_aspect >= target_aspect:
            new_w = OUT_W
            new_h = int(H * (new_w / W))
        else:
            new_h = OUT_H
            new_w = int(W * (new_h / H))
        fg = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x0 = (OUT_W - new_w) // 2
        y0 = (OUT_H - new_h) // 2
        out = bg.copy()
        out[y0:y0 + new_h, x0:x0 + new_w] = fg
        return out

    return frame_rgb


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Karaoke Captions (No Background)", layout="wide")
st.title("🎬 Karaoke Captions (4 words) — stroke + pop highlight (MoviePy v2)")

with st.expander("Debug"):
    st.write("Python:", sys.version)
    st.write("Default font path:", DEFAULT_FONT_PATH)
    st.write("Font exists?", os.path.exists(DEFAULT_FONT_PATH))

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("1) Upload")
    video_file = st.file_uploader("Upload video (mp4/mov/mkv)", type=["mp4", "mov", "mkv"])

    st.subheader("2) Whisper")
    whisper_size = st.selectbox("Model size (Cloud-friendly)", ["tiny", "base", "small"], index=1)
    language_hint = st.text_input("Optional language hint (e.g., id, en)", value="en")

    st.subheader("3) Captions")
    words_per_chunk = st.slider("Words per chunk", 2, 8, 3, 1)

    enable_translate = st.checkbox("Translate captions (timing becomes approximate)", value=True)
    lang_keys = list(LANG_OPTIONS.keys())
    lang_name = st.selectbox("Target language", options=lang_keys, index=lang_keys.index("Bahasa Melayu"))
    target_lang = LANG_OPTIONS[lang_name]
    # Store for font auto-pick (no UI change)
    st.session_state["target_lang"] = target_lang

    st.subheader("4) Style")
    base_color = st.color_picker("Base text color", "#FFFFFF")
    highlight_color = st.color_picker("Highlight color", "#FF8200")
    font_size = st.slider("Font size", 16, 90, 64, 1)

    stroke_width = st.slider("Stroke width", 0, 12, 2, 1)
    stroke_color = st.color_picker("Stroke color", "#000000")

    pop_peak = st.slider("Pop peak scale", 1.0, 1.6, 1.0, 0.02)
    extra_pad_y = st.slider("Anti-clip vertical padding", 6, 30, 12, 1)

    st.subheader("5) Position")
    y_mode = st.radio("Y mode", ["Percent", "Pixels"], index=0, horizontal=True)
    if y_mode == "Percent":
        y_percent = st.slider("Caption Y (% from top)", 55, 82, 65, 1)
        y_pos_px = None
    else:
        y_pos_px = st.slider("Caption Y (px from top)", 0, 2000, 900, 10)
        y_percent = None

    st.subheader("6) Preview Controls")
    preview_scale = st.slider("Preview scale", 0.2, 1.0, 0.45, 0.05)
    show_guides = st.checkbox("Show guide line (preview)", value=True)
    preview_time = st.slider("Preview time (seconds)", 0.0, 120.0, 0.0, 0.1)

    # NEW (vertical output options)
    st.subheader("7) Output (Vertical)")
    out_preset = st.selectbox("Output size", ["1080x1920 (9:16)", "720x1280 (9:16)"], index=0)
    vertical_mode = st.selectbox(
        "Vertical mode",
        ["Crop to fill (center)", "Fit with black bars", "Fit with blur background"],
        index=2
    )
    if out_preset.startswith("1080"):
        OUT_W, OUT_H = 1080, 1920
    else:
        OUT_W, OUT_H = 720, 1280

    render_btn = st.button("🚀 Render MP4", type="primary", use_container_width=True)

with colR:
    st.subheader("Video preview")
    if video_file is not None:
        st.video(video_file)

    st.subheader("Small live preview (single frame)")
    st.caption("Updates when sliders change. Shows caption + highlighted word.")
    preview_slot = st.empty()


# -----------------------------
# Session cache
# -----------------------------
if "cache_key" not in st.session_state:
    st.session_state.cache_key = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""


def build_cache_key() -> Optional[str]:
    if video_file is None:
        return None
    return f"{video_file.name}|{whisper_size}|{language_hint.strip()}|{int(words_per_chunk)}|{enable_translate}|{target_lang}"


def ensure_ready():
    key = build_cache_key()
    if key is None:
        return
    if st.session_state.cache_key == key and st.session_state.video_path and st.session_state.chunks:
        return

    st.session_state.video_path = save_uploaded_file(video_file, suffix=".mp4")

    whisper = load_whisper(whisper_size)
    hint = language_hint.strip() or None

    words, full_text = transcribe_words(st.session_state.video_path, whisper, language_hint=hint)
    chunks = build_chunks_from_words(words, n=int(words_per_chunk))

    if enable_translate:
        chunks = translate_chunks_keep_timing(chunks, target_lang)

    st.session_state.cache_key = key
    st.session_state.chunks = chunks
    st.session_state.full_text = full_text


# -----------------------------
# Live preview (now shows vertical output)
# -----------------------------
if video_file is not None:
    try:
        ensure_ready()
        vp = st.session_state.video_path
        chunks = st.session_state.chunks

        frame = get_frame_at(vp, preview_time)
        if frame is not None:
            # Convert preview frame to vertical canvas first
            frame_v = make_vertical_frame_preview(frame, OUT_W, OUT_H, vertical_mode)

            H, W = frame_v.shape[:2]
            font = load_font(int(font_size))

            chunk = find_chunk_at_time(chunks, preview_time)
            preview_img = frame_v.copy()

            if chunk:
                words_pos = compute_word_positions_centered(chunk["words"], font, W)
                base_strip, strip_h, y_text = render_full_line_rgba(
                    words_pos, W, font, base_color,
                    int(stroke_width), stroke_color,
                    extra_pad_y=int(extra_pad_y),
                )

                if y_mode == "Percent":
                    y = int(H * (y_percent / 100.0))
                else:
                    y = int(y_pos_px if y_pos_px is not None else int(H * 0.82))
                y = max(0, min(H - strip_h, y))

                preview_img = overlay_rgba_on_rgb(preview_img, base_strip, 0, y)

                curw = find_word_at_time(chunk["words"], preview_time)
                if curw:
                    match = None
                    for wp in words_pos:
                        if abs(wp["start"] - curw["start"]) < 1e-6 and wp["word"] == curw["word"]:
                            match = wp
                            break
                    if match:
                        word_rgba, ww, wh, pad = render_word_rgba(
                            match["word"], font, highlight_color,
                            int(stroke_width), stroke_color,
                            extra_pad_y=int(extra_pad_y),
                        )
                        wx = int(match["x"]) - pad
                        wy = y + (y_text - pad)
                        preview_img = overlay_rgba_on_rgb(preview_img, word_rgba, wx, wy)

                if show_guides:
                    cv2.line(preview_img, (0, y), (W - 1, y), (0, 255, 255), 2)

            new_w = int(W * preview_scale)
            new_h = int(H * preview_scale)
            small = cv2.resize(preview_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            preview_slot.image(small, caption="Preview (vertical output + captions)")
    except Exception as e:
        preview_slot.warning(f"Preview error: {e}")


# -----------------------------
# Final render (vertical output + chunk animations)
# -----------------------------
if render_btn:
    if video_file is None:
        st.error("Please upload a video first.")
        st.stop()

    with st.status("Rendering…", expanded=True) as status:
        ensure_ready()
        video_path = st.session_state.video_path
        chunks = st.session_state.chunks

        status.write("Opening video…")
        with VideoFileClip(video_path) as base_raw:
            # Make the base vertical first
            base = make_vertical_base_clip(base_raw, OUT_W, OUT_H, vertical_mode)

            W, H = base.w, base.h
            fps = base.fps if base.fps else 30
            font = load_font(int(font_size))

            overlays: List[ImageClip] = []
            status.write("Building caption clips…")

            for c in chunks:
                start = float(c["start"])
                end = float(c["end"])
                dur = max(0.10, end - start)

                words_pos = compute_word_positions_centered(c["words"], font, W)
                base_strip, strip_h, y_text = render_full_line_rgba(
                    words_pos, W, font, base_color,
                    int(stroke_width), stroke_color,
                    extra_pad_y=int(extra_pad_y),
                )

                if y_mode == "Percent":
                    y = int(H * (y_percent / 100.0))
                else:
                    y = int(y_pos_px if y_pos_px is not None else int(H * 0.82))
                y = max(0, min(H - strip_h, y))

                # Chunk-level animation: fade in/out + subtle scale
                base_clip = (
                    ImageClip(base_strip)
                    .with_start(start)
                    .with_duration(dur)
                    .with_position((0, y))
                    .with_effects([
                        FadeIn(0.14),
                        FadeOut(0.14),
                        # Resize(lambda t, d=dur: chunk_scale(t, d, 0.14, 0.14, 1.0, 1.06))
                    ])
                )
                overlays.append(base_clip)

                # Word highlight + pop (still your existing behavior)
                for wp in words_pos:
                    ws = float(wp["start"])
                    we = float(wp["end"])
                    wdur = max(0.06, we - ws)

                    word_rgba, ww, wh, pad = render_word_rgba(
                        wp["word"], font, highlight_color,
                        int(stroke_width), stroke_color,
                        extra_pad_y=int(extra_pad_y),
                    )
                    wx = int(wp["x"]) - pad
                    wy = y + (y_text - pad)

                    wc = (
                        ImageClip(word_rgba)
                        .with_start(ws)
                        .with_duration(wdur)
                        .with_position((wx, wy))
                        .with_effects([
                            Resize(lambda t, d=wdur, peak=float(pop_peak):
                                   pop_scale(t, d, base=1.0, peak=peak))
                        ])
                    )
                    overlays.append(wc)

            status.write("Compositing & exporting MP4…")
            final = CompositeVideoClip([base, *overlays])

            out_path = os.path.join(tempfile.gettempdir(), "captioned_output.mp4")
            final.write_videofile(
                out_path,
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium",
            )

        status.update(label="Done ✅", state="complete")

    st.success("Video generated!")
    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button(
            "⬇️ Download output MP4",
            data=f,
            file_name="captioned_output.mp4",
            mime="video/mp4",
        )

    with st.expander("Transcript (original)"):
        st.write(st.session_state.full_text or "_(empty)_")
