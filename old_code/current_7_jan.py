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
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip
from moviepy.video.fx import Resize  # v2 effect for animation resizing


# -----------------------------
# Config
# -----------------------------
DEFAULT_FONT_PATH = "fonts/Inter-SemiBold.ttf"

FALLBACK_FONT_CANDIDATES = [
    DEFAULT_FONT_PATH,
    "font/Montserrat-Bold.ttf",
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


def load_font(font_size: int) -> ImageFont.ImageFont:
    for fp in FALLBACK_FONT_CANDIDATES:
        try:
            if fp and os.path.exists(fp):
                return ImageFont.truetype(fp, font_size)
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
    """
    Base caption strip with generous padding (prevents clipping).
    Returns (strip_rgba, strip_h, y_text_offset).
    """
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
    extra_pad_y: int,  # <-- MATCH base strip
) -> Tuple[np.ndarray, int, int, int]:
    """
    Highlight word RGBA with SAME padding model as base strip (fix yellow clipping).
    Returns (rgba, img_w, img_h, pad).
    """
    ascent, descent = font.getmetrics()
    line_h = ascent + descent

    sw = int(stroke_width)
    pad = sw + int(extra_pad_y)  # SAME as base

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
    language_hint = st.text_input("Optional language hint (e.g., id, en)", value="")

    st.subheader("3) Captions")
    words_per_chunk = st.slider("Words per chunk", 2, 8, 4, 1)

    enable_translate = st.checkbox("Translate captions (timing becomes approximate)", value=True)
    lang_keys = list(LANG_OPTIONS.keys())
    lang_name = st.selectbox("Target language", options=lang_keys, index=lang_keys.index("English"))
    target_lang = LANG_OPTIONS[lang_name]

    st.subheader("4) Style")
    base_color = st.color_picker("Base text color", "#FFFFFF")
    highlight_color = st.color_picker("Highlight color", "#FFD400")
    font_size = st.slider("Font size", 16, 90, 44, 1)

    stroke_width = st.slider("Stroke width", 0, 12, 3, 1)
    stroke_color = st.color_picker("Stroke color", "#000000")

    pop_peak = st.slider("Pop peak scale", 1.0, 1.6, 1.0, 0.02)
    extra_pad_y = st.slider("Anti-clip vertical padding", 6, 30, 12, 1)

    st.subheader("5) Position")
    y_mode = st.radio("Y mode", ["Percent", "Pixels"], index=0, horizontal=True)
    if y_mode == "Percent":
        y_percent = st.slider("Caption Y (% from top)", 55, 95, 82, 1)
        y_pos_px = None
    else:
        y_pos_px = st.slider("Caption Y (px from top)", 0, 2000, 900, 10)
        y_percent = None

    st.subheader("6) Preview Controls")
    preview_scale = st.slider("Preview scale", 0.2, 1.0, 0.45, 0.05)
    show_guides = st.checkbox("Show guide line (preview)", value=True)
    preview_time = st.slider("Preview time (seconds)", 0.0, 120.0, 0.0, 0.1)

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
# Live preview
# -----------------------------
if video_file is not None:
    try:
        ensure_ready()
        vp = st.session_state.video_path
        chunks = st.session_state.chunks

        frame = get_frame_at(vp, preview_time)
        if frame is not None:
            H, W = frame.shape[:2]
            font = load_font(int(font_size))

            chunk = find_chunk_at_time(chunks, preview_time)
            preview_img = frame.copy()

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
            preview_slot.image(small, caption="Preview (caption + highlighted word)")
    except Exception as e:
        preview_slot.warning(f"Preview error: {e}")


# -----------------------------
# Final render
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
        with VideoFileClip(video_path) as base:
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

                base_clip = (
                    ImageClip(base_strip)
                    .with_start(start)
                    .with_duration(dur)
                    .with_position((0, y))
                )
                overlays.append(base_clip)

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

