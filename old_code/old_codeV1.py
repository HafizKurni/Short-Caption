# app.py  (Streamlit + Faster-Whisper + Translate + MoviePy v2)
# Improvements included:
# 1) Small preview image with caption overlay + bounding box (updates when sliders change)
# 2) Custom font upload (.ttf)
# 3) Stroke + padding + rounded box effects
# 4) Karaoke-style word color animation using word timestamps (best with no-translate;
#    with translate we do an approximate timing-to-words mapping)

import os
import sys
import math
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


# -----------------------------
# Language dropdown (name -> code)
# -----------------------------
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


# -----------------------------
# Utilities: files, colors, fonts
# -----------------------------
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


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    # If user uploaded a font, use it. Otherwise use safe fallbacks.
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass

    font_candidates = [
        "DejaVuSans.ttf",  # Linux common
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "arial.ttf",  # Windows
    ]
    for fp in font_candidates:
        try:
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
    if stroke_width > 0:
        # radial-ish stroke
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx * dx + dy * dy <= stroke_width * stroke_width:
                    draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)
    draw.text((x, y), text, font=font, fill=fill)


def measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# -----------------------------
# Renderer: caption box (full line) and word overlay (single word)
# -----------------------------
def render_caption_box_rgba(
    text: str,
    width: int,
    font_path: Optional[str],
    font_size: int,
    color_hex: str,
    box: bool,
    box_color_hex: str,
    box_alpha: int,
    padding: int,
    line_spacing: int,
    max_lines: int,
    radius: int,
    stroke_width: int,
    stroke_color_hex: str,
) -> np.ndarray:
    """
    Render the entire caption line(s) into a full-width RGBA image with optional rounded box + stroke.
    This is used as the "base" layer for the chunk duration.
    """
    font = load_font(font_path, font_size)
    dummy = Image.new("RGBA", (width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)

    # For micro captions, keep it compact; still wrap minimally if needed.
    est_chars_per_line = max(8, int((width - 2 * padding) / (0.60 * font_size)))
    words = text.split()
    lines = []
    cur = []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) >= est_chars_per_line:
            lines.append(" ".join(cur))
            cur = []
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    if not lines:
        lines = [""]

    sizes = [measure_text(d, ln, font) for ln in lines]
    text_h = sum(h for _, h in sizes) + line_spacing * (len(lines) - 1)
    img_h = text_h + 2 * padding

    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if box:
        box_rgba = _hex_to_rgba(box_color_hex, int(box_alpha))
        r = max(0, int(radius))
        # Full-width box (to keep position stable)
        draw.rounded_rectangle(
            [padding // 2, padding // 2, width - padding // 2, img_h - padding // 2],
            radius=r,
            fill=box_rgba,
        )

    fill = _hex_to_rgba(color_hex, 255)
    stroke_fill = _hex_to_rgba(stroke_color_hex, 255)

    y = padding
    for i, ln in enumerate(lines):
        w, h = sizes[i]
        x = (width - w) // 2
        draw_text_with_stroke(
            draw, (x, y), ln, font,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_width=int(stroke_width),
        )
        y += h + line_spacing

    return np.array(img)


def render_word_rgba(
    word: str,
    font_path: Optional[str],
    font_size: int,
    color_hex: str,
    stroke_width: int,
    stroke_color_hex: str,
) -> Tuple[np.ndarray, int, int]:
    """
    Render a single word into a tight RGBA image. Returns (rgba, w, h).
    Used for karaoke highlight overlays.
    """
    font = load_font(font_path, font_size)
    # measure
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)
    w, h = measure_text(d, word, font)

    # add stroke padding
    sw = int(stroke_width)
    pad = max(1, sw + 2)
    img = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    fill = _hex_to_rgba(color_hex, 255)
    stroke_fill = _hex_to_rgba(stroke_color_hex, 255)
    draw_text_with_stroke(
        draw, (pad, pad), word, font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=sw,
    )
    arr = np.array(img)
    return arr, img.size[0], img.size[1]


# -----------------------------
# Watermark
# -----------------------------
def apply_logo_opacity(logo_pil: Image.Image, opacity: float) -> Image.Image:
    opacity = max(0.0, min(1.0, float(opacity)))
    logo = logo_pil.convert("RGBA")
    r, g, b, a = logo.split()
    a = a.point(lambda px: int(px * opacity))
    return Image.merge("RGBA", (r, g, b, a))


# -----------------------------
# Preview helpers (small preview with bbox + overlay)
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


def caption_chunk_at_time(chunks: List[Dict], t: float) -> Optional[Dict]:
    for c in chunks:
        if c["start"] <= t < c["end"]:
            return c
    return None


# -----------------------------
# Whisper + chunking with word timestamps
# -----------------------------
@st.cache_resource
def load_whisper(model_size: str) -> WhisperModel:
    # Cloud-friendly CPU mode
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_with_words(
    video_path: str,
    whisper_model: WhisperModel,
    language_hint: Optional[str] = None,
) -> Tuple[List[Dict], str]:
    """
    Returns:
    - words: list of {start, end, word}
    - full_text: transcript
    """
    segments, _info = whisper_model.transcribe(
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

    full_text = " ".join(full_parts).strip()
    return words, full_text


def build_chunks_from_words(words: List[Dict], words_per_chunk: int = 4) -> List[Dict]:
    """
    Build chunk objects from word timestamps:
    chunk = {start, end, text, words:[{start,end,word}, ...]}
    """
    chunks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) == words_per_chunk:
            start = cur[0]["start"]
            end = cur[-1]["end"]
            text = " ".join([x["word"] for x in cur])
            chunks.append({"start": start, "end": end, "text": text, "words": cur})
            cur = []
    if cur:
        start = cur[0]["start"]
        end = cur[-1]["end"]
        text = " ".join([x["word"] for x in cur])
        chunks.append({"start": start, "end": end, "text": text, "words": cur})
    return chunks


def translate_chunk_text(text: str, target_lang: str) -> str:
    if not text.strip():
        return ""
    return GoogleTranslator(source="auto", target=target_lang).translate(text).strip()


def approximate_word_timing_for_translation(orig_words: List[Dict], translated_text: str) -> List[Dict]:
    """
    When translating, word timestamps no longer match. We approximate by:
    - splitting translated_text into words
    - distributing original total duration across translated words evenly
    """
    trans_words = translated_text.split()
    if not trans_words:
        return []

    start = orig_words[0]["start"]
    end = orig_words[-1]["end"]
    dur = max(0.10, end - start)
    step = dur / len(trans_words)

    out = []
    for i, tw in enumerate(trans_words):
        ws = start + i * step
        we = min(end, ws + step)
        out.append({"start": ws, "end": we, "word": tw})
    return out


# -----------------------------
# Karaoke layout: compute word x positions centered
# -----------------------------
def compute_word_positions_centered(
    words: List[Dict],
    font_path: Optional[str],
    font_size: int,
    video_width: int,
) -> List[Dict]:
    """
    Adds x offsets for each word in a single-line caption, centered.
    Returns list of word dict with extra field: x (pixel)
    """
    font = load_font(font_path, font_size)
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(tmp)

    space_w, _ = measure_text(d, " ", font)
    widths = []
    for w in words:
        ww, _ = measure_text(d, w["word"], font)
        widths.append(ww)

    total_w = sum(widths) + space_w * (max(0, len(words) - 1))
    start_x = (video_width - total_w) // 2

    x = int(start_x)
    out = []
    for i, w in enumerate(words):
        out.append({**w, "x": x})
        x += widths[i] + space_w
    return out


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Short Captions + Preview + Karaoke", layout="wide")
st.title("🎬 Short Captions (Whisper timestamps) + Small Preview + Karaoke Highlight + Watermark (MoviePy v2)")

with st.expander("Debug"):
    st.write("Python:", sys.version)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("1) Upload")
    video_file = st.file_uploader("Upload video (mp4/mov/mkv)", type=["mp4", "mov", "mkv"])

    st.subheader("2) Whisper")
    whisper_size = st.selectbox("Whisper model size (Cloud-friendly)", ["tiny", "base", "small"], index=1)
    language_hint = st.text_input("Optional language hint (e.g., id, en). Leave blank for auto-detect.", value="")

    st.subheader("3) Translation")
    enable_translate = st.checkbox("Translate captions", value=False)
    lang_keys = list(LANG_OPTIONS.keys())
    default_idx = lang_keys.index("English") if "English" in lang_keys else 0
    lang_name = st.selectbox("Target language", options=lang_keys, index=default_idx)
    target_lang = LANG_OPTIONS[lang_name]
    st.caption(f"Selected: {lang_name} ({target_lang})")

    st.subheader("4) Caption Content")
    words_per_chunk = st.slider("Words per chunk", 2, 8, 4, 1)

    st.subheader("5) Caption Styling")
    font_file = st.file_uploader("Custom font (.ttf)", type=["ttf"])
    font_size = st.slider("Font size", 16, 72, 34, 1)
    caption_color = st.color_picker("Base text color", "#FFFFFF")

    karaoke = st.checkbox("Karaoke highlight (word color follows timestamps)", value=True)
    highlight_color = st.color_picker("Highlight word color", "#FFD400")

    # Stroke + box effects
    stroke_width = st.slider("Stroke width", 0, 8, 2, 1)
    stroke_color = st.color_picker("Stroke color", "#000000")

    box = st.checkbox("Use rounded background box", value=True)
    box_color = st.color_picker("Box color", "#000000")
    box_alpha = st.slider("Box opacity (0 transparent - 255 solid)", 0, 255, 120, 5)
    padding = st.slider("Box padding", 8, 60, 18, 2)
    radius = st.slider("Box corner radius", 0, 40, 18, 1)

    # Position controls
    st.subheader("6) Position")
    y_mode = st.radio("Y positioning mode", ["Percent", "Pixels"], index=0, horizontal=True)
    if y_mode == "Percent":
        y_percent = st.slider("Caption Y (% from top)", 60, 95, 82, 1)
        y_pos_px = None
    else:
        y_pos_px = st.slider("Caption Y (px from top)", 0, 2000, 900, 10)
        y_percent = None

    st.subheader("7) Watermark / Logo (Optional)")
    logo_file = st.file_uploader("Upload logo (png recommended)", type=["png", "jpg", "jpeg"])
    logo_opacity = st.slider("Logo opacity", 0.0, 1.0, 0.35, 0.05)
    logo_scale = st.slider("Logo scale (relative to video width)", 0.05, 0.6, 0.18, 0.01)
    logo_pos = st.selectbox("Logo position", ["top-left", "top-right", "bottom-left", "bottom-right"], index=3)

    st.subheader("8) Preview Controls")
    preview_scale = st.slider("Preview scale", 0.2, 1.0, 0.45, 0.05)
    show_bbox = st.checkbox("Show caption area box (preview)", value=True)
    preview_time = st.slider("Preview time (seconds)", 0.0, 120.0, 0.0, 0.1)

    render_btn = st.button("🚀 Generate captioned video", type="primary", use_container_width=True)

with colR:
    st.subheader("Video Preview")
    if video_file is not None:
        st.video(video_file)

    st.subheader("Small Live Preview (frame + overlay)")
    st.caption("This preview updates instantly when you change sliders (it’s a single frame).")
    preview_slot = st.empty()


# -----------------------------
# Processing (transcribe once per upload + settings)
# We'll store results in session_state so sliders can update preview without re-transcribing.
# -----------------------------
if "cache_key" not in st.session_state:
    st.session_state.cache_key = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "font_path" not in st.session_state:
    st.session_state.font_path = None


def build_cache_key() -> Optional[str]:
    if video_file is None:
        return None
    # Key depends on: file name + whisper size + language hint + words_per_chunk + translate settings + target lang
    return f"{video_file.name}|{whisper_size}|{language_hint.strip()}|{int(words_per_chunk)}|{enable_translate}|{target_lang}"


def ensure_transcription_ready():
    key = build_cache_key()
    if key is None:
        return

    # Save font if uploaded
    if font_file is not None:
        st.session_state.font_path = save_uploaded_file(font_file, suffix=".ttf")
    else:
        st.session_state.font_path = None

    # If already computed with same key, skip
    if st.session_state.cache_key == key and st.session_state.video_path and st.session_state.chunks:
        return

    # Save video to temp (new each time)
    st.session_state.video_path = save_uploaded_file(video_file, suffix=".mp4")

    whisper = load_whisper(whisper_size)
    hint = language_hint.strip() or None

    words, full_text = transcribe_with_words(st.session_state.video_path, whisper, language_hint=hint)
    chunks = build_chunks_from_words(words, words_per_chunk=int(words_per_chunk))

    # Translate (keeps chunk timings; karaoke uses approximate mapping if enabled)
    if enable_translate:
        translated_chunks = []
        for c in chunks:
            t = translate_chunk_text(c["text"], target_lang)
            # approximate word timings for karaoke on translated text
            approx_words = approximate_word_timing_for_translation(c["words"], t) if t else []
            translated_chunks.append({
                "start": c["start"],
                "end": c["end"],
                "text": t if t else c["text"],
                "words": approx_words if approx_words else c["words"],  # fallback
            })
        chunks = translated_chunks

    st.session_state.cache_key = key
    st.session_state.chunks = chunks
    st.session_state.full_text = full_text


# Build live preview if we have video & transcription ready (even before final render)
if video_file is not None:
    try:
        ensure_transcription_ready()

        vp = st.session_state.video_path
        chunks = st.session_state.chunks
        font_path = st.session_state.font_path

        frame = get_frame_at(vp, preview_time)
        if frame is not None:
            H0, W0 = frame.shape[:2]

            if y_mode == "Percent":
                safe_y = int(H0 * (y_percent / 100.0))
            else:
                safe_y = int(y_pos_px if y_pos_px is not None else int(H0 * 0.82))

            safe_y = max(0, min(H0 - 1, safe_y))

            chunk_now = caption_chunk_at_time(chunks, preview_time)
            text_now = chunk_now["text"] if chunk_now else ""

            # base caption box overlay (full-width image)
            base_rgba = render_caption_box_rgba(
                text=text_now if text_now else " ",
                width=W0,
                font_path=font_path,
                font_size=int(font_size),
                color_hex=caption_color,
                box=box,
                box_color_hex=box_color,
                box_alpha=int(box_alpha),
                padding=int(padding),
                line_spacing=6,
                max_lines=2,
                radius=int(radius),
                stroke_width=int(stroke_width),
                stroke_color_hex=stroke_color,
            )

            oh, ow = base_rgba.shape[:2]
            x = 0  # full-width overlay
            y = safe_y

            preview_img = frame.copy()
            preview_img = overlay_rgba_on_rgb(preview_img, base_rgba, x, y)

            # If karaoke enabled, highlight current word at preview time
            if karaoke and chunk_now and chunk_now.get("words"):
                # pick the current word by time
                current = None
                for w in chunk_now["words"]:
                    if w["start"] <= preview_time < w["end"]:
                        current = w
                        break

                if current:
                    # compute x positions for words centered, then render only that word with highlight color
                    # note: base overlay is full-width with centered text, so we align highlight by word x positions
                    word_layout = compute_word_positions_centered(
                        chunk_now["words"], font_path, int(font_size), W0
                    )
                    # find current word x
                    for w in word_layout:
                        if w["word"] == current["word"] and abs(w["start"] - current["start"]) < 1e-3:
                            word_rgba, ww, wh = render_word_rgba(
                                w["word"], font_path, int(font_size),
                                color_hex=highlight_color,
                                stroke_width=int(stroke_width),
                                stroke_color_hex=stroke_color,
                            )
                            # y position inside the box: approximate by padding
                            wy = y + int(padding)
                            wx = w["x"]
                            preview_img = overlay_rgba_on_rgb(preview_img, word_rgba, wx, wy)
                            break

            if show_bbox:
                if y + oh < H0:
                    cv2.rectangle(preview_img, (x, y), (x + ow - 1, y + oh - 1), (255, 255, 0), 3)

            new_w = int(W0 * preview_scale)
            new_h = int(H0 * preview_scale)
            preview_small = cv2.resize(preview_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            preview_slot.image(preview_small, caption="Small preview (frame + caption overlay + bbox)")
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
        ensure_transcription_ready()

        video_path = st.session_state.video_path
        chunks = st.session_state.chunks
        font_path = st.session_state.font_path

        status.write("Opening video…")
        with VideoFileClip(video_path) as base:
            W, H = base.w, base.h
            fps = base.fps if base.fps else 30

            # compute caption y
            if y_mode == "Percent":
                safe_y = int(H * (y_percent / 100.0))
            else:
                safe_y = int(y_pos_px if y_pos_px is not None else int(H * 0.82))
            safe_y = max(0, min(H - 1, safe_y))

            overlays = []

            status.write("Building caption clips…")

            # For each chunk:
            # - base caption box clip spans chunk duration
            # - karaoke highlight word clips span per word (colored)
            for c in chunks:
                start = float(c["start"])
                end = float(c["end"])
                dur = max(0.10, end - start)
                text = c["text"] or " "

                base_rgba = render_caption_box_rgba(
                    text=text,
                    width=W,
                    font_path=font_path,
                    font_size=int(font_size),
                    color_hex=caption_color,
                    box=box,
                    box_color_hex=box_color,
                    box_alpha=int(box_alpha),
                    padding=int(padding),
                    line_spacing=6,
                    max_lines=2,
                    radius=int(radius),
                    stroke_width=int(stroke_width),
                    stroke_color_hex=stroke_color,
                )

                base_clip = (
                    ImageClip(base_rgba)
                    .with_start(start)
                    .with_duration(dur)
                    .with_position((0, safe_y))  # full-width overlay at x=0
                )
                overlays.append(base_clip)

                # karaoke highlight
                if karaoke and c.get("words"):
                    word_layout = compute_word_positions_centered(
                        c["words"], font_path, int(font_size), W
                    )
                    # y inside the box: approximate by padding
                    wy = safe_y + int(padding)

                    for w in word_layout:
                        ws = float(w["start"])
                        we = float(w["end"])
                        wdur = max(0.06, we - ws)

                        word_rgba, ww, wh = render_word_rgba(
                            w["word"],
                            font_path,
                            int(font_size),
                            color_hex=highlight_color,
                            stroke_width=int(stroke_width),
                            stroke_color_hex=stroke_color,
                        )
                        wc = (
                            ImageClip(word_rgba)
                            .with_start(ws)
                            .with_duration(wdur)
                            .with_position((int(w["x"]), int(wy)))
                        )
                        overlays.append(wc)

            # Optional watermark/logo
            if logo_file is not None:
                status.write("Applying watermark/logo…")
                logo_path = save_uploaded_file(logo_file, suffix=".png")
                logo_pil = Image.open(logo_path)
                logo_pil = apply_logo_opacity(logo_pil, logo_opacity)

                target_logo_w = max(20, int(W * float(logo_scale)))
                ratio = target_logo_w / max(1, logo_pil.size[0])
                target_logo_h = max(20, int(logo_pil.size[1] * ratio))
                logo_pil = logo_pil.resize((target_logo_w, target_logo_h))

                logo_np = np.array(logo_pil)
                logo_clip = ImageClip(logo_np).with_duration(base.duration)

                pad = 20
                if logo_pos == "top-left":
                    pos_xy = (pad, pad)
                elif logo_pos == "top-right":
                    pos_xy = (W - target_logo_w - pad, pad)
                elif logo_pos == "bottom-left":
                    pos_xy = (pad, H - target_logo_h - pad)
                else:
                    pos_xy = (W - target_logo_w - pad, H - target_logo_h - pad)

                overlays.append(logo_clip.with_position(pos_xy))

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

    with st.expander("Caption chunks (debug)"):
        st.write([
            {
                "start": round(c["start"], 2),
                "end": round(c["end"], 2),
                "text": c["text"],
                "words": len(c.get("words", [])),
            }
            for c in st.session_state.chunks[:50]
        ])
