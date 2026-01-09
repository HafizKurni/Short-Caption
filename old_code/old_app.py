import os
import sys
import math
import tempfile
from typing import List, Dict, Optional

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

# MoviePy v2 imports
from moviepy import VideoFileClip, CompositeVideoClip, ImageClip

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
# PIL-based text renderer (stable on Streamlit Cloud)
# -----------------------------
def _hex_to_rgba(hex_color: str, alpha: int = 255):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


def render_text_rgba(
    text: str,
    width: int,
    font_size: int = 36,
    color_hex: str = "#FFFFFF",
    box: bool = True,
    box_color_hex: str = "#000000",
    box_alpha: int = 140,
    margin: int = 20,
    line_spacing: int = 8,
    max_lines: int = 2,
):
    """
    Render short caption text into a transparent RGBA image (numpy array).
    Designed for micro-captions (e.g., 4 words), so it won't cover video.
    """
    # Font fallback order
    font_candidates = [
        "DejaVuSans.ttf",  # Linux common
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "arial.ttf",  # Windows
    ]
    font = None
    for fp in font_candidates:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    # Keep text short; still guard for overflow by basic wrap if needed
    # Estimate chars per line from width and font size
    est_chars_per_line = max(8, int((width - 2 * margin) / (0.60 * font_size)))
    # Simple wrap
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

    dummy = Image.new("RGBA", (width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)

    line_sizes = []
    for ln in lines:
        bbox = d.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_sizes.append((w, h))

    text_h = sum(h for _, h in line_sizes) + line_spacing * (len(lines) - 1)
    img_h = text_h + 2 * margin

    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if box:
        box_rgba = _hex_to_rgba(box_color_hex, box_alpha)
        draw.rounded_rectangle(
            [margin // 2, margin // 2, width - margin // 2, img_h - margin // 2],
            radius=18,
            fill=box_rgba,
        )

    text_rgba = _hex_to_rgba(color_hex, 255)
    y = margin
    for i, ln in enumerate(lines):
        w, h = line_sizes[i]
        x = (width - w) // 2
        draw.text((x, y), ln, font=font, fill=text_rgba)
        y += h + line_spacing

    return np.array(img)


def apply_logo_opacity(logo_pil: Image.Image, opacity: float):
    opacity = max(0.0, min(1.0, float(opacity)))
    logo = logo_pil.convert("RGBA")
    r, g, b, a = logo.split()
    a = a.point(lambda px: int(px * opacity))
    return Image.merge("RGBA", (r, g, b, a))


# -----------------------------
# Whisper + chunking
# -----------------------------
@st.cache_resource
def load_whisper(model_size: str):
    # Streamlit Cloud is typically CPU only; int8 is faster & lighter
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


def transcribe_video_segments(
    video_path: str,
    whisper_model: WhisperModel,
    language_hint: Optional[str] = None,
) -> (List[Dict], str):
    segments, info = whisper_model.transcribe(
        video_path,
        language=language_hint,
        vad_filter=True,
    )
    seg_list = []
    for s in segments:
        seg_list.append(
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": (s.text or "").strip(),
            }
        )
    full_text = " ".join([x["text"] for x in seg_list]).strip()
    return seg_list, full_text


def chunk_words_with_timing(seg_list: List[Dict], words_per_chunk: int = 4) -> List[Dict]:
    """
    Split each Whisper segment into N-word chunks and distribute the segment duration evenly.
    Returns list of {start, end, text} chunks.
    """
    out = []
    for seg in seg_list:
        start, end, text = seg["start"], seg["end"], seg["text"]
        if not text:
            continue
        words = text.split()
        if not words:
            continue

        chunks = [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
        dur = max(0.10, end - start)  # avoid zero / too-short
        chunk_dur = dur / len(chunks)

        for i, c in enumerate(chunks):
            c_start = start + i * chunk_dur
            c_end = min(end, c_start + chunk_dur)
            out.append({"start": c_start, "end": c_end, "text": c})
    return out


def translate_chunks(chunks: List[Dict], target_lang: str) -> List[Dict]:
    """
    Translate each chunk text (keeps timestamps).
    """
    translated = []
    translator = GoogleTranslator(source="auto", target=target_lang)
    for c in chunks:
        try:
            t = translator.translate(c["text"])
            t = (t or "").strip()
        except Exception:
            t = c["text"]
        translated.append({**c, "text": t})
    return translated


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Short Captions (4 words) + Translate + Watermark", layout="wide")
st.title("🎬 Short Captions (4 words) from Whisper timestamps + Translate + Watermark (MoviePy v2)")

with st.expander("Debug (optional)"):
    st.write("Python:", sys.version)

colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("1) Upload")
    video_file = st.file_uploader("Upload video (mp4/mov/mkv)", type=["mp4", "mov", "mkv"])

    st.subheader("2) Whisper Settings")
    whisper_size = st.selectbox("Whisper model size (Cloud-friendly)", ["tiny", "base", "small"], index=1)
    language_hint = st.text_input("Optional language hint (e.g., id, en). Leave blank for auto-detect.", value="")

    st.subheader("3) Translation")
    enable_translate = st.checkbox("Translate captions", value=True)

    lang_keys = list(LANG_OPTIONS.keys())
    default_idx = lang_keys.index("English") if "English" in lang_keys else 0
    lang_name = st.selectbox("Target language", options=lang_keys, index=default_idx)
    target_lang = LANG_OPTIONS[lang_name]
    st.caption(f"Selected: {lang_name} ({target_lang})")

    st.subheader("4) Caption Style")
    words_per_chunk = st.slider("Words per caption chunk", 2, 8, 4, 1)  # user asked 4; allow adjust
    font_size = st.slider("Font size", 16, 72, 34, 1)
    caption_color = st.color_picker("Caption color", "#FFFFFF")

    # Y position: percentage and pixel
    y_mode = st.radio("Y positioning mode", ["Percent", "Pixels"], index=0, horizontal=True)
    if y_mode == "Percent":
        y_percent = st.slider("Caption Y position (% from top)", 60, 95, 82, 1)
        y_pos_px = None
    else:
        y_pos_px = st.slider("Caption Y position (px from top)", 0, 2000, 900, 10)
        y_percent = None

    box = st.checkbox("Use background box behind text", value=True)
    box_color = st.color_picker("Box color", "#000000")
    box_alpha = st.slider("Box transparency (0=transparent, 255=solid)", 0, 255, 120, 5)

    st.subheader("5) Optional Watermark / Logo")
    logo_file = st.file_uploader("Upload logo (png recommended)", type=["png", "jpg", "jpeg"])
    logo_opacity = st.slider("Logo opacity", 0.0, 1.0, 0.35, 0.05)
    logo_scale = st.slider("Logo scale (relative to video width)", 0.05, 0.6, 0.18, 0.01)
    logo_pos = st.selectbox("Logo position", ["top-left", "top-right", "bottom-left", "bottom-right"], index=3)

    render_btn = st.button("🚀 Generate captioned video", type="primary", use_container_width=True)

with colR:
    st.subheader("Preview")
    if video_file is not None:
        st.video(video_file)


# -----------------------------
# Main render
# -----------------------------
if render_btn:
    if video_file is None:
        st.error("Please upload a video first.")
        st.stop()

    with st.status("Processing…", expanded=True) as status:
        status.write("Saving uploaded video…")
        video_path = save_uploaded_file(video_file, suffix=".mp4")

        status.write("Loading Whisper model…")
        whisper = load_whisper(whisper_size)

        status.write("Transcribing with timestamps…")
        hint = language_hint.strip() or None
        seg_list, full_transcript = transcribe_video_segments(video_path, whisper, language_hint=hint)

        status.write("Creating short caption chunks (time-synced)…")
        chunks = chunk_words_with_timing(seg_list, words_per_chunk=int(words_per_chunk))

        if enable_translate:
            status.write("Translating caption chunks… (requires internet)")
            chunks = translate_chunks(chunks, target_lang)

        # Show some text preview
        st.divider()
        st.subheader("Transcript (full, original)")
        st.write(full_transcript if full_transcript else "_(empty)_")
        st.subheader("Caption chunks preview (first 20)")
        st.write([{"start": round(c["start"], 2), "end": round(c["end"], 2), "text": c["text"]} for c in chunks[:20]])

        status.write("Rendering video with burn-in captions…")
        with VideoFileClip(video_path) as base:
            W, H = base.w, base.h
            fps = base.fps if base.fps else 30

            # Compute safe Y
            if y_mode == "Percent":
                safe_y = int(H * (y_percent / 100.0))
            else:
                safe_y = int(y_pos_px if y_pos_px is not None else int(H * 0.82))

            # Hard safety: ensure not outside frame
            safe_y = max(0, min(H - 1, safe_y))

            # Create per-chunk ImageClips
            caption_clips = []
            # Cap width usage: full W works, but box becomes wide. Use full W for consistent centering.
            # If you want narrower box, you could render at narrower width and position center.
            for c in chunks:
                dur = max(0.10, float(c["end"]) - float(c["start"]))
                # Render tiny caption image
                caption_img = render_text_rgba(
                    text=c["text"],
                    width=W,
                    font_size=int(font_size),
                    color_hex=caption_color,
                    box=box,
                    box_color_hex=box_color,
                    box_alpha=int(box_alpha),
                    margin=18,
                    line_spacing=6,
                    max_lines=2,
                )
                clip = (
                    ImageClip(caption_img)
                    .with_start(float(c["start"]))
                    .with_duration(dur)
                    .with_position(("center", safe_y))
                )
                caption_clips.append(clip)

            overlays = caption_clips

            # Optional watermark
            if logo_file is not None:
                status.write("Applying watermark/logo…")
                logo_path = save_uploaded_file(logo_file, suffix=".png")
                logo_pil = Image.open(logo_path)
                logo_pil = apply_logo_opacity(logo_pil, logo_opacity)

                target_logo_w = int(W * float(logo_scale))
                target_logo_w = max(20, target_logo_w)
                ratio = target_logo_w / max(1, logo_pil.size[0])
                target_logo_h = int(logo_pil.size[1] * ratio)
                target_logo_h = max(20, target_logo_h)

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

                logo_clip = logo_clip.with_position(pos_xy)
                overlays.append(logo_clip)

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
