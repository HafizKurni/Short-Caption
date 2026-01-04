import os
import tempfile
import streamlit as st

from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

from moviepy import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image

from utils_render import render_text_rgba, apply_logo_opacity

st.set_page_config(page_title="Video Caption + Translate + Watermark (MoviePy v2)", layout="wide")
st.title("🎬 Video → Transcribe → Translate → Caption + Watermark (Streamlit + MoviePy v2)")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource
def load_whisper(model_size: str = "small"):
    # device="auto" bisa pakai GPU kalau tersedia
    return WhisperModel(model_size, device="auto", compute_type="int8")

def save_uploaded_file(uploaded_file, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name

def transcribe_video(video_path: str, whisper_model: WhisperModel, language_hint: str | None = None):
    # Faster-whisper menerima audio langsung dari file video via ffmpeg
    segments, info = whisper_model.transcribe(
        video_path,
        language=language_hint,      # None = autodetect
        vad_filter=True
    )
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text, info

def translate_text(text: str, target_lang: str):
    if not text.strip():
        return ""
    return GoogleTranslator(source="auto", target=target_lang).translate(text)

# -----------------------------
# UI
# -----------------------------
colL, colR = st.columns([1, 1], gap="large")

with colL:
    st.subheader("1) Upload Video")
    video_file = st.file_uploader("Upload video (mp4/mov)", type=["mp4", "mov", "mkv"])
    whisper_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=2)
    language_hint = st.text_input("Optional: language hint (contoh: 'id', 'en'). Kosongkan untuk auto-detect.", value="")

    st.subheader("2) Translate Target")
    # deep-translator GoogleTranslator memakai kode bahasa (contoh: 'id','en','ms','ar','zh-CN')
    target_lang = st.text_input("Target language code", value="en")

    st.subheader("3) Caption Style")
    font_size = st.slider("Font size", 18, 120, 52, 2)
    caption_color = st.color_picker("Caption color", "#FFFFFF")
    y_pos = st.slider("Caption Y position (px from top)", 0, 2000, 900, 10)

    box = st.checkbox("Use background box behind text", value=True)
    box_color = st.color_picker("Box color", "#000000")
    box_alpha = st.slider("Box transparency (0=transparent, 255=solid)", 0, 255, 140, 5)

    st.subheader("4) Optional Watermark / Logo")
    logo_file = st.file_uploader("Upload logo (png recommended)", type=["png", "jpg", "jpeg"])
    logo_opacity = st.slider("Logo opacity", 0.0, 1.0, 0.35, 0.05)
    logo_scale = st.slider("Logo scale (relative)", 0.05, 0.8, 0.18, 0.01)
    logo_pos = st.selectbox("Logo position", ["top-left", "top-right", "bottom-left", "bottom-right"], index=3)

    render_btn = st.button("🚀 Generate captioned video", type="primary", use_container_width=True)

with colR:
    st.subheader("Preview / Output")
    if video_file:
        st.video(video_file)

# -----------------------------
# Main action
# -----------------------------
if render_btn:
    if not video_file:
        st.error("Please upload a video first.")
        st.stop()

    with st.status("Processing…", expanded=True) as status:
        status.write("Saving uploaded files…")
        video_path = save_uploaded_file(video_file, suffix=".mp4")

        status.write("Loading Whisper model…")
        whisper = load_whisper(whisper_size)

        status.write("Transcribing audio → text…")
        hint = language_hint.strip() or None
        transcript, info = transcribe_video(video_path, whisper, language_hint=hint)

        status.write("Translating text…")
        try:
            translated = translate_text(transcript, target_lang.strip())
        except Exception as e:
            translated = ""
            st.warning(f"Translate failed (check internet / language code). Error: {e}")

        st.divider()
        st.subheader("Transcript (original)")
        st.write(transcript if transcript else "_(empty)_")
        st.subheader("Translated (target)")
        st.write(translated if translated else "_(empty)_")

        status.write("Rendering caption overlay…")
        with VideoFileClip(video_path) as base:
            W, H = base.w, base.h

            caption_img = render_text_rgba(
                text=translated if translated else transcript,
                width=W,
                font_size=font_size,
                color_hex=caption_color,
                box=box,
                box_color_hex=box_color,
                box_alpha=box_alpha,
                margin=30,
                line_spacing=10,
                max_lines=8
            )

            caption_clip = ImageClip(caption_img).with_position(("center", y_pos)).with_duration(base.duration)

            overlays = [caption_clip]

            # Optional logo
            if logo_file is not None:
                status.write("Applying watermark/logo…")
                logo_path = save_uploaded_file(logo_file, suffix=".png")
                logo_pil = Image.open(logo_path)
                logo_pil = apply_logo_opacity(logo_pil, logo_opacity)

                # Resize logo relative to video width
                target_logo_w = int(W * float(logo_scale))
                ratio = target_logo_w / logo_pil.size[0]
                target_logo_h = int(logo_pil.size[1] * ratio)
                logo_pil = logo_pil.resize((target_logo_w, target_logo_h))

                logo_np = __import__("numpy").array(logo_pil)
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

            out_path = os.path.join(tempfile.gettempdir(), "captioned_output_v2.mp4")
            # write_videofile akan pakai ffmpeg
            final.write_videofile(
                out_path,
                fps=base.fps if base.fps else 30,
                codec="libx264",
                audio_codec="aac",
                threads=4,
                preset="medium"
            )

        status.update(label="Done ✅", state="complete")

    st.success("Video generated!")
    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button("⬇️ Download output MP4", data=f, file_name="captioned_output.mp4", mime="video/mp4")
