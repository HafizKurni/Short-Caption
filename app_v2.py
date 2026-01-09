# ============================
# KARAOKE CAPTION APP (FINAL)
# ============================

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

from moviepy import VideoFileClip, CompositeVideoClip, ImageClip, ColorClip
from moviepy.video.fx import Resize, FadeIn, FadeOut


# -----------------------------
# CONFIG
# -----------------------------
LATIN_FONT = "fonts/Montserrat-Bold.ttf"
CJK_FONT = "fonts/Noto Sans CJK Regular.otf"

LANG_OPTIONS = {
    "English": "en",
    "Bahasa Indonesia": "id",
    "Bahasa Melayu": "ms",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Japanese": "ja",
    "Korean": "ko",
}


# -----------------------------
# FONT SELECTION
# -----------------------------
def is_cjk(lang: str) -> bool:
    return lang in ("zh-CN", "zh-TW", "ja", "ko")


def load_font(size: int) -> ImageFont.ImageFont:
    lang = st.session_state.get("target_lang", "en")
    font_path = CJK_FONT if is_cjk(lang) else LATIN_FONT
    return ImageFont.truetype(font_path, size)


# -----------------------------
# UTILITIES
# -----------------------------
def save_uploaded(file, suffix=".mp4"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.read())
    tmp.close()
    return tmp.name


def hex_rgba(hex_color, alpha=255):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0,2,4)) + (alpha,)


# -----------------------------
# WHISPER
# -----------------------------
@st.cache_resource
def load_whisper(model):
    return WhisperModel(model, device="cpu", compute_type="int8")


def transcribe_words(path, model):
    segs, _ = model.transcribe(path, word_timestamps=True)
    words = []
    for s in segs:
        if s.words:
            for w in s.words:
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "word": w.word.strip()
                })
    return words


def build_chunks(words, n=4):
    out, buf = [], []
    for w in words:
        buf.append(w)
        if len(buf) == n:
            out.append(buf)
            buf = []
    if buf:
        out.append(buf)
    return out


# -----------------------------
# TEXT RENDERING
# -----------------------------
def draw_text(draw, x, y, txt, font, fill, stroke, sw):
    for dx in range(-sw, sw+1):
        for dy in range(-sw, sw+1):
            draw.text((x+dx,y+dy), txt, font=font, fill=stroke)
    draw.text((x,y), txt, font=font, fill=fill)


def render_line(words, W, font, color, stroke, stroke_color, pad_y):
    ascent, descent = font.getmetrics()
    h = ascent + descent + 2*pad_y
    img = Image.new("RGBA", (W, h), (0,0,0,0))
    d = ImageDraw.Draw(img)

    x = 0
    for w in words:
        draw_text(d, x, pad_y, w["word"], font,
                  hex_rgba(color), hex_rgba(stroke_color), stroke)
        w["x"] = x
        w["w"] = d.textlength(w["word"], font=font)
        x += w["w"] + d.textlength(" ", font=font)
    return np.array(img), h, pad_y


# -----------------------------
# KARAOKE MASK
# -----------------------------
def karaoke_x(words, t):
    for w in words:
        if w["start"] <= t < w["end"]:
            frac = (t - w["start"]) / (w["end"] - w["start"])
            return w["x"] + frac * w["w"]
    if t >= words[-1]["end"]:
        return words[-1]["x"] + words[-1]["w"]
    return 0


def mask_highlight(img, x):
    out = img.copy()
    out[:, int(x):, 3] = 0
    return out


# -----------------------------
# VERTICAL VIDEO
# -----------------------------
def make_vertical(base, W, H, mode):
    ar = base.w / base.h
    tar = W / H

    if mode == "Crop":
        scaled = base.resized(height=H) if ar > tar else base.resized(width=W)
        return scaled.cropped(x_center=scaled.w/2, y_center=scaled.h/2, width=W, height=H)

    bg = ColorClip((W,H), color=(0,0,0)).with_duration(base.duration)

    fg = base.resized(height=H) if ar < tar else base.resized(width=W)
    return CompositeVideoClip([bg, fg.with_position("center")]).with_audio(base.audio)


# -----------------------------
# UI
# -----------------------------
st.set_page_config("Karaoke Caption App", layout="wide")
st.title("🎤 Karaoke Caption Generator")

video = st.file_uploader("Upload video", ["mp4"])
model = st.selectbox("Whisper model", ["tiny","base","small"], index=1)
lang = st.selectbox("Target language", list(LANG_OPTIONS.keys()))
st.session_state["target_lang"] = LANG_OPTIONS[lang]

font_size = st.slider("Font size", 20, 80, 44)
base_color = st.color_picker("Base color", "#FFFFFF")
hl_color = st.color_picker("Highlight color", "#FFD400")
stroke = st.slider("Stroke width", 0, 6, 3)
stroke_color = st.color_picker("Stroke color", "#000000")
pad_y = st.slider("Padding Y", 6, 30, 12)

out_mode = st.selectbox("Vertical mode", ["Crop","Fit"])
render = st.button("🚀 Render")


# -----------------------------
# RENDER
# -----------------------------
if render and video:
    path = save_uploaded(video)
    whisper = load_whisper(model)
    words = transcribe_words(path, whisper)
    chunks = build_chunks(words)

    with VideoFileClip(path) as raw:
        base = make_vertical(raw, 1080, 1920, out_mode)
        font = load_font(font_size)
        clips = [base]

        for chunk in chunks:
            start, end = chunk[0]["start"], chunk[-1]["end"]
            dur = end - start

            base_img, h, ytxt = render_line(chunk, base.w, font,
                                             base_color, stroke, stroke_color, pad_y)
            hl_img, _, _ = render_line(chunk, base.w, font,
                                       hl_color, stroke, stroke_color, pad_y)

            y = int(base.h * 0.82)

            base_clip = ImageClip(base_img).with_start(start).with_duration(dur).with_position((0,y))
            clips.append(base_clip)

            def frame(t, img=hl_img, words=chunk):
                return mask_highlight(img, karaoke_x(words, start+t))

            hl_clip = ImageClip(frame, ismask=False)\
                .with_start(start)\
                .with_duration(dur)\
                .with_position((0,y))
            clips.append(hl_clip)

        final = CompositeVideoClip(clips)
        out = os.path.join(tempfile.gettempdir(), "karaoke.mp4")
        final.write_videofile(out, codec="libx264", audio_codec="aac")

    st.video(out)
    st.download_button("Download", open(out,"rb"), "karaoke.mp4")
