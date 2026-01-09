from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import cv2
import os

def hex_to_rgba(hex_color: str, alpha: int = 255):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)

def render_text_rgba(
    text: str,
    width: int,
    font_size: int = 48,
    color_hex: str = "#FFFFFF",
    box: bool = True,
    box_color_hex: str = "#000000",
    box_alpha: int = 140,
    margin: int = 30,
    line_spacing: int = 10,
    max_lines: int = 6,
):
    """
    Render teks ke image RGBA transparan (numpy array).
    Lebih stabil daripada TextClip/ImageMagick.
    """
    # Coba font umum; ganti sesuai OS kalau perlu
    font_candidates = [
        "DejaVuSans.ttf",          # Linux
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
        "arial.ttf",               # Windows
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

    # Wrap teks agar muat
    # Estimasi kasar: 1 karakter ~ 0.55 * font_size px
    est_chars_per_line = max(10, int((width - 2 * margin) / (0.55 * font_size)))
    lines = textwrap.wrap(text, width=est_chars_per_line)[:max_lines]
    if not lines:
        lines = [""]

    # Hitung tinggi total
    dummy = Image.new("RGBA", (width, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(dummy)
    line_heights = []
    line_widths = []
    for ln in lines:
        bbox = d.textbbox((0, 0), ln, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_widths.append(w)
        line_heights.append(h)

    text_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    img_h = text_h + 2 * margin
    img = Image.new("RGBA", (width, img_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # background box
    if box:
        box_rgba = hex_to_rgba(box_color_hex, box_alpha)
        draw.rounded_rectangle(
            [margin//2, margin//2, width - margin//2, img_h - margin//2],
            radius=20,
            fill=box_rgba
        )

    # Draw centered per line
    y = margin
    text_rgba = hex_to_rgba(color_hex, 255)
    for i, ln in enumerate(lines):
        w = line_widths[i]
        h = line_heights[i]
        x = (width - w) // 2
        draw.text((x, y), ln, font=font, fill=text_rgba)
        y += h + line_spacing

    return np.array(img)

def apply_logo_opacity(logo_pil: Image.Image, opacity: float):
    """
    opacity: 0.0 - 1.0
    """
    opacity = max(0.0, min(1.0, float(opacity)))
    logo = logo_pil.convert("RGBA")
    r, g, b, a = logo.split()
    a = a.point(lambda px: int(px * opacity))
    return Image.merge("RGBA", (r, g, b, a))

import subprocess
import os

def split_video_ffmpeg(input_path, start_ts, end_ts, clip_duration, output_folder="output"):
    """
    Splits video using FFmpeg. Fast, preserves audio, and follows the 45s discard rule.
    """
    
    def get_seconds(ts):
        if isinstance(ts, (int, float)): return float(ts)
        parts = list(map(int, ts.split(':')))
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return float(ts)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_sec = get_seconds(start_ts)
    end_sec = get_seconds(end_ts)
    current_ptr = start_sec
    clip_count = 1

    print(f"Using FFmpeg to process: {input_path}")

    while True:
        time_left = end_sec - current_ptr
        
        # 45-second discard rule
        if time_left < 45:
            print(f"Stopping: Only {time_left:.2f}s left (below 45s threshold).")
            break

        output_name = os.path.join(output_folder, f"segment_{clip_count}.mp4")

        # FFmpeg Command Explanation:
        # -ss: Start time
        # -t: Duration
        # -i: Input file
        # -c copy: "Stream Copy" - no re-encoding (Instant speed + Audio preserved)
        # -avoid_negative_ts make_zero: Helps sync audio/video timing on cuts
        cmd = [
            'ffmpeg',
            '-ss', str(current_ptr),
            '-t', str(clip_duration),
            '-i', input_path,
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            output_name,
            '-y', # Overwrite existing files
            '-loglevel', 'error' # Keeps the console clean
        ]

        try:
            print(f"Creating Clip {clip_count}: {current_ptr}s -> {current_ptr + clip_duration}s")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing clip {clip_count}: {e}")
            break

        current_ptr += clip_duration
        clip_count += 1

    print("\nProcess Complete. Audio should now be present in all clips.")
    return True