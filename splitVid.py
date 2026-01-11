from utils_render import split_video_ffmpeg

# --- YOUR PARAMETERS ---
MY_VIDEO = "videoSplit/Gravity FallsS01E01.mp4"
START_AT = "1:23"       
END_AT = "22:37"        
DURATION = 90.0        
FOLDER = "videoSplit/EPS1_clips"    
    
# --- EXECUTION ---
print("Starting the video splitter...")

success = split_video_ffmpeg(
    input_path=MY_VIDEO,
    start_ts=START_AT,
    end_ts=END_AT,
    clip_duration=DURATION,
    output_folder=FOLDER
)

if success:
    print("Process finished successfully!")