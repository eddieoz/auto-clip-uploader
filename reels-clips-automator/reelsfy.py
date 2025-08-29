import random
import sys
import textwrap
import time
import numpy as np
from pytube import YouTube
import cv2
import subprocess
import openai
import json
from datetime import datetime
import os
from os import path
import shutil
from dotenv import load_dotenv
from dotenv import dotenv_values
import whisper
import tempfile
import ffmpeg
from enhanced_audio_analyzer import EnhancedAudioAnalyzer
from topic_analyzer import TopicAnalyzer

# Define workspace directory
WORKSPACE_DIR = os.getcwd()  # Use current working directory instead of script location
REELS_BACKGROUNDS_DIR = os.path.join(WORKSPACE_DIR, "workspace", "backgrounds", "reels")

# CUDA environment settings for better performance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["GGML_CUDA_NO_PINNED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

scale = 4
target_face_size = 345

# Global variables for conditional loading
upsampler = None
face_enhancer = None

import argparse
import glob

# Get the absolute path to the reels-clips-automator directory (where the .env file is)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

# Load environment variables from reels-clips-automator/.env file
if not load_dotenv(env_path):
    print(f"Warning: Could not load .env file from {env_path}")

# Set OpenAI API key
config = dotenv_values(env_path)
api_key = config.get("OPENAI_API_KEY")

if not api_key:
    print(f"Warning: OPENAI_API_KEY not found in environment variables")
    print(f"Looking for .env file at: {env_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Environment variables: {dict(os.environ)}")
    raise ValueError(
        f"OpenAI API key not found in environment variables. Please check your .env file at {env_path}"
    )

# Remove any quotes from the API key if present
api_key = api_key.strip("'")
openai.api_key = api_key


def cleanup_tmp_directory(tmp_dir="tmp/", preserve_outputs=True):
    """
    Clean up temporary files after successful processing.
    
    Args:
        tmp_dir (str): Path to temporary directory
        preserve_outputs (bool): Whether to preserve output files (move them instead of delete)
    
    Returns:
        bool: True if cleanup successful, False otherwise
    """
    try:
        if not os.path.exists(tmp_dir):
            print(f"✅ Cleanup: tmp directory '{tmp_dir}' doesn't exist")
            return True
            
        print(f"🧹 Starting cleanup of temporary directory: {tmp_dir}")
        
        # Get all files in tmp directory
        tmp_files = glob.glob(os.path.join(tmp_dir, "*"))
        
        if not tmp_files:
            print(f"✅ Cleanup: tmp directory is already empty")
            return True
        
        files_cleaned = 0
        files_preserved = 0
        
        for file_path in tmp_files:
            filename = os.path.basename(file_path)
            
            try:
                # Check if this is an output file we might want to preserve
                is_output_file = (
                    filename.startswith("final-") or
                    filename.endswith(".mp4") and "cropped" in filename or
                    filename.endswith(".srt") and "final" in filename
                )
                
                if preserve_outputs and is_output_file:
                    # Check if outputs directory exists
                    outputs_dir = "outputs/"
                    if not os.path.exists(outputs_dir):
                        os.makedirs(outputs_dir, exist_ok=True)
                    
                    # Move output file to outputs directory
                    output_path = os.path.join(outputs_dir, filename)
                    
                    # Avoid overwriting existing files
                    counter = 1
                    base_name, ext = os.path.splitext(filename)
                    while os.path.exists(output_path):
                        output_path = os.path.join(outputs_dir, f"{base_name}_{counter}{ext}")
                        counter += 1
                    
                    shutil.move(file_path, output_path)
                    print(f"  📦 Preserved: {filename} → {output_path}")
                    files_preserved += 1
                else:
                    # Remove temporary file
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"  🗑️  Removed: {filename}")
                    files_cleaned += 1
                    
            except Exception as e:
                print(f"  ⚠️  Warning: Could not handle {filename}: {str(e)}")
        
        print(f"✅ Cleanup completed:")
        print(f"   • Files cleaned: {files_cleaned}")
        print(f"   • Files preserved: {files_preserved}")
        print(f"   • Temporary directory ready for next processing")
        
        return True
        
    except Exception as e:
        print(f"❌ Cleanup failed: {str(e)}")
        return False


def cleanup_on_success(video_id, preserve_outputs=True):
    """
    Perform cleanup operations after successful video processing.
    
    Args:
        video_id (str): The processed video identifier
        preserve_outputs (bool): Whether to preserve output files
    """
    print(f"\n🎉 Processing completed successfully for: {video_id}")
    print("=" * 50)
    
    # Clean up tmp directory
    cleanup_success = cleanup_tmp_directory(preserve_outputs=preserve_outputs)
    
    if cleanup_success:
        print("✅ All cleanup operations completed successfully")
        print("🚀 System ready for next video processing")
    else:
        print("⚠️  Some cleanup operations failed - manual cleanup may be needed")
    
    return cleanup_success


def cleanup_on_failure(video_id, preserve_logs=True):
    """
    Perform cleanup operations after failed video processing.
    
    Args:
        video_id (str): The video identifier that failed processing
        preserve_logs (bool): Whether to preserve log files for debugging
    """
    print(f"\n❌ Processing failed for: {video_id}")
    print("=" * 50)
    
    if preserve_logs:
        print("🔍 Preserving temporary files for debugging")
        print("   Manual cleanup of tmp/ directory may be needed later")
        return True
    else:
        print("🧹 Cleaning up after failed processing...")
        return cleanup_tmp_directory(preserve_outputs=False)


def initialize_upsampler():
    """
    Initialize the RealESRGAN upsampler only when needed.
    
    Returns:
        RealESRGANer: The initialized upsampler object, or None if initialization fails
    """
    global upsampler
    
    if upsampler is not None:
        return upsampler
    
    try:
        print("🔧 Initializing RealESRGAN upsampler...")
        
        # Check if weights file exists
        weights_path = "weights/realesr-general-x4v3.pth"
        if not os.path.exists(weights_path):
            print(f"❌ Weights file not found: {weights_path}")
            print("   Please download the weights file or run without --upscale")
            return None
        
        # Import dependencies only when needed
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.download_util import load_file_from_url
        from realesrgan import RealESRGANer
        from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        
        upsampler = RealESRGANer(
            scale=scale,
            model_path=weights_path,
            dni_weight=1,
            model=SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
            ),
            tile=512,  # Use tile approach for better quality on large images
            tile_pad=10,
            pre_pad=0,
            half=not True,
            gpu_id=0,
        )
        
        print("✅ RealESRGAN upsampler initialized successfully")
        return upsampler
        
    except Exception as e:
        print(f"❌ Failed to initialize RealESRGAN upsampler: {str(e)}")
        print("   Please check your dependencies or run without --upscale")
        return None


def initialize_face_enhancer():
    """
    Initialize the GFPGAN face enhancer only when needed.
    
    Returns:
        GFPGANer: The initialized face enhancer object, or None if initialization fails
    """
    global face_enhancer
    
    if face_enhancer is not None:
        return face_enhancer
    
    try:
        print("🔧 Initializing GFPGAN face enhancer...")
        
        # Initialize upsampler first (face_enhancer depends on it)
        bg_upsampler = initialize_upsampler()
        if bg_upsampler is None:
            print("❌ Cannot initialize face enhancer without upsampler")
            return None
        
        # Import GFPGAN only when needed
        from gfpgan import GFPGANer
        
        face_enhancer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            upscale=scale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
        )
        
        print("✅ GFPGAN face enhancer initialized successfully")
        return face_enhancer
        
    except Exception as e:
        print(f"❌ Failed to initialize GFPGAN face enhancer: {str(e)}")
        print("   Please check your dependencies or run without --enhance")
        return None


# Download video
def download_video(url, filename, tmp_dir):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension="mp4").get_highest_resolution()

    # Download the video
    video.download(filename=filename, output_path=tmp_dir)


# Segment Video function
def generate_segments(response):
    for i, segment in enumerate(response):
        print(i, segment)

        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", 0)
        title = segment.get("title", 0)

        # Handle SRT format timestamps (HH:MM:SS,mmm)
        def parse_srt_time(time_str):
            # Split into time and milliseconds
            time_parts, ms = time_str.split(",")
            # Split time into hours, minutes, seconds
            h, m, s = map(int, time_parts.split(":"))
            # Convert to total seconds
            total_seconds = h * 3600 + m * 60 + s + int(ms) / 1000
            return total_seconds

        # Convert timestamps to seconds
        start_time = parse_srt_time(start_time)
        end_time = parse_srt_time(end_time)

        # Adjust duration if too short
        if end_time - start_time < 30:
            end_time += 30 - (end_time - start_time)

        output_file = f"{str(title).replace(' ', '_').replace(':', '').replace('?', '').replace('!', '')}{str(i).zfill(3)}.mp4"
        command = f"ffmpeg -y -hwaccel cuda -i tmp/input_video.mp4 -vf scale='1920:1080' -c:v h264_nvenc -preset slow -profile:v high -rc:v vbr_hq -qp 18 -b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -c:a aac -b:a 192k -ss {start_time} -to {end_time} tmp/{str(output_file)}"
        subprocess.call(command, shell=True)


def find_smile(frame, output_file):
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_smile.xml"
    )

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the smile
    smile = smile_cascade.detectMultiScale(
        img_gray, scaleFactor=1.8, minNeighbors=25, flags=cv2.CASCADE_SCALE_IMAGE
    )
    # print(f"smile: {smile}")

    if len(smile) > 0:
        for x, y, w, h in smile:
            print("[INFO] Smile found. Saving locally.")
            # resize image
            # resizeimg = cv2.resize(frame, (400, 400), interpolation = cv2.INTER_CUBIC)
            # cv2.imwrite(f"./tmp/smiling-{output_file}.png", frame)
            return True, frame
    else:
        return False, ""


def generate_thumbnail(face, text, thumb_dir):
    from PIL import Image, ImageFont, ImageDraw

    # Create a blank image
    W, H = (1080, 1920)
    newthumb = Image.new("RGBA", size=(W, H), color=(0, 0, 0, 0))

    # Use the correct backgrounds directory
    bg_dir = REELS_BACKGROUNDS_DIR
    background = Image.open(
        os.path.join(bg_dir, random.choice(os.listdir(bg_dir)))
    ).convert("RGBA")
    background = background.resize((W, H))
    # Paste background
    newthumb.paste(background, (0, 0))

    # resizing image
    img_w, img_h = (500, 500)
    face = cv2.resize(face, (img_w, img_h))

    # Define the border
    top = bottom = left = right = 20
    color = [255, 255, 255]
    bordered_face = cv2.copyMakeBorder(
        face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    bordered_face = cv2.cvtColor(bordered_face, cv2.COLOR_BGR2RGB)

    bordered_image_pil = Image.fromarray(bordered_face)

    # add title
    title_font = ImageFont.truetype(font="fonts/Arial_Black.ttf", size=100)
    title_text = textwrap.fill(text, width=15)
    image_editable = ImageDraw.Draw(newthumb)

    # get sizes using textbbox instead of textsize
    bbox = image_editable.textbbox((0, 0), title_text, font=title_font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    print(f"1) text w: {text_w}, text h: {text_h}")

    # adjust text size
    if text_h > 400:
        title_font = ImageFont.truetype(font="fonts/Arial_Black.ttf", size=85)
        title_text = textwrap.fill(text, width=20)
        image_editable = ImageDraw.Draw(newthumb)

        # get sizes
        bbox = image_editable.textbbox((0, 0), title_text, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        print(f"2) text w: {text_w}, text h: {text_h}")

    if text_h > 400:
        title_font = ImageFont.truetype(font="fonts/Arial_Black.ttf", size=70)
        title_text = textwrap.fill(text, width=25)
        image_editable = ImageDraw.Draw(newthumb)

        # get sizes
        bbox = image_editable.textbbox((0, 0), title_text, font=title_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        print(f"3) text w: {text_w}, text h: {text_h}")

    face_text_h = img_h + 80 + text_h

    # New positions
    new_face_h = int((H - face_text_h) / 2)
    new_face_w = int((W - img_w - left - right) / 2)
    new_text_h = new_face_h + face_text_h - text_h
    new_text_w = int((W - text_w) / 2)

    # Paste face, write text
    newthumb.paste(bordered_image_pil, box=(new_face_w, new_face_h))
    image_editable.text(
        (new_text_w, new_text_h),
        title_text,
        (255, 255, 255),
        align="center",
        font=title_font,
        stroke_width=10,
        stroke_fill=(0, 0, 0),
    )

    # Create thumbs directory if it doesn't exist
    thumbs_dir = os.path.join(thumb_dir, "thumbs")
    os.makedirs(thumbs_dir, exist_ok=True)

    # Save thumbnail in the thumbs subdirectory
    thumbnail_path = os.path.join(thumbs_dir, f"thumb-{text}.png")
    newthumb.save(thumbnail_path, "PNG")
    print(f"Thumbnail saved to: {thumbnail_path}")


def calculate_target_face_size(frame_width, frame_height):
    """Calculate a dynamic threshold for face size based on video resolution."""
    # For 1920x1080 videos, use 345 as base
    # Scale proportionally for other resolutions
    base_resolution = 1920 * 1080
    current_resolution = frame_width * frame_height
    scale_factor = current_resolution / base_resolution

    # Add minimum and maximum bounds for the threshold
    min_threshold = 200  # Minimum face size threshold
    max_threshold = 500  # Maximum face size threshold
    base_threshold = 345

    # Calculate scaled threshold with bounds
    scaled_threshold = int(base_threshold * scale_factor)
    return max(min_threshold, min(scaled_threshold, max_threshold))

def generate_short(
    input_file,
    output_file,
    title="",
    upscale=False,
    enhance=False,
    thumb=False,
    thumb_dir="",
):
    try:
        # Sanitize input and output file names
        sanitized_input = input_file.replace(":", "").replace("?", "").replace("!", "")
        sanitized_output = (
            output_file.replace(":", "").replace("?", "").replace("!", "")
        )

        # Frame counter
        frame_count = 0

        # Index of the currently displayed face
        current_face_index = 0

        # Constants for cropping
        CROP_RATIO_BIG = 1  # Adjust the ratio to control how much of the image (around face) is visible in the cropped video
        CROP_RATIO_SMALL = 0.5
        CROP_RATIO_WIDE = 0.3  # For extra wide shots
        VERTICAL_RATIO = 9 / 16  # Aspect ratio for the vertical video
        
        # Constants for dynamic tracking behavior
        TRACKING_STATIC_PROBABILITY = 0.4  # 40% chance for static tracking
        ZOOM_CLOSEUP_PROBABILITY = 0.5  # 50% chance for close-up shots
        ZOOM_IN_PROBABILITY = 0.4  # 40% chance for smooth zoom-in effect

        # Load pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Open video file
        cap = cv2.VideoCapture(f"tmp/{sanitized_input}")

        # Get the frame dimensions and frame rate
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Input video: {frame_width}x{frame_height} at {input_fps} fps")

        # Interval to switch faces (calculated based on input fps for 5 seconds)
        switch_interval = int(input_fps * 5)  # 5 seconds at input fps
        print(
            f"Face switch interval: {switch_interval} frames ({switch_interval/input_fps:.1f} seconds)"
        )

        # Calculate dynamic target face size based on resolution
        target_face_size = calculate_target_face_size(frame_width, frame_height)
        print(f"Dynamic target face size threshold: {target_face_size}")

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            f"tmp/{sanitized_output}", fourcc, input_fps, (1080, 1920), True
        )  # Use input fps
        face_positions = []
        smile_found = False
        boxes = None
        trackers = None
        
        # Dynamic tracking variables
        tracking_mode = "static"  # Values: "static", "frame_by_frame"
        zoom_mode = "close_up"  # Values: "close_up", "wide_shot"
        frame_tracking_enabled = False
        zoom_in_enabled = False  # Whether to apply smooth zoom-in effect
        
        # Smoothing variables to reduce camera shake
        prev_face_center = None
        smoothing_factor = 0.7  # 0.7 = 70% previous position, 30% new position
        movement_threshold = 5  # Ignore movements smaller than 5 pixels

        while cap.isOpened():
            # Read frame from video
            ret, frame = cap.read()

            if ret == True:
                # If we don't have any face positions, detect the faces
                # Switch faces if it's time to do so
                if frame_count % switch_interval == 0:
                    # Convert color style from BGR to RGB
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Perform face detection
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100)
                    )

                    if len(faces) > 0:
                        # Initialize trackers and variable to hold face positions
                        trackers = cv2.legacy.MultiTracker_create()
                        face_positions.clear()

                        for x, y, w, h in faces:
                            face_positions.append((x, y, w, h))
                            tracker = cv2.legacy.TrackerKCF_create()
                            tracker.init(frame, (x, y, w, h))
                            trackers.add(tracker, frame, (x, y, w, h))

                        # Update trackers and get updated positions
                        try:
                            success, boxes = trackers.update(frame)
                        except Exception as e:
                            print(f"Error update trackers: {e}")

                    # Randomly select tracking and zoom modes for this 5-second interval
                    if random.random() < TRACKING_STATIC_PROBABILITY:
                        tracking_mode = "static"
                    else:
                        tracking_mode = "frame_by_frame"
                    
                    if random.random() < ZOOM_CLOSEUP_PROBABILITY:
                        zoom_mode = "close_up"
                    else:
                        zoom_mode = "wide_shot"
                    
                    # Randomly decide whether to apply smooth zoom-in effect
                    zoom_in_enabled = random.random() < ZOOM_IN_PROBABILITY
                    
                    frame_tracking_enabled = (tracking_mode == "frame_by_frame")
                    
                    print(f"Interval {frame_count // switch_interval}: tracking_mode={tracking_mode}, zoom_mode={zoom_mode}, zoom_in={zoom_in_enabled}")

                    # Switch faces if it's time to do so
                    current_face_index = (current_face_index + 1) % len(face_positions)
                    x, y, w, h = [int(v) for v in boxes[current_face_index]]

                    print(
                        f"Frame: {frame_count}, Current Face index {current_face_index} height {h} width {w} total faces {len(face_positions)}"
                    )

                # Update face positions for frame-by-frame tracking (not in 5-second intervals)
                elif frame_tracking_enabled and len(face_positions) > 0 and trackers is not None:
                    try:
                        # Update trackers to get current face positions
                        success, boxes = trackers.update(frame)
                        if success and len(boxes) > current_face_index:
                            # Update current face position from tracker
                            x, y, w, h = [int(v) for v in boxes[current_face_index]]
                            print(f"Frame-by-frame tracking - Frame: {frame_count}, Face: ({x}, {y}, {w}, {h})")
                        else:
                            # Fallback to static positioning if tracking fails
                            print(f"Frame-by-frame tracking failed, falling back to static mode")
                            frame_tracking_enabled = False
                    except Exception as e:
                        print(f"Error in frame-by-frame tracking: {e}, falling back to static mode")
                        frame_tracking_enabled = False

                # Calculate face center and crop parameters (for both modes)
                if len(face_positions) > 0:
                    # Calculate raw face center
                    raw_face_center = (x + w // 2, y + h // 2)
                    
                    # Apply smoothing to reduce camera shake
                    if prev_face_center is not None and frame_tracking_enabled:
                        # Calculate distance moved
                        dx = raw_face_center[0] - prev_face_center[0]
                        dy = raw_face_center[1] - prev_face_center[1]
                        movement_distance = (dx**2 + dy**2)**0.5
                        
                        # Only apply movement if it's above threshold (dead zone)
                        if movement_distance > movement_threshold:
                            # Apply exponential smoothing
                            face_center = (
                                int(prev_face_center[0] * smoothing_factor + raw_face_center[0] * (1 - smoothing_factor)),
                                int(prev_face_center[1] * smoothing_factor + raw_face_center[1] * (1 - smoothing_factor))
                            )
                        else:
                            # Movement too small, keep previous position
                            face_center = prev_face_center
                    else:
                        # First frame or static mode, use raw position
                        face_center = raw_face_center
                    
                    # Update previous position for next frame
                    prev_face_center = face_center

                    if w * 16 > h * 9:
                        w_916 = w
                        h_916 = int(w * 16 / 9)
                    else:
                        h_916 = h
                        w_916 = int(h * 9 / 16)

                    # Calculate the target width and height for cropping (vertical format)
                    # Dynamic zoom based on randomly selected zoom mode, but respect face size
                    base_crop_ratio = CROP_RATIO_BIG  # Default base ratio
                    
                    if zoom_mode == "close_up":
                        # For close-up: use smaller crop ratio for small faces, bigger for large faces
                        if max(h, w) < target_face_size:
                            base_crop_ratio = CROP_RATIO_SMALL  # 0.5 - more zoom for small faces
                        else:
                            base_crop_ratio = CROP_RATIO_BIG    # 1.0 - normal zoom for large faces
                    elif zoom_mode == "wide_shot":
                        # For wide shots: always use less zoom (wider view)
                        base_crop_ratio = CROP_RATIO_BIG        # 1.0 - always wide view
                    else:
                        # Fallback to original face size threshold logic
                        if max(h, w) < target_face_size:
                            base_crop_ratio = CROP_RATIO_SMALL
                        else:
                            base_crop_ratio = CROP_RATIO_BIG
                    
                    # Apply progressive zoom-in effect if enabled
                    final_crop_ratio = base_crop_ratio
                    if zoom_in_enabled:
                        # Calculate progress within current 5-second interval (0.0 to 1.0)
                        interval_frame = frame_count % switch_interval
                        zoom_progress = interval_frame / switch_interval if switch_interval > 0 else 0
                        
                        # Start from base ratio and gradually zoom in (reduce ratio) over 5 seconds
                        # Zoom in by reducing crop ratio by up to 30% over the interval
                        zoom_reduction = 0.3 * zoom_progress
                        final_crop_ratio = base_crop_ratio * (1 - zoom_reduction)
                        
                        # Ensure minimum crop ratio to prevent over-zoom
                        final_crop_ratio = max(final_crop_ratio, 0.3)
                    
                    target_height = int(frame_height * final_crop_ratio)
                    target_width = int(target_height * VERTICAL_RATIO)

                    # Debug output for cropping parameters
                    crop_ratio = target_height / frame_height if frame_height > 0 else 0
                    if zoom_in_enabled:
                        interval_frame = frame_count % switch_interval
                        zoom_progress = interval_frame / switch_interval if switch_interval > 0 else 0
                        print(f"Cropping: target_width={target_width}, target_height={target_height}, crop_ratio={crop_ratio:.3f}, zoom_mode={zoom_mode}, zoom_in_progress={zoom_progress:.2f}")
                    else:
                        print(f"Cropping: target_width={target_width}, target_height={target_height}, crop_ratio={crop_ratio:.3f}, zoom_mode={zoom_mode}")

                # Calculate the top-left corner of the 9:16 rectangle (only if we have face positions)
                if len(face_positions) > 0:
                    x_916 = face_center[0] - w_916 // 2
                    y_916 = face_center[1] - h_916 // 2

                    crop_x = max(
                        0, x_916 + (w_916 - target_width) // 2
                    )  # Adjust the crop region to center the face
                    crop_y = max(0, y_916 + (h_916 - target_height) // 2)
                    crop_x2 = min(crop_x + target_width, frame_width)
                    crop_y2 = min(crop_y + target_height, frame_height)

                    # Crop the frame to the face region
                    crop_img = frame[crop_y:crop_y2, crop_x:crop_x2]

                    # Upscale the cropped image if the face is too small
                    if upscale or enhance:
                        if max(h, w) < target_face_size:
                            if upscale:
                                # Higher quality upscaling
                                active_upsampler = initialize_upsampler()
                                if active_upsampler is not None:
                                    crop_img, _ = active_upsampler.enhance(
                                        crop_img, outscale=scale
                                    )
                                else:
                                    print("⚠️  Upscaling requested but upsampler not available, skipping...")
                            if enhance:
                                # Enhanced face
                                active_face_enhancer = initialize_face_enhancer()
                                if active_face_enhancer is not None:
                                    _, _, crop_img = active_face_enhancer.enhance(
                                        crop_img,
                                        has_aligned=False,
                                        only_center_face=False,
                                        paste_back=True,
                                    )
                                else:
                                    print("⚠️  Face enhancement requested but enhancer not available, skipping...")

                    resized = cv2.resize(
                        crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                    )

                    if thumb and not smile_found:
                        smile_found, smile_frame = find_smile(
                            frame[y : y + h, x : x + w], output_file
                        )

                    out.write(resized)
                else:
                    # If no faces detected, center crop the frame
                    crop_height = int(frame_height * CROP_RATIO_BIG)
                    crop_width = int(crop_height * VERTICAL_RATIO)
                    crop_x = (frame_width - crop_width) // 2
                    crop_y = (frame_height - crop_height) // 2
                    crop_img = frame[
                        crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
                    ]
                    resized = cv2.resize(
                        crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                    )
                    out.write(resized)

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Extract audio from original video
        command = f"ffmpeg -y -hwaccel cuda -i tmp/{sanitized_input} -vn -c:a aac -b:a 192k tmp/output-audio.aac"
        subprocess.call(command, shell=True)

        # Merge audio and processed video with better quality
        command = f"ffmpeg -y -hwaccel cuda -i tmp/{sanitized_output} -i tmp/output-audio.aac -c:v h264_nvenc -preset slow -profile:v high -rc:v vbr_hq -qp 18 -b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -c:a copy tmp/final-{sanitized_output}"
        subprocess.call(command, shell=True)

        if thumb and smile_found:
            generate_thumbnail(smile_frame, title, thumb_dir)

    except Exception as e:
        print(f"Error during video cropping: {str(e)}")


def generate_metadata(transcript):
    json_template = """
        {
            "title": "Video Title",
            "description": "Video Description",
            "tags": "Video hashtags"
        }
    """

    prompt = f"""Based on the following video transcript, create a title and a complete, engaging description for YouTube. Make sure to:

1. Engagement: Include calls to action and elements that encourage viewers to interact, such as questions or invitations to subscribe to the channel.
2. SEO: Optimize the text by incorporating keywords and relevant phrases related to the video's content.
3. Chapters: List the chapters with their respective timestamps to facilitate navigation.
4. Tags: Provide a list of relevant tags in one word, separated by commas, covering the main topics and keywords of the video.
5. Do not use (),[] or other special characters on title or description.

The provided transcript is as follows: {transcript}

Return a JSON document following this format: {json_template}
Please replace the placeholder values with the actual results from your analysis. Return ONLY the JSON object, without any markdown formatting or code block markers. Ensure all property names and string values are enclosed in double quotes."""

    system = """You are a YouTube Content Optimizer, an AI system that creates engaging titles, descriptions, and metadata for YouTube videos. 
    You analyze video transcripts and create optimized content that helps videos perform better on YouTube's platform."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-5-mini'), messages=messages, n=1, stop=None
        )

        content = response.choices[0].message.content
        # Clean up the response
        content = content.replace("```json", "").replace("```", "").strip()

        # Validate JSON structure
        try:
            json.loads(content)  # This will raise an error if the JSON is invalid
            return {"content": content}
        except json.JSONDecodeError as e:
            # If JSON is invalid, try to fix common issues
            content = content.replace(
                "'", '"' # Replace single quotes with double quotes
            )
            content = content.replace("\n", " ")  # Remove newlines
            content = content.replace("  ", " ")  # Remove extra spaces
            try:
                json.loads(content)  # Validate again
                return {"content": content}
            except json.JSONDecodeError:
                # If still invalid, return a default structure
                default_content = {
                    "title": "Untitled Video",
                    "description": "No description available",
                }
                return {"content": json.dumps(default_content)}

    except Exception as e:
        print(f"Error generating metadata: {str(e)}")
        default_content = {
            "title": "Untitled Video",
            "description": "No description available",
        }
        return {"content": json.dumps(default_content)}


def check_audio_file(audio_path):
    """
    Verify if an audio file exists and can be read by librosa.
    Returns a tuple of (is_valid, error_message)
    """
    if not audio_path:
        return False, "No audio path provided"

    if not os.path.exists(audio_path):
        return False, f"Audio file doesn't exist at path: {audio_path}"

    if os.path.getsize(audio_path) == 0:
        return False, f"Audio file is empty: {audio_path}"

    try:
        import librosa

        y, sr = librosa.load(
            audio_path, sr=None, duration=2
        )  # Try to load just the first 2 seconds
        if y.size == 0:
            return False, f"Audio file could not be read or is corrupted: {audio_path}"
        return True, "Audio file is valid"
    except Exception as e:
        return False, f"Error loading audio with librosa: {str(e)}"


def create_content_centric_prompt(segment_info, concluding_topic, supporting_segments, title_language, json_template):
    """
    Create content-centric prompt focused on concluding topic representation. 
    
    This replaces the traditional viral detection prompt with a focus on:
    - Best representation of the concluding topic
    - Complete narrative with supporting context
    - Content quality over viral tricks
    """
    supporting_context = ""
    if supporting_segments:
        supporting_context = "\nSupporting Context:\n"
        for i, seg in enumerate(supporting_segments[:3]):  # Show top 3
            supporting_context += f"• {seg['text'][:80]}...\n"
    
    prompt = f"""You are analyzing a video segment that has been optimized for content completeness and narrative flow. 

SEGMENT ANALYSIS:
• Duration: {segment_info['duration']:.1f} seconds (optimized for complete storytelling)
• Time Range: {segment_info['start_time']} → {segment_info['end_time']}
• Pre-generated Title: {segment_info['title']}

CONCLUDING TOPIC (Primary Focus):
"{concluding_topic['text']}"

Key Elements: {', '.join(concluding_topic['topic_keywords'][:5])}
Confidence Score: {concluding_topic['topic_confidence']:.2f}/1.0
{supporting_context}

CONTENT STRATEGY:
This segment represents the CONCLUDING topic from a 90-second conversation. Your task is to create content that:
1. Emphasizes the concluding message/topic as the primary value
2. Uses supporting context to provide necessary background
3. Creates a complete, standalone narrative
4. Focuses on content quality and information completeness over viral tricks
5. Ensures the audience gets the full context and conclusion

Create a title and Instagram description in {title_language} that presents this concluding topic with proper context and narrative completeness.

The Instagram description should:
1. Lead with the concluding topic/message
2. Provide necessary context from supporting segments
3. Use clear, informative language
4. Include natural engagement elements
5. Focus on educational/informational value
6. Use relevant emojis and hashtags appropriately
7. Do not use (),[] or other special characters on title or description.

Return the response as JSON following this format: {json_template}

Replace placeholder values with content that emphasizes the CONCLUDING TOPIC while providing complete context."""

    return prompt


def create_content_centric_system_prompt(title_language):
    """Create system prompt for content-centric approach."""
    return f"""You are a Content Optimization Specialist focused on creating complete, informative social media content.

Your role is to transform video segments that have been pre-analyzed for concluding topics into engaging {title_language} social media posts.

KEY PRINCIPLES:
- Prioritize content completeness over viral tricks
- Focus on the concluding topic as the main value proposition
- Ensure supporting context enhances understanding
- Create educational and informative content
- Maintain narrative flow and logical progression

CONTENT APPROACH:
- Lead with the concluding message/topic
- Provide necessary background context
- Create standalone, complete narratives
- Use clear, accessible language
- Include natural engagement elements
- Do not use (),[] or other special characters on title or description.

For {title_language} content, ensure cultural appropriateness and natural language flow.

Return ONLY the JSON object without markdown formatting."""

def generate_viral(
    transcript_content,
    audio_path=None,
    output_dir=None,
    min_duration=35.0,
    max_duration=59.0,
):
    # GEMINI WAS HERE

    """
    Generate optimized content segments focusing on concluding topics. 
    
    This function has been refactored as part of the improve-extraction EPIC
    to prioritize content completeness and concluding topic identification
    over traditional viral detection.
    """
    # Import TopicAnalyzer for advanced topic detection
    from topic_analyzer import TopicAnalyzer
    # Initialize TopicAnalyzer for content-centric approach
    print("\n==== Topic-Focused Content Analysis ====")
    topic_analyzer = TopicAnalyzer()
    
    # First generate metadata
    metadata = generate_metadata(transcript_content)
    metadata_json = json.loads(metadata["content"])

    # Use TopicAnalyzer to analyze transcript and identify concluding topics
    print("Analyzing transcript for concluding topics and supporting context...")
    topic_analysis = topic_analyzer.analyze_transcript(transcript_content)
    
    if 'error' in topic_analysis:
        print(f"Topic analysis failed: {topic_analysis['error']}")
        # Fallback to original parsing for compatibility
        segments = []
        current_segment = {}

        for line in transcript_content.split("\n"):
            line = line.strip()
            if not line:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = {}
                continue

            if "-->" in line:
                start_time, end_time = line.split(" --> ")
                start_seconds = sum(
                    float(x) * 60**i
                    for i, x in enumerate(reversed(start_time.replace(",", ".").split(":")))
                )
                end_seconds = sum(
                    float(x) * 60**i
                    for i, x in enumerate(reversed(end_time.replace(",", ".").split(":")))
                )
                current_segment["start_time"] = start_time
                current_segment["end_time"] = end_time
                current_segment["start"] = start_seconds
                current_segment["end"] = end_seconds
            elif line and not line.isdigit() and not current_segment.get("text"):
                current_segment["text"] = line

        if current_segment:
            segments.append(current_segment)
    else:
        # Use TopicAnalyzer results
        segments = topic_analysis['segments']
        concluding_topic = topic_analysis['concluding_topic']
        supporting_segments = topic_analysis['supporting_segments']
        optimized_segment = topic_analysis['optimized_segment']
        
        print(f"✅ Topic Analysis Results:")
        print(f"   • Total segments: {len(segments)}")
        print(f"   • Concluding topic: \"{concluding_topic['text'][:50]}...\"")
        print(f"   • Supporting segments: {len(supporting_segments)}")
        print(f"   • Optimized duration: {optimized_segment['duration']:.1f}s")
        print(f"   • Target range: 35-59 seconds")

    # Debug: Print segment information
    print(f"DEBUG: Parsed {len(segments)} segments from transcript")
    for i, seg in enumerate(segments[:3]):  # Show first 3 segments
        print(
            f"DEBUG: Segment {i}: keys={list(seg.keys())}, has_text={seg.get('text', 'MISSING')}"
        )

    # Then generate viral segments
    json_template = """
    {
        "segments": [
            {
                "start_time": "00:00:00,000",
                "end_time": "00:00:00,000",
                "title": "Title of the reels",
                "description": "Description of the reels",
                "tags": "Three relevant and objective one-word hashtags"
            }
        ]
    }
    """

    title_language = "Portuguese"
    
    # If we have topic analysis results, use the optimized approach
    if 'topic_analysis' in locals() and 'error' not in topic_analysis:
        print("\n==== Using Topic-Focused Content Generation ====")
        
        # Use the pre-analyzed optimized segment from TopicAnalyzer
        optimized_segment = topic_analysis['optimized_segment']
        concluding_topic = topic_analysis['concluding_topic']
        supporting_segments = topic_analysis['supporting_segments']
        
        # Create content description based on optimized segment
        segment_info = {
            "start_time": optimized_segment['start_time'],
            "end_time": optimized_segment['end_time'],
            "title": optimized_segment['title'],
            "duration": optimized_segment['duration'],
            "concluding_topic_text": concluding_topic['text'],
            "supporting_count": len(supporting_segments)
        }
        
        print(f"✅ Using optimized segment:")
        print(f"   • Duration: {segment_info['duration']:.1f}s")
        print(f"   • Title: {segment_info['title']}")
        print(f"   • Time: {segment_info['start_time']} → {segment_info['end_time']}")
        
        # Use content-centric prompt instead of viral detection
        prompt = create_content_centric_prompt(
            segment_info, concluding_topic, supporting_segments, title_language, json_template
        )
        
        system = create_content_centric_system_prompt(title_language)
        
        # Skip audio analysis since we already have optimized content
        print("Skipping traditional audio analysis - using topic-focused approach")
        
    else:
        print("\n==== Fallback to Traditional Analysis ====")
        print("Topic analysis unavailable, using original viral detection approach")

        # Add debug information
        print("Audio Analysis Debug Information:")
        print(f"Audio path: {audio_path}")
        print(f"Output directory: {output_dir}")
        print(
            f"Audio file exists: {os.path.exists(audio_path) if audio_path else 'No audio path provided'}"
        )
        print(f"Number of segments parsed from transcript: {len(segments)}")

        # Continue with traditional analysis
        # Check output directory exists
        if output_dir and not os.path.exists(output_dir):
            try:
                print(f"Creating output directory: {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating output directory: {str(e)}")

    # Check audio file validity
    valid_audio, audio_message = check_audio_file(audio_path)
    print(f"Audio validity check: {audio_message}")

    # Initialize audio_context as empty in case it's not set in the try block
    audio_context = ""

    # If audio analysis is available, use it to enhance segment selection
    if valid_audio:
        try:
            print("Analyzing audio...")
            # Debug import statement
            print("Importing AudioAnalyzer...")
            from enhanced_audio_analyzer import EnhancedAudioAnalyzer

            print("AudioAnalyzer imported successfully")

            # Debug audio file information
            file_size = os.path.getsize(audio_path) / (1024 * 1024)  # Size in MB
            print(f"Audio file size: {file_size:.2f} MB")

            # Initialize analyzer
            print("Creating EnhancedAudioAnalyzer instance...")
            analyzer = EnhancedAudioAnalyzer(audio_path)
            print("EnhancedAudioAnalyzer instance created")

            # Find narrative moments
            print("Finding narrative moments...")
            narrative_segments = analyzer.find_narrative_moments(
                segments, min_duration=min_duration, max_duration=max_duration
            )
            print(f"Found {len(narrative_segments)} narrative segments")

            if len(narrative_segments) > 0:
                # Get optimal boundaries
                optimal_boundaries = analyzer.get_optimal_boundaries(
                    segments, target_duration_range=(35.0, 59.0)
                )

                if optimal_boundaries:
                    print(f"Optimal boundaries found: {optimal_boundaries}")
                    # TODO: Use these boundaries to create the video segment

            if len(narrative_segments) == 0:
                print(
                    "Warning: No viral segments found in audio analysis, using a fallback approach"
                )
                # Fallback: just use the 3 longest segments or all segments if fewer than 3
                print("Using longest segments as fallback...")

                # Sort segments by duration
                sorted_segments = sorted(
                    segments,
                    key=lambda x: float(x.get("end", x.get("end_time", 0))) 
                    - float(x.get("start", x.get("start_time", 0))),
                    reverse=True,
                )

                # Take up to 1 longest segments
                fallback_segments = sorted_segments[: min(1, len(sorted_segments))]

                # Create mock audio_viral_segments with basic features
                audio_viral_segments = []
                for i, seg in enumerate(fallback_segments):
                    # Create mock features - use simple placeholder values
                    mock_features = {
                        "energy": 0.5,
                        "speech_rate": 0.5,
                        "emotional_content": 0.5,
                        "onset_strength": 0.5,
                    }

                    # Add to audio_viral_segments with a decreasing score
                    audio_viral_segments.append(
                        {
                            "segment": seg,
                            "score": 1.0
                            - (i * 0.1),  # First has score 1.0, second 0.9, third 0.8
                            "features": mock_features,
                        }
                    )

                print(f"Created {len(audio_viral_segments)} fallback segments")

            # Get top 1 segment from audio analysis (most viral)
            top_audio_segments = audio_viral_segments[:1]
            print(f"Top 1 segment selected: {len(top_audio_segments)}")

            # Format segments for GPT prompt
            audio_context = "\nAudio Analysis Results:\n"
            for i, seg in enumerate(top_audio_segments):
                audio_context += f"\nSegment {i+1}:\n"
                audio_context += f"Start: {seg['segment']['start_time']}, End: {seg['segment']['end_time']}\n"
                audio_context += f"Energy: {seg['features']['energy']:.2f}\n"
                audio_context += f"Speech Rate: {seg['features']['speech_rate']:.2f}\n"
                audio_context += (
                    f"Emotional Content: {seg['features']['emotional_content']:.2f}\n"
                )
                audio_context += (
                    f"Onset Strength: {seg['features']['onset_strength']:.2f}\n"
                )
                audio_context += f"Combined Score: {seg['score']:.2f}\n"

            # Save audio analysis results for each segment
            if output_dir:
                # Create an audio analysis summary file
                summary_file = os.path.join(output_dir, "audio_analysis_summary.txt")
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write("Audio Analysis Summary\n")
                    f.write("=====================\n\n")
                    f.write(f"Audio file: {audio_path}\n")
                    f.write(f"File size: {file_size:.2f} MB\n")
                    f.write(f"Total segments analyzed: {len(segments)}\n")
                    f.write(f"Viral segments found: {len(audio_viral_segments)}\n\n")
                    f.write(audio_context)
                print(f"Saved audio analysis summary to {summary_file}")

        except Exception as e:
            import traceback

            print(f"Warning: Audio analysis failed: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            audio_context = ""
    else:
        print("No audio analysis performed - audio path missing or file doesn't exist")
        audio_context = ""

    # Format segments for GPT prompt
    formatted_segments = "\n".join(
        [
            f"Segment {i+1}:\n"
            f"Time: {seg.get('start_time', 'Unknown')} --> {seg.get('end_time', 'Unknown')}\n"
            f"Text: {seg.get('text', '[No text available]')}\n"
            for i, seg in enumerate(segments)
            if seg  # Skip empty segments
        ]
    )

    prompt = f"""Given the following video transcript and audio analysis results, analyze each part for potential virality and identify the single most viral segment of maximum 59 seconds. 
The segment should be complete with the full explanation for the audience to comprehend the entire information or message of the segment, 
and have a minimum of {min_duration} seconds and maximum of 59 seconds in duration.

Do not select a segment with partial or incomplete information.

The provided transcript is as follows:

{formatted_segments}
{audio_context}

Based on your analysis, return a JSON document containing the timestamps (start and end) in SRT format (HH:MM:SS,mmm). 
You will create a title and a complete, engaging Instagram post description in {title_language} language. 

The Instagram description should:
1. Start with an engaging hook or question to capture attention
2. Present the main content using bullet points or lists for easy reading
3. Include relevant keywords naturally within the content
4. End with clear calls to action (like, save, comment, follow)
5. Be written as if it's the actual caption that will be posted on Instagram
6. Use emojis strategically to increase engagement
7. Include relevant hashtags at the end

The JSON document should follow this format: {json_template}

Please replace the placeholder values with the actual results from your analysis. Return ONLY the JSON object, without any markdown formatting or code block markers. Ensure all property names and string values are enclosed in double quotes."""

    system = f"""You are a Viral Segment Identifier and Instagram Content Creator. You analyze video transcripts and audio features to identify viral segments, then create ready-to-post Instagram captions.

Your analysis considers:
- Emotional impact and engagement potential
- Humor and unexpected content
- Relevance to current trends
- Audio characteristics (energy, speech rate, emotional content)

For the Instagram description, write the actual caption text that would be posted directly on Instagram in {title_language} language. Do NOT include meta-instructions, tips for content creation, or explanations about how to use the content. Write as if you are the content creator posting directly to your audience.

The description should be engaging, informative, and include natural calls to action. Use bullet points, emojis, and hashtags appropriately for Instagram format.

Return ONLY the JSON object, without any markdown formatting or code block markers. Ensure all property names and string values are enclosed in double quotes."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv('OPENAI_MODEL', 'gpt-5-mini'), messages=messages, n=1, stop=None
    )

    content = response.choices[0].message.content
    content = content.replace("```json", "").replace("```", "").strip()

    # Clean up the JSON response
    try:
        # First try to parse the content as is
        viral_json = json.loads(content)
    except json.JSONDecodeError:
        # If that fails, try to clean up the content
        content = content.replace("'", '"')  # Replace single quotes with double quotes
        content = content.replace("\n", " ")  # Remove newlines
        content = content.replace("  ", " ")  # Remove extra spaces
        try:
            viral_json = json.loads(content)
        except json.JSONDecodeError:
            # If still invalid, create a default structure
            viral_json = {
                "segments": [
                    {
                        "start_time": "00:00:00,000",
                        "end_time": "00:00:00,000",
                        "title": "Untitled Segment",
                        "description": "No description available",
                    }
                ]
            }

    # Merge metadata and viral segments
    merged_json = {"metadata": metadata_json, "segments": viral_json["segments"]}

    return {"content": json.dumps(merged_json, indent=2)}


def write_srt(segments, file):
    for i, segment in enumerate(segments, start=1):
        # Convert timestamps to SRT format (HH:MM:SS,mmm)
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])

        # Write the SRT entry
        file.write(f"{i}\n")
        file.write(f"{start} --> {end}\n")
        file.write(f"{segment['text'].strip()}\n\n")


def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def parse_srt(srt_content):
    """Parse SRT content into a list of segments."""
    segments = []
    current_segment = {}

    for line in srt_content.split("\n"):
        line = line.strip()
        if not line:
            if current_segment:
                segments.append(current_segment)
                current_segment = {}
            continue

        if "-->" in line:
            # Parse timestamp line
            start_time, end_time = line.split(" --> ")
            current_segment["start_time"] = start_time
            current_segment["end_time"] = end_time
        elif line and not line.isdigit() and not current_segment.get("text"):
            # This is the text line
            current_segment["text"] = line

    # Add the last segment if exists
    if current_segment:
        segments.append(current_segment)

    return segments


def generate_transcript(input_file):
    # Extract audio for transcription
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(
        temp_dir, f"{os.path.basename(input_file).split('.')[0]}.wav"
    )

    print(f"Extracting audio from {input_file}...")
    # Use subprocess instead of ffmpeg-python
    audio_cmd = f"ffmpeg -y -i tmp/{input_file} -vn -acodec pcm_s16le -ac 1 -ar 16000 {audio_path}"
    subprocess.call(audio_cmd, shell=True)

    # Load Whisper model and transcribe
    print("Loading Whisper model...")
    # model = whisper.load_model("medium")
    model = whisper.load_model(os.getenv('WHISPER_MODEL', 'small'))
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    print("Transcription done")

    # Return both the segments and the full transcript
    return {
        "segments": result["segments"],
        "full_transcript": " ".join(
            segment["text"].strip() for segment in result["segments"]
        ),
    }


def save_metadata(metadata, output_dir):
    """Save metadata in a copy-paste friendly format."""
    try:
        # Create a formatted text file for easy copy-paste
        formatted_text = f"""Title:
{metadata['title']}

Description:
{metadata['description']}

"""
        # Save to file
        metadata_file = os.path.join(output_dir, "description.txt")
        with open(metadata_file, "w", encoding="utf-8") as file:
            file.write(formatted_text)
        print(f"Metadata saved to {metadata_file}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")


def direct_process_file(
    input_file,
    output_dir, upscale=False, enhance=False, thumb=False
):
    """Process a file directly without viral segment detection"""
    print(f"\n==== Direct File Processing ====")
    print(f"Processing file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Generate a unique job_id based on timestamp
    job_id = f"direct_{int(time.time())}"

    # Get base filename and create output path
    filename = os.path.basename(input_file)
    video_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{video_name}_reel.mp4")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process the file
    print(f"Generating reel from {filename}...")
    result = generate_short(
        filename, output_file, video_name, upscale, enhance, thumb, output_dir
    )

    if result:
        print(f"Successfully created reel at: {output_file}")
    else:
        print(f"Error: Failed to create reel from {input_file}")

    return output_file

def __main__():
    # Check command line argument
    parser = argparse.ArgumentParser(
        description="Create 3 reels or tiktoks from Youtube video"
    )
    parser.add_argument(
        "-v",
        "--video_id",
        required=False,
        help="Youtube video id. Ex: Cuptv7-A4p0 in https://www.youtube.com/watch?v=Cuptv7-A4p0",
    )
    parser.add_argument("-f", "--file", required=False, help="Video file to be used")
    parser.add_argument(
        "-u",
        "--upscale",
        action="store_true",
        default=False,
        required=False,
        help="Upscale small faces",
    )
    parser.add_argument(
        "-e",
        "--enhance",
        action="store_true",
        default=False,
        required=False,
        help="Upscale and enhance small faces",
    )
    parser.add_argument(
        "-t",
        "--thumb",
        action="store_true",
        default=False,
        required=False,
        help="Create thumbnail",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum duration in seconds for viral segments (default: 1.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=120.0,
        help="Maximum duration in seconds for viral segments (default: 120.0)",
    )
    parser.add_argument("--job-id", required=False, help="Job ID for organization")
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Custom output directory for reels and thumbnails",
    )
    args = parser.parse_args()
    print(args)

    if not args.video_id and not args.file:
        print("Needed at least one argument. <command> --help for help")
        sys.exit(1)

    if args.upscale and args.enhance:
        print("You can use --upcale or --enhance. Not both")
        sys.exit(1)

    if args.upscale or args.enhance:
        if (
            not os.path.exists("weights")
            or not os.path.exists("gfpgan")
            or not os.path.exists("realesrgan")
        ):
            args.upscale = args.enhance = False
            print("Upscale and Enhance require utils/Real-ESRGAN installed and working")
            sys.exit(1)

    if args.thumb:
        if (
            not os.path.exists(REELS_BACKGROUNDS_DIR)
            or len(os.listdir(REELS_BACKGROUNDS_DIR)) == 0
        ):
            args.thumb = False
            print(
                f"Thumbnail requires background images in {REELS_BACKGROUNDS_DIR} directory"
            )
            sys.exit(1)

    if args.video_id and args.file:
        print("use --video_id or --file")
        sys.exit(1)

    # Create temp folder
    tmp_dir = "tmp/"
    try:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
    except OSError as error:
        print(error)

    # Define the output folder
    output_folder = "outputs"
    print(f"Base output folder: {output_folder}")

    filename = "input_video.mp4"
    if args.video_id:
        video_id = args.video_id
        url = (
            "https://www.youtube.com/watch?v=" + video_id
        )  # Replace with your video's URL
        # Download video
        download_video(url, filename)

    if args.file:
        # Validate the input file
        if not os.path.exists(args.file):
            print(f"Error: Input file '{args.file}' not found")
            sys.exit(1)

        if not os.path.isfile(args.file):
            print(f"Error: '{args.file}' is not a valid file")
            sys.exit(1)

        file_size = os.path.getsize(args.file)
        if file_size == 0:
            print(f"Error: Input file '{args.file}' is empty (0 bytes)")
            sys.exit(1)

        print(f"Input file validation: {args.file} exists and is {file_size} bytes")

        video_id = os.path.basename(args.file).split(".")[0]
        print(f"Using video_id: {video_id}")
        if path.exists(args.file) == True:
            command = f"cp {args.file} tmp/input_video.mp4"
            subprocess.call(command, shell=True)
        else:
            print(f"File {args.file} does not exist")
            sys.exit(1)

    # Determine the output directory
    if args.output_dir:
        output_dir = args.output_dir
        print(f"Using custom output directory: {output_dir}")
    elif args.job_id:
        output_dir = f"{output_folder}/jobs/{args.job_id}"
        print(f"Using job-specific output directory: {output_dir}")
    else:
        output_dir = f"{output_folder}/{video_id}"
        print(f"Using standard output directory: {output_dir}")

    # Create outputs folder
    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as error:
        print(error)

    # Create the output directory and subdirectories
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Create subdirectories
        os.makedirs(os.path.join(output_dir, "thumbs"), exist_ok=True)
        print(f"Created output directory structure: {output_dir}")
    except OSError as error:
        print(f"Error creating output directory: {error}")

    # Extract audio for analysis
    audio_path = f"tmp/audio_analysis.wav"
    print(f"\n==== Audio Extraction ====")
    print(f"Extracting audio for analysis to: {audio_path}")

    # Use a more reliable audio extraction command compatible with librosa
    # - Use mono (1 channel) audio
    # - Use 22050Hz sample rate which works well with librosa
    # - Use PCM 16-bit format
    # - Don't use hardware acceleration for audio extraction
    command = f"ffmpeg -y -i tmp/input_video.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 1 {audio_path}"
    print(f"Running audio extraction command: {command}")
    subprocess.call(command, shell=True)

    # Verify the audio file was created correctly
    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        print(f"Audio extraction successful, file size: {audio_size_mb:.2f} MB")

        # Test loading with librosa to verify compatibility
        try:
            import librosa

            y, sr = librosa.load(audio_path, sr=None, duration=1)
            print(
                f"Successfully loaded audio with librosa, duration: 1 sec, sample rate: {sr}Hz"
            )
            print(f"Audio data shape: {y.shape}")
        except Exception as e:
            print(f"Warning: Could not load audio with librosa: {str(e)}")
    else:
        print("WARNING: Audio extraction failed or produced an empty file")

        # Try a different extraction approach as fallback
        fallback_audio_path = f"tmp/audio_fallback.wav"
        print(f"Trying alternative extraction approach to: {fallback_audio_path}")
        command = f"ffmpeg -y -i tmp/input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 {fallback_audio_path}"
        subprocess.call(command, shell=True)

        if (
            os.path.exists(fallback_audio_path)
            and os.path.getsize(fallback_audio_path) > 0
        ):
            print(f"Fallback audio extraction successful, using: {fallback_audio_path}")
            audio_path = fallback_audio_path
        else:
            print("WARNING: Audio extraction failed even with fallback approach")
            # Create an empty file to prevent errors
            with open(audio_path, "w"):
                pass

    # Copy the audio file to the output directory for reference
    output_audio_path = f"{output_dir}/audio_analysis.wav"
    try:
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            shutil.copy2(audio_path, output_audio_path)
            print(f"Copied audio file to output directory: {output_audio_path}")
    except Exception as e:
        print(f"Warning: Could not copy audio file to output directory: {str(e)}")

    # Verifies if output_file exists, or create it. If exists, it doesn't call OpenAI APIs
    output_file = f"{output_dir}/content.txt"
    transcript_file = f"{output_dir}/transcript.txt"
    transcript_done = False

    print(f"\n==== Transcript Generation ====")
    # It creates the transcription and tests it. If it is possible to read the json at the end, if exits the loop. Else it do it again
    transcript_result = generate_transcript(filename)
    print(
        f"Transcript generation complete with {len(transcript_result['segments'])} segments"
    )

    # Save transcript in SRT format
    try:
        with open(transcript_file, "w", encoding="utf-8") as file:
            write_srt(transcript_result["segments"], file)
        print(f"Full transcription written to {transcript_file}")
    except IOError as e:
        print(f"Error: Failed to write the output file: {str(e)}")
        sys.exit(1)

    # Read the contents of the input file
    try:
        with open(transcript_file, "r", encoding="utf-8") as file:
            transcript = file.read()
            print(f"Transcript read from file, length: {len(transcript)} characters")
    except IOError:
        print("Error: Failed to read the input file.")
        sys.exit(1)

    while not transcript_done:
        print(f"\n==== Viral Segment Generation ====")

        if not path.exists(output_file):
            print(
                f"Output file {output_file} does not exist, generating viral segments..."
            )
            # Generate viral segments
            viral_segments = generate_viral(
                transcript,
                audio_path,
                f"{output_dir}",
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            content = viral_segments["content"]
            print(
                f"Viral segments generation complete, content size: {len(content)} characters"
            )

            try:
                # Validate JSON before writing
                print("Validating JSON output...")
                parsed_json = json.loads(content)
                if not isinstance(parsed_json, dict):
                    raise ValueError("Invalid JSON structure: not a dictionary")
                if "metadata" not in parsed_json or "segments" not in parsed_json:
                    raise ValueError(
                        "Invalid JSON structure: missing 'metadata' or 'segments' keys"
                    )
                if not isinstance(parsed_json["metadata"], dict):
                    raise ValueError(
                        "Invalid JSON structure: 'metadata' is not a dictionary"
                    )
                if not isinstance(parsed_json["segments"], list):
                    raise ValueError("Invalid JSON structure: 'segments' is not a list")

                print(f"JSON validation successful, writing content to {output_file}")
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(content)
                print(f"Content written to {output_file}")
                transcript_done = True
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON format: {str(e)}")
                print(
                    "Content received:",
                    content[:500] + "..." if len(content) > 500 else content,
                )
            except ValueError as e:
                print(f"Error: {str(e)}")
                print(
                    "Content received:",
                    content[:500] + "..." if len(content) > 500 else content,
                )
            except IOError as e:
                print(f"Error: Failed to write output file: {str(e)}")
                sys.exit(1)
        else:
            print(
                f"Output file {output_file} already exists, reading existing content..."
            )
            # Read the contents of the input file
            try:
                with open(output_file, "r", encoding="utf-8") as file:
                    content = file.read()
                    print(f"Content read from file, length: {len(content)} characters")
                    try:
                        # Validate JSON structure
                        print("Validating existing JSON content...")
                        parsed_json = json.loads(content)
                        if not isinstance(parsed_json, dict):
                            raise ValueError("Invalid JSON structure: not a dictionary")
                        if (
                            "metadata" not in parsed_json
                            or "segments" not in parsed_json
                        ):
                            raise ValueError(
                                "Invalid JSON structure: missing 'metadata' or 'segments' keys"
                            )
                        if not isinstance(parsed_json["metadata"], dict):
                            raise ValueError(
                                "Invalid JSON structure: 'metadata' is not a dictionary"
                            )
                        if not isinstance(parsed_json["segments"], list):
                            raise ValueError(
                                "Invalid JSON structure: 'segments' is not a list"
                            )
                        transcript_done = True
                        print("Existing content is valid JSON")
                    except json.JSONDecodeError as e:
                        print(f"Error: Invalid JSON format in existing file: {str(e)}")
                        os.remove(output_file)
                        print("Removed invalid file, will regenerate")
                    except ValueError as e:
                        print(f"Error: {str(e)}")
                        os.remove(output_file)
                        print("Removed invalid file, will regenerate")
            except IOError as e:
                print(f"Error: Failed to read the input file: {str(e)}")
                sys.exit(1)

    print(f"\n==== Processing Generated Content ====")
    try:
        parsed_content = json.loads(content)
        # Save metadata in a copy-paste friendly format
        save_metadata(parsed_content["metadata"], f"{output_dir}")
        print(f"Processing {len(parsed_content['segments'])} generated segments...")
        generate_segments(parsed_content["segments"])
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error: Failed to process content: {str(e)}")
        cleanup_on_failure(video_id if 'video_id' in locals() else 'unknown', preserve_logs=True)
        sys.exit(1)

    # Loop through each segment
    print(f"\n==== Generating Final Video Segments ====")
    for i, segment in enumerate(parsed_content["segments"]):
        print(
            f"Processing segment {i+1}/{len(parsed_content['segments'])}: {segment['title']}"
        )
        # Just pass the output_dir, the generate_thumbnail function will handle the thumbs subdirectory
        safe_title = (
            str(segment["title"])
            .replace(" ", "_")
            .replace(":", "")
            .replace("?", "")
            .replace("!", "")
        )
        input_file = f"{safe_title}{str(i).zfill(3)}.mp4"
        output_file = f"{safe_title}_cropped{str(i).zfill(3)}.mp4"

        try:
            print(f"Generating short video for segment {i+1}...")
            generate_short(
                input_file,
                output_file,
                str(segment["title"]),
                args.upscale,
                args.enhance,
                args.thumb,
                output_dir,
            )
            print(f"Adding subtitles to video for segment {i+1}...")
            generate_subtitle(
                f"final-{output_file.replace(':', '').replace('?', '').replace('!', '')}",
                video_id,
                output_dir,
            )
            print(f"Segment {i+1} processing complete")
        except Exception as e:
            print(f"Error processing segment {i}: {str(e)}")
            continue

    print(f"\n==== Processing Complete ====")
    print(f"Results saved to: {output_dir}/")

    # Clean up temporary files after successful processing
    try:
        cleanup_on_success(video_id, preserve_outputs=True)
    except Exception as e:
        print(f"⚠️  Warning: Cleanup failed: {str(e)}")
        print("   Manual cleanup of tmp/ directory may be needed")

    # Check if backgrounds directory exists
    if (
        not os.path.exists(REELS_BACKGROUNDS_DIR)
        or len(os.listdir(REELS_BACKGROUNDS_DIR)) == 0
    ):
        print(f"Warning: {REELS_BACKGROUNDS_DIR} directory is empty or does not exist")
        if args.thumb:
            print(
                "Thumbnail requires background images in workspace/backgrounds/reels/ directory"
            )
            args.thumb = False


# __main__()


def generate_subtitle(input_file, video_id, output_dir):
    """Generate subtitle for a video using Whisper and FFmpeg"""
    # Get channel name from root .env file
    root_env_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), ".env")
    print(f"Looking for .env file at: {root_env_path}")
    if os.path.exists(root_env_path):
        # Load environment variables from root .env
        root_env = dotenv_values(root_env_path)
        channel_name = root_env.get("CHANNEL_NAME", "").strip('"')
        print(f"Channel name from root .env: {channel_name}")
    else:
        # Fallback to current environment
        channel_name = os.getenv("CHANNEL_NAME", "").strip('"')
        print(f"Channel name from current env: {channel_name}")

    # Get current date
    from datetime import datetime

    current_time = datetime.now()
    date_text = current_time.strftime("%d/%m/%Y")

    # Use Whisper to transcribe the final cropped video segment
    # This ensures subtitles are perfectly synced to the segment timing

    def transcribe_with_whisper(video_path):
        """Transcribe video using Whisper and return SRT content"""
        try:
            # Try different model sizes based on available memory
            models_to_try = ["small", "medium", "base"]

            for model_name in models_to_try:
                try:
                    print(f"Loading Whisper {model_name} model...")
                    model = whisper.load_model(os.getenv('WHISPER_MODEL', model_name))

                    print(f"Transcribing video with {model_name} model...")
                    result = model.transcribe(video_path, word_timestamps=True)

                    # Generate SRT content
                    srt_content = ""
                    for i, segment in enumerate(result["segments"]):
                        start_time = segment["start"]
                        end_time = segment["end"]
                        text = segment["text"].strip()

                        # Convert seconds to SRT time format
                        start_h, start_remainder = divmod(start_time, 3600)
                        start_m, start_s = divmod(start_remainder, 60)
                        start_ms = int((start_s - int(start_s)) * 1000)

                        end_h, end_remainder = divmod(end_time, 3600)
                        end_m, end_s = divmod(end_remainder, 60)
                        end_ms = int((end_s - int(end_s)) * 1000)

                        srt_content += f"{i+1}\n"
                        srt_content += f"{int(start_h):02d}:{int(start_m):02d}:{int(start_s):02d},{start_ms:03d} --> {int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d},{end_ms:03d}\n"
                        srt_content += f"{text}\n\n"

                    print(f"Successfully transcribed with {model_name} model")
                    return srt_content

                except Exception as e:
                    print(f"Failed with {model_name} model: {str(e)}")
                    if "CUDA out of memory" in str(e):
                        print(
                            f"GPU memory issue with {model_name}, trying smaller model..."
                        )
                        continue
                    else:
                        break

            print("All Whisper models failed")
            return None

        except Exception as e:
            print(f"Error in Whisper transcription: {str(e)}")
            return None

    # Generate subtitles using Whisper directly
    print(f"Transcribing video: tmp/{input_file}")
    srt_content = transcribe_with_whisper(f"tmp/{input_file}")

    # Create temporary SRT file
    input_base = os.path.basename(input_file).split(".")[0]
    temp_srt_file = f"tmp/{input_base}.srt"

    if srt_content:
        with open(temp_srt_file, "w", encoding="utf-8") as f:
            f.write(srt_content)
        print(f"Subtitle file created: {temp_srt_file}")
    else:
        print("No subtitles generated, will create video without subtitles")
        temp_srt_file = None

    # Create output path for final video with overlays
    output_filename = os.path.basename(input_file)
    output_filename = (
        output_filename.replace(" ", "_")
        .replace(":", "")
        .replace("?", "")
        .replace("!", "")
        .replace("-", "_")
    )
    output_path = f"{output_dir}/{output_filename}"

    # Check if subtitle file was generated
    if not temp_srt_file or not os.path.exists(temp_srt_file):
        print(f"Warning: No subtitle file available, creating video without subtitles")
        # Create video with only overlays (no subtitles)
        filter_parts = []
        last_filter = "0:v"
    else:
        print(f"Using subtitle file: {temp_srt_file}")
        # Build filter complex starting with subtitles
        filter_parts = [
            f"subtitles='{temp_srt_file}':force_style='Alignment=2,MarginV=40,MarginL=55,MarginR=55,Fontname=Noto Sans,Fontsize=14,PrimaryColour=&H00ffffff,SecondaryColour=&H000000ff,OutlineColour=&H00000000,BackColour=&H80000000,Outline=1.5,Shadow=1.5,BorderStyle=1'[v1]"
        ]
        last_filter = "v1"

    # Add date overlay (top-left)
    if date_text:
        filter_parts.append(
            f"[{last_filter}]drawtext=text='{date_text}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=24:fontcolor=white:x=30:y=30:shadowcolor=black:shadowx=2:shadowy=2[v2]"
        )
        last_filter = "v2"

    # Add channel name overlay (bottom-right, above subtitles)
    if channel_name:
        filter_parts.append(
            f"[{last_filter}]drawtext=text='{channel_name}':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontsize=20:fontcolor=white:x=W-tw-30:y=30:shadowcolor=black:shadowx=2:shadowy=2[v3]"
        )
        last_filter = "v3"

    # Build final FFmpeg command with proper spacing and quotes
    if filter_parts:
        filter_complex = ";".join(filter_parts)
        ffmpeg_command = (
            f'ffmpeg -y -hwaccel cuda -i "tmp/{input_file}" '  # Added space and quotes
            f'-filter_complex "{filter_complex}" '  # Added space
            f'-map "[{last_filter}]" -map 0:a '
            f'-c:v h264_nvenc -preset slow -profile:v high -rc:v vbr_hq -qp 18 '
            f'-b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p '
            f'-c:a copy "{output_path}"'  # Added quotes around output path
        )
    else:
        # No overlays, just copy
        ffmpeg_command = (
            f'ffmpeg -y -hwaccel cuda -i "tmp/{input_file}" '  # Added space and quotes
            f'-c copy "{output_path}"'  # Added quotes around output path
        )

    print(f"Applying overlays with command: {ffmpeg_command}")
    
    # Add error handling and logging
    try:
        result = subprocess.call(ffmpeg_command, shell=True)
        if result == 0:
            print(f"Video with subtitles and overlays saved to: {output_path}")
            return True
        else:
            print(f"Error applying overlays, result code: {result}")
            # Log the failed command
            with open(f"{output_dir}/ffmpeg_errors.log", "a") as f:
                f.write(f"Failed command: {ffmpeg_command}\n")
                f.write(f"Error code: {result}\n")
                f.write("=" * 80 + "\n")
            return False
    except Exception as e:
        print(f"Exception during FFmpeg execution: {str(e)}")
        return False


if __name__ == "__main__":
    __main__()
