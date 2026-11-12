import random
import re
import shlex
import sys
import textwrap
import time
import math
import numpy as np
import yt_dlp
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
import tempfile
import ffmpeg
from enhanced_audio_analyzer import EnhancedAudioAnalyzer
from topic_analyzer import TopicAnalyzer
from talking_face_detector import TalkingFaceDetector
from performance_profiler import PerformanceProfiler
from face_quality_scorer import FaceQualityScorer

# Define workspace directory
WORKSPACE_DIR = os.getcwd()  # Use current working directory instead of script location
REELS_BACKGROUNDS_DIR = os.path.join(WORKSPACE_DIR, "workspace", "backgrounds", "reels")

# CUDA environment settings for better performance
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["GGML_CUDA_NO_PINNED"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# STT Model Configuration
def get_stt_model_config():
    """
    Get STT model configuration from environment variables with validation and fallback.

    Returns:
        str: Valid STT model name ('whisper' or 'faster-whisper')
    """
    stt_model = os.getenv('STT_MODEL', 'whisper').lower().strip()
    valid_models = ['whisper', 'faster-whisper']

    if stt_model in valid_models:
        print(f"Using STT model: {stt_model}")
        return stt_model
    else:
        print(f"Warning: Invalid STT_MODEL '{stt_model}'. Valid options are: {', '.join(valid_models)}")
        print("Falling back to default: whisper")
        return 'whisper'

scale = 4
target_face_size = 345

# Global variables for conditional loading
upsampler = None
face_enhancer = None

import argparse
import glob
import re


def _contained_path(base_dir, filename):
    """Join filename under base_dir, refusing escapes outside it.

    Returns the same joined form the call site would have built (relative
    stays relative), so existing behavior is unchanged; the check only
    refuses paths that would land outside base_dir. Defense in depth for
    writes whose base directory ultimately derives from watched-folder
    names: even with upstream sanitization, a write must never land
    outside the directory it was meant for.
    """
    joined = os.path.join(str(base_dir), str(filename))
    base = os.path.abspath(str(base_dir))
    full = os.path.abspath(joined)
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(
            f"path escapes its base directory: {filename!r} under {base_dir!r}"
        )
    return joined


def sanitize_filename(filename):
    """
    Sanitize filename by removing or replacing problematic characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename safe for filesystem
    """
    if not filename:
        return "untitled"
    
    # Convert to string if not already
    filename = str(filename)
    
    # Replace problematic characters with underscores or remove them
    # Remove/replace: parentheses, brackets, colons, question marks, exclamation marks, quotes
    filename = filename.replace("(", "").replace(")", "")
    filename = filename.replace("[", "").replace("]", "")
    filename = filename.replace("{", "").replace("}", "")
    filename = filename.replace(":", "").replace(";", "")
    filename = filename.replace("?", "").replace("!", "")
    filename = filename.replace('"', "").replace("'", "")
    filename = filename.replace("/", "_").replace("\\", "_")
    filename = filename.replace("|", "_").replace("*", "_")
    filename = filename.replace("<", "_").replace(">", "_")
    filename = filename.replace("&", "and")
    filename = filename.replace("#", "")
    filename = filename.replace("%", "")
    filename = filename.replace("@", "at")
    filename = filename.replace("$", "")
    
    # Replace multiple spaces with single space, then replace spaces with underscores
    filename = re.sub(r'\s+', ' ', filename.strip())
    filename = filename.replace(" ", "_")
    
    # Replace multiple underscores with single underscore
    filename = re.sub(r'_+', '_', filename)
    
    # Remove leading/trailing underscores
    filename = filename.strip('_')
    
    # Ensure filename is not empty after sanitization
    if not filename:
        filename = "untitled"
    
    # Limit length to prevent filesystem issues
    max_length = 200  # Leave room for extensions and prefixes
    if len(filename) > max_length:
        filename = filename[:max_length].rstrip('_')
    
    return filename

# Get the absolute path to the reels-clips-automator directory (where the .env file is)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")

# Load environment variables from reels-clips-automator/.env file
if not load_dotenv(env_path):
    print(f"Warning: Could not load .env file from {env_path}")

# Set OpenAI API key
config = dotenv_values(env_path)
api_key = config.get("OPENAI_API_KEY")
ai_endpoint = (config.get("AI_ENDPOINT") or os.getenv("AI_ENDPOINT") or "").strip("'\"")
openai_base_url = ai_endpoint or None

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
# Per-client base_url is set explicitly below via openai_base_url (None => default endpoint)


def copy_input_to_tmp(source_path, dest_path):
    """Stage the input video without a shell so paths with spaces stay intact."""
    shutil.copy2(source_path, dest_path)


def cleanup_tmp_directory(tmp_dir="tmp/", preserve_outputs=True):
    """
    Clean up temporary files after successful processing.

    Args:
        tmp_dir (str): Path to temporary directory
        preserve_outputs (bool): Deprecated, kept for compatibility. All tmp files
                                are deleted since the final complete video is already
                                in the output directory created by generate_subtitle().

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

        for file_path in tmp_files:
            filename = os.path.basename(file_path)

            try:
                # Delete all temporary files. The final complete video
                # (with audio, subtitles, overlays) is already in output_dir.
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
        preserve_outputs (bool): Deprecated, kept for compatibility.
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
def download_video(url, filename, tmp_dir="tmp/"):
    """Download a YouTube video to tmp/{filename} using yt-dlp."""
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(tmp_dir, filename),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# Segment Video function
def parse_srt_time(time_str):
    """Parse an SRT timestamp (HH:MM:SS,mmm) into total seconds."""
    time_parts, ms = str(time_str).split(",")
    h, m, s = map(int, time_parts.split(":"))
    return h * 3600 + m * 60 + s + int(ms) / 1000


# Jump cuts (Story 5 / P1.3): remove intra-segment dead air at the
# generate_segments stage, then remap the word timeline so captions stay in
# sync with the condensed clip. When nothing is cut (or the escape hatch is
# off) every path below is byte-identical to the previous pipeline.

JUMP_CUT_DEFAULT_PAUSE_THRESHOLD = 0.6
JUMP_CUT_DEFAULT_PADDING = 0.12


def get_jump_cut_config():
    """Get jump-cut configuration from environment variables with fallback.

    Returns:
        tuple: (enabled, pause_threshold). JUMP_CUTS_ENABLED defaults to
        "true"; only the literal "false" (any case) disables jump cuts.
        JUMP_CUT_PAUSE_THRESHOLD defaults to 0.6 seconds.
    """
    enabled_raw = os.getenv('JUMP_CUTS_ENABLED', 'true').lower().strip()
    enabled = enabled_raw != 'false'
    if not enabled:
        print("Jump cuts disabled (JUMP_CUTS_ENABLED=false); keeping full-length segments")

    threshold_raw = os.getenv(
        'JUMP_CUT_PAUSE_THRESHOLD', str(JUMP_CUT_DEFAULT_PAUSE_THRESHOLD)
    )
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        print(
            f"Warning: Invalid JUMP_CUT_PAUSE_THRESHOLD '{threshold_raw}'. "
            f"Falling back to default: {JUMP_CUT_DEFAULT_PAUSE_THRESHOLD}"
        )
        threshold = JUMP_CUT_DEFAULT_PAUSE_THRESHOLD
    return enabled, threshold


def build_jump_cut_plan(words, seg_start, seg_end, pause_threshold=0.6, padding=0.12):
    """Build the jump-cut plan for one segment.

    Args:
        words: flat word list (Story 3 contract) restricted to the segment,
            in clip-relative time exactly as collect_segment_words returns
            them (i.e. already shifted by -seg_start). May be None/empty.
        seg_start, seg_end: segment boundaries in source time (seconds).
        pause_threshold: gaps strictly above this many seconds are cut.
        padding: seconds kept on each side of a cut.

    Returns:
        tuple: (pieces, remapped_words).
        pieces: list of (start, end) source-time tuples covering the segment
            with intra-segment pauses removed; leading material before the
            first word and trailing material after the last word are kept.
        remapped_words: each word shifted left by the total removed duration
            before it (clamped >= 0), or None when no words were given.
        With no removable pauses (or no words) this is a single piece
        (seg_start, seg_end) and words identical to today's -seg_start shift.
    """
    if not words:
        return [(seg_start, seg_end)], None

    # Sort defensively and restrict to the segment (clip-relative bounds).
    seg_duration = seg_end - seg_start
    clip_words = sorted(
        (w for w in words if seg_start <= seg_start + float(w["start"]) <= seg_end),
        key=lambda w: float(w["start"]),
    )
    if not clip_words or seg_duration <= 0:
        return [(seg_start, seg_end)], list(words)

    # Cut ranges in source time: between consecutive words, the range
    # [prev_end + padding, next_start - padding] is removed when the gap
    # exceeds the threshold and the cut range is positive.
    cuts = []  # (cut_start, cut_end) in source time
    for prev_w, next_w in zip(clip_words, clip_words[1:]):
        prev_end_src = seg_start + float(prev_w["end"])
        next_start_src = seg_start + float(next_w["start"])
        if next_start_src - prev_end_src > pause_threshold:
            cut_start = prev_end_src + padding
            cut_end = next_start_src - padding
            if cut_end > cut_start:
                cuts.append((cut_start, cut_end))

    if not cuts:
        return [(seg_start, seg_end)], list(words)

    pieces = []
    cursor = seg_start
    for cut_start, cut_end in cuts:
        pieces.append((cursor, cut_start))
        cursor = cut_end
    pieces.append((cursor, seg_end))

    remapped = []
    for w in clip_words:
        start_src = seg_start + float(w["start"])
        removed_before = sum(
            c_end - c_start for c_start, c_end in cuts if c_start <= start_src
        )
        remapped.append(
            {
                "word": w.get("word", ""),
                "start": max(0.0, float(w["start"]) - removed_before),
                "end": max(0.0, float(w["end"]) - removed_before),
            }
        )
    return pieces, remapped


def _build_multi_piece_command(pieces, output_file):
    """Shell command cutting every piece from tmp/input_video.mp4 and
    concatenating them into tmp/{output_file}, re-encoded with the NVENC
    settings used by the single-cut path."""
    inputs = []
    parts = []
    for j, (piece_start, piece_end) in enumerate(pieces):
        inputs.append(f"-hwaccel cuda -ss {piece_start} -to {piece_end} -i tmp/input_video.mp4")
        parts.append(f"[{j}:v][{j}:a]")
    filter_complex = (
        f"{''.join(parts)}concat=n={len(pieces)}:v=1:a=1[out_v][out_a]"
    )
    return (
        "ffmpeg -y "
        + " ".join(inputs)
        + f" -filter_complex \"{filter_complex}\" -map \"[out_v]\" -map \"[out_a]\" "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -c:a aac -b:a 192k "
        f"tmp/{shlex.quote(output_file)}"
    )


def build_single_piece_command(start_time, end_time, output_file):
    """Shell command cutting one segment from tmp/input_video.mp4.

    -ss/-to sit BEFORE -i (fast seek; frame-accurate when re-encoding) so
    ffmpeg no longer decodes the entire source per cut. NVENC output
    settings are identical to the multi-piece builder."""
    return (
        "ffmpeg -y -hwaccel cuda "
        f"-ss {start_time} -to {end_time} -i tmp/input_video.mp4 "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -c:a aac -b:a 192k "
        f"tmp/{shlex.quote(output_file)}"
    )


def generate_segments(response, jump_cut_plans=None):
    plans = jump_cut_plans or {}
    for i, segment in enumerate(response):
        print(i, segment)

        start_time = segment.get("start_time", 0)
        end_time = segment.get("end_time", 0)
        title = segment.get("title", 0)

        # Convert timestamps to seconds (SRT format HH:MM:SS,mmm)
        start_time = parse_srt_time(start_time)
        end_time = parse_srt_time(end_time)

        # Adjust duration if too short
        if end_time - start_time < 30:
            end_time += 30 - (end_time - start_time)

        output_file = f"{sanitize_filename(title)}{str(i).zfill(3)}.mp4"
        entry = plans.get(i)
        pieces = entry[0] if entry else None
        if pieces and len(pieces) > 1:
            # Keep the short-segment end_time extension on the trailing piece.
            pieces = list(pieces)
            pieces[-1] = (pieces[-1][0], end_time)
            print(
                f"Jump cuts: cutting segment {i} as {len(pieces)} pieces "
                f"(removed {pieces[-1][0] - start_time:.2f}s of pauses)"
            )
            command = _build_multi_piece_command(pieces, str(output_file))
        else:
            command = build_single_piece_command(start_time, end_time, str(output_file))
        run_ffmpeg_command(command)


def compute_jump_cut_plans(segments, transcript_segments, pause_threshold):
    """Compute {index: (pieces, remapped_words)} for cuttable segments.

    Only segments whose plan actually removes material are stored; every
    other segment keeps today's single-cut / plain-shift behavior.
    """
    plans = {}
    for i, segment in enumerate(segments or []):
        seg_words = collect_segment_words(
            transcript_segments, segment.get("start_time"), segment.get("end_time")
        )
        if not seg_words:
            continue
        try:
            seg_start = parse_srt_time(segment.get("start_time"))
            seg_end = parse_srt_time(segment.get("end_time"))
        except (ValueError, AttributeError, TypeError):
            continue
        pieces, remapped = build_jump_cut_plan(
            seg_words, seg_start, seg_end, pause_threshold=pause_threshold
        )
        if pieces and len(pieces) > 1:
            plans[i] = (pieces, remapped)
            print(
                f"Jump cuts: segment {i} ({segment.get('start_time')} - "
                f"{segment.get('end_time')}) -> {len(pieces)} pieces"
            )
    return plans


def resolve_subtitle_words(jump_cut_plans, index, transcript_segments, start_time, end_time):
    """Subtitle words for one segment.

    Returns the jump-cut remapped words when a plan exists for the segment,
    otherwise the plain collect_segment_words shift (Story 4 behavior)."""
    entry = (jump_cut_plans or {}).get(index)
    if entry is not None:
        return entry[1]
    return collect_segment_words(transcript_segments, start_time, end_time)


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
    thumbnail_path = os.path.join(thumbs_dir, f"thumb-{sanitize_filename(text)}.png")
    newthumb.save(thumbnail_path, "PNG")
    print(f"Thumbnail saved to: {thumbnail_path}")


# Polish features (Story 11, P3.5): instead of taking the first smiling frame
# (early exit), smile candidates are COLLECTED across the clip (up to
# SMILE_CANDIDATE_LIMIT) and the one with the best face-quality score wins.
SMILE_CANDIDATE_LIMIT = 5


def select_best_smile_frame(candidates):
    """
    Pick the smile candidate with the best FaceQualityScorer score.

    Scoring failures (exception or no usable scores) fall back to the FIRST
    candidate — the previous first-smile behavior.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    scores = []
    try:
        scorer = FaceQualityScorer()
        for candidate in candidates:
            ch, cw = candidate.shape[:2]
            scores.append(scorer.score_face_quality(candidate, (0, 0, cw, ch)))
    except Exception as e:
        print(f"Warning: smile quality scoring failed ({e}); using first smile frame")
        return candidates[0]
    usable = [
        (i, s)
        for i, s in enumerate(scores)
        if isinstance(s, (int, float)) and not isinstance(s, bool)
    ]
    if not usable:
        return candidates[0]
    best_index = max(usable, key=lambda item: item[1])[0]
    print(f"[INFO] Selected smile frame {best_index + 1}/{len(candidates)} (best quality)")
    return candidates[best_index]


# ---------------------------------------------------------------------------
# Layout-aware reframing (Story 8, P2.1): screencast-first
#
# Screencast-style sources (camera in the top third, screen content below) are
# rendered as screen-filled 9:16 video with a circular tracking face-cam badge
# (PiP) alternating with motion-following screen zoom. Talking-head sources
# keep the pre-existing face-crop behavior unchanged.
# ---------------------------------------------------------------------------

# Layout classification
LAYOUT_SCREENCAST = "screencast"
LAYOUT_TALKING_HEAD = "talking_head"

# Screencast rendering modes
SCREENCAST_PIP = "pip"
SCREENCAST_SCREEN_ZOOM = "screen_zoom"
SCREENCAST_FACE_ZOOM = "face_zoom"
SCREENCAST_FACE_WIDE = "face_wide"
SCREENCAST_CLOSEUP = "closeup"

# Modes the screencast render loop can draw. FRAMING_MODE accepts any of them
# to pin the clip to one framing for the whole run.
SCREENCAST_RENDER_MODES = (
    SCREENCAST_PIP,
    SCREENCAST_SCREEN_ZOOM,
    SCREENCAST_FACE_ZOOM,
    SCREENCAST_FACE_WIDE,
)

# FRAMING_MODE env override: auto (default) uses detection; the others force
# the framing mode for the clip.
FRAMING_MODE_ENV_VAR = "FRAMING_MODE"
FRAMING_MODE_VALID = ("auto",) + SCREENCAST_RENDER_MODES + (SCREENCAST_CLOSEUP,)

# Detection thresholds (binding per design)
LAYOUT_SAMPLE_COUNT = 30            # frames sampled for layout detection
LAYOUT_MIN_FACE_SAMPLE_RATIO = 0.30 # face found in >= 30% of samples
LAYOUT_FACE_TOP_ZONE_RATIO = 0.40   # median face center-y in the top 40%
LAYOUT_MAX_FACE_HEIGHT_RATIO = 0.35 # median face height <= 35% of frame height
LAYOUT_MIN_EDGE_DENSITY = 0.05      # Canny non-zero ratio below the face > 0.05
LAYOUT_SCREEN_MARGIN_RATIO = 0.10   # small margin below the median face box

# PiP rendering specs
PIP_BADGE_DIAMETER_RATIO = 0.28     # badge diameter ~28% of frame width (~300 px)
PIP_FACE_MARGIN_RATIO = 0.25        # face crop expanded by ~25% margin
PIP_BADGE_MARGIN_RATIO = 0.03       # badge distance from the frame edges

# PiP badge stabilization: the Haar detector jitters its box every frame, so the
# badge EMA is strong (90% previous / 10% new) with a generous dead-zone, the
# crop SIZE is EMA-smoothed too (box-scale jitter would pulse the badge). The
# badge is cropped LIVE from every frame at the smoothed center — the EMA +
# dead-zone absorb detector jitter without freezing the face into stills.
PIP_BADGE_SMOOTHING_FACTOR = 0.9    # badge EMA: 90% previous / 10% new
PIP_BADGE_MOVEMENT_THRESHOLD = 15   # detector jitter (px) ignored by the EMA

# Deterministic variety schedule: rotate pip -> screen_zoom -> face_zoom ->
# face_wide on 5 s boundaries. The two face modes are what keep a screencast
# from reading as one static composition: they put a tracked face crop back in
# the rotation, so the clip alternates between "screen with a badge" and
# "face-forward" the way the talking-head path always did.
SCREENCAST_SEGMENT_SECONDS = 5
SCREENCAST_MODE_CYCLE = (
    SCREENCAST_PIP,
    SCREENCAST_SCREEN_ZOOM,
    SCREENCAST_FACE_ZOOM,
    SCREENCAST_FACE_WIDE,
)

# Face-mode crop widths, as a multiple of the smoothed face side. face_zoom is
# a tight talking-head crop; face_wide keeps the face plus enough screen around
# it to stay oriented.
FACE_ZOOM_WIDTH_RATIO = 1.6
FACE_WIDE_WIDTH_RATIO = 3.2

# Hard cap on a face-mode crop width, as a fraction of the frame width. The
# ratios above are keyed to the face, which on a portrait screencast can be
# large enough that face_wide asks for a window wider than the frame. Clamping
# BEFORE the push-in (rather than after) is what keeps the cap from collapsing
# the crop to the full frame and cancelling the motion: without it, face_wide
# on a 1080-wide source became 1080x1920 with no push-in at all, i.e. a static
# duplicate of the un-cropped frame.
FACE_MODE_MAX_WIDTH_RATIO = 0.85

# In-segment push-in for the face modes: the crop eases tighter across the
# segment (100% -> 92%), giving each face segment its own motion instead of a
# frozen crop that snaps at the mode boundary.
FACE_MODE_PUSH_IN_START_SCALE = 1.0
FACE_MODE_PUSH_IN_END_SCALE = 0.92

# Whole-clip push-in (PiP only): background zoom eases 100% -> 96%.
PUSH_IN_START_SCALE = 1.0
PUSH_IN_END_SCALE = 0.96

# Motion energy search grid for screen-zoom mode
MOTION_GRID_SIZE = 16

def resolve_framing_override():
    """Read the FRAMING_MODE env override. Invalid values fall back to auto."""
    mode = os.environ.get(FRAMING_MODE_ENV_VAR, "auto").strip().lower()
    if mode not in FRAMING_MODE_VALID:
        print(f"FRAMING_MODE={mode!r} is not one of {FRAMING_MODE_VALID}, using auto")
        return "auto"
    return mode

def median_face_box(face_boxes_per_sample):
    """Median face box across samples (None samples ignored) or None."""
    found = [
        tuple(int(v) for v in box)
        for box in (face_boxes_per_sample or [])
        if box is not None
    ]
    if not found:
        return None
    med = np.median(np.array(found, dtype=np.float64), axis=0)
    return tuple(int(round(v)) for v in med)

def screen_region_from_median_box(median_box, frame_shape,
                                  margin_ratio=LAYOUT_SCREEN_MARGIN_RATIO):
    """
    Screen payload region: everything below the median face box (with a small
    margin) as (x, y, width, height). Falls back to the full frame when no
    face box was established.
    """
    frame_height, frame_width = frame_shape[0], frame_shape[1]
    if median_box is None:
        return (0, 0, frame_width, frame_height)
    _, face_y, _, face_h = median_box
    margin = int(face_h * margin_ratio)
    sy = min(max(face_y + face_h + margin, 0), max(frame_height - 1, 0))
    return (0, sy, frame_width, frame_height - sy)

def detect_layout(face_boxes_per_sample, frame_shape, edge_density_below_face,
                  forced_mode="auto"):
    """
    Classify the clip layout from precomputed observations.

    Screencast when: a face is found in >= 30% of samples AND the median face
    box has center-y in the top 40% of the frame AND median face height <= 35%
    of frame height AND the region below the median face box has meaningful
    edge density (Canny non-zero ratio > 0.05). Otherwise talking_head.

    A forced mode (FRAMING_MODE override) bypasses the classification.
    """
    if forced_mode in SCREENCAST_RENDER_MODES:
        return LAYOUT_SCREENCAST
    if forced_mode == SCREENCAST_CLOSEUP:
        return LAYOUT_TALKING_HEAD

    samples = list(face_boxes_per_sample or [])
    if not samples:
        return LAYOUT_TALKING_HEAD

    found = [box for box in samples if box is not None]
    if len(found) / len(samples) < LAYOUT_MIN_FACE_SAMPLE_RATIO:
        return LAYOUT_TALKING_HEAD

    med = median_face_box(found)
    if med is None:
        return LAYOUT_TALKING_HEAD
    _, med_y, _, med_h = med
    frame_height = frame_shape[0]
    center_y = med_y + med_h / 2.0

    if center_y / frame_height > LAYOUT_FACE_TOP_ZONE_RATIO:
        return LAYOUT_TALKING_HEAD
    if med_h / frame_height > LAYOUT_MAX_FACE_HEIGHT_RATIO:
        return LAYOUT_TALKING_HEAD
    if edge_density_below_face <= LAYOUT_MIN_EDGE_DENSITY:
        return LAYOUT_TALKING_HEAD
    return LAYOUT_SCREENCAST

def collect_layout_observations(cap, face_cascade, frame_width, frame_height,
                                sample_count=LAYOUT_SAMPLE_COUNT,
                                debug_faces=False):
    """
    Sample ~30 evenly spaced frames cheaply (seek via grab/retrieve), run Haar
    face detection on them, and collect the observations detect_layout needs:
    (face_boxes_per_sample, edge_density_below_face).

    The edge density is computed on the first sampled frame that had a face,
    over the region below the median face box.
    """
    boxes = []
    sampled = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = sample_count
        sample_count = max(min(sample_count, total_frames), 1)

        frames_read = 0
        for i in range(sample_count):
            target = int(i * total_frames / sample_count)
            while frames_read < target:
                if not cap.grab():
                    break
                frames_read += 1
            if not cap.grab():
                break
            frames_read += 1
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                continue
            sampled.append(frame)

        for frame in sampled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100)
            )
            if faces is not None and len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                boxes.append((int(fx), int(fy), int(fw), int(fh)))
            else:
                boxes.append(None)

        edge_density = 0.0
        med = median_face_box(boxes)
        if med is not None:
            frame_shape = (frame_height, frame_width)
            region_x, region_y, region_w, region_h = screen_region_from_median_box(
                med, frame_shape
            )
            for frame in sampled:
                sub = frame[region_y:region_y + region_h, region_x:region_x + region_w]
                if sub.size == 0:
                    break
                edges = cv2.Canny(sub, 50, 150)
                if edges.size > 0:
                    edge_density = np.count_nonzero(edges) / edges.size
                break
        if debug_faces:
            print(f"Layout sampling: {len(sampled)} frames, faces in {sum(1 for b in boxes if b is not None)}, edge_density={edge_density:.4f}")
        return boxes, edge_density
    except Exception as e:
        print(f"Layout sampling failed ({e}), treating as talking-head")
        return [], 0.0

def compute_push_in_scale(progress,
                          start_scale=PUSH_IN_START_SCALE,
                          end_scale=PUSH_IN_END_SCALE):
    """Whole-clip push-in: smoothstep easing from start_scale to end_scale."""
    p = min(max(progress, 0.0), 1.0)
    smooth = p * p * (3.0 - 2.0 * p)
    return start_scale + (end_scale - start_scale) * smooth

def screencast_mode_for_frame(frame_count, fps, segment_seconds=SCREENCAST_SEGMENT_SECONDS):
    """Deterministic variety: rotate through SCREENCAST_MODE_CYCLE on segment
    boundaries (fixed pattern, pip first)."""
    if fps <= 0:
        fps = 30.0
    segment_frames = max(int(fps * segment_seconds), 1)
    return SCREENCAST_MODE_CYCLE[(frame_count // segment_frames) % len(SCREENCAST_MODE_CYCLE)]

def resolve_screencast_mode(frame_count, fps, forced="auto",
                            segment_seconds=SCREENCAST_SEGMENT_SECONDS):
    """Per-frame screencast mode: forced override or the deterministic schedule."""
    if forced in SCREENCAST_RENDER_MODES:
        return forced
    return screencast_mode_for_frame(frame_count, fps, segment_seconds)

def compute_face_push_in(progress,
                         start_scale=FACE_MODE_PUSH_IN_START_SCALE,
                         end_scale=FACE_MODE_PUSH_IN_END_SCALE):
    """Smoothstep easing of the face-mode crop scale across one segment."""
    p = min(max(progress, 0.0), 1.0)
    smooth = p * p * (3.0 - 2.0 * p)
    return start_scale + (end_scale - start_scale) * smooth

def compute_face_crop(center, face_side, frame_shape, width_ratio, push_scale=1.0,
                      max_width_ratio=FACE_MODE_MAX_WIDTH_RATIO):
    """
    9:16 crop window centered on the tracked face, sized from the smoothed
    face side times `width_ratio`. Clamped to FACE_MODE_MAX_WIDTH_RATIO of the
    frame width BEFORE the push-in is applied, so the in-segment motion always
    survives the clamp. Clamped to stay inside the frame.

    Returns (x, y, width, height).
    """
    frame_height, frame_width = frame_shape[0], frame_shape[1]

    side = max(int(face_side or 0), 16)
    max_width = max(int(frame_width * max_width_ratio), 16)
    crop_w = min(max(int(side * width_ratio), 16), max_width)
    crop_w = max(int(crop_w * push_scale), 16)
    crop_w = min(crop_w, frame_width)
    crop_h = int(crop_w * 16 / 9)
    if crop_h > frame_height:
        crop_h = frame_height
        crop_w = max(int(frame_height * 9 / 16), 16)

    cx = int(center[0])
    cy = int(center[1])
    cx = max(cx, crop_w // 2)
    cx = min(cx, frame_width - crop_w // 2)
    cy = max(cy, crop_h // 2)
    cy = min(cy, frame_height - crop_h // 2)
    return (cx - crop_w // 2, cy - crop_h // 2, crop_w, crop_h)

def smooth_center(prev_center, raw_center, smoothing_factor=0.7, movement_threshold=5):
    """
    Existing EMA smoothing semantics (70% previous / 30% new) with the
    existing dead-zone threshold. Applied EVERY frame now, also when tracking
    is not frame-by-frame. Returns None only when both inputs are None.
    """
    if raw_center is None:
        return prev_center
    if prev_center is None:
        return raw_center
    dx = raw_center[0] - prev_center[0]
    dy = raw_center[1] - prev_center[1]
    movement_distance = (dx ** 2 + dy ** 2) ** 0.5
    if movement_distance > movement_threshold:
        return (
            int(prev_center[0] * smoothing_factor + raw_center[0] * (1 - smoothing_factor)),
            int(prev_center[1] * smoothing_factor + raw_center[1] * (1 - smoothing_factor)),
        )
    return prev_center

def compute_motion_center(prev_gray, gray, frame_shape, grid=MOTION_GRID_SIZE):
    """
    Point of maximum motion energy between two downscaled gray frames, mapped
    back to original frame coordinates. Returns None when motion is
    negligible or there is no previous frame (caller falls back to the
    screen-region center).
    """
    if prev_gray is None or gray is None:
        return None
    frame_height, frame_width = frame_shape[0], frame_shape[1]
    prev_small = cv2.resize(prev_gray, (grid, grid), interpolation=cv2.INTER_AREA)
    small = cv2.resize(gray, (grid, grid), interpolation=cv2.INTER_AREA)
    diff = cv2.absdiff(small, prev_small)
    if float(np.sum(diff)) <= 0.0:
        return None
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(diff)
    col, row = max_loc
    cell_w = frame_width / grid
    cell_h = frame_height / grid
    return (int((col + 0.5) * cell_w), int((row + 0.5) * cell_h))

def compute_screen_zoom_crop(screen_region, center, frame_shape):
    """
    9:16 crop window over the screen region, centered on `center` (the
    motion point or the screen-region center fallback), clamped to the frame.
    Returns (x, y, width, height).
    """
    frame_height, frame_width = frame_shape[0], frame_shape[1]
    sx, sy, sw, sh = screen_region

    crop_w = min(max(sw, 16), frame_width)
    crop_h = int(crop_w * 16 / 9)
    if crop_h > frame_height:
        crop_h = frame_height
        crop_w = max(int(frame_height * 9 / 16), 16)

    cx = int(center[0])
    cy = int(center[1])
    if sw >= crop_w:
        cx = max(cx, sx + crop_w // 2)
        cx = min(cx, sx + sw - crop_w // 2)
    cx = max(cx, crop_w // 2)
    cx = min(cx, frame_width - crop_w // 2)
    if crop_h <= frame_height - sy:
        cy = max(cy, sy + crop_h // 2)
    cy = max(cy, crop_h // 2)
    cy = min(cy, frame_height - crop_h // 2)
    return (cx - crop_w // 2, cy - crop_h // 2, crop_w, crop_h)

def smooth_badge_side(prev_side, raw_side, smoothing_factor=PIP_BADGE_SMOOTHING_FACTOR):
    """
    EMA on the badge crop SIZE so detector box-scale jitter does not pulse the
    badge content. None handling mirrors smooth_center: a missing raw keeps the
    previous size; the first raw is taken as-is.
    """
    if raw_side is None:
        return prev_side
    if prev_side is None:
        return raw_side
    return int(prev_side * smoothing_factor + raw_side * (1 - smoothing_factor))


def render_pip_frame(frame, screen_region, face_box, face_center, push_scale=1.0,
                     last_badge=None, output_size=(1080, 1920), face_side=None):
    """
    Screencast PiP rendering: background = the screen region (below the median
    face box) as an aspect-preserving 9:16 crop scaled by push_scale, resized
    to output_size; foreground = circular face-cam badge in the TOP-RIGHT
    (face crop with margin, white ring border).

    When the face is missing in this frame, the last badge crop is held.
    Returns (output_frame, badge_crop_or_None).
    """
    out_w, out_h = output_size
    frame_height, frame_width = frame.shape[:2]
    sx, sy, sw, sh = screen_region

    # Background: largest 9:16 window in the screen region, shrunk by the
    # push-in scale, centered in the region, clamped to the frame.
    if sw * 16 / 9 <= sh:
        crop_w = sw
        crop_h = int(sw * 16 / 9)
    else:
        crop_h = max(sh, 1)
        crop_w = int(sh * 9 / 16)
    crop_w = max(int(crop_w * push_scale), 16)
    crop_h = max(int(crop_h * push_scale), 16)
    bg_cx = sx + sw // 2
    bg_cy = sy + sh // 2
    bg_x = min(max(bg_cx - crop_w // 2, 0), max(frame_width - crop_w, 0))
    bg_y = min(max(bg_cy - crop_h // 2, 0), max(frame_height - crop_h, 0))
    background = frame[bg_y:bg_y + crop_h, bg_x:bg_x + crop_w]
    output = cv2.resize(background, (out_w, out_h), interpolation=cv2.INTER_CUBIC)

    # Foreground badge: face crop with margin (or the held last badge).
    badge = None
    if face_box is not None and face_center is not None:
        fx, fy, fw, fh = face_box
        side = face_side if face_side else int(max(fw, fh) * (1.0 + 2 * PIP_FACE_MARGIN_RATIO))
        side = max(side, 16)
        half = side // 2
        bc_x = int(face_center[0])
        bc_y = int(face_center[1])
        bx = min(max(bc_x - half, 0), max(frame_width - side, 0))
        by = min(max(bc_y - half, 0), max(frame_height - side, 0))
        badge = frame[by:by + side, bx:bx + side].copy()
    elif last_badge is not None:
        badge = last_badge

    if badge is not None:
        d = max(int(out_w * PIP_BADGE_DIAMETER_RATIO), 32)
        margin = max(10, int(out_w * PIP_BADGE_MARGIN_RATIO))
        badge_cx = out_w - d // 2 - margin
        badge_cy = d // 2 + margin
        badge_sq = cv2.resize(badge, (d, d), interpolation=cv2.INTER_CUBIC)

        mask = np.zeros((d, d), dtype=np.uint8)
        cv2.circle(mask, (d // 2, d // 2), d // 2, 255, -1)
        x0 = badge_cx - d // 2
        y0 = badge_cy - d // 2
        region = output[y0:y0 + d, x0:x0 + d]
        sel = mask > 0
        region[sel] = badge_sq[sel]

        # White ring border on top of the badge edge
        ring_w = max(2, d // 30)
        cv2.circle(output, (badge_cx, badge_cy), d // 2 - ring_w // 2,
                   (255, 255, 255), ring_w)

    return output, badge

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


def build_pipe_ffmpeg_command(sanitized_input, sanitized_output, input_fps):
    """Single-encode command for generate_short (Story 9).

    Rendered BGR frames arrive on stdin (`-f rawvideo`); the segment input
    supplies the audio via the second -i + -map 1:a. Video is encoded ONCE
    with the existing NVENC settings and normalized to ASSEMBLY_FPS; audio is
    re-encoded to AAC 192k in the same pass. No mp4v intermediate, no
    separate merge step.

    The frame rate and timescale are pinned rather than left to the encoder
    because this file is not always an intermediate: when generate_subtitle
    has no overlays it stream-copies it straight into the published path (C3),
    so its parameters would otherwise reach the assembly verbatim."""
    return (
        "ffmpeg -y "
        f"-f rawvideo -pix_fmt bgr24 -s 1080x1920 -r {input_fps} -i pipe:0 "
        f"-i tmp/{shlex.quote(sanitized_input)} "
        "-map 0:v -map 1:a "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k "
        f"-r {ASSEMBLY_FPS} -pix_fmt yuv420p "
        f"-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE} "
        f"-c:a aac -b:a 192k -ar {ASSEMBLY_AUDIO_RATE} "
        f"-ac {ASSEMBLY_AUDIO_CHANNELS} "
        f"tmp/{shlex.quote(sanitized_output)}"
    )


class _PipeFrameWriter:
    """Frame sink that pipes raw BGR frames into a running ffmpeg process.

    Replaces cv2.VideoWriter on the primary (single-encode) path; both the
    screencast continue-branch and the talking-head path call write() with
    a BGR numpy frame, which is streamed to the encoder's stdin. A broken
    pipe is remembered (failed) instead of spewing per-frame errors; the
    caller decides whether to fall back to the legacy encode chain."""

    def __init__(self, proc):
        self.proc = proc
        self.failed = False

    def write(self, frame):
        if self.failed:
            return
        try:
            self.proc.stdin.write(frame.tobytes())
        except Exception as e:
            self.failed = True
            print(f"Warning: frame pipe broke ({e}); skipping remaining frames")

    def release(self):
        try:
            self.proc.stdin.close()
        except Exception as e:
            print(f"Warning: closing frame pipe failed: {e}")
        self.proc.wait()


# Polish features (Story 11, P3.4): the center-crop fallback (no face in the
# current frame) no longer hard-centers when the clip HAS seen a face earlier;
# the largest 9:16 window is positioned on the last known face center instead.
# Pure helper so the math is testable; crop_ratio=1 matches the local
# CROP_RATIO_BIG inside generate_short.
def compute_fallback_crop(
    frame_width, frame_height, last_face_center=None, crop_ratio=1
):
    """
    Compute the top-left corner and size of the fallback 9:16 crop window.

    last_face_center is the (x, y) of the most recently tracked face center
    (the EMA-smoothed prev_face_center when available). None keeps the
    current centered behavior exactly.
    """
    vertical_ratio = 9 / 16

    # Largest possible 9:16 box that fits in the frame (same max-window math
    # as the face-path crop).
    max_possible_width = frame_width
    max_possible_height = int(frame_width / vertical_ratio)
    if max_possible_height > frame_height:
        max_possible_height = frame_height
        max_possible_width = int(frame_height * vertical_ratio)

    crop_height = int(max_possible_height * crop_ratio)
    crop_width = int(max_possible_width * crop_ratio)

    if last_face_center is None:
        # No face history: hard-centered crop (previous behavior).
        crop_x = max(0, (frame_width - crop_width) // 2)
        crop_y = max(0, (frame_height - crop_height) // 2)
    else:
        # Position the window on the last known face center, clamped to
        # frame bounds (same positioning as the face-path crop).
        crop_x = last_face_center[0] - crop_width // 2
        crop_y = last_face_center[1] - crop_height // 2
        if crop_x < 0:
            crop_x = 0
        elif crop_x + crop_width > frame_width:
            crop_x = frame_width - crop_width
        if crop_y < 0:
            crop_y = 0
        elif crop_y + crop_height > frame_height:
            crop_y = frame_height - crop_height
        crop_x = max(0, crop_x)
        crop_y = max(0, crop_y)

    return crop_x, crop_y, crop_width, crop_height


def generate_short(
    input_file,
    output_file,
    title="",
    upscale=False,
    enhance=False,
    thumb=False,
    thumb_dir="",
    debug_faces=False,
    performance_profile=False,
    thumb_text=None,
    _pipe=True,
):
    # Fix C: initialized up-front so the error handler can clean up the
    # encoder child (and the capture/writer) no matter where the failure
    # happens — including inside the frame loop.
    cap = None
    out = None
    pipe_proc = None
    try:
        # Sanitize input and output file names
        sanitized_input = sanitize_filename(os.path.splitext(input_file)[0]) + os.path.splitext(input_file)[1]
        sanitized_output = sanitize_filename(os.path.splitext(output_file)[0]) + os.path.splitext(output_file)[1]

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
        TRACKING_STATIC_PROBABILITY = 0.3  # 30% chance for static tracking
        ZOOM_CLOSEUP_PROBABILITY = 0.6  # 60% chance for close-up shots
        ZOOM_IN_PROBABILITY = 0.2  # 20% chance for smooth zoom-in effect

        # Load pre-trained face detector from OpenCV
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Initialize performance profiler if enabled
        profiler = PerformanceProfiler() if performance_profile else None

        # Initialize talking face detector with updated parameters
        # FIX: Increased movement weight (0.7) and added min_movement_threshold (0.15)
        # to prevent selection of static faces like YouTube thumbnails
        talking_detector = TalkingFaceDetector(
            movement_weight=0.7,
            quality_weight=0.3,
            min_score_threshold=0.3,
            hysteresis_threshold=0.2,
            min_movement_threshold=0.09,
            static_penalty_threshold=0.1,
            static_frame_count=20,
            verbose=True,  # Set to True for debugging face selection
            performance_profiler=profiler
        )

        # Open video file
        cap = cv2.VideoCapture(f"tmp/{sanitized_input}")

        # Get the frame dimensions and frame rate
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Input video: {frame_width}x{frame_height} at {input_fps} fps")

        # Story 8: layout detection, once per clip, before the frame loop.
        # Screencasts render as screen-filled 9:16 with a face-cam badge /
        # motion-following zoom; talking heads keep the existing face crop.
        framing_override = resolve_framing_override()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        layout = LAYOUT_TALKING_HEAD
        screen_region = (0, 0, frame_width, frame_height)
        obs_boxes, obs_edge_density = [], 0.0
        if framing_override == SCREENCAST_CLOSEUP:
            print(f"FRAMING_MODE=closeup override: using talking-head framing")
        else:
            obs_boxes, obs_edge_density = collect_layout_observations(
                cap, face_cascade, frame_width, frame_height, debug_faces=debug_faces
            )
            layout = detect_layout(
                obs_boxes,
                (frame_height, frame_width),
                obs_edge_density,
                forced_mode=framing_override,
            )
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if layout == LAYOUT_SCREENCAST:
                med = median_face_box(obs_boxes)
                screen_region = screen_region_from_median_box(
                    med, (frame_height, frame_width)
                )
                print(
                    f"Layout: {layout} (median face box={med}, "
                    f"screen region={screen_region}, "
                    f"edge_density={obs_edge_density:.4f}, override={framing_override})"
                )
            else:
                print(f"Layout: {layout} (override={framing_override})")

        # Screencast render state
        last_badge = None            # held badge crop for detection misses
        prev_badge_center = None     # EMA-smoothed badge center
        prev_badge_side = None       # EMA-smoothed badge crop size
        prev_zoom_gray = None        # previous gray frame for motion energy
        prev_zoom_center = None      # EMA-smoothed screen-zoom center
        prev_screencast_mode = None  # for mode-switch debug prints

        # Interval to switch faces (calculated based on input fps for 5 seconds)
        switch_interval = int(input_fps * 5)  # 5 seconds at input fps
        print(
            f"Face switch interval: {switch_interval} frames ({switch_interval/input_fps:.1f} seconds)"
        )

        # Calculate dynamic target face size based on resolution
        target_face_size = calculate_target_face_size(frame_width, frame_height)
        print(f"Dynamic target face size threshold: {target_face_size}")

        # Story 9: single-encode path. OpenCV stays the reader/decoder and
        # all rendering logic is unchanged; rendered BGR frames are piped to
        # ONE ffmpeg pass (NVENC + muxed audio + 30 fps normalize). The
        # legacy mp4v VideoWriter + audio-merge chain is kept intact and
        # used when the pipe process cannot start (and, via re-render, when
        # it exits nonzero).
        pipe_proc = None
        out = None
        if _pipe:
            pipe_command = build_pipe_ffmpeg_command(
                sanitized_input, f"final-{sanitized_output}", input_fps
            )
            print(f"Single-pass encode command: {pipe_command}")
            try:
                pipe_proc = subprocess.Popen(
                    shlex.split(pipe_command), shell=False, stdin=subprocess.PIPE
                )
                out = _PipeFrameWriter(pipe_proc)
            except Exception as e:
                pipe_proc = None
                print(
                    f"Warning: could not start single-pass encoder ({e}); "
                    "falling back to legacy mp4v + merge path"
                )
        if pipe_proc is None:
            # Legacy path: define the codec and create VideoWriter object.
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                f"tmp/{sanitized_output}", fourcc, input_fps, (1080, 1920), True
            )  # Use input fps
        face_positions = []
        # Story 11: collected smile candidates (no first-smile early exit);
        # the best-quality candidate becomes the thumbnail at the end.
        smile_candidates = []
        boxes = None
        trackers = None

        # Explicit face-box state. Initialized to None so downstream code never
        # relies on leaked loop variables: if no face box has been established
        # yet (detection or tracking), the center-crop branch takes over.
        x, y, w, h = None, None, None, None

        # Dynamic tracking variables
        tracking_mode = "static"  # Values: "static", "frame_by_frame"
        zoom_mode = "close_up"  # Values: "close_up", "wide_shot"
        frame_tracking_enabled = False
        zoom_in_enabled = False  # Whether to apply smooth zoom-in effect
        
        # Smoothing variables to reduce camera shake
        prev_face_center = None
        smoothing_factor = 0.7  # 0.7 = 70% previous position, 30% new position
        movement_threshold = 5  # Ignore movements smaller than 5 pixels

        # Previous frame buffer for talking face detection
        prev_frame = None

        while cap.isOpened():
            # Read frame from video
            ret, frame = cap.read()

            if ret == True:
                # Story 8: screencast rendering path (guarded; the talking-head
                # path below is unchanged).
                if layout == LAYOUT_SCREENCAST:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100)
                    )
                    face_box = None
                    if faces is not None and len(faces) > 0:
                        face_box = max(
                            (tuple(int(v) for v in f) for f in faces),
                            key=lambda b: b[2] * b[3],
                        )

                    sc_mode = resolve_screencast_mode(
                        frame_count, input_fps, forced=framing_override
                    )
                    if sc_mode != prev_screencast_mode:
                        if debug_faces:
                            print(
                                f"Screencast mode switch: {prev_screencast_mode} -> "
                                f"{sc_mode} at frame {frame_count}"
                            )
                        prev_screencast_mode = sc_mode

                    raw_center = None
                    raw_side = None
                    if face_box is not None:
                        fb_x, fb_y, fb_w, fb_h = face_box
                        raw_center = (fb_x + fb_w // 2, fb_y + fb_h // 2)
                        raw_side = int(
                            max(fb_w, fb_h) * (1.0 + 2 * PIP_FACE_MARGIN_RATIO)
                        )
                    # Shared face tracking: the badge crop and the full-frame
                    # face modes read the SAME smoothed center and side, so a
                    # mode switch reuses the current estimate instead of
                    # re-snapping to a raw (jittery) detector box.
                    face_center = smooth_center(
                        prev_badge_center,
                        raw_center,
                        PIP_BADGE_SMOOTHING_FACTOR,
                        PIP_BADGE_MOVEMENT_THRESHOLD,
                    )
                    prev_badge_center = face_center
                    face_side = smooth_badge_side(prev_badge_side, raw_side)
                    prev_badge_side = face_side

                    if sc_mode == SCREENCAST_PIP:
                        # Background push-in over the whole clip duration
                        progress = frame_count / total_frames if total_frames > 0 else 0.0
                        push_scale = compute_push_in_scale(progress)

                        # Live crop every frame at the smoothed center: the
                        # EMA + dead-zone absorb detector jitter, while the
                        # badge content stays real video (no frozen hold).
                        rendered, badge_crop = render_pip_frame(
                            frame,
                            screen_region,
                            face_box,
                            face_center,
                            push_scale=push_scale,
                            last_badge=last_badge,
                            face_side=face_side,
                        )
                        if badge_crop is not None:
                            last_badge = badge_crop
                        out.write(rendered)

                        if (
                            thumb
                            and len(smile_candidates) < SMILE_CANDIDATE_LIMIT
                            and face_box is not None
                        ):
                            fb_x, fb_y, fb_w, fb_h = face_box
                            candidate_found, candidate_frame = find_smile(
                                frame[fb_y : fb_y + fb_h, fb_x : fb_x + fb_w],
                                output_file,
                            )
                            if candidate_found:
                                smile_candidates.append(candidate_frame)
                    elif sc_mode in (SCREENCAST_FACE_ZOOM, SCREENCAST_FACE_WIDE):
                        # Face-forward modes: the tracked face fills the frame
                        # instead of sitting in a badge. The crop eases tighter
                        # across the segment so the segment has its own motion.
                        if face_center is None:
                            # No face tracked yet: fall back to the screen
                            # region so the segment still renders.
                            ms_x, ms_y, ms_w, ms_h = screen_region
                            face_center = (ms_x + ms_w // 2, ms_y + ms_h // 2)
                        if face_side is None:
                            face_side = int(
                                frame_height * FACE_ZOOM_WIDTH_RATIO * 0.5
                            )

                        segment_frames = max(int(input_fps * SCREENCAST_SEGMENT_SECONDS), 1)
                        seg_progress = (frame_count % segment_frames) / segment_frames
                        push_scale = compute_face_push_in(seg_progress)

                        width_ratio = (
                            FACE_ZOOM_WIDTH_RATIO
                            if sc_mode == SCREENCAST_FACE_ZOOM
                            else FACE_WIDE_WIDTH_RATIO
                        )
                        crop_x, crop_y, crop_w9, crop_h9 = compute_face_crop(
                            face_center,
                            face_side,
                            (frame_height, frame_width),
                            width_ratio,
                            push_scale=push_scale,
                        )
                        crop_img = frame[crop_y : crop_y + crop_h9, crop_x : crop_x + crop_w9]
                        out.write(
                            cv2.resize(crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC)
                        )
                    else:
                        # Screen-zoom: follow the point of maximum motion energy,
                        # fallback to the screen-region center when negligible.
                        motion_center = compute_motion_center(
                            prev_zoom_gray, gray, (frame_height, frame_width)
                        )
                        if motion_center is None:
                            ms_x, ms_y, ms_w, ms_h = screen_region
                            motion_center = (ms_x + ms_w // 2, ms_y + ms_h // 2)
                        zoom_center = smooth_center(prev_zoom_center, motion_center)
                        prev_zoom_center = zoom_center

                        crop_x, crop_y, crop_w9, crop_h9 = compute_screen_zoom_crop(
                            screen_region, zoom_center, (frame_height, frame_width)
                        )
                        crop_img = frame[crop_y : crop_y + crop_h9, crop_x : crop_x + crop_w9]
                        resized = cv2.resize(
                            crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                        )
                        out.write(resized)

                    prev_zoom_gray = gray.copy()
                    frame_count += 1
                    continue

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
                        # Store face positions for this frame
                        face_positions.clear()
                        for x, y, w, h in faces:
                            face_positions.append((x, y, w, h))

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

                    if debug_faces:
                        print(f"Interval {frame_count // switch_interval}: tracking_mode={tracking_mode}, zoom_mode={zoom_mode}, zoom_in={zoom_in_enabled}")

                    # Intelligently select best talking face using TalkingFaceDetector
                    # Convert faces to list of tuples for detector
                    if len(face_positions) > 0:
                        face_list = list(face_positions)

                        # FIX: Use look-ahead window for better face selection
                        # Read next N frames to determine which face is actually talking
                        # This prevents static faces (thumbnails) from being selected
                        lookahead_frames = []
                        saved_position = cap.get(cv2.CAP_PROP_POS_FRAMES)

                        # Read ahead up to static_frame_count frames
                        for _ in range(talking_detector.static_frame_count):
                            ret_ahead, frame_ahead = cap.read()
                            if ret_ahead:
                                lookahead_frames.append(frame_ahead)
                            else:
                                break

                        # Restore video position
                        cap.set(cv2.CAP_PROP_POS_FRAMES, saved_position)

                        # Use look-ahead if we got enough frames, otherwise fall back to single-frame
                        if len(lookahead_frames) >= 2:
                            best_face_idx = talking_detector.get_best_talking_face_with_lookahead(
                                frames=lookahead_frames,
                                faces=face_list,
                                current_face_index=current_face_index if current_face_index < len(face_list) else None
                            )
                        else:
                            # Fallback to single-frame evaluation (end of video or read error)
                            best_face_idx = talking_detector.get_best_talking_face(
                                frame=frame,
                                faces=face_list,
                                previous_frame=prev_frame,
                                current_face_index=current_face_index if current_face_index < len(face_list) else None
                            )

                        if best_face_idx is not None:
                            current_face_index = best_face_idx
                            x, y, w, h = face_positions[current_face_index]

                            # Get scores for logging
                            scores = talking_detector.get_face_scores()
                            print(
                                f"Frame: {frame_count}, Selected Face {current_face_index}/{len(face_positions)} "
                                f"(talking={scores['movement']:.2f}, quality={scores['quality']:.2f}, combined={scores['combined']:.2f}) "
                                f"height={h} width={w}"
                            )
                        else:
                            # FIX: No suitable face found in look-ahead window
                            # This could be temporary silence - keep previous face if we have one
                            if current_face_index is not None and current_face_index < len(face_positions):
                                # Keep tracking the previous face during temporary silence
                                print(f"Frame: {frame_count}, No talking face in look-ahead window (temporary silence), "
                                      f"keeping previous face {current_face_index}")
                                # Don't clear face_positions - continue tracking current face
                            else:
                                # No previous face to fall back to - use center crop
                                print(f"Frame: {frame_count}, No suitable talking face found (all below threshold), using center crop")
                                face_positions.clear()
                    else:
                        # No faces detected, clear positions to use center crop fallback
                        print(f"Frame: {frame_count}, No faces detected, using center crop fallback")
                        face_positions.clear()

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
                            # Fallback to static positioning if tracking fails.
                            # Keep the last known good face box (x, y, w, h)
                            # deliberately - it is explicit state initialized
                            # before the loop, not a leaked loop variable.
                            print(f"Frame-by-frame tracking failed, falling back to static mode")
                            frame_tracking_enabled = False
                    except Exception as e:
                        print(f"Error in frame-by-frame tracking: {e}, falling back to static mode")
                        frame_tracking_enabled = False

                # Calculate face center and crop parameters (for both modes)
                # Only when a face box has actually been established; otherwise
                # fall back to the center-crop branch below.
                if len(face_positions) > 0 and x is not None:
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
                        linear_progress = interval_frame / switch_interval if switch_interval > 0 else 0
                        
                        # Apply ease-in-out cubic easing for smooth zoom motion
                        # Formula: t * t * (3.0 - 2.0 * t) creates smooth acceleration/deceleration
                        zoom_progress = linear_progress * linear_progress * (3.0 - 2.0 * linear_progress)
                        
                        # Start from base ratio and gradually zoom in (reduce ratio) over 5 seconds
                        # Zoom in by reducing crop ratio by up to 30% over the interval
                        zoom_reduction = 0.3 * zoom_progress
                        final_crop_ratio = base_crop_ratio * (1 - zoom_reduction)
                        
                        # Ensure minimum crop ratio to prevent over-zoom
                        final_crop_ratio = max(final_crop_ratio, 0.3)
                    
                    # Calculate crop dimensions ensuring proper 9:16 aspect ratio
                    # and preventing content stretching
                    
                    # Calculate the maximum possible 9:16 box that fits in the frame
                    max_possible_width = frame_width
                    max_possible_height = int(frame_width / VERTICAL_RATIO)
                    
                    if max_possible_height > frame_height:
                        max_possible_height = frame_height
                        max_possible_width = int(frame_height * VERTICAL_RATIO)
                        
                    # Now apply the crop ratio
                    target_height = int(max_possible_height * final_crop_ratio)
                    
                    # Ensure minimum dimensions
                    target_height = max(target_height, int(100 / VERTICAL_RATIO))
                    
                    # Calculate width to exactly match VERTICAL_RATIO
                    target_width = int(target_height * VERTICAL_RATIO)
                    
                    # Safety bounds check
                    target_width = min(target_width, frame_width)
                    target_height = min(target_height, frame_height)

                    # Debug output for cropping parameters with aspect ratio tracking
                    crop_ratio = target_height / frame_height if frame_height > 0 else 0
                    final_aspect_ratio = target_width / target_height if target_height > 0 else VERTICAL_RATIO
                    face_size_category = "small" if max(h, w) < target_face_size else "large"
                    
                    if zoom_in_enabled:
                        interval_frame = frame_count % switch_interval
                        zoom_progress = interval_frame / switch_interval if switch_interval > 0 else 0
                        if debug_faces:
                            print(f"Cropping: {target_width}x{target_height}, crop_ratio={crop_ratio:.3f}, aspect={final_aspect_ratio:.3f}, face={face_size_category}, zoom_mode={zoom_mode}, zoom_in_progress={zoom_progress:.2f}")
                    else:
                        if debug_faces:
                            print(f"Cropping: {target_width}x{target_height}, crop_ratio={crop_ratio:.3f}, aspect={final_aspect_ratio:.3f}, face={face_size_category}, zoom_mode={zoom_mode}")

                # Calculate the top-left corner of the 9:16 rectangle (only if we have face positions)
                if len(face_positions) > 0 and x is not None:
                    # Ideal top-left corner to center the face
                    crop_x = face_center[0] - target_width // 2
                    crop_y = face_center[1] - target_height // 2

                    # Shift the crop window if it goes out of bounds
                    if crop_x < 0:
                        crop_x = 0
                    elif crop_x + target_width > frame_width:
                        crop_x = frame_width - target_width
                        
                    if crop_y < 0:
                        crop_y = 0
                    elif crop_y + target_height > frame_height:
                        crop_y = frame_height - target_height
                        
                    # Final safety check in case target dimensions are larger than frame
                    crop_x = max(0, crop_x)
                    crop_y = max(0, crop_y)
                    
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
                                    if debug_faces:
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
                                    if debug_faces:
                                        print("⚠️  Face enhancement requested but enhancer not available, skipping...")

                    # Ensure proper aspect ratio preservation during resize
                    crop_h, crop_w = crop_img.shape[:2]
                    crop_aspect_ratio = crop_w / crop_h if crop_h > 0 else VERTICAL_RATIO
                    target_aspect_ratio = VERTICAL_RATIO
                    
                    if abs(crop_aspect_ratio - target_aspect_ratio) < 0.01:
                        # Aspect ratios are close enough, direct resize
                        resized = cv2.resize(
                            crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                        )
                    else:
                        # Aspect ratios don't match, need to pad or crop to prevent stretching
                        if debug_faces:
                            print(f"🔧 Aspect ratio correction: crop={crop_aspect_ratio:.3f} -> target={target_aspect_ratio:.3f}")
                        if crop_aspect_ratio > target_aspect_ratio:
                            # Crop is wider than 9:16, crop width to match
                            new_width = int(crop_h * target_aspect_ratio)
                            x_offset = (crop_w - new_width) // 2
                            crop_img = crop_img[:, x_offset:x_offset + new_width]
                            if debug_faces:
                                print(f"   Cropped width: {crop_w} -> {new_width}")
                        else:
                            # Crop is taller than 9:16, crop height to match  
                            new_height = int(crop_w / target_aspect_ratio)
                            y_offset = (crop_h - new_height) // 2
                            crop_img = crop_img[y_offset:y_offset + new_height, :]
                            if debug_faces:
                                print(f"   Cropped height: {crop_h} -> {new_height}")
                        
                        # Now resize the properly cropped image
                        resized = cv2.resize(
                            crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                        )

                    if (
                        thumb
                        and len(smile_candidates) < SMILE_CANDIDATE_LIMIT
                    ):
                        candidate_found, candidate_frame = find_smile(
                            frame[y : y + h, x : x + w], output_file
                        )
                        if candidate_found:
                            smile_candidates.append(candidate_frame)

                    out.write(resized)
                else:
                    # If no faces detected, use the largest possible crop.
                    # Story 11 (P3.4): when the clip HAS seen a face earlier,
                    # position that window on the last known face center so
                    # off-center speakers keep their framing through
                    # detection gaps; with no face history the helper keeps
                    # the previous hard-centered behavior exactly.
                    last_face_center = prev_face_center
                    crop_x, crop_y, crop_width, crop_height = compute_fallback_crop(
                        frame_width, frame_height, last_face_center, CROP_RATIO_BIG
                    )
                    crop_img = frame[
                        crop_y : crop_y + crop_height, crop_x : crop_x + crop_width
                    ]
                    resized = cv2.resize(
                        crop_img, (1080, 1920), interpolation=cv2.INTER_CUBIC
                    )
                    out.write(resized)

                # Store current frame for next iteration's movement detection
                prev_frame = frame.copy()

                frame_count += 1
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        if pipe_proc is not None:
            exit_code = pipe_proc.returncode
            # Fix B: a zero exit code with a failed frame pipe means ffmpeg
            # died mid-run and the output is truncated — it must fall back
            # to the legacy re-render instead of publishing as success.
            if exit_code is None or exit_code != 0 or out.failed:
                print(
                    f"Single-pass encode failed (exit code {exit_code}, "
                    f"frame pipe failed: {out.failed}); "
                    "falling back to legacy mp4v + merge path"
                )
                # The truncated partial must not survive as a stale
                # "success" if the fallback below fails.
                partial_output = f"tmp/final-{sanitized_output}"
                if os.path.exists(partial_output):
                    os.remove(partial_output)
                # Re-render through the legacy chain; the pipe consumed the
                # frames, so the legacy writer needs a fresh pass. The
                # recursive call skips the pipe entirely.
                return generate_short(
                    input_file,
                    output_file,
                    title,
                    upscale,
                    enhance,
                    thumb,
                    thumb_dir,
                    debug_faces,
                    performance_profile,
                    thumb_text=thumb_text,
                    _pipe=False,
                )
            # Success: the single pass already produced tmp/final-{output}.
        else:
            # Legacy encode chain (fallback / _pipe=False): extract audio
            # from the original video, then merge audio and processed video.
            # C3: this rung is reached whenever the single-pass pipe encoder
            # fails, and it produces the SAME tmp/final-* file the teaser burn
            # and the main burn both read — so its parameters reach the concat
            # whenever the no-overlay `-c copy` rung is taken. It must declare
            # the shared assembly shape like every other piece: the cv2 mp4v
            # writer runs at input_fps (often 60) with no pinned timescale and
            # an unpinned pixel format, and the audio extraction kept the
            # capture's 44100. Left alone, that is the original freeze, plus a
            # silent 0.40 s desync, re-introduced on the fallback path only.
            command = (
                f"ffmpeg -y -hwaccel cuda -i tmp/{shlex.quote(sanitized_input)} -vn "
                f"-c:a aac -b:a 192k -ar {ASSEMBLY_AUDIO_RATE} "
                f"-ac {ASSEMBLY_AUDIO_CHANNELS} tmp/output-audio.aac"
            )
            run_ffmpeg_command(command)

            command = (
                f"ffmpeg -y -hwaccel cuda -i tmp/{shlex.quote(sanitized_output)} "
                f"-i tmp/output-audio.aac "
                f"-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 "
                f"-b:v 10000k -maxrate:v 12000k -bufsize:v 15000k "
                f"-pix_fmt yuv420p -r {ASSEMBLY_FPS} "
                f"-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE} "
                f"-c:a copy tmp/final-{shlex.quote(sanitized_output)}"
            )
            run_ffmpeg_command(command)

        # Save performance report if profiling is enabled
        if profiler is not None:
            report_path = os.path.join(thumb_dir if thumb_dir else ".", "performance_report.json")
            profiler.save_report(report_path)
            profiler.print_summary()

        if thumb and smile_candidates:
            # Story 11 (P3.5): best face-quality candidate wins; hook text
            # (when present) is preferred over the raw title.
            smile_frame = select_best_smile_frame(smile_candidates)
            generate_thumbnail(smile_frame, thumb_text or title, thumb_dir)

        # Return the final merged output path so callers can report success
        return f"tmp/final-{sanitized_output}"

    except Exception as e:
        print(f"Error during video cropping: {str(e)}")
        # Fix C: never leak the NVENC child (it holds tmp/final-* open).
        # Closing stdin gives ffmpeg EOF; kill+wait guarantees the process
        # is gone and reaped. The legacy writer just needs releasing.
        if cap is not None:
            cap.release()
        if out is not None:
            if pipe_proc is not None:
                try:
                    pipe_proc.stdin.close()
                except Exception as close_error:
                    print(f"Warning: closing frame pipe failed: {close_error}")
                pipe_proc.kill()
                pipe_proc.wait()
            out.release()
        cv2.destroyAllWindows()


def build_segment_json_template():
    """Shared segment JSON template for BOTH generate_viral prompt paths.

    Placeholder values are kept plain (no parenthetical guidance inside
    values) so GPT cannot echo instructions into a real value, which
    parse_srt_time would reject. The hook_* fields are OPTIONAL at parse
    time: when GPT omits them (or they fail validation) the pipeline
    behaves exactly as before Story 7 (no cold-open teaser).
    """
    return """
    {
        "segments": [
            {
                "start_time": "00:00:00,000",
                "end_time": "00:00:00,000",
                "title": "Title of the reels",
                "description": "Description of the reels",
                "tags": "Three relevant and objective one-word hashtags",
                "hook_start_time": "00:00:05,000",
                "hook_end_time": "00:00:08,000",
                "hook_text": "Punchy 4-7 word hook"
            }
        ]
    }
    """


def request_json_completion(client, model, messages):
    """Chat completion preferring structured JSON output, with degradation.

    Both JSON-returning LLM calls (``generate_metadata`` and
    ``generate_viral``) ask the endpoint for a structured JSON object via
    ``response_format={"type": "json_object"}``. Custom OPENAI_BASE_URL
    endpoints may not support the parameter: when the call carrying it
    fails (TypeError/KeyError/400-class status error), retry ONCE without
    it. The caller's existing JSON-repair chain stays as the last-resort
    path for whatever comes back.
    """
    base_kwargs = {"model": model, "messages": messages, "n": 1, "stop": None}
    try:
        return client.chat.completions.create(
            response_format={"type": "json_object"}, **base_kwargs
        )
    except Exception as exc:
        print(
            f"Structured output rejected by endpoint ({type(exc).__name__}); "
            "retrying once without response_format"
        )
        return client.chat.completions.create(**base_kwargs)


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
        client = openai.OpenAI(api_key=api_key, base_url=openai_base_url)
        response = request_json_completion(
            client, os.getenv('OPENAI_MODEL', 'gpt-5-mini'), messages
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


def resolve_topic_boundary_llm_enabled():
    """S2 gate: TOPIC_BOUNDARY_LLM defaults to "true"; only the literal
    "false" (any case) disables the topic-start LLM pass."""
    enabled_raw = os.getenv('TOPIC_BOUNDARY_LLM', 'true').lower().strip()
    enabled = enabled_raw != 'false'
    if not enabled:
        print("Topic boundary LLM disabled (TOPIC_BOUNDARY_LLM=false); keeping optimized window unchanged")
    return enabled


def _srt_like_to_seconds(value):
    """Convert an SRT timestamp (HH:MM:SS,mmm) or bare seconds value into
    float seconds. Returns None when the value cannot be interpreted."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip() if value is not None else ""
    if not text:
        return None
    try:
        if "," in text or ":" in text:
            return parse_srt_time(text)
        return float(text)
    except (ValueError, TypeError):
        return None


def _segment_start_srt(segment):
    """Pick a printable start timestamp for a cue entry.

    TopicAnalyzer's segment dicts carry SRT strings under 'start_time_srt'
    and float seconds under 'start_time'; both shapes are handled so the
    cue list always shows a timestamp."""
    if not isinstance(segment, dict):
        return ""
    for key in ('start_time_srt', 'start_time'):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    start = segment.get('start_time')
    if isinstance(start, (int, float)):
        return format_timestamp(float(start))
    return ""


def _segment_start_seconds(segment):
    """Numeric start (float seconds) of a segment dict, or None."""
    if not isinstance(segment, dict):
        return None
    value = segment.get('start_time')
    if value is None:
        value = segment.get('start_time_srt')
    return _srt_like_to_seconds(value)


def _segment_end_seconds(segment):
    """Numeric end (float seconds) of a segment dict, or None."""
    if not isinstance(segment, dict):
        return None
    value = segment.get('end_time')
    if value is None:
        value = segment.get('end_time_srt')
    if value is None:
        value = segment.get('end')
    return _srt_like_to_seconds(value)


def _segment_end_srt(segment):
    """Pick a printable end timestamp for a cue entry.

    Mirror of _segment_start_srt for the END edge: SRT strings first, then
    a formatted float, then an empty string when nothing parses."""
    if not isinstance(segment, dict):
        return ""
    for key in ('end_time_srt', 'end_time'):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    end = segment.get('end_time')
    if isinstance(end, (int, float)):
        return format_timestamp(float(end))
    return ""


def _parse_hook_candidates(value):
    """Well-formed hook candidate dicts from the boundary reply (S5).

    Only the SHAPE is validated here (dict with int cue_start/cue_end and
    non-empty text); range/window/duration validation happens in
    select_hook_candidate. Malformed entries are skipped, never raising.
    """
    if not isinstance(value, list):
        return []
    parsed = []
    for item in value:
        try:
            if not isinstance(item, dict):
                continue
            cue_start = item.get('cue_start')
            cue_end = item.get('cue_end')
            text = item.get('text')
            if isinstance(cue_start, bool) or not isinstance(cue_start, int):
                continue
            if isinstance(cue_end, bool) or not isinstance(cue_end, int):
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            parsed.append(
                {
                    "cue_start": cue_start,
                    "cue_end": cue_end,
                    "text": text.strip(),
                }
            )
        except Exception:
            continue
    return parsed


def _parse_boundary_reply(content, cue_count):
    """Parse the boundary reply into (cue_index, topic_complete, earlier_start_cue, hook_candidates).

    - ``cue_index`` — the 1-based cue where the final topic begins, or None.
    - ``topic_complete`` — the model's own verdict on whether ``cue_index``
      captures the whole topic, or None when the reply carried no verdict.
      Kept separate from ``earlier_start_cue`` because the two can disagree:
      an incomplete topic whose earlier cue was rejected still has
      ``earlier_start_cue is None``, and the observability log (A5) must
      report the model's actual answer rather than an inferred one.
    - ``earlier_start_cue`` — when the topic is incomplete, the cue where it
      actually begins (earlier than ``cue_index``); else None. Callers only
      extend the window when it is set, so a legacy or malformed reply
      degrades silently.
    - ``hook_candidates`` — a possibly empty list of cue dicts.

    The cue number and the hook candidates are parsed INDEPENDENTLY: a
    missing/invalid candidates list never invalidates the cue number and
    vice versa. A malformed reply with no parseable cue yields
    ``(None, None, None, [])`` so the pipeline degrades to today's untrimmed
    window.
    """
    cue_index = None
    topic_complete = None
    earlier_start_cue = None
    hook_candidates = []
    text = str(content) if content is not None else ""
    text = text.replace("```json", "").replace("```", "").strip()
    payload = None
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict):
        raw_cue = payload.get('cue_number')
        if (
            isinstance(raw_cue, int)
            and not isinstance(raw_cue, bool)
            and 1 <= raw_cue <= cue_count
        ):
            cue_index = raw_cue
        earlier_start_cue = _parse_earlier_start_cue(payload, cue_index, cue_count)
        if isinstance(payload.get('topic_complete'), bool):
            topic_complete = payload['topic_complete']
        hook_candidates = _parse_hook_candidates(payload.get('hook_candidates'))
    else:
        match = re.search(r"-?\d+", text)
        if match:
            try:
                value = int(match.group())
                if 1 <= value <= cue_count:
                    cue_index = value
            except ValueError:
                cue_index = None
    return cue_index, topic_complete, earlier_start_cue, hook_candidates


def _parse_earlier_start_cue(payload, cue_index, cue_count):
    """Validate the earlier-cue answer; return an earlier cue index or None.

    ``earlier_start_cue`` is honored whenever the model supplies a usable
    index strictly before ``cue_index``, whether or not it also set
    ``topic_complete`` to a literal false. It used to require that boolean,
    which silently dropped the answer whenever the model named an earlier cue
    and omitted (or mangled) the redundant flag — the very reply shape that
    carries the most actionable signal, and the one behind a clip starting
    mid-topic. An explicit, in-range, strictly-earlier cue is the stronger and
    more specific instruction, and honouring it only ever moves the window
    earlier (bounded by the duration cap). Every other shape — a missing cue,
    a non-integer, an out-of-range or later index — returns None.

    The ``topic_complete`` VERDICT itself is parsed separately and still only
    accepts a literal false, so the observability log never reports a
    completeness the model did not give.
    """
    if not isinstance(payload, dict):
        return None
    earlier = payload.get('earlier_start_cue')
    if (
        isinstance(earlier, int)
        and not isinstance(earlier, bool)
        and 1 <= earlier <= cue_count
        and cue_index is not None
        and earlier < cue_index
    ):
        return earlier
    return None


def resolve_topic_start_cue(transcript_content, segments):
    """S2/S3/S5: ask the LLM where the FINAL topic begins + hook candidates.

    Replay-buffer captures are <= 90 s and the final topic concludes at the
    end, but the capture's start often holds the PREVIOUS topic's tail. The
    numbered cue list below covers the full capture (nothing is truncated)
    and the model must answer with the first cue that belongs to the final
    topic; on top of that cue number it proposes up to 3 hook candidates
    inside the final topic (S5 hook-first selection). It also reports whether
    that cue already captures the whole topic, and if not an earlier cue where
    the topic actually begins (S3 completeness signal).

    Returns (cue_index, topic_complete, earlier_start_cue, hook_candidates):
    cue_index is the 1-based cue index or None on any failure (exception,
    unparseable reply, out-of-range index) so the pipeline degrades to today's
    untrimmed window; topic_complete is the model's own completeness verdict,
    or None when the reply carried none; earlier_start_cue is an earlier valid
    cue index when the topic is incomplete, else None; hook_candidates is a
    possibly empty list of {"cue_start", "cue_end", "text"} dicts. A
    missing/invalid candidates list never invalidates the cue number and vice
    versa. Never raises.
    """
    if not segments:
        print("Topic boundary LLM: no segments available; keeping window unchanged")
        return None, None, None, []
    if not resolve_topic_boundary_llm_enabled():
        return None, None, None, []

    # Cues come from the analyzed segments, which span the entire capture.
    cue_lines = []
    for index, segment in enumerate(segments, start=1):
        text = segment.get('text', '') if isinstance(segment, dict) else ''
        cue_lines.append(f"cue {index}: {_segment_start_srt(segment)} {text}")
    prompt = f"""The following numbered cues cover a short video capture from its start to its end (nothing was cut):

{chr(10).join(cue_lines)}

The FINAL topic of this capture CONCLUDES at the end of the capture. Any cue text at the very beginning may still belong to the PREVIOUS topic.

Return ONLY a JSON object (no markdown, no code block markers) with EXACTLY this shape:

{{"topic_complete": <bool>, "cue_number": <integer>, "earlier_start_cue": <integer or null>, "hook_candidates": [{{"cue_start": <integer>, "cue_end": <integer>, "text": "<hook sentence>"}}]}}

- "topic_complete": true when the cue you pick for "cue_number" already captures the whole final topic; false when it does NOT — the topic actually began earlier than that cue and context is missing.
- "cue_number": the cue number where the FINAL topic BEGINS — the first cue that belongs to the final topic, not the previous topic's tail.
- "earlier_start_cue": only when "topic_complete" is false, the earlier cue number where the final topic actually BEGINS (an integer strictly less than "cue_number"); otherwise null.
- "hook_candidates": up to 3 hook candidates INSIDE the final topic, best first. Each candidate is a self-contained, curiosity-driving stretch of 3-9 seconds: "cue_start" is the first cue number of that stretch, "cue_end" is the last cue number of it, and "text" is the hook wording verbatim. Cues here run several seconds each, so aim for a stretch of 1-3 consecutive cues and ALWAYS return a span that is at least 3 seconds and at most 9 seconds long — a shorter span cannot be used as a cold open. A candidate MUST NOT be the first sentence of the final topic (the clip already opens there, so a hook there only repeats it) — prefer an engaging stretch from mid-topic instead. Use an empty list when no clear hook exists."""

    system = """You identify topic boundaries in video transcripts and propose hook candidates. You receive numbered cues covering a full short capture and must report whether the final topic is captured in full, the cue number where the final topic begins, an earlier cue when it is not, plus hook candidates inside the final topic."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    try:
        print(f"Topic boundary LLM: resolving final-topic start across {len(segments)} cues...")
        client = openai.OpenAI(api_key=api_key, base_url=openai_base_url)
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-5-mini'), messages=messages, n=1, stop=None
        )

        content = response.choices[0].message.content
        cue_index, topic_complete, earlier_start_cue, hook_candidates = (
            _parse_boundary_reply(content, len(segments))
        )
        if cue_index is None:
            print(
                f"Topic boundary LLM: unparseable or out-of-range reply "
                f"\"{str(content)[:80]}\"; keeping window unchanged"
            )
        else:
            print(f"Topic boundary LLM: final topic begins at cue {cue_index}")
        if earlier_start_cue is not None:
            print(
                f"Topic boundary LLM: topic incomplete — it actually begins "
                f"earlier at cue {earlier_start_cue}"
            )
        if hook_candidates:
            print(
                f"Topic boundary LLM: {len(hook_candidates)} hook candidate(s) proposed"
            )
        return cue_index, topic_complete, earlier_start_cue, hook_candidates
    except Exception as e:
        print(f"Topic boundary LLM pass failed ({e}); keeping window unchanged")
        return None, None, None, []


def trim_window_to_topic_start(
    cue_index,
    optimized_segment,
    supporting_segments,
    segments,
    min_duration=None,
    max_duration=None,
):
    """Move the optimized window start to the resolved final-topic cue.

    The end time and concluding topic are untouched — the recency anchor
    (the final topic still concludes near the capture's end) is preserved.

    The cue is applied in EITHER direction. It used to be applied only when it
    started strictly AFTER the window, which silently threw away the boundary
    model's most valuable answer: "the final topic actually begins earlier than
    your window". That left the previous topic's tail inside the clip — the
    exact symptom of the topic not being identified from the capture start. A
    cue at the window start is still a no-op.

    The window stays inside its bounds whichever way the start moves:
    ``min_duration`` (seconds, None = no floor) stops a cue near the capture end
    from collapsing the clip, and ``max_duration`` (None = no cap) stops an
    early cue from growing the clip past the cap. Both clamp the start and are
    reported. The start is never before 0.0 s.

    Moving the start EARLIER only grows the window, so supporting segments are
    kept as-is; moving it LATER drops the ones that begin before the new start.
    Returns the (optimized, supporting) pair.
    """
    if not isinstance(cue_index, int) or isinstance(cue_index, bool):
        return optimized_segment, supporting_segments
    if not (1 <= cue_index <= len(segments)):
        return optimized_segment, supporting_segments

    cue = segments[cue_index - 1]
    cue_seconds = _segment_start_seconds(cue)
    window_seconds = _srt_like_to_seconds(optimized_segment.get('start_time'))
    end_seconds = _srt_like_to_seconds(optimized_segment.get('end_time'))
    if cue_seconds is None or window_seconds is None or end_seconds is None:
        print("Topic boundary LLM: could not compare cue/window timestamps; keeping window unchanged")
        return optimized_segment, supporting_segments
    if abs(cue_seconds - window_seconds) < 1e-9:
        print(f"Topic boundary LLM: cue {cue_index} is already the window start; keeping window unchanged")
        return optimized_segment, supporting_segments

    moving_earlier = cue_seconds < window_seconds
    new_start = max(0.0, cue_seconds)
    if (
        isinstance(min_duration, (int, float))
        and not isinstance(min_duration, bool)
        and end_seconds - new_start < float(min_duration)
    ):
        clamped = max(0.0, end_seconds - float(min_duration))
        print(
            f"Topic boundary trim: cue {cue_index} at {new_start:.1f}s would leave "
            f"{end_seconds - new_start:.1f}s, under the {float(min_duration):.0f}s "
            f"floor; clamping the window start to {clamped:.1f}s"
        )
        new_start = clamped
    # The cap is enforced here as well as in
    # extend_window_to_earlier_topic_start: moving the start earlier is now
    # reachable from this call, and without the cap an early cue would produce
    # a clip longer than max_duration.
    if (
        isinstance(max_duration, (int, float))
        and not isinstance(max_duration, bool)
        and end_seconds - new_start > float(max_duration)
    ):
        clamped = max(0.0, end_seconds - float(max_duration))
        print(
            f"Topic boundary trim: cue {cue_index} at {new_start:.1f}s would exceed "
            f"the {float(max_duration):.0f}s cap; clamping the window start to {clamped:.1f}s"
        )
        new_start = clamped

    previous_start = optimized_segment.get('start_time')
    if abs(new_start - cue_seconds) < 1e-9:
        optimized_segment['start_time'] = _segment_start_srt(cue)
    else:
        optimized_segment['start_time'] = format_timestamp(round(new_start, 3))
    optimized_segment['duration'] = end_seconds - new_start

    if moving_earlier:
        # The window only grew, so every supporting segment is still inside it.
        direction = "extend"
        trimmed_supporting = list(supporting_segments)
    else:
        direction = "trim"
        trimmed_supporting = []
        for segment in supporting_segments:
            segment_seconds = _segment_start_seconds(segment)
            if segment_seconds is None or segment_seconds >= new_start:
                trimmed_supporting.append(segment)

    print(f"✂️ Topic boundary {direction}: window start {previous_start} → {optimized_segment['start_time']} (cue {cue_index}); supporting segments {len(supporting_segments)} → {len(trimmed_supporting)}")
    return optimized_segment, trimmed_supporting


def extend_window_to_earlier_topic_start(
    earlier_start_cue, optimized_segment, supporting_segments, segments, max_duration=59.0
):
    """Move the optimized window start EARLIER to the reported earlier cue.

    Sibling of ``trim_window_to_topic_start``, moving in the opposite
    direction: when the boundary LLM pass says the final topic is
    incomplete, the earlier cue it names is where the topic actually
    begins, so the clip must carry that context.

    The end time and concluding topic are untouched — the recency anchor
    (the final topic still concludes near the capture's end) is preserved.
    The window never grows past ``max_duration``: because the end stays
    anchored, a start that would breach the cap is clamped to
    ``end - max_duration``. A cue that does not start strictly before the
    current window start leaves the window unchanged. A cue that would start
    before the capture start is floored to 0.0 s. Because the extension
    only ever GROWS the window (the start moves earlier, the end never
    moves), it never drops a supporting segment: every segment that is still
    on the list is inside the new, larger window, so the list is returned
    unchanged.
    Returns the (optimized, supporting) pair.
    """
    if not isinstance(earlier_start_cue, int) or isinstance(earlier_start_cue, bool):
        return optimized_segment, supporting_segments
    if not (1 <= earlier_start_cue <= len(segments)):
        return optimized_segment, supporting_segments

    cue = segments[earlier_start_cue - 1]
    cue_seconds = _segment_start_seconds(cue)
    window_seconds = _srt_like_to_seconds(optimized_segment.get('start_time'))
    end_seconds = _srt_like_to_seconds(optimized_segment.get('end_time'))
    if cue_seconds is None or window_seconds is None or end_seconds is None:
        print("Topic boundary LLM: could not compare cue/window timestamps; keeping window unchanged")
        return optimized_segment, supporting_segments
    if cue_seconds >= window_seconds:
        print(f"Topic boundary LLM: earlier cue {earlier_start_cue} does not start before the window start; keeping window unchanged")
        return optimized_segment, supporting_segments

    new_start_seconds = max(0.0, cue_seconds)
    floored = new_start_seconds != cue_seconds
    capped = end_seconds - new_start_seconds > max_duration
    if capped:
        new_start_seconds = max(0.0, end_seconds - max_duration)
        print(
            f"Topic boundary extend: earlier cue {earlier_start_cue} would exceed the "
            f"{max_duration:.0f}s cap; clamping the window start to {end_seconds - max_duration:.1f}s"
        )

    previous_start = optimized_segment.get('start_time')
    optimized_segment['start_time'] = (
        _segment_start_srt(cue)
        if not (capped or floored)
        else format_timestamp(round(new_start_seconds, 3))
    )
    optimized_segment['duration'] = end_seconds - new_start_seconds

    print(f"✂️ Topic boundary extend: window start {previous_start} → {optimized_segment['start_time']} (earlier cue {earlier_start_cue})")
    return optimized_segment, supporting_segments


def _cue_bounds(cue):
    """(start, end) float seconds of a TopicAnalyzer cue dict, or (None, None).

    Cue dicts are dual-shape: floats under 'start_time'/'end_time' (or bare
    'start'/'end'), SRT strings under 'start_time_srt'/'end_time_srt'."""
    if not isinstance(cue, dict):
        return None, None
    start = _srt_like_to_seconds(cue.get('start_time'))
    if start is None:
        start = _srt_like_to_seconds(cue.get('start_time_srt'))
    if start is None:
        start = _srt_like_to_seconds(cue.get('start'))
    end = _srt_like_to_seconds(cue.get('end_time'))
    if end is None:
        end = _srt_like_to_seconds(cue.get('end_time_srt'))
    if end is None:
        end = _srt_like_to_seconds(cue.get('end'))
    return start, end


def _cue_containing(segments, when):
    """First cue whose [start, end) span contains `when`, or None."""
    for cue in segments:
        start, end = _cue_bounds(cue)
        if start is not None and end is not None and start <= when < end:
            return cue
    return None


def _last_sentence_start_offset(text):
    """Character offset just after the LAST sentence terminator that still
    leaves a non-empty remainder, or None."""
    for index in range(len(text) - 2, -1, -1):
        if text[index] in '.!?':
            if text[index + 1:].strip():
                return index + 1
    return None


def _last_terminator_offset(text):
    """Character offset just after the LAST sentence terminator in `text`
    (a terminator may be the final character), or None."""
    for index in range(len(text) - 1, -1, -1):
        if text[index] in '.!?':
            return index + 1
    return None


def _offset_to_seconds(cue_start, cue_end, text, offset):
    """Map a character offset inside a cue's text onto its time span.

    Whisper cues carry no word timestamps here, so the offset is placed
    proportionally across the cue's duration."""
    cue_duration = cue_end - cue_start
    if cue_duration <= 0 or not text:
        return None
    return cue_start + (offset / len(text)) * cue_duration


def snap_window_to_boundaries(window, segments, min_start_srt=None, min_duration=35.0):
    """S3: refine an optimized window using transcript TEXT only (pure).

    - START: within the cue containing the window start, move the start to
      just after the LAST sentence terminator that still leaves a non-empty
      remainder (open at a sentence start). The start NEVER moves earlier
      than the window start, nor earlier than `min_start_srt` (the Task-13
      topic-start cue's start; None = unbounded).
    - END: within the cue containing the window end, move the end to just
      after the last sentence terminator inside that cue (trim a trailing
      partial sentence), but never below start + `min_duration`; if that
      guard binds, the edge is left unchanged. The end never moves later.
    - Any parse failure returns the input unchanged. Never raises.

    Returns the refined window dict (a copy) or the input when unchanged.
    """
    try:
        if not isinstance(window, dict) or not segments:
            return window
        start_seconds = _srt_like_to_seconds(window.get('start_time'))
        end_seconds = _srt_like_to_seconds(window.get('end_time'))
        if start_seconds is None or end_seconds is None:
            return window

        new_start = start_seconds
        start_cue = _cue_containing(segments, start_seconds)
        if start_cue is not None:
            cue_start, cue_end = _cue_bounds(start_cue)
            text = str(start_cue.get('text') or '')
            offset = _last_sentence_start_offset(text)
            if offset is not None and cue_start is not None and cue_end is not None:
                candidate = _offset_to_seconds(cue_start, cue_end, text, offset)
                if candidate is not None and start_seconds < candidate < end_seconds:
                    new_start = candidate

        min_start_seconds = _srt_like_to_seconds(min_start_srt) if min_start_srt is not None else None
        if min_start_seconds is not None and new_start < min_start_seconds:
            new_start = min_start_seconds

        new_end = end_seconds
        end_cue = _cue_containing(segments, end_seconds)
        if end_cue is not None:
            cue_start, cue_end = _cue_bounds(end_cue)
            text = str(end_cue.get('text') or '')
            offset = _last_terminator_offset(text)
            if offset is not None and cue_start is not None and cue_end is not None:
                candidate = _offset_to_seconds(cue_start, cue_end, text, offset)
                if (
                    candidate is not None
                    and new_start < candidate < end_seconds
                    and candidate >= new_start + min_duration
                ):
                    new_end = candidate

        if abs(new_start - start_seconds) < 1e-9 and abs(new_end - end_seconds) < 1e-9:
            return window
        snapped = dict(window)
        snapped['start_time'] = format_timestamp(new_start)
        snapped['end_time'] = format_timestamp(new_end)
        snapped['duration'] = new_end - new_start
        print(
            f"✂️ Boundary snap: start {window.get('start_time')} → {snapped['start_time']}, "
            f"end {window.get('end_time')} → {snapped['end_time']}"
        )
        return snapped
    except Exception as e:
        print(f"Boundary snap failed ({e}); keeping window unchanged")
        return window


def build_sentence_spans(transcript_segments):
    """Sentence units with exact word-timestamp bounds (pure).

    Whisper cues are short (~2 s on a real 90 s capture, measured 0.92-3.5 s)
    and routinely break mid-sentence, so a hook cut on cue boundaries starts
    and ends mid-thought. The transcribers already emit word-level timestamps
    for every word, so the real sentence boundaries are recoverable: walk the
    flattened word stream and close a sentence after each word whose text ends
    one. A trailing partial sentence is flushed too, so the last one is not
    lost.

    Returns a chronological list of {"index" (1-based), "start", "end",
    "text"}, or [] when the transcript carries no usable word data so callers
    fall back to the cue-based path. Never raises.
    """
    sentences = []
    pending = []

    def _close():
        if not pending:
            return
        start = pending[0]["start"]
        end = pending[-1]["end"]
        text = " ".join(word["word"] for word in pending).strip()
        pending.clear()
        if text and end > start:
            sentences.append(
                {
                    "index": len(sentences) + 1,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

    try:
        for segment in transcript_segments or []:
            if not isinstance(segment, dict):
                continue
            for word in segment.get("words") or []:
                if not isinstance(word, dict):
                    continue
                try:
                    start = float(word.get("start"))
                    end = float(word.get("end"))
                except (TypeError, ValueError):
                    continue
                text = str(word.get("word") or "").strip()
                if not text or end <= start:
                    continue
                pending.append({"word": text, "start": start, "end": end})
                if text[-1] in ".!?":
                    _close()
        _close()
    except Exception as e:
        print(f"Sentence span build failed ({e}); keeping the cue-based hook")
        return []
    return sentences


def _sentence_hook_overlap_score(sentence, hook_text):
    """How much of `hook_text` a sentence covers, as a 0-1 ratio.

    Pure: the ratio of hook words present in the sentence, so the sentence the
    model actually quoted outranks one that merely overlaps in time."""
    try:
        wanted = {word for word in _HOOK_WORD_RE.findall(hook_text or "" if isinstance(hook_text, str) else "")}
        if not wanted:
            return 0.0
        have = set(_HOOK_WORD_RE.findall(sentence.get("text") or ""))
        return len(wanted & have) / len(wanted)
    except Exception:
        return 0.0


def refine_hook_span_to_sentences(
    hook_start,
    hook_end,
    hook_text,
    sentences,
    min_duration=None,
    max_duration=None,
    window_end=None,
):
    """Snap a cue-derived hook span onto real sentence boundaries (pure).

    The cue span is only an approximation of the hook — it starts wherever the
    cue started, mid-sentence. This picks the sentence inside that span which
    best covers the hook wording, then extends forward across WHOLE following
    sentences until the hook is long enough to read as a cold open. Every
    candidate span is therefore sentence-aligned on both edges.

    Returns (start, end) in float seconds, or the input span unchanged when
    nothing better is available: no sentences, no sentence overlapping the
    span, or no sentence-aligned span that satisfies the duration band. Never
    raises.
    """
    try:
        if not sentences or hook_start is None or hook_end is None:
            return hook_start, hook_end
        if hook_end <= hook_start:
            return hook_start, hook_end
        band_min = min_duration if min_duration else HOOK_TEASER_MIN_DURATION
        band_max = max_duration if max_duration else HOOK_TEASER_MAX_DURATION
        target = min(band_max, HOOK_TARGET_DURATION)

        overlapping = [
            sentence
            for sentence in sentences
            if sentence["end"] > hook_start + 1e-9
            and sentence["start"] < hook_end - 1e-9
            # A sentence that runs past the resolved window would put hook
            # audio outside the clip, so it is not a candidate at all.
            and (window_end is None or sentence["end"] <= window_end + 1e-9)
        ]
        if not overlapping:
            return hook_start, hook_end

        def _rank(sentence):
            shared = min(hook_end, sentence["end"]) - max(hook_start, sentence["start"])
            return (_sentence_hook_overlap_score(sentence, hook_text), shared)

        chosen = max(overlapping, key=_rank)
        position = sentences.index(chosen)

        # A run of whole sentences, grown only while it stays inside the band
        # and inside the resolved window.
        run_start, run_end, cursor = chosen["start"], chosen["end"], position
        while (run_end - run_start) < target - 1e-9 and cursor + 1 < len(sentences):
            following = sentences[cursor + 1]
            if window_end is not None and following["end"] > window_end + 1e-9:
                break
            if (following["end"] - run_start) > band_max + 1e-9:
                break
            run_end = following["end"]
            cursor += 1

        for candidate_start, candidate_end in (
            (run_start, run_end),
            (chosen["start"], chosen["end"]),
        ):
            if (
                candidate_end - candidate_start >= band_min - 1e-9
                and candidate_end - candidate_start <= band_max + 1e-9
            ):
                return candidate_start, candidate_end
        return hook_start, hook_end
    except Exception as e:
        print(f"Hook span refinement failed ({e}); keeping the cue-based span")
        return hook_start, hook_end


def select_hook_candidate(
    candidates, window_start_srt, window_end_srt, segments, sentences=None
):
    """S5: first hook candidate inside the resolved window (pure).

    Candidates come from the boundary LLM pass and reference cues by their
    1-based numbers ("cue_start"/"cue_end"), so the cue→seconds mapping
    walks `segments`. A candidate survives when its span (start of cue_start
    to end of cue_end) falls inside [window_start_srt, window_end_srt] and
    spans HOOK_TEASER_MIN_DURATION..HOOK_TEASER_MAX_DURATION seconds. The
    model ranks the candidates, so the FIRST survivor wins.

    B1: a candidate overlapping the first HOOK_DUPLICATE_GUARD_SECONDS of
    the resolved window is rejected even though it passes every other check.
    The teaser is prepended to the clip, so a candidate drawn from the clip's
    own opening replays the first sentence the viewer is about to hear; the
    window starts at the topic start, which is exactly where the LLM's
    top-ranked candidate tends to sit. With no survivor the teaser is simply
    skipped. Malformed candidates are skipped, never raising. Returns
    {"hook_start_time", "hook_end_time", "hook_text"} (SRT + text) or None.

    B2: surviving candidates are PREFERRED when they land in the middle band
    of the window — from window_start + HOOK_MID_BAND_START_FRACTION to
    window_start + HOOK_MID_BAND_END_FRACTION of its duration, inclusive.
    A teaser at the very start leads with a truncated thought and one at the
    very end pre-empts the payoff; a mid-clip teaser adds information
    instead. This is a preference, not a filter: with no mid-band survivor
    the first survivor in the LLM's ranking order is returned, exactly as
    before B2. Both tiers keep the ranking order, so "first" always means
    first in the candidate list.
    """
    try:
        if not isinstance(candidates, list) or not segments:
            return None
        window_start = _srt_like_to_seconds(window_start_srt)
        window_end = _srt_like_to_seconds(window_end_srt)
        if window_start is None or window_end is None or window_end <= window_start:
            return None
        # Invariants of the window, computed once (B2) rather than per candidate.
        window_duration = window_end - window_start
        guard_end = window_start + HOOK_DUPLICATE_GUARD_SECONDS
        band_start = window_start + HOOK_MID_BAND_START_FRACTION * window_duration
        band_end = window_start + HOOK_MID_BAND_END_FRACTION * window_duration
        first_survivor = None
        for candidate in candidates:
            try:
                if not isinstance(candidate, dict):
                    continue
                cue_start = candidate.get('cue_start')
                cue_end = candidate.get('cue_end')
                text = candidate.get('text')
                if isinstance(cue_start, bool) or not isinstance(cue_start, int):
                    continue
                if isinstance(cue_end, bool) or not isinstance(cue_end, int):
                    continue
                if not isinstance(text, str) or not text.strip():
                    continue
                if not (1 <= cue_start <= len(segments)) or not (
                    1 <= cue_end <= len(segments)
                ):
                    continue
                span_start, _ = _cue_bounds(segments[cue_start - 1])
                _, span_end = _cue_bounds(segments[cue_end - 1])
                if span_start is None or span_end is None:
                    continue
                if span_end <= span_start:
                    continue
                if span_start < window_start - 1e-9 or span_end > window_end + 1e-9:
                    continue
                duration = span_end - span_start
                if (
                    duration < HOOK_TEASER_MIN_DURATION - 1e-9
                    or duration > HOOK_TEASER_MAX_DURATION + 1e-9
                ):
                    continue
                if span_start < guard_end - 1e-9:
                    continue
                survivor = {
                    "hook_start_time": _segment_start_srt(segments[cue_start - 1])
                    or format_timestamp(span_start),
                    "hook_end_time": _segment_end_srt(segments[cue_end - 1])
                    or format_timestamp(span_end),
                    "hook_text": text.strip(),
                }
                # Snap the cue-derived span onto real sentence boundaries when
                # word timestamps are available, so the teaser neither opens nor
                # closes mid-sentence. Falls back to the cue span unchanged.
                if sentences:
                    refined_start, refined_end = refine_hook_span_to_sentences(
                        span_start,
                        span_end,
                        text,
                        sentences,
                        min_duration=HOOK_TEASER_MIN_DURATION,
                        max_duration=HOOK_TEASER_MAX_DURATION,
                        window_end=window_end,
                    )
                    if (refined_start, refined_end) != (span_start, span_end):
                        survivor["hook_start_time"] = format_timestamp(
                            round(refined_start, 3)
                        )
                        survivor["hook_end_time"] = format_timestamp(
                            round(refined_end, 3)
                        )
                # B2: the first mid-band survivor wins outright; an edge
                # survivor is only kept as the fallback.
                if band_start - 1e-9 <= span_start <= band_end + 1e-9:
                    return survivor
                if first_survivor is None:
                    first_survivor = survivor
            except Exception:
                continue
        return first_survivor
    except Exception as e:
        print(f"Hook candidate selection failed ({e}); keeping the hook optional")
        return None


def validate_segment_against_window(
    segment, resolved_window, min_duration=35.0, max_duration=59.0
):
    """S4: deterministic validation of the LLM-returned segment (pure).

    With a usable `resolved_window` (the heuristic + trimmed + snapped
    window from the primary path) the returned start/end are clamped into
    it and the duration recomputed; a clamped duration outside
    [min_duration, max_duration] REPLACES the segment's times with the
    resolved window's times — the resolved window is trusted over an
    out-of-range LLM answer.

    Without a usable resolved window (fallback path, gate off, LLM pass
    failed) only the duration-range check applies: an out-of-range segment
    is left unchanged with a warning note (pre-S4 behavior).

    Any unparseable returned time leaves the segment unchanged (downstream
    parsers already tolerate that). Pure: no I/O; never raises.

    Returns (segment, notes) — the (possibly corrected) segment plus a
    list of human-readable notes describing every correction/warning.
    """
    notes = []
    try:
        if not isinstance(segment, dict):
            return segment, ["S4 validation: segment is not a dict; left unchanged"]
        start_seconds = _srt_like_to_seconds(segment.get('start_time'))
        end_seconds = _srt_like_to_seconds(segment.get('end_time'))
        if start_seconds is None or end_seconds is None:
            return segment, [
                "S4 validation: returned segment times are unparseable; leaving unchanged"
            ]

        window_start = None
        window_end = None
        if isinstance(resolved_window, dict):
            window_start = _srt_like_to_seconds(resolved_window.get('start_time'))
            window_end = _srt_like_to_seconds(resolved_window.get('end_time'))
        has_window = (
            window_start is not None
            and window_end is not None
            and window_end > window_start
        )

        if has_window:
            clamped_start = min(max(start_seconds, window_start), window_end)
            clamped_end = min(max(end_seconds, window_start), window_end)
            duration = clamped_end - clamped_start
            if duration < min_duration - 1e-9 or duration > max_duration + 1e-9:
                corrected = dict(segment)
                corrected['start_time'] = resolved_window['start_time']
                corrected['end_time'] = resolved_window['end_time']
                corrected['duration'] = window_end - window_start
                notes.append(
                    f"S4 validation: clamped duration {duration:.1f}s is outside "
                    f"[{min_duration}, {max_duration}]s; replacing segment with the "
                    f"resolved window {resolved_window['start_time']} --> "
                    f"{resolved_window['end_time']}"
                )
                return corrected, notes
            if (
                abs(clamped_start - start_seconds) > 1e-9
                or abs(clamped_end - end_seconds) > 1e-9
            ):
                corrected = dict(segment)
                corrected['start_time'] = format_timestamp(clamped_start)
                corrected['end_time'] = format_timestamp(clamped_end)
                corrected['duration'] = duration
                notes.append(
                    f"S4 validation: segment clamped into the resolved window "
                    f"({corrected['start_time']} --> {corrected['end_time']}, "
                    f"{duration:.1f}s)"
                )
                return corrected, notes
            return segment, notes

        # Fallback path: no usable resolved window — duration check only.
        duration = end_seconds - start_seconds
        if duration < min_duration - 1e-9 or duration > max_duration + 1e-9:
            notes.append(
                f"Warning: no resolved window; returned segment duration "
                f"{duration:.1f}s is outside [{min_duration}, {max_duration}]s; "
                f"leaving unchanged"
            )
        return segment, notes
    except Exception as e:
        return segment, [f"S4 validation failed ({e}); leaving segment unchanged"]


def _first_sentence_ends_complete(text):
    """True when `text` opens with a sentence that ends in . / ! / ?.

    The ratified rule is literally "the first sentence ends with a
    terminator", and the scan is flat: a terminator anywhere in the text ends
    the first sentence, so "One. Two." opens with a finished sentence even
    though it goes on.

    KNOWN BIAS, deliberately not "fixed" (the rule was ratified as-is; these
    are false positives that run SYSTEMATICALLY toward `complete`, and this
    fact is the tuning signal for A1-A4):

      * decimals — "the matrix is 3.14 wide and it keeps going" is reported
        complete. Render/settings screencasts mention numbers constantly, so
        this is the common case, not an edge case.
      * abbreviations — "e.g." reads as a terminator.
      * ellipses — "the pipeline is slow... and then it dropped" is reported
        complete.

    Tests in tests/test_selection_summary.py pin these; if one flips, the rule
    changed on purpose, not by accident.
    """
    return any(char in '.!?' for char in text)


def _cue_opens_new_sentence(cue):
    """True when `cue`'s text visibly starts a new sentence.

    Used as the second branch of the first-caption heuristic: Whisper cues
    frequently carry no terminal punctuation, so a following cue that opens
    with a capital is evidence the previous sentence ended at the cue
    boundary.

    Only a capital letter counts. Quotes and mid-sentence continuation marks
    (`,)]}` and the dashes) are NOT new-sentence openers: a cue beginning
    ", which is why the render stalled" continues the previous sentence, and
    treating it as a fresh one claims a sentence finished when it did not.
    """
    if not isinstance(cue, dict):
        return False
    text = str(cue.get('text') or '').lstrip()
    if not text:
        return False
    first = text[0]
    return first.isalpha() and first.isupper()


def build_selection_summary(
    segments, resolved_window, chosen_segment, capture_span, topic_complete
):
    """A5: the five completeness facts about the chosen clip (pure).

    Returns a dict with exactly:

      capture_span                     float seconds, or None when unknown
      resolved_window                  {"start", "end"} SRT strings, or None
                                       (None on the fallback path, where no
                                       window was resolved)
      chosen_duration                  float seconds, or None
      topic_complete                   the boundary LLM's own verdict, passed
                                       through verbatim (True / False / None)
      first_caption_sentence_complete  True / False / None

    `segments` is the transcript cue list in either dual shape (floats under
    start_time/end_time with SRT strings under start_time_srt/end_time_srt, or
    the SRT-fallback shape with strings under start_time/end_time and bare
    floats under start/end); `_cue_bounds` absorbs both.

    Cues are selected by OVERLAP, not full containment, with the resolved
    window: `snap_window_to_boundaries` can land the window start mid-cue, and
    the cue that OWNS the window start begins before the window does — a
    containment filter would drop exactly the cue whose first sentence the
    check is about.

    The first-caption verdict is the simple heuristic: the first in-window
    cue's first sentence is complete when it ends in . / ! / ?, or when the
    following cue's text opens a new sentence. No auxiliary LLM call, no
    capitalization or pause heuristics. With no window, no overlapping cue,
    or no text there is nothing to judge and the fact is reported as None
    rather than guessed.

    `chosen_duration` is recomputed from the chosen segment's times, which are
    always the post-validation ones. `validate_segment_against_window` only
    sets `duration` when it CORRECTS a segment, so an uncorrected one may
    carry whatever the LLM put there (or none at all); the segment's own
    `duration` is used only as a last resort.

    Read `chosen_duration` as the PRE-CUT duration: the length of the window
    the cutter starts from. It is NOT the length of the published file, and
    the two routinely differ in both directions. Cutting happens after this
    summary is built — jump cuts (JUMP_CUTS_ENABLED, on by default) remove
    pause material of 0.6 s or more, so a screencast with pauses publishes
    SHORTER than `chosen_duration`, and `generate_segments` pads any sub-30 s
    segment up to 30 s, so a very short fallback-path segment publishes
    LONGER. There is deliberately no `published_duration` companion field
    (the story is scoped to these five facts): treat this one as the cutter's
    input, not a measurement of its output.

    `first_caption_sentence_complete` is judged at the RESOLVED WINDOW's
    start, not at the chosen segment's start. The LLM's chosen segment may
    begin anywhere inside that window, so when it starts later than the
    window does, this fact describes a sentence the published clip does not
    begin with. The two fields sit side by side in one JSON object but are
    NOT both attributes of the same clip: `chosen_duration` describes the
    chosen segment, this one describes the window. Neither the name nor the
    window was changed; this note is the honest account of what is computed.

    Pure: no I/O, no printing, no globals mutation. Never raises — any
    missing or unparseable input degrades that one field to None and the rest
    of the summary still emits. This is a diagnostics path; a partial line
    is far more useful than a crash.
    """
    try:
        return {
            "capture_span": _summary_field(
                lambda: _coerce_seconds(capture_span)
            ),
            "resolved_window": _summary_field(
                lambda: _summary_window(resolved_window)
            ),
            "chosen_duration": _summary_field(
                lambda: _summary_chosen_duration(chosen_segment)
            ),
            "topic_complete": topic_complete,
            "first_caption_sentence_complete": _summary_field(
                lambda: _summary_first_caption_complete(segments, resolved_window)
            ),
        }
    except Exception:
        # Belt and braces: each fact above is already computed behind
        # _summary_field, so this is unreachable in practice — but a logging
        # path must never be the thing that breaks a run.
        return {
            "capture_span": None,
            "resolved_window": None,
            "chosen_duration": None,
            "topic_complete": topic_complete,
            "first_caption_sentence_complete": None,
        }


def _summary_field(compute):
    """One summary fact, isolated: any exception degrades it to None alone so
    the other four still emit. The never-raises contract is per-fact, not
    all-or-nothing."""
    try:
        return compute()
    except Exception:
        return None


def _coerce_seconds(value):
    """Float seconds for a FINITE numeric value, else None.

    NaN / +inf / -inf degrade to None rather than being carried through: the
    summary line is advertised as one strictly-parseable JSON object, and
    `json.dumps` writes a bare NaN / Infinity, which jq and JSON.parse both
    reject. It is reachable — `_srt_like_to_seconds("nan")` returns nan and
    `capture_span` is a difference of two of its results. A degraded field is
    honest; a line no parser can read is not.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    return seconds if math.isfinite(seconds) else None


def _summary_window(resolved_window):
    """{"start", "end"} of a usable resolved window, or None."""
    try:
        if not isinstance(resolved_window, dict):
            return None
        start = resolved_window.get("start_time")
        end = resolved_window.get("end_time")
        start_seconds = _srt_like_to_seconds(start)
        end_seconds = _srt_like_to_seconds(end)
        if start_seconds is None or end_seconds is None:
            return None
        # A non-finite bound ("inf" parses as a float) is not a usable window;
        # report the fact as unknown rather than echoing a nonsense bound.
        if not (math.isfinite(start_seconds) and math.isfinite(end_seconds)):
            return None
        return {"start": start, "end": end}
    except Exception:
        return None


def _summary_chosen_duration(chosen_segment):
    """Duration of the chosen (post-validation) segment in seconds, or None."""
    try:
        if not isinstance(chosen_segment, dict):
            return None
        start = _srt_like_to_seconds(chosen_segment.get("start_time"))
        end = _srt_like_to_seconds(chosen_segment.get("end_time"))
        if start is not None and end is not None:
            duration = end - start
        else:
            duration = _coerce_seconds(chosen_segment.get("duration"))
        # Either operand can be non-finite (see _coerce_seconds), so the
        # difference can be too.
        if duration is None or not math.isfinite(duration):
            return None
        return duration
    except Exception:
        return None


def _summary_in_window_cues(segments, window_start, window_end):
    """Cues OVERLAPPING [window_start, window_end], unparseable ones skipped.

    Overlap, not containment: a cue that begins before the window start and
    runs into it owns that start, and its text is the first thing the clip
    shows.
    """
    cues = []
    for cue in segments or []:
        start, end = _cue_bounds(cue)
        if start is None or end is None:
            continue
        if end > window_start and start < window_end:
            cues.append(cue)
    return cues


def _summary_first_caption_complete(segments, resolved_window):
    """True/False/None: is the first in-window cue's first sentence complete.

    Judged at the RESOLVED WINDOW's start, deliberately: the chosen segment
    is chosen by the LLM and may start anywhere inside that window, so
    anchoring this to the segment would make the answer depend on a segment
    the caller may not even have. The consequence is stated rather than
    hidden — when the chosen segment starts later than the window, this
    describes a sentence the published clip does not begin with.
    """
    try:
        if not isinstance(segments, list) or not segments:
            return None
        if not isinstance(resolved_window, dict):
            # No window -> no defensible in-window cue list; report unknown.
            return None
        window_start = _srt_like_to_seconds(resolved_window.get("start_time"))
        window_end = _srt_like_to_seconds(resolved_window.get("end_time"))
        if window_start is None or window_end is None or window_end <= window_start:
            return None

        cues = _summary_in_window_cues(segments, window_start, window_end)
        if not cues:
            return None

        first = cues[0]
        if not isinstance(first, dict):
            return None
        text = str(first.get("text") or "").strip()
        if not text:
            return None

        if _first_sentence_ends_complete(text):
            return True
        if len(cues) > 1 and _cue_opens_new_sentence(cues[1]):
            return True
        return False
    except Exception:
        return None


def refine_window_start_with_pause(
    window, audio_path, min_start_srt=None, neighborhood=1.5, max_duration=None
):
    """S3: prosodic refinement of the text-snapped START point (never raises).

    Builds an EnhancedAudioAnalyzer for `audio_path` and asks
    detect_natural_boundaries for pauses inside a ±`neighborhood` window
    around the current start; the nearest pause becomes the new start,
    still respecting `min_start_srt` (None = the pause must simply stay
    within the neighborhood). The nearest pause can lie BEFORE the current
    start, so when `max_duration` is given, a pause that would push the
    start earlier than ``end - max_duration`` is rejected: the hard clip cap
    binds the whole chain, not just this function's caller. Audio
    load/analysis failures degrade to the text-only snapped window. Keep it
    cheap: only the neighborhood matters (a <=90 s capture)."""
    try:
        if not isinstance(window, dict):
            return window
        start_seconds = _srt_like_to_seconds(window.get('start_time'))
        end_seconds = _srt_like_to_seconds(window.get('end_time'))
        if start_seconds is None or end_seconds is None:
            return window
        if not audio_path or not os.path.isfile(str(audio_path)):
            return window

        from enhanced_audio_analyzer import EnhancedAudioAnalyzer

        analyzer = EnhancedAudioAnalyzer(audio_path)
        sample_rate = getattr(analyzer, 'sr', 22050)
        if not sample_rate or sample_rate <= 0:
            return window
        lo = max(0.0, start_seconds - neighborhood)
        hi = start_seconds + neighborhood
        pauses = analyzer.detect_natural_boundaries(int(lo * sample_rate), int(hi * sample_rate))
        if not pauses:
            return window

        min_start_seconds = _srt_like_to_seconds(min_start_srt) if min_start_srt is not None else None
        # Hard cap: a pause here would make the window longer than
        # `max_duration`, so it is not a candidate at all.
        cap_start_seconds = (
            end_seconds - max_duration
            if isinstance(max_duration, (int, float)) and not isinstance(max_duration, bool)
            else None
        )
        candidates = []
        for pause in pauses:
            try:
                absolute = lo + float(pause)
            except (TypeError, ValueError):
                continue
            if absolute < lo or absolute > hi:
                continue
            if min_start_seconds is not None and absolute < min_start_seconds:
                continue
            if cap_start_seconds is not None and absolute < cap_start_seconds:
                continue
            candidates.append(absolute)
        if not candidates:
            return window

        nearest = min(candidates, key=lambda t: abs(t - start_seconds))
        if abs(nearest - start_seconds) < 0.01:
            return window
        refined = dict(window)
        refined['start_time'] = format_timestamp(nearest)
        refined['duration'] = end_seconds - nearest
        print(
            f"🎵 Prosodic snap: start {window.get('start_time')} → {refined['start_time']} "
            f"(nearest pause within ±{neighborhood}s)"
        )
        return refined
    except Exception as e:
        print(f"Prosodic start refinement failed ({e}); keeping text-snapped start")
        return window


def generate_viral(
    transcript_content,
    audio_path=None,
    output_dir=None,
    min_duration=35.0,
    max_duration=59.0,
    source_link=None,
    word_segments=None,
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

    # Capture-length guard: the OBS replay buffer bounds how much topic context
    # exists. If the capture is shorter than the minimum clip window plus margin,
    # the clip will feel truncated — raise the OBS replay buffer (90 s default).
    # Seeded to None so the A5 selection summary can report it even when the
    # guard below does not bind it (empty or unparseable cue list).
    capture_span = None
    if segments:
        capture_start = _segment_start_seconds(segments[0])
        capture_end = _segment_end_seconds(segments[-1])
        if capture_start is not None and capture_end is not None:
            capture_span = capture_end - capture_start
            if capture_span < min_duration + 10.0:
                print(
                    f"⚠️ Capture span {capture_span:.0f}s is shorter than the "
                    f"minimum clip window ({min_duration:.0f}s) + margin — the clip "
                    f"may be truncated. Check the OBS replay buffer length."
                )

    # Then generate viral segments
    json_template = build_segment_json_template()

    title_language = "Portuguese"
    
    # If we have topic analysis results, use the optimized approach
    # Seeded to None so the A5 selection summary can report the boundary LLM's
    # verdict even on the fallback path, where no verdict exists.
    topic_complete = None
    if 'topic_analysis' in locals() and 'error' not in topic_analysis:
        print("\n==== Using Topic-Focused Content Generation ====")
        
        # Use the pre-analyzed optimized segment from TopicAnalyzer
        optimized_segment = topic_analysis['optimized_segment']
        concluding_topic = topic_analysis['concluding_topic']
        supporting_segments = topic_analysis['supporting_segments']

        # S2: LLM topic-start pass — the capture's start often holds the tail
        # of the previous topic. Resolve where the final topic begins and trim
        # the window start to that cue; any failure keeps the window exactly
        # as analyzed above. The end time (recency anchor) is untouched.
        # S3: the same pass also reports when the topic is incomplete and the
        # earlier cue where it actually begins. That cue extends the window
        # start backwards so the clip carries the full topic context; the end
        # stays anchored and the extension is bounded by `max_duration`. The
        # completeness verdict itself is bound to `topic_complete` and is read
        # by the A5 selection summary, which logs it verbatim as
        # SELECTION_SUMMARY.topic_complete.
        # S5: it also proposes hook candidates inside the final topic;
        # selection against the snapped window happens below.
        cue_index, topic_complete, earlier_start_cue, hook_candidates = (
            resolve_topic_start_cue(transcript_content, topic_analysis['segments'])
        )
        if cue_index is not None:
            optimized_segment, supporting_segments = trim_window_to_topic_start(
                cue_index,
                optimized_segment,
                supporting_segments,
                topic_analysis['segments'],
                min_duration=min_duration,
                max_duration=max_duration,
            )
        if earlier_start_cue is not None:
            optimized_segment, supporting_segments = (
                extend_window_to_earlier_topic_start(
                    earlier_start_cue,
                    optimized_segment,
                    supporting_segments,
                    topic_analysis['segments'],
                    max_duration=max_duration,
                )
            )

        # S3: refine the trimmed window to sentence/pause boundaries. The
        # former content-centric prompt built here was dead code (the
        # traditional prompt below is the one actually sent), so its guidance
        # lives on as the resolved-window section injected into that live
        # prompt: lead with the concluding topic, keep the narrative
        # complete, and respect the resolved start/end boundaries.
        #
        # The snap floor must come from whichever cue ended up OWNING the
        # window start: the earlier cue when the extension above applied,
        # otherwise the trimmed cue. Using the trimmed cue's start here
        # would let the snapper drag the start forward again and silently
        # undo the extension.
        min_start_srt = None
        # Latent seam, deliberately not coded around: this picks the owner by
        # `earlier_start_cue is not None`, not by whether extend_window_to_
        # earlier_topic_start actually applied. Were the extension ever to
        # decline on an unparseable cue timestamp, _segment_start_srt would
        # yield None and the snap floor would degrade to None — LOOSER than
        # the pre-A4 floor, which came from the trimmed cue's timestamp.
        # Currently unreachable: resolve_topic_start_cue only returns an
        # earlier_start_cue that is a valid int strictly below a non-None
        # cue_index, so the extension does apply whenever this branch is taken.
        start_owner_cue = earlier_start_cue if earlier_start_cue is not None else cue_index
        if (
            isinstance(start_owner_cue, int)
            and not isinstance(start_owner_cue, bool)
            and 1 <= start_owner_cue <= len(topic_analysis['segments'])
        ):
            min_start_srt = _segment_start_srt(
                topic_analysis['segments'][start_owner_cue - 1]
            )
        optimized_segment = snap_window_to_boundaries(
            optimized_segment,
            topic_analysis['segments'],
            min_start_srt=min_start_srt,
            min_duration=min_duration,
        )
        optimized_segment = refine_window_start_with_pause(
            optimized_segment,
            audio_path,
            min_start_srt=min_start_srt,
            max_duration=max_duration,
        )

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
                summary_file = _contained_path(output_dir, "audio_analysis_summary.txt")
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

    # The resolved topic window (Task-13 trim + S3 snapping) reaches the LIVE
    # prompt as instruction prose — the single enforcement channel for
    # Tasks 13+14 (deterministic validation of the returned window lands in
    # S4). Empty on the fallback path so that path's prompt stays
    # byte-identical.
    resolved_window_section = ""
    if 'topic_analysis' in locals() and 'error' not in topic_analysis:
        # S5: hook-first selection — the first boundary-pass hook candidate
        # inside the snapped resolved window becomes a MANDATORY hook in the
        # live prompt, so the burned cold-open teaser and the returned
        # hook_text are the same sentence. With no survivor (gate off, LLM
        # pass failed, no valid candidate) nothing is appended: the section
        # stays byte-identical to Task 14's and the hook remains optional.
        selected_hook = select_hook_candidate(
            hook_candidates,
            optimized_segment['start_time'],
            optimized_segment['end_time'],
            topic_analysis['segments'],
            sentences=build_sentence_spans(word_segments),
        )
        resolved_window_section = (
            "\nRESOLVED TOPIC WINDOW:\n"
            f"- Resolved start: {optimized_segment['start_time']}\n"
            f"- Resolved end: {optimized_segment['end_time']}\n"
        )
        if cue_index is not None:
            resolved_window_section += (
                f"- The final topic begins at cue {cue_index} of the numbered transcript cues above\n"
            )
        resolved_window_section += (
            "The returned segment MUST start at or after the resolved start time and end at or "
            "before the resolved end time. Prefer exactly the resolved boundaries; only a clear "
            "sentence boundary justifies a slight adjustment within them.\n"
        )
        if selected_hook:
            resolved_window_section += (
                "\nMANDATORY HOOK:\n"
                "- You MUST return hook_start_time / hook_end_time / hook_text with EXACTLY these values:\n"
                f"- hook_start_time: {selected_hook['hook_start_time']}\n"
                f"- hook_end_time: {selected_hook['hook_end_time']}\n"
                f'- hook_text: "{selected_hook["hook_text"]}"\n'
                "Return these hook fields unchanged and keep the hook span fully inside the "
                "returned segment.\n"
            )

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

    prompt = f"""Given the following video transcript and audio analysis results, analyze each part for potential virality and identify the single most viral segment of maximum {max_duration} seconds.
The segment should be complete with the full explanation for the audience to comprehend the entire information or message of the segment,
and have a minimum of {min_duration} seconds and maximum of {max_duration} seconds in duration.

Do not select a segment with partial or incomplete information.

HOOK TEASER (optional):
- "hook_start_time" and "hook_end_time": SRT timestamps (HH:MM:SS,mmm) INSIDE the segment, spanning an engaging, self-contained stretch of 3-9 seconds. A span shorter than 3 seconds is unusable as a cold open, so pick a stretch that carries a complete point.
- "hook_text": a punchy 4-7 word hook in {title_language} that will be burned as text over the teaser.
- These fields are optional: only include them when a clear hook sentence exists inside the segment.

The provided transcript is as follows:

{formatted_segments}
{resolved_window_section}{audio_context}

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

    client = openai.OpenAI(api_key=api_key, base_url=openai_base_url)
    response = request_json_completion(
        client, os.getenv('OPENAI_MODEL', 'gpt-5-mini'), messages
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

    # S4: deterministic validation of the returned segment. The primary
    # path carries a resolved topic window (Task-13 trim + S3 snapping), so
    # the returned times are clamped into it and an out-of-range clamped
    # duration is replaced by the window itself. The fallback path keeps
    # today's behavior: only the duration range is checked and out-of-range
    # segments are left unchanged with a warning.
    resolved_window = None
    if 'topic_analysis' in locals() and 'error' not in topic_analysis:
        resolved_window = optimized_segment
    if isinstance(viral_json, dict) and isinstance(viral_json.get("segments"), list):
        validated_segments = []
        for returned_segment in viral_json["segments"]:
            returned_segment, validation_notes = validate_segment_against_window(
                returned_segment,
                resolved_window,
                min_duration=min_duration,
                max_duration=max_duration,
            )
            for note in validation_notes:
                print(note)
            validated_segments.append(returned_segment)
        viral_json["segments"] = validated_segments

        # A5: one structured line per chosen segment reporting the five
        # completeness facts (capture span, resolved window, chosen duration,
        # the boundary LLM's topic_complete verdict, and whether the first
        # caption sentence is grammatically complete). This is the feedback
        # signal for tuning A1-A4 — it changes no behavior, so the builder is
        # pure and the print is wrapped: diagnostics must never break a run.
        # An empty selection still gets ONE line (built with a None chosen
        # segment, so chosen_duration is null): a run that selected nothing is
        # exactly the run worth reading, and it used to log nothing at all.
        # ONE LINE PER VALIDATED SEGMENT, not one per run: an LLM reply with
        # N segments emits N lines that share capture_span, resolved_window
        # and topic_complete and differ only in chosen_duration. A scraper
        # keying these lines by run must not assume a single line per run.
        for validated_segment in validated_segments or [None]:
            try:
                summary = build_selection_summary(
                    segments,
                    resolved_window,
                    validated_segment,
                    capture_span,
                    topic_complete,
                )
            except Exception as e:
                # Deliberately NOT prefixed with SELECTION_SUMMARY: that marker
                # means "one parseable JSON object", so a scraper never sees a
                # partial line.
                print(f"⚠️ Selection summary unavailable (summary builder failed: {e})")
                continue
            print("SELECTION_SUMMARY " + json.dumps(summary))

    # Merge metadata and viral segments
    # Merge metadata and viral segments
    if source_link:
        metadata_json["source_link"] = source_link
        
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
    """
    Generate transcript using the new transcription service.
    Supports both Whisper and Faster Whisper with automatic fallback.
    """
    import logging
    from transcription_service import get_transcriber

    logger = logging.getLogger(__name__)

    # Extract audio for transcription
    temp_dir = "tmp/"
    audio_path = os.path.join(
        temp_dir, f"{os.path.basename(input_file).split('.')[0]}.wav"
    )

    print(f"Extracting audio from {input_file}...")
    # Use subprocess instead of ffmpeg-python
    audio_cmd = (
        f"ffmpeg -y -i tmp/{shlex.quote(input_file)} "
        f"-vn -acodec pcm_s16le -ac 1 -ar 16000 {shlex.quote(audio_path)}"
    )
    run_ffmpeg_command(audio_cmd)

    try:
        # Get transcriber through the service factory
        print("Initializing transcription service...")
        transcriber = get_transcriber()
        print(f"Using {transcriber.name} for transcription...")
        print(f"Audio file: {audio_path}")

        # Check if audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
        print(f"Audio file size: {file_size:.2f} MB")

        # Transcribe using the service
        print("Transcribing audio...")
        import time
        start_time = time.time()
        result = transcriber.transcribe(audio_path)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Transcription completed in {processing_time:.2f} seconds")
        print(f"Generated {len(result)} segments")

        # Clean up temporary audio file
        try:
            os.remove(audio_path)
        except OSError:
            pass  # File might not exist or already removed

        # Convert to legacy format for compatibility
        return {"segments": result}

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        print(f"Error during transcription: {str(e)}")

        # Fallback to legacy Whisper implementation
        print("Falling back to legacy Whisper implementation...")
        try:
            import whisper
            model = whisper.load_model(os.getenv('WHISPER_MODEL', 'small'))

            # Use configured language or auto-detect
            transcription_language = os.getenv('TRANSCRIPTION_LANGUAGE', None)
            if transcription_language:
                print(f"Fallback: Using configured language: {transcription_language}")
                result = model.transcribe(audio_path, language=transcription_language)
            else:
                print("Fallback: Using automatic language detection")
                result = model.transcribe(audio_path)

            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except OSError:
                pass

            # Return in expected format
            return {
                "segments": result["segments"],
                "full_transcript": " ".join(
                    segment["text"].strip() for segment in result["segments"]
                ),
            }
        except Exception as fallback_error:
            logger.error(f"Fallback transcription also failed: {str(fallback_error)}")
            print(f"Fallback transcription failed: {str(fallback_error)}")
            raise RuntimeError("All transcription methods failed") from fallback_error


def save_metadata(metadata, output_dir, date_text=None, overlay_title=None):
    """Save metadata in a copy-paste friendly format.

    When date_text/overlay_title are provided, the burned-overlay info is
    appended as a description fallback so platforms that re-crop or strip
    burned-in text keep it.
    """
    try:
        # Create a formatted text file for easy copy-paste
        formatted_text = f"""Title:
{metadata['title']}

Description:
{metadata['description']}

Source: {metadata.get('source_link', 'N/A')}
"""
        extra_lines = ""
        if date_text is not None:
            extra_lines += f"\nDate: {date_text}"
        if overlay_title is not None:
            extra_lines += f"\nOverlay title: {overlay_title}"
        if extra_lines:
            formatted_text = formatted_text.rstrip("\n") + extra_lines + "\n"
        # Save to file
        metadata_file = _contained_path(output_dir, "description.txt")
        with open(metadata_file, "w", encoding="utf-8") as file:
            file.write(formatted_text)
        print(f"Metadata saved to {metadata_file}")
    except Exception as e:
        print(f"Error saving metadata: {str(e)}")


def direct_process_file(
    input_file,
    output_dir, upscale=False, enhance=False, thumb=False, debug_faces=False, performance_profile=False
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
        filename, output_file, video_name, upscale, enhance, thumb, output_dir, debug_faces, performance_profile
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
        default=35.0,
        help="Minimum duration in seconds for viral segments (default: 35.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=59.0,
        help="Maximum duration in seconds for viral segments (default: 59.0)",
    )
    parser.add_argument("--job-id", required=False, help="Job ID for organization")
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Custom output directory for reels and thumbnails",
    )
    parser.add_argument(
        "--debug-faces",
        action="store_true",
        default=False,
        required=False,
        help="Enable debug visualization for face detection and scoring",
    )
    parser.add_argument(
        "--performance-profile",
        action="store_true",
        default=False,
        required=False,
        help="Enable performance profiling and save detailed timing report",
    )
    parser.add_argument(
        "--video-title",
        required=False,
        help="Specific video title/identifier to use in overlay (e.g. '[Show][Ep 1]')",
    )
    parser.add_argument(
        "--source-link",
        required=False,
        help="Original video source link to be included in metadata",
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
            copy_input_to_tmp(args.file, "tmp/input_video.mp4")
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
    command = (
        "ffmpeg -y -i tmp/input_video.mp4 -vn -acodec pcm_s16le -ar 22050 -ac 1 "
        f"{shlex.quote(audio_path)}"
    )
    print(f"Running audio extraction command: {command}")
    run_ffmpeg_command(command)

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
        command = (
            "ffmpeg -y -i tmp/input_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 "
            f"{shlex.quote(fallback_audio_path)}"
        )
        run_ffmpeg_command(command)

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
    output_audio_path = _contained_path(output_dir, "audio_analysis.wav")
    try:
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            shutil.copy2(audio_path, output_audio_path)
            print(f"Copied audio file to output directory: {output_audio_path}")
    except Exception as e:
        print(f"Warning: Could not copy audio file to output directory: {str(e)}")

    # Verifies if output_file exists, or create it. If exists, it doesn't call OpenAI APIs
    output_file = _contained_path(output_dir, "content.txt")
    transcript_file = _contained_path(output_dir, "transcript.txt")
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
                source_link=args.source_link,
                word_segments=transcript_result.get("segments"),
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
        # Save metadata in a copy-paste friendly format. Anything burned in
        # the video (date + overlay title) must also be in the description —
        # and both overlays are off by default, so neither line is added
        # unless the matching OVERLAY_*_ENABLED flag turned the burn on.
        save_metadata(
            parsed_content["metadata"],
            f"{output_dir}",
            date_text=(
                datetime.now().strftime("%d/%m/%Y")
                if resolve_overlay_date_enabled()
                else None
            ),
            overlay_title=(
                resolve_overlay_title(args.video_title, get_channel_name())
                if resolve_overlay_title_enabled()
                else None
            ),
        )
        print(f"Processing {len(parsed_content['segments'])} generated segments...")
        # Jump cuts (Story 5): plan the intra-segment pause removal once per
        # segment so generate_segments (pieces) and the subtitle words below
        # share the same plan. Escape hatch off -> no plans at all.
        jump_cut_enabled, jump_cut_threshold = get_jump_cut_config()
        jump_cut_plans = {}
        if jump_cut_enabled:
            jump_cut_plans = compute_jump_cut_plans(
                parsed_content["segments"],
                transcript_result["segments"],
                jump_cut_threshold,
            )
        generate_segments(parsed_content["segments"], jump_cut_plans=jump_cut_plans)
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
        safe_title = sanitize_filename(segment["title"])
        input_file = f"{safe_title}{str(i).zfill(3)}.mp4"
        output_file = f"{safe_title}_cropped{str(i).zfill(3)}.mp4"

        try:
            print(f"Generating short video for segment {i+1}...")
            # Story 11 (P3.5): hook fields are evaluated up-front so the hook
            # text (when present) is used as the thumbnail text; missing or
            # invalid fields fall through to today's title behavior exactly.
            try:
                seg_start_sec = parse_srt_time(segment["start_time"])
                seg_end_sec = parse_srt_time(segment["end_time"])
                hook = parse_hook_fields(segment, seg_start_sec, seg_end_sec)
            except Exception as e:
                print(f"Hook teaser: could not evaluate hook fields: {str(e)}")
                hook = None
            generate_short(
                input_file,
                output_file,
                str(segment["title"]),
                args.upscale,
                args.enhance,
                args.thumb,
                output_dir,
                args.debug_faces,
                args.performance_profile,
                thumb_text=(hook[2] if hook else None),
            )
            print(f"Adding subtitles to video for segment {i+1}...")
            final_output_name = sanitize_filename(f"final-{os.path.splitext(output_file)[0]}") + os.path.splitext(output_file)[1]
            # Reuse the master transcript word timestamps (clip-relative);
            # None when the transcriber has no word data -> SRT fallback.
            # With jump cuts (Story 5) the remapped word timeline is used so
            # captions stay in sync with the condensed clip.
            subtitle_words = resolve_subtitle_words(
                jump_cut_plans,
                i,
                transcript_result["segments"],
                segment["start_time"],
                segment["end_time"],
            )
            if subtitle_words:
                print(
                    f"Using {len(subtitle_words)} words from master transcript for subtitles"
                )
            # Hook teaser (Story 7): optional cold-open. `hook` was parsed
            # above (before generate_short) for the thumbnail text; every
            # teaser failure mode ends in the direct publish, so the segment
            # is never left without a published file.
            # B4: the clip's own start time is handed over so the publish
            # step can refuse a teaser that would replay the clip opening.
            publish_segment_with_optional_teaser(
                final_output_name,
                video_id,
                output_dir,
                video_title=args.video_title,
                subtitle_words=subtitle_words,
                hook=hook,
                clip_start_time=segment["start_time"],
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


# Assembly contract (Story C3). Every piece that assemble_hook_teaser_final
# concatenates — teaser, main, end card — must declare the SAME frame rate,
# video timescale and audio shape, or the concat demuxer silently rescales
# the odd one out. That is not hypothetical: the end card used to omit -r,
# so its `color` lavfi source defaulted to 25 fps / 1-12800 while everything
# else was 30 fps / 1-15360, and `concat -c copy` produced a valid-looking
# file whose end card collapsed into 2.5 ms on the main clip's last frames.
# Instagram's transcoder stalls there; desktop players show an invisible
# flash. See the Theme C appendix in the EPIC.
#
# Defined ONCE and used by every encoder so a retune cannot leave two
# pieces disagreeing again (the HOOK_DUPLICATE_GUARD_SECONDS lesson).
ASSEMBLY_FPS = 30
# 15360 = ASSEMBLY_FPS * 512, the timescale NVENC picks for 30 fps. Pinned
# explicitly because leaving it to the encoder is what caused the bug.
ASSEMBLY_VIDEO_TIMESCALE = 15360
ASSEMBLY_AUDIO_RATE = 48000
ASSEMBLY_AUDIO_CHANNELS = 2

# Audio mastering for the final published file: loudness-normalize to the
# streaming-platform target (-14 LUFS) then gentle compression. Applied ONLY
# in generate_subtitle's final mux; earlier pipeline stages pass audio through
# untouched so the chain is applied exactly once.
AUDIO_MASTERING_FILTER = (
    "loudnorm=I=-14:LRA=11:tp=-1.5,acompressor=threshold=0.05:ratio=3:makeup=1.2"
)


def build_audio_filter_args():
    """Shell fragment mastering the audio stream and re-encoding it to AAC.

    The rate/channel pin is not only about loudnorm: it is what keeps every
    assembly piece byte-compatible, so `concat -c copy` cannot silently
    adopt one piece's parameters and desync the rest (measured: a 44100 main
    against a 48000 end card yields a container 0.40 s longer than the sum of
    the pieces, with no error anywhere)."""
    return (
        f'-af "{AUDIO_MASTERING_FILTER}" -c:a aac -b:a 192k '
        f"-ar {ASSEMBLY_AUDIO_RATE} -ac {ASSEMBLY_AUDIO_CHANNELS}"
    )


# Polish features (Story 10 / P3.1-P3.3): mild visual grade on the main clip,
# a 1.5 s end-card CTA appended at assembly, and an optional music bed mixed
# under the speech. Each feature is env-gated and must be inert (commands
# byte-identical to the previous behavior) whenever it is off/unset.

# P3.1: exact grade chain from the brief — mild saturation/contrast lift plus
# a light sharpen. Prepended as the FIRST video filter in generate_subtitle so
# captions and overlays burn ON TOP of the graded image.
VIDEO_GRADE_FILTER = "eq=saturation=1.06:contrast=1.02,unsharp=5:5:0.6"

# P3.2: end card CTA constants.
END_CARD_FILE = "end-card.mp4"
END_CARD_DURATION_SECONDS = 1.5
END_CARD_FADE_SECONDS = 0.4
END_CARD_FONT_SIZE = 64
END_CARD_DEFAULT_COLOR = "0x14213D"
END_CARD_DEFAULT_TEXT = "Segue para mais!"


def resolve_video_grade():
    """Resolve the P3.1 visual grade from the environment.

    Returns the grade filter string that opens generate_subtitle's video
    chain, or None when grading is disabled. VIDEO_GRADE defaults to "true"
    (only the literal "false" disables, like JUMP_CUTS_ENABLED); when
    VIDEO_LUT is set to a path, lut3d=<path> replaces the eq/unsharp pair.
    """
    lut_path = os.getenv('VIDEO_LUT', '').strip()
    if lut_path:
        print(
            f"Visual grade: using VIDEO_LUT '{lut_path}' instead of eq/unsharp"
        )
        return f"lut3d='{escape_filter_option(lut_path)}'"
    grade_raw = os.getenv('VIDEO_GRADE', 'true').lower().strip()
    if grade_raw == 'false':
        print("Visual grade disabled (VIDEO_GRADE=false)")
        return None
    return VIDEO_GRADE_FILTER


def resolve_music_bed():
    """Resolve the P3.3 music bed path from MUSIC_BED.

    Unset/empty (the default) means OFF and every command stays identical
    to the no-music-bed behavior.
    """
    path = os.getenv('MUSIC_BED', '').strip()
    if not path:
        return None
    print(f"Music bed enabled: {path}")
    return path


def build_music_bed_audio_filters():
    """Audio half of the music-bed -filter_complex: the bed is mixed UNDER
    the speech (-24 dB) and the SAME mastering chain is applied to the mix,
    so the audio is still mastered exactly once."""
    return (
        "[1:a]volume=-24dB[m];[0:a][m]amix=inputs=2:duration=first[aout];"
        f"[aout]{AUDIO_MASTERING_FILTER}[aout2]"
    )


def resolve_end_card_enabled():
    """P3.2 gate. Default "true" as of Story C4.

    It was default-off only as an interim mitigation: the assembly froze
    before the end card on Instagram. That cause is fixed (Story C3) — the
    end card was the one piece that never declared a frame rate, so concat
    -c copy collapsed it into 2.5 ms on the main clip's last frames, which
    a desktop player showed as an invisible flash and a social transcoder
    stalled on. Measured on real capture after the fix, the end card
    occupies its full 1.5 s and every parameter matches the main clip.

    Set END_CARD_ENABLED=false to opt out; the end card is a CTA, so it is
    a product decision as much as a technical one, and this is the one-line
    revert if it does not earn its place."""
    enabled_raw = os.getenv('END_CARD_ENABLED', 'true').lower().strip()
    enabled = enabled_raw == 'true'
    if not enabled:
        print("End card disabled (END_CARD_ENABLED=false); publishing without it")
    return enabled


def escape_filter_option(text):
    """Escape a user-set string for use inside a single-quoted ffmpeg
    filter option value (drawtext text=..., lut3d file=...) that sits in
    a shell double-quoted argument.

    The value crosses the shell, the filtergraph parser (whose option
    splitter is quote-blind) and the filter's own option parser, so each
    special is written in the doubled form verified byte-exact against
    ffmpeg 6.1 for BOTH drawtext and lut3d: backslash and colon doubled,
    a literal quote as close-escape-reopen, and a double quote escaped
    for the shell. '%' is left alone — pair with drawtext's
    expansion=none, which keeps it verbatim."""
    text = str(text)
    text = text.replace("\\", r"\\\\")
    text = text.replace("'", r"\\'\''")
    text = text.replace(":", r"\\:")
    text = text.replace('"', r'\"')
    return text


def build_end_card_command():
    """Pure builder for the 1.5 s end-card clip command.

    1080x1920 solid color (END_CARD_COLOR) with the CTA text (END_CARD_TEXT)
    centered via drawtext (OVERLAY_FONTFILE, fontsize 64, white, black
    outline, expansion=none so the CTA text renders verbatim) plus a
    0.4 s fade-in. Audio is a silent anullsrc (44.1k) encoded
    as AAC 48000 Hz stereo so both sides of the concat agree with the
    mastering args' pinned -ar. The lavfi color source is pinned to
    ASSEMBLY_FPS and ASSEMBLY_VIDEO_TIMESCALE, matching the other concat
    pieces (C3). The end card is silent-sourced so no mastering filter
    ever touches it.
    """
    color = os.getenv('END_CARD_COLOR', END_CARD_DEFAULT_COLOR).strip()
    text = escape_filter_option(
        os.getenv('END_CARD_TEXT', END_CARD_DEFAULT_TEXT)
    )
    drawtext = (
        f"drawtext=text='{text}':expansion=none:fontfile={OVERLAY_FONTFILE}:"
        f"fontsize={END_CARD_FONT_SIZE}:fontcolor=white:"
        "borderw=3:bordercolor=black:x=(w-tw)/2:y=(h-th)/2"
    )
    return (
        "ffmpeg -y -hwaccel cuda "
        f"-f lavfi -i color=c={color}:s=1080x1920:d={END_CARD_DURATION_SECONDS} "
        "-f lavfi -i anullsrc=r=44100:cl=stereo "
        # C3: the `color` source defaults to 25 fps, and without this the
        # concat demuxer collapses this piece into ~2.5 ms (the Instagram
        # freeze). ASSEMBLY_FPS/ASSEMBLY_VIDEO_TIMESCALE are shared with the
        # other two pieces so the three cannot drift apart.
        f"-r {ASSEMBLY_FPS} "
        f"-vf \"{drawtext},fade=t=in:st=0:d={END_CARD_FADE_SECONDS}\" "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p "
        f"-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE} "
        f"-c:a aac -b:a 192k -ar {ASSEMBLY_AUDIO_RATE} "
        f"-ac {ASSEMBLY_AUDIO_CHANNELS} -shortest "
        f"tmp/{END_CARD_FILE}"
    )


def generate_end_card():
    """Build tmp/end-card.mp4 once per segment; returns the path or None
    when the build fails (the caller assembles without the end card)."""
    try:
        command = build_end_card_command()
        print(f"End card command: {command}")
        result = run_ffmpeg_command(command)
        end_card_path = f"tmp/{END_CARD_FILE}"
        if (
            result != 0
            or not os.path.exists(end_card_path)
            or os.path.getsize(end_card_path) == 0
        ):
            print(f"End card: build failed (rc={result}); skipping end card")
            return None
        return end_card_path
    except Exception as e:
        print(f"End card generation failed: {str(e)}")
        return None


# Karaoke ASS captions: large, centered, word-by-word captions built from the
# master transcript word timestamps (no re-transcription of the cropped clip).

ASS_DEFAULT_FONT_FAMILY = "Noto Sans"
ASS_DEFAULT_HIGHLIGHT_COLOR = "&H0000FFFF"  # ASS is &HAABBGGRR; this is yellow
ASS_PRIMARY_COLOR = "&H00FFFFFF"  # white
ASS_MAX_WORDS_PER_GROUP = 3
ASS_PUNCTUATION_BREAK = (".", "!", "?")


def _ass_timestamp(seconds):
    """Format seconds as an ASS timestamp (H:MM:SS.cc)."""
    seconds = max(0.0, float(seconds))
    total_cs = int(seconds * 100)
    h = total_cs // 360000
    m = (total_cs // 6000) % 60
    s = (total_cs // 100) % 60
    cs = total_cs % 100
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text):
    """Escape word text so braces cannot start ASS override blocks.

    Commas need no escaping: the Dialogue Text field is the last field, so
    commas cannot break field parsing.
    """
    return str(text).replace("{", "\\{").replace("}", "\\}")


def build_karaoke_ass(words, font_family=None, highlight_color=None):
    """Build a complete ASS karaoke subtitle document from word timestamps.

    Args:
        words: flat list of {"word": str, "start": float, "end": float}
            (clip-relative seconds), sorted by start.
        font_family: optional font family; defaults to env
            CAPTION_FONT_FAMILY, falling back to "Noto Sans".
        highlight_color: optional ASS color for the current word; defaults to
            env CAPTION_HIGHLIGHT_COLOR, falling back to &H0000FFFF (yellow).

    Returns the ASS document string, or None when no words are available
    (the caller falls back to the legacy SRT path).
    """
    if not words:
        return None

    words = sorted(
        (w for w in words if w.get("word")),
        key=lambda w: w.get("start", 0.0),
    )
    if not words:
        return None

    font = font_family or os.getenv("CAPTION_FONT_FAMILY", ASS_DEFAULT_FONT_FAMILY)
    highlight = highlight_color or os.getenv(
        "CAPTION_HIGHLIGHT_COLOR", ASS_DEFAULT_HIGHLIGHT_COLOR
    )

    # Group words into caption lines of at most 3 words; a new group also
    # starts after a word ending with sentence punctuation.
    groups = []
    current_group = []
    for word in words:
        current_group.append(word)
        if (
            len(current_group) >= ASS_MAX_WORDS_PER_GROUP
            or str(word["word"]).strip().endswith(ASS_PUNCTUATION_BREAK)
        ):
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
# MarginV 600 on the 1920 px-tall PlayRes puts the caption block in the
# upper part of the bottom half (below the frame middle), clear of the
# platform UI that covers the bottom ~250 px.
Style: Karaoke,{font},58,{ASS_PRIMARY_COLOR},&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,3,1,2,60,60,600,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def display_word(word):
        return _ass_escape(str(word["word"]).strip().upper())

    events = []
    for group in groups:
        group_end = group[-1]["end"]

        # One dialogue per word: the full group text with the current word
        # highlighted, the other words in the default white. Each dialogue
        # spans until the next word starts (group end for the last word) so
        # coverage is continuous with no gaps and no overlapping lines.
        for i, word in enumerate(group):
            if i + 1 < len(group):
                event_end = max(float(group[i + 1]["start"]), float(word["start"]))
            else:
                event_end = group_end
            tokens = [display_word(w) for w in group]
            tokens[i] = (
                f"{{\\c{highlight}&}}{tokens[i]}{{\\c{ASS_PRIMARY_COLOR}&}}"
            )
            events.append(
                f"Dialogue: 0,{_ass_timestamp(word['start'])},"
                f"{_ass_timestamp(event_end)},"
                f"Karaoke,,0,0,0,," + " ".join(tokens)
            )

    return header + "\n".join(events) + "\n"


def collect_segment_words(segments, start_time, end_time):
    """Collect master-transcript words inside an SRT segment time range.

    Picks the words (midpoint inside [start_time, end_time], SRT
    HH:MM:SS,mmm timestamps) from the in-memory transcript segments and
    shifts them to clip-relative time. Returns None when the transcript has
    no word data so the caller falls back to re-transcription.
    """
    try:
        seg_start = parse_srt_time(start_time)
        seg_end = parse_srt_time(end_time)
    except (ValueError, AttributeError):
        return None

    words = []
    for segment in segments or []:
        for word in segment.get("words") or []:
            # A malformed entry must not defeat the documented fallback: skip
            # it and let an empty result degrade to re-transcription.
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                continue
            midpoint = (start + end) / 2
            if seg_start <= midpoint <= seg_end:
                words.append(
                    {
                        "word": str(word["word"]).strip(),
                        "start": max(0.0, start - seg_start),
                        "end": max(0.0, end - seg_start),
                    }
                )
    return words if words else None


def run_ffmpeg_command(command):
    """Run a shell-built ffmpeg command WITHOUT a shell.

    Every command string this module executes is POSIX-exact, so
    shlex.split yields exactly the argv a shell would produce — minus the
    injection surface: an interpolated path or text can no longer add shell
    syntax, it can only change its own argument.
    """
    return subprocess.call(shlex.split(command), shell=False)


def get_channel_name():
    """Look up CHANNEL_NAME from the root .env file.

    Falls back to the current environment when the root .env does not
    exist. Shared by the burned overlays and the description fallback so
    both carry the same identity.
    """
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
    return channel_name


def resolve_overlay_title(video_title, channel_name):
    """Overlay-title precedence: video_title (arg) -> channel_name -> None."""
    return video_title or (channel_name or None)


# Burned overlays: subtle, semi-transparent, kept inside the top safe zone
# (y=120 clears platform UI). Style values are binding for Story 6.
OVERLAY_FONTFILE = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
OVERLAY_FONT_SIZE = 32
OVERLAY_FONT_COLOR = "white@0.65"
OVERLAY_Y = 120
OVERLAY_X_MARGIN = 30


def _overlay_enabled(name):
    """Shared gate for the burned top overlays. Only the literal "true"
    enables one, so an unset or misspelled value leaves it off."""
    enabled = os.getenv(name, 'false').lower().strip() == 'true'
    if not enabled:
        print(f"Burned overlay off ({name} is not 'true')")
    return enabled


def resolve_overlay_date_enabled():
    """Gate for the top-left burned date. Default OFF."""
    return _overlay_enabled('OVERLAY_DATE_ENABLED')


def resolve_overlay_title_enabled():
    """Gate for the top-right burned title/channel (chapter) overlay.

    Default OFF.
    """
    return _overlay_enabled('OVERLAY_TITLE_ENABLED')


def build_overlay_filters(date_text, title_text, input_label="0:v"):
    """Build the chained drawtext filters for the burned date/title overlays.

    Pure helper so the exact style stays testable. Returns
    (list_of_filter_strings, output_label): the date overlay sits top-left
    (x=30) and the title/channel overlay top-right (x=W-tw-30), both at
    y=120, fontsize=32, semi-transparent white. input_label is the video
    stream label the first filter chains from; the date filter emits [v2]
    and the title filter [v3], matching the historical chain labels.
    output_label is the label of the final video stream (input_label
    unchanged when neither overlay applies).
    """
    filters = []
    last_label = input_label
    # The filter strings land inside a shell double-quoted -filter_complex:
    # strip the characters that stay shell-alive there (env-derived title,
    # e.g. CHANNEL_NAME, must not be able to break out).
    if date_text:
        date_text = re.sub(r'["`$\\]', "", str(date_text))
    if title_text:
        title_text = re.sub(r'["`$\\]', "", str(title_text))
    if date_text:
        filters.append(
            f"[{last_label}]drawtext=text='{date_text}':fontfile={OVERLAY_FONTFILE}:fontsize={OVERLAY_FONT_SIZE}:fontcolor={OVERLAY_FONT_COLOR}:x={OVERLAY_X_MARGIN}:y={OVERLAY_Y}:shadowcolor=black:shadowx=2:shadowy=2[v2]"
        )
        last_label = "v2"
    if title_text:
        filters.append(
            f"[{last_label}]drawtext=text='{title_text}':fontfile={OVERLAY_FONTFILE}:fontsize={OVERLAY_FONT_SIZE}:fontcolor={OVERLAY_FONT_COLOR}:x=W-tw-{OVERLAY_X_MARGIN}:y={OVERLAY_Y}:shadowcolor=black:shadowx=2:shadowy=2[v3]"
        )
        last_label = "v3"
    return filters, last_label


# Cold-open hook teaser (Story 7 / P2.2): when GPT returns valid hook fields
# for a segment, a HOOK_TEASER_MIN_DURATION..HOOK_TEASER_MAX_DURATION s teaser
# cut of that moment is burned with the hook
# text, cropped to 9:16, and prepended to the main clip. Audio is mastered
# exactly once per output stream: the teaser gets the chain in its burn step
# and the main clip keeps it in generate_subtitle's final mux; the concat
# between them is -c copy (or a video-only re-encode fallback) and NEVER
# re-masters audio.

# The teaser length band. These are set from the CUE GRID, not from taste:
# Whisper cues on a 90 s capture measure 3.7-10.9 s, so with the old 1.5-6.0 s
# band most cues were longer than the maximum and could never be a hook, the
# shortest two-cue span was 10.4 s (so a hook was ALWAYS a single cue), and the
# reachable durations collapsed to five values topping out at 5.7 s -- a teaser
# that reads as a flash. The floor also rejects the 1-2 s blink outright.
HOOK_TEASER_MIN_DURATION = 2.5
HOOK_TEASER_MAX_DURATION = 9.0
# How long a sentence-aligned hook should run before it is considered long
# enough. Below this the refine step keeps adding whole following sentences,
# which is what stops the teaser being a 1.5 s flash.
HOOK_TARGET_DURATION = 4.5
# Word tokenizer for matching the quoted hook text against candidate sentences.
_HOOK_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
# Story B1: the cold-open teaser is prepended to the clip, so a teaser taken
# from the clip's own first seconds plays the same words twice. Any candidate
# overlapping this opening span of the resolved window is rejected; the clip
# then simply publishes without a teaser.
# Story B4: the publish step re-uses this SAME span as its defense-in-depth
# check, so the two layers can never drift apart.
HOOK_DUPLICATE_GUARD_SECONDS = 5.0
# B4's caption net: how many leading words a hook must share with the clip's
# opening before the teaser counts as a repeat. A false match here drops a
# fine teaser, so the floor is set above a single coincidental word; two
# shared leading words is also the width of the short exact-match case the
# net replaced, so nothing that used to be caught is lost.
HOOK_CAPTION_MIN_MATCH_WORDS = 2
# The floor alone is not enough: screencast captions open with formulaic
# bigrams ("so today", "in this video", "welcome back"), so a genuinely
# different teaser can open with the same two words by coincidence. A real
# repeat restates the opening, so its shared prefix covers most of the shorter
# of the two sequences; a coincidental bigram collision does not.
HOOK_CAPTION_MIN_MATCH_COVERAGE = 0.6
# Per-word punctuation stripped when the caption net compares words; the
# sentence normalizer above only strips the very end of the text.
_CAPTION_PUNCTUATION = ".,!?;:…\"'’-()"
# Story B2: among candidates that already survive every check, prefer the ones
# landing in the middle of the clip — not in the first 20% (leads with a
# truncated thought) nor the last 10% (pre-empts the payoff). A teaser from the
# middle is complementary instead of repetitive. This is a preference, not a
# filter: an edge candidate is still used when no mid-band one exists.
HOOK_MID_BAND_START_FRACTION = 0.20
HOOK_MID_BAND_END_FRACTION = 0.90
HOOK_TEASER_FONT_SIZE = 64
# drawtext cannot wrap text, so long hooks must be pre-split into lines that
# fit the 1080 px frame (fontsize 64 uppercase ≈ 40 px/char).
HOOK_TEXT_MAX_LINE_CHARS = 18
HOOK_TEXT_LINE_SPACING = 14
HOOK_TEASER_RAW_FILE = "hook-teaser-raw.mp4"
HOOK_TEASER_CROPPED_FILE = "hook-teaser.mp4"
HOOK_TEASER_OUTPUT_FILE = "hook-teaser.mp4"
HOOK_TEASER_MAIN_OUTPUT_FILE = "hook-main.mp4"
HOOK_TEASER_CONCAT_LIST = "tmp/hook-concat.txt"


def parse_hook_fields(segment, seg_start_sec, seg_end_sec):
    """Parse and validate the optional GPT hook fields for one segment.

    Returns (hook_start_sec, hook_end_sec, hook_text) or None. All three
    fields must be present with non-empty text; timestamps are parsed with
    parse_srt_time and the range must sit inside the hook teaser band, be
    strictly increasing, and
    fully inside the segment boundaries.
    """
    if not isinstance(segment, dict):
        return None
    hook_start_raw = segment.get("hook_start_time")
    hook_end_raw = segment.get("hook_end_time")
    hook_text_raw = segment.get("hook_text")
    if not hook_start_raw or not hook_end_raw or hook_text_raw is None:
        return None
    hook_text = str(hook_text_raw).strip()
    if not hook_text:
        return None
    try:
        hook_start = parse_srt_time(hook_start_raw)
        hook_end = parse_srt_time(hook_end_raw)
    except (ValueError, AttributeError, TypeError):
        return None
    if hook_end <= hook_start:
        return None
    duration = hook_end - hook_start
    if duration < HOOK_TEASER_MIN_DURATION or duration > HOOK_TEASER_MAX_DURATION:
        return None
    if hook_start < seg_start_sec or hook_end > seg_end_sec:
        return None
    return hook_start, hook_end, hook_text


def sanitize_hook_text(text):
    """Make hook text safe inside a drawtext text='...' filter string.

    Uppercases (hook style), strips apostrophes (drawtext would otherwise
    break the quoting), and escapes ':' and '%' like drawtext requires.
    Also strips the characters that stay shell-alive inside the double
    quotes wrapping the -vf argument (" ` $ \\), so LLM-derived hook text
    can never break out of the ffmpeg command.
    """
    text = str(text).strip().upper()
    text = text.replace("'", "")
    for ch in ('"', "`", "$", "\\"):
        text = text.replace(ch, "")
    text = text.replace(":", r"\:")
    text = text.replace("%", r"\%")
    return text


def wrap_hook_text(text, max_chars=HOOK_TEXT_MAX_LINE_CHARS):
    """Word-wrap hook text into lines that fit the frame width.

    drawtext renders a single line per filter and has no wrapping, so an
    unsplit hook sentence overflows the 1080 px frame. Returns a list of
    lines (never empty while text is non-empty).
    """
    lines = []
    for word in str(text).split():
        if lines and len(lines[-1]) + 1 + len(word) <= max_chars:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def build_hook_drawtext_filter(hook_text):
    """Burned hook text: large white with black border, dead-center.

    The text is word-wrapped into fixed-width lines burned as one centered
    drawtext per line, stacked vertically around the frame middle.
    """
    text = sanitize_hook_text(hook_text)
    lines = wrap_hook_text(text)
    line_height = HOOK_TEASER_FONT_SIZE + HOOK_TEXT_LINE_SPACING
    total_height = len(lines) * line_height - HOOK_TEXT_LINE_SPACING
    block_top = (1920 - total_height) // 2
    filters = []
    for i, line in enumerate(lines):
        filters.append(
            f"drawtext=text='{line}':fontfile={OVERLAY_FONTFILE}:"
            f"fontsize={HOOK_TEASER_FONT_SIZE}:fontcolor=white:"
            f"borderw=3:bordercolor=black:x=(w-tw)/2:"
            f"y={block_top + i * line_height}"
        )
    return ",".join(filters)


def build_teaser_cut_command(hook_start, hook_end):
    """Cut the hook moment from tmp/input_video.mp4 (NVENC style as
    generate_segments' single-cut path, audio re-encoded to AAC)."""
    return (
        "ffmpeg -y -hwaccel cuda -i tmp/input_video.mp4 "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -c:a aac -b:a 192k "
        f"-ss {hook_start} -to {hook_end} tmp/{HOOK_TEASER_RAW_FILE}"
    )


def build_teaser_burn_command(hook_text):
    """Burn the hook text over the cropped teaser and master its audio
    exactly once (same AUDIO_MASTERING_FILTER chain as the main clip).
    Input is generate_short's audio-muxed final-<cropped> output."""
    return (
        f"ffmpeg -y -hwaccel cuda -i tmp/final-{HOOK_TEASER_CROPPED_FILE} "
        f'-vf "{build_hook_drawtext_filter(hook_text)}" '
        f"{build_audio_filter_args()} "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 "
        "-b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p "
        f"-r {ASSEMBLY_FPS} "
        f"-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE} "
        f"tmp/{HOOK_TEASER_OUTPUT_FILE}"
    )


def build_teaser_concat_command(concat_list_path, published_path):
    """Stream-copy concat of teaser + main into the published path."""
    return (
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(concat_list_path)} "
        f"-c copy {shlex.quote(published_path)}"
    )


def build_teaser_concat_fallback_command(concat_list_path, published_path):
    """Fallback when -c copy concat fails: re-encode VIDEO only; audio is
    copied through, never re-mastered over the concatenation."""
    return (
        f'ffmpeg -y -f concat -safe 0 -i "{concat_list_path}" '
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p "
        f'-c:a copy "{published_path}"'
    )


def build_teaser_concat_full_fallback_command(concat_list_path, published_path):
    """Last-resort assembly fallback (end-card assemblies only): full
    re-encode of video (NVENC) AND audio (plain AAC re-encode) when both
    -c copy and the video-only re-encode fail. Audio is re-encoded WITHOUT
    the mastering chain — never re-mastered over the concatenation."""
    return (
        f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(concat_list_path)} "
        "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 -b:v 10000k "
        "-maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p "
        f"-c:a aac -b:a 192k {shlex.quote(published_path)}"
    )


def generate_hook_teaser(hook_start, hook_end, hook_text):
    """Build the 9:16 hook teaser from tmp/input_video.mp4.

    Cut at [hook_start, hook_end] -> crop to 9:16 via generate_short -> burn
    the hook text + master audio -> tmp/hook-teaser.mp4. Returns the teaser
    path or None when any step fails (caller falls back to today's behavior).
    """
    try:
        command = build_teaser_cut_command(hook_start, hook_end)
        print(f"Hook teaser cut command: {command}")
        result = run_ffmpeg_command(command)
        raw_path = f"tmp/{HOOK_TEASER_RAW_FILE}"
        if (
            result != 0
            or not os.path.exists(raw_path)
            or os.path.getsize(raw_path) == 0
        ):
            print(
                f"Hook teaser: cutting failed (rc={result}); skipping teaser"
            )
            return None

        print("Hook teaser piece cut, cropping to 9:16...")
        cropped = generate_short(HOOK_TEASER_RAW_FILE, HOOK_TEASER_CROPPED_FILE)
        if not cropped:
            print("Hook teaser: 9:16 crop failed; skipping teaser")
            return None

        command = build_teaser_burn_command(hook_text)
        print(f"Hook teaser burn command: {command}")
        result = run_ffmpeg_command(command)
        teaser_path = f"tmp/{HOOK_TEASER_OUTPUT_FILE}"
        if (
            result != 0
            or not os.path.exists(teaser_path)
            or os.path.getsize(teaser_path) == 0
        ):
            print(
                f"Hook teaser: burning hook text failed (rc={result}); "
                "skipping teaser"
            )
            return None
        return teaser_path
    except Exception as e:
        print(f"Hook teaser generation failed: {str(e)}")
        return None


def compute_published_output_path(output_dir, input_file):
    """Published path generate_subtitle derives from the cropped input name
    (shared by the main loop so the teaser concat targets the same file and
    the published `final-*` naming stays unchanged)."""
    output_filename = os.path.basename(input_file)
    base_name = os.path.splitext(output_filename)[0]
    extension = os.path.splitext(output_filename)[1]
    output_filename = sanitize_filename(base_name) + extension
    return f"{output_dir}/{output_filename}"


def assemble_hook_teaser_final(
    teaser_path, main_path, published_path, concat_list_path=None, end_card_path=None
):
    """Concat teaser (if any) + main + end card (if any) into the published
    path with `-c copy`.

    Falls back to a video-only NVENC re-encode (-c:a copy) when the
    stream-copy concat fails, and — only when an end card is part of the
    assembly — to a final full re-encode (NVENC + plain AAC). Audio is never
    re-mastered here.
    """
    concat_list_path = concat_list_path or HOOK_TEASER_CONCAT_LIST
    try:
        # ffmpeg's concat demuxer resolves relative entry paths against the
        # LIST FILE's directory (not the cwd), so the entries must be
        # absolute or both concat attempts would look for tmp/tmp/... .
        entries = [path for path in (teaser_path, main_path, end_card_path) if path]
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(f"file '{os.path.abspath(entry)}'\n")
        command = build_teaser_concat_command(concat_list_path, published_path)
        print(f"Hook teaser concat command: {command}")
        result = run_ffmpeg_command(command)
        if (
            result == 0
            and os.path.exists(published_path)
            and os.path.getsize(published_path) > 0
        ):
            print(f"Hook teaser assembly complete: {published_path}")
            return True
        print(
            f"Hook teaser: concat -c copy failed (rc={result}); "
            "falling back to video re-encode"
        )
        command = build_teaser_concat_fallback_command(
            concat_list_path, published_path
        )
        print(f"Hook teaser fallback command: {command}")
        result = run_ffmpeg_command(command)
        if result == 0 and os.path.exists(published_path):
            print(f"Hook teaser fallback assembly complete: {published_path}")
            return True
        if end_card_path:
            # The silent end card can force a -c:a copy failure (sample-rate
            # mismatch); one last rung re-encodes video AND audio (no
            # mastering) so the assembly still succeeds.
            command = build_teaser_concat_full_fallback_command(
                concat_list_path, published_path
            )
            print(f"Hook teaser full fallback command: {command}")
            result = run_ffmpeg_command(command)
            if result == 0 and os.path.exists(published_path):
                print(
                    f"Hook teaser full fallback assembly complete: "
                    f"{published_path}"
                )
                return True
        print(f"Hook teaser: fallback assembly also failed (rc={result})")
        return False
    except Exception as e:
        print(f"Hook teaser assembly failed: {str(e)}")
        return False


def teaser_duplicates_clip_opening(
    clip_start, hook_start, hook_end, guard=HOOK_DUPLICATE_GUARD_SECONDS
):
    """True when the teaser span replays the clip's own first `guard` seconds.

    B4, defense in depth for the selection-layer guard (B1): the teaser is
    prepended to the clip, so a teaser overlapping the clip's opening plays
    the first words the viewer is about to hear, twice. The teaser range and
    the clip's opening [clip_start, clip_start + guard) must therefore
    intersect; a hook starting exactly at clip_start + guard is already
    outside the opening span and is NOT a duplicate.

    Pure; never raises. An unusable clip_start (None, or an unparseable
    timestamp) means the overlap cannot be evaluated, so the check reports
    "not a duplicate" and the caller behaves exactly as it did before B4.
    """
    try:
        clip_start_s = _srt_like_to_seconds(clip_start)
        hook_start_s = _srt_like_to_seconds(hook_start)
        hook_end_s = _srt_like_to_seconds(hook_end)
        if clip_start_s is None or hook_start_s is None or hook_end_s is None:
            return False
        if hook_end_s <= hook_start_s:
            return False
        opening_end = clip_start_s + guard
        return hook_start_s < opening_end - 1e-9 and hook_end_s > clip_start_s + 1e-9
    except Exception:
        return False


def _normalize_comparable_text(text):
    """Lowercase, collapse whitespace, strip trailing sentence punctuation."""
    collapsed = " ".join(str(text or "").split()).strip().lower()
    return collapsed.rstrip(".,!?;:…\"'")


def opening_caption_prefix(subtitle_words, guard=HOOK_DUPLICATE_GUARD_SECONDS):
    """The clip's opening words, normalized (pure; "" when unknown).

    Reads the karaoke word list (clip-relative seconds) and joins every word
    that starts inside the first `guard` seconds, in order. The window is
    TIME-based, not sentence-based: a terminator-delimited "first sentence"
    is unusable on real content here, because _first_sentence_ends_complete is
    biased toward reporting "complete" on decimals, abbreviations and
    ellipses, and screencast captions are full of numbers — the extract would
    be a fragment ("we set version 2.") that can never equal a hook text. A
    time window also removes the second weakness of the old rule: an opening
    sentence running past 5 s used to make the check inert.
    Words without a usable numeric start are kept (unknown position, not
    known-outside); the scan stops at the first word at or past `guard`.
    Returns "" when there are no usable words, so the caller can treat
    "unknown" as "no match".
    """
    try:
        if not isinstance(subtitle_words, list) or not subtitle_words:
            return ""
        parts = []
        for word in subtitle_words:
            if not isinstance(word, dict):
                continue
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            start = word.get("start")
            if isinstance(start, (int, float)) and not isinstance(start, bool):
                if start >= guard:
                    break
            parts.append(text)
        return _normalize_comparable_text(" ".join(parts))
    except Exception:
        return ""


def _comparable_word_list(text):
    """Normalized text as a list of words, punctuation stripped per word.

    Tokens that normalize away entirely (punctuation-only, e.g. "…" or "--")
    are dropped: left in, two of them aligned at positions 0 and 1 would
    compare equal and fake a two-word match.
    """
    normalized = _normalize_comparable_text(text)
    if not normalized:
        return []
    return [
        stripped
        for stripped in (word.strip(_CAPTION_PUNCTUATION) for word in normalized.split())
        if stripped
    ]


def teaser_repeats_first_caption(hook_text, subtitle_words):
    """True when the hook text restates the clip's own opening words.

    B4's second net for the case where timings shifted between selection and
    publish (e.g. jump cuts) but the words did not. This is the case B1's
    timing net cannot see: a hook cut from the MIDDLE of the clip does not
    overlap the opening span, yet it can still replay the clip's first words.

    Both sides are normalized (lowercased, whitespace collapsed, punctuation
    stripped) and a match only counts when both are non-empty, so a missing
    caption list or a missing hook text leaves the check inert.

    OVERLAP RULE (deliberately conservative, because a false match is the
    expensive direction — it drops a teaser that was fine, while a false miss
    only lets through the repetition this net exists to stop):

      * the two word sequences must share a COMMON PREFIX of at least
        HOOK_CAPTION_MIN_MATCH_WORDS words — one side may be a longer
        version of the other, but the overlap is always anchored at the
        opening's first word, never anywhere in the window. A hook that
        merely repeats a later sentence ("Then it broke.") is not the
        opening and must not trip the net.
      * that word floor is only the absolute minimum; a hook sharing a
        single leading word with the opening is coincidental phrasing, not a
        repetition of the opening. Two is the floor because the exact-match
        rule this replaces fired on two-word openings, so nothing that used
        to be caught is lost.
      * on top of the floor, the shared prefix must cover at least
        HOOK_CAPTION_MIN_MATCH_COVERAGE of the SHORTER of the two sequences.
        Screencast captions open with formulaic bigrams ("so today", "in
        this video"), so two shared words alone no longer mean a repeat — a
        real restatement of the opening covers most of the shorter side, a
        coincidental bigram collision does not.

    Pure; never raises.
    """
    try:
        hook_words = _comparable_word_list(hook_text)
        caption_words = _comparable_word_list(
            opening_caption_prefix(subtitle_words)
        )
        if not hook_words or not caption_words:
            return False
        shared = 0
        for hook_word, caption_word in zip(hook_words, caption_words):
            if hook_word != caption_word:
                break
            shared += 1
        shorter = min(len(hook_words), len(caption_words))
        return shared >= HOOK_CAPTION_MIN_MATCH_WORDS and (
            shared / shorter >= HOOK_CAPTION_MIN_MATCH_COVERAGE
        )
    except Exception:
        return False


def publish_segment_with_optional_teaser(
    final_output_name,
    video_id,
    output_dir,
    video_title=None,
    subtitle_words=None,
    hook=None,
    clip_start_time=None,
):
    """Publish one segment, prepending the hook teaser when hook is valid.

    hook is a parsed (hook_start, hook_end, hook_text) tuple from
    parse_hook_fields, or None. Every failure mode degrades to today's
    direct-publish behavior (generate_subtitle writing the published file):
    teaser generation failure, missing burned main file, and — as the last
    fallback — teaser assembly failure, so the segment never ends without
    a published file.

    clip_start_time is the main clip's own start time (any SRT-like or float
    value) and is B4's only new input. It defaults to None, which leaves the
    overlap check — and therefore the teaser's timing — byte-identical to
    pre-B4 for every existing caller; the caption net below runs whatever
    this is, since it only needs subtitle_words. A teaser that would replay
    the clip's opening — by overlapping its first
    HOOK_DUPLICATE_GUARD_SECONDS, or by restating its opening words — is
    skipped before anything is generated, and the segment falls through to
    the ordinary publish path (direct publish, or the end-card assembly when
    the end card is on).
    """
    direct_publish_kwargs = {
        "video_title": video_title,
        "subtitle_words": subtitle_words,
    }
    published_path = compute_published_output_path(output_dir, final_output_name)

    # P3.2: build the end card once per segment. A failed build degrades to
    # today's behavior (assembly without the end card / direct publish).
    end_card_path = None
    if resolve_end_card_enabled():
        end_card_path = generate_end_card()
        if end_card_path:
            print(f"End card ready: {end_card_path}")

    main_output_path = None
    teaser_path = None
    if hook:
        hook_start, hook_end, hook_text = hook
        # Print via the seconds helper so an unexpected hook field shape
        # cannot turn the skip message into a raise (both predicates are
        # documented as never raising).
        _hook_start_s = _srt_like_to_seconds(hook_start)
        _hook_end_s = _srt_like_to_seconds(hook_end)
        hook_span_log = (
            f"{_hook_start_s:.2f}s-{_hook_end_s:.2f}s"
            if _hook_start_s is not None and _hook_end_s is not None
            else f"{hook_start}-{hook_end}"
        )
        # B4: refuse a teaser that would repeat the clip's own opening. Both
        # checks are pure and the skip is not an error: no teaser file, no
        # burned main for the teaser, straight to the publish path below.
        if teaser_duplicates_clip_opening(
            clip_start_time, hook_start, hook_end
        ):
            print(
                f"Hook teaser: {hook_span_log} overlaps the "
                f"clip's first {HOOK_DUPLICATE_GUARD_SECONDS:.0f}s; publishing "
                "without teaser"
            )
        elif teaser_repeats_first_caption(hook_text, subtitle_words):
            print(
                f"Hook teaser: '{hook_text}' repeats the clip's opening "
                "caption; publishing without teaser"
            )
        else:
            print(f"Hook teaser: {hook_span_log} '{hook_text}'")
            teaser_path = generate_hook_teaser(hook_start, hook_end, hook_text)
            if teaser_path:
                main_output_path = f"tmp/{HOOK_TEASER_MAIN_OUTPUT_FILE}"
            else:
                print("Hook teaser: generation failed; publishing without teaser")

    if not main_output_path and not end_card_path:
        generate_subtitle(
            final_output_name, video_id, output_dir, **direct_publish_kwargs
        )
        return

    if not main_output_path:
        # No teaser but an end card: burn the main clip to tmp/ so the end
        # card can be appended by the concat assembly.
        main_output_path = f"tmp/{HOOK_TEASER_MAIN_OUTPUT_FILE}"

    # Write the burned main clip to tmp/, then concat teaser (if any) and the
    # end card (if any) around it into the published path.
    burned_ok = generate_subtitle(
        final_output_name,
        video_id,
        output_dir,
        main_output_path=main_output_path,
        **direct_publish_kwargs,
    )
    if (
        not burned_ok
        or not os.path.exists(main_output_path)
        or os.path.getsize(main_output_path) == 0
    ):
        print(
            "Hook teaser: burned main file missing; falling back to "
            "direct publish without teaser"
        )
        generate_subtitle(
            final_output_name, video_id, output_dir, **direct_publish_kwargs
        )
        return

    if assemble_hook_teaser_final(
        teaser_path, main_output_path, published_path, end_card_path=end_card_path
    ):
        return
    print(
        "Hook teaser: assembly failed; publishing the video directly "
        "without the teaser"
    )
    generate_subtitle(
        final_output_name, video_id, output_dir, **direct_publish_kwargs
    )


def generate_subtitle(
    input_file,
    video_id,
    output_dir,
    video_title=None,
    subtitle_words=None,
    main_output_path=None,
):
    """Generate subtitle for a video using Whisper and FFmpeg

    When subtitle_words (flat word list already offset to clip-relative time)
    is provided and non-empty, word-level karaoke captions built from the
    master transcript are burned instead of re-transcribing the clip. The
    legacy re-transcription + SRT path is kept as the fallback.

    When main_output_path is provided (hook teaser assembly, Story 7), the
    burned main clip is written to that tmp path INSTEAD of the published
    output_dir; the caller then concats the teaser in front of it. Without
    it the behavior and output path are identical to before.
    """
    # Get channel name from root .env file (shared helper, current-env fallback)
    channel_name = get_channel_name()

    # P3.1/P3.3: resolve the polish features. Both are inert when off/unset
    # (commands byte-identical to the previous behavior).
    grade_filter = resolve_video_grade()
    music_bed = resolve_music_bed()

    # Get current date
    from datetime import datetime

    current_time = datetime.now()
    date_text = (
        current_time.strftime("%d/%m/%Y") if resolve_overlay_date_enabled() else None
    )

    # Use the configured STT model to transcribe the final cropped video segment
    # This ensures subtitles are perfectly synced to the segment timing and uses the same model as the main transcription

    def transcribe_with_configured_model(video_path):
        """Transcribe video using the configured STT model and return SRT content"""
        try:
            # Extract audio from video for transcription
            audio_path = f"tmp/subtitle_audio_{os.path.basename(video_path)}.wav"
            print(f"Extracting audio for subtitle transcription: {audio_path}")
            audio_cmd = (
                f"ffmpeg -y -i {shlex.quote(video_path)} "
                f"-vn -acodec pcm_s16le -ac 1 -ar 16000 {shlex.quote(audio_path)}"
            )
            run_ffmpeg_command(audio_cmd)

            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                print(f"Failed to extract audio from {video_path}")
                return None

            # Use the same transcription service as the main transcript generation
            from transcription_service import get_transcriber

            print("Getting configured transcriber for subtitle generation...")
            transcriber = get_transcriber()
            print(f"Using {transcriber.name} for subtitle transcription...")

            # Transcribe using the configured service
            result = transcriber.transcribe(audio_path)

            # Clean up temporary audio file
            try:
                os.remove(audio_path)
            except OSError:
                pass

            # Generate SRT content from transcriber result
            srt_content = ""
            for i, segment in enumerate(result):
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

            print(f"Successfully transcribed with {transcriber.name}")
            return srt_content

        except Exception as e:
            print(f"Error in subtitle transcription with configured model: {str(e)}")
            print("Falling back to legacy Whisper implementation...")

            # Fallback to legacy Whisper implementation
            try:
                import whisper
                models_to_try = ["medium", "small", "base"]

                for model_name in models_to_try:
                    try:
                        print(f"Loading Whisper {model_name} model...")
                        model = whisper.load_model(os.getenv('WHISPER_MODEL', model_name))

                        print(f"Transcribing video with {model_name} model...")

                        # Use configured language or auto-detect
                        transcription_language = os.getenv('TRANSCRIPTION_LANGUAGE', None)
                        if transcription_language:
                            print(f"Using configured language: {transcription_language}")
                            result = model.transcribe(video_path, word_timestamps=True, language=transcription_language)
                        else:
                            print("Using automatic language detection")
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

                        print(f"Successfully transcribed with fallback {model_name} model")
                        return srt_content

                    except Exception as fallback_e:
                        print(f"Failed with {model_name} model: {str(fallback_e)}")
                        if "CUDA out of memory" in str(fallback_e):
                            print(f"GPU memory issue with {model_name}, trying smaller model...")
                            continue
                        else:
                            break

                print("All fallback Whisper models failed")
                return None
            except Exception as fallback_error:
                print(f"Error in fallback Whisper transcription: {str(fallback_error)}")
                return None

    # Prefer karaoke captions from the master transcript words (no
    # re-transcription); fall back to the legacy SRT path otherwise.
    input_base = sanitize_filename(os.path.basename(input_file).split(".")[0])
    temp_ass_file = None
    if subtitle_words:
        ass_content = build_karaoke_ass(subtitle_words)
        if ass_content:
            temp_ass_file = _contained_path("tmp", f"{input_base}.ass")
            try:
                with open(temp_ass_file, "w", encoding="utf-8") as f:
                    f.write(ass_content)
                print(f"Karaoke subtitle file created: {temp_ass_file}")
            except IOError as e:
                print(f"Error writing karaoke subtitle file: {str(e)}")
                temp_ass_file = None
        else:
            print("No word timestamps available, falling back to SRT subtitles")

    temp_srt_file = _contained_path("tmp", f"{input_base}.srt")
    if temp_ass_file:
        srt_content = None
        temp_srt_file = None
    else:
        # Generate subtitles using the configured STT model
        print(f"Transcribing video: tmp/{input_file}")
        srt_content = transcribe_with_configured_model(f"tmp/{input_file}")

        if srt_content:
            with open(temp_srt_file, "w", encoding="utf-8") as f:
                f.write(srt_content)
            print(f"Subtitle file created: {temp_srt_file}")
        else:
            print("No subtitles generated, will create video without subtitles")
            temp_srt_file = None

    # Create output path for final video with overlays
    output_path = compute_published_output_path(output_dir, input_file)
    if main_output_path:
        print(f"Hook teaser: writing burned main clip to {main_output_path}")
        output_path = main_output_path

    # Check if subtitle file was generated
    if temp_ass_file:
        print(f"Using karaoke subtitle file: {temp_ass_file}")
        # libass reads styles from the ASS file itself, so no force_style.
        if grade_filter:
            # P3.1: grade FIRST, captions burn on top of the graded image.
            filter_parts = [
                f"{grade_filter}[v0]",
                f"[v0]subtitles='{temp_ass_file}'[v1]",
            ]
        else:
            filter_parts = [f"subtitles='{temp_ass_file}'[v1]"]
        last_filter = "v1"
    elif not temp_srt_file or not os.path.exists(temp_srt_file):
        print(f"Warning: No subtitle file available, creating video without subtitles")
        # Create video with only overlays (no subtitles)
        if grade_filter:
            filter_parts = [f"{grade_filter}[v0]"]
            last_filter = "v0"
        else:
            filter_parts = []
            last_filter = "0:v"
    else:
        print(f"Using subtitle file: {temp_srt_file}")
        # Build filter complex starting with subtitles
        srt_subtitles = f"subtitles='{temp_srt_file}':force_style='Alignment=2,MarginV=130,MarginL=55,MarginR=55,Fontname=Noto Sans,Fontsize=6,PrimaryColour=&H00ffffff,SecondaryColour=&H000000ff,OutlineColour=&H00000000,BackColour=&H80000000,Outline=1.5,Shadow=1.5,BorderStyle=1'[v1]"
        if grade_filter:
            filter_parts = [f"{grade_filter}[v0]", f"[v0]{srt_subtitles}"]
        else:
            filter_parts = [srt_subtitles]
        last_filter = "v1"

    # Add the restyled overlays: date (top-left) and video title or channel
    # name (top-right), semi-transparent inside the top safe zone.
    if video_title:
        print(f"Using video title for overlay: {video_title}")
    title_text = (
        resolve_overlay_title(video_title, channel_name)
        if resolve_overlay_title_enabled()
        else None
    )
    overlay_filters, last_filter = build_overlay_filters(
        date_text, title_text, input_label=last_filter
    )
    filter_parts.extend(overlay_filters)

    # Build final FFmpeg command with proper spacing and quotes
    if filter_parts or music_bed:
        if music_bed:
            # P3.3: bed mixed UNDER the speech, then mastered exactly once;
            # the video chain itself is unchanged.
            audio_complex = build_music_bed_audio_filters()
            video_complex = ";".join(filter_parts) if filter_parts else None
            filter_complex = (
                f"{video_complex};{audio_complex}" if video_complex else audio_complex
            )
            video_map = (
                f'-map "[{last_filter}]"' if filter_parts else "-map 0:v"
            )
            video_codec = (
                "-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 "
                "-b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p "
                f"-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE}"
                if filter_parts
                else "-c copy"
            )
            ffmpeg_command = (
                f"ffmpeg -y -hwaccel cuda -i tmp/{shlex.quote(input_file)} "
                f'-stream_loop -1 -i {shlex.quote(music_bed)} '
                f'-filter_complex "{filter_complex}" '
                f'{video_map} -map "[aout2]" '
                f"{video_codec} "
                # C3: this rung used to emit bare `-c:a aac -b:a 192k`, so it
                # kept the capture's 44100 rate while every other piece was
                # 48000. `concat -c copy` does not fail on that — it adopts
                # the first stream's parameters and emits a container longer
                # than the sum of the pieces. Pin the shared shape here.
                f"-c:a aac -b:a 192k -ar {ASSEMBLY_AUDIO_RATE} "
                f"-ac {ASSEMBLY_AUDIO_CHANNELS} {shlex.quote(output_path)}"
            )
        elif filter_parts:
            filter_complex = ";".join(filter_parts)
            ffmpeg_command = (
                f"ffmpeg -y -hwaccel cuda -i tmp/{shlex.quote(input_file)} "
                f'-filter_complex "{filter_complex}" '
                f'-map "[{last_filter}]" -map 0:a '
                f'-c:v h264_nvenc -profile:v high -rc:v vbr -qp 18 '
                f'-b:v 10000k -maxrate:v 12000k -bufsize:v 15000k -pix_fmt yuv420p '
                f'-video_track_timescale {ASSEMBLY_VIDEO_TIMESCALE} '
                f'{build_audio_filter_args()} {shlex.quote(output_path)}'
            )
    else:
        # No overlays, just copy
        ffmpeg_command = (
            f"ffmpeg -y -hwaccel cuda -i tmp/{shlex.quote(input_file)} "
            f"-c copy {shlex.quote(output_path)}"
        )

    print(f"Applying overlays with command: {ffmpeg_command}")
    
    # Add error handling and logging
    try:
        result = run_ffmpeg_command(ffmpeg_command)
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
