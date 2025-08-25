# Snapdragon Hardware Compatibility Fix

## Problem Description

When running reelsfy.py on Snapdragon devices, the script failed because it was trying to use CUDA hardware acceleration, which is not available on ARM-based Snapdragon processors:

```
[AVHWDeviceContext @ 0x3000094880] Cannot load libcuda.so.1
[AVHWDeviceContext @ 0x3000094880] Could not dynamically load CUDA
Device creation failed: -1.
[h264 @ 0x30001fb100] No device available for decoder: device type cuda needed for codec h264.
Device setup failed for decoder on input stream #0:0 : Operation not permitted
```

This occurred because all FFmpeg commands were hardcoded to use:
- `-hwaccel cuda` (CUDA hardware acceleration)
- `-c:v h264_nvenc` (NVIDIA NVENC encoder)

## Solution Implemented

### 1. Hardware Detection System

Added intelligent hardware detection that automatically configures FFmpeg parameters based on:
- **Snapdragon Detection**: Uses `USE_QAI_HUB_WHISPER=true` as indicator
- **CUDA Availability**: Tests if CUDA is actually available
- **Alternative Hardware**: Detects VideoToolbox, VAAPI, Quick Sync, etc.
- **Software Fallback**: Uses libx264 when no hardware acceleration available

### 2. Dynamic FFmpeg Command Generation

**Before (Hardcoded CUDA):**
```python
command = f"ffmpeg -y -hwaccel cuda -i tmp/input.mp4 -c:v h264_nvenc -preset slow ..."
```

**After (Dynamic Hardware):**
```python
command = f"ffmpeg -y {HW_CONFIG['hwaccel']} -i tmp/input.mp4 -c:v {HW_CONFIG['video_encoder']} -preset {HW_CONFIG['preset']} {HW_CONFIG['additional_params']} ..."
```

### 3. Hardware Configuration Profiles

#### Snapdragon/QAI Hub Mode (`USE_QAI_HUB_WHISPER=true`)

**V4L2 Hardware Acceleration (when available):**
```python
{
    'hwaccel': '-hwaccel v4l2m2m',      # V4L2 hardware acceleration
    'video_encoder': 'h264_v4l2m2m',    # V4L2 H.264 encoder
    'preset': 'medium',                 # V4L2 optimized
    'additional_params': '-b:v 5000k -maxrate:v 6000k -bufsize:v 8000k'
}
```

**Software Fallback:**
```python
{
    'hwaccel': '',                      # No hardware acceleration
    'video_encoder': 'libx264',         # Software H.264 encoder
    'preset': 'faster',                 # Mobile-optimized preset
    'additional_params': '-crf 25 -tune zerolatency'  # Mobile settings
}
```

#### CUDA Available
```python
{
    'hwaccel': '-hwaccel cuda',
    'video_encoder': 'h264_nvenc',
    'preset': 'slow',
    'additional_params': '-rc:v vbr_hq -qp 18 -b:v 10000k -maxrate:v 12000k -bufsize:v 15000k'
}
```

#### Software Fallback
```python
{
    'hwaccel': '',
    'video_encoder': 'libx264',
    'preset': 'medium', 
    'additional_params': '-crf 23'
}
```

## Changes Made

### File: `reelsfy.py`

1. **Added `detect_hardware_acceleration()` function** (Lines 127-196)
   - Detects Snapdragon via QAI Hub environment variable
   - Tests CUDA availability with actual device creation
   - Detects alternative hardware acceleration (VideoToolbox, VAAPI, etc.)
   - Provides software fallback

2. **Updated all FFmpeg commands to use dynamic configuration:**
   - **Line 240**: `generate_segments()` - Video segmentation
   - **Line 573**: `generate_short()` - Audio extraction
   - **Line 577**: `generate_short()` - Video + audio merging
   - **Line 1709**: `generate_subtitle()` - Video with subtitles/overlays
   - **Line 1716**: `generate_subtitle()` - Video copy operation

3. **Global hardware configuration** (Line 200)
   - `HW_CONFIG` initialized once at startup
   - Used throughout all video processing operations

## Testing Results

### Hardware Detection Test Results:
```
✅ QAI Hub Detection: Correctly uses software encoding
✅ CUDA Detection: Properly detects and uses NVIDIA hardware
✅ Fallback: Uses libx264 when no hardware available
✅ Command Generation: Creates proper FFmpeg commands
```

### Encoder Availability:
```
✅ Software H.264 (libx264) - Always available
✅ NVIDIA NVENC (h264_nvenc) - CUDA systems only
✅ Intel/AMD VAAPI (h264_vaapi) - Linux systems
❌ Apple VideoToolbox (h264_videotoolbox) - macOS/iOS only
❌ Intel Quick Sync (h264_qsv) - Intel systems only
```

## Usage

### For Snapdragon Devices:
```bash
export USE_QAI_HUB_WHISPER=true
python reelsfy.py --file video.mp4
```

Expected output:
```
QAI Hub detected - using software encoding for Snapdragon compatibility
```

### For Regular Systems:
```bash
# Auto-detects best available hardware
python reelsfy.py --file video.mp4
```

Expected output:
```
Testing CUDA availability...
CUDA hardware acceleration available
```

## Performance Impact

### Snapdragon (Software Encoding):
- **Pros**: Compatible with ARM processors, no driver dependencies
- **Cons**: Slower encoding, higher CPU usage
- **Quality**: Excellent with CRF 23 setting

### CUDA (Hardware Encoding):
- **Pros**: Fast encoding, low CPU usage
- **Cons**: Requires NVIDIA GPU and CUDA drivers
- **Quality**: High with VBR rate control

## Backwards Compatibility

✅ **Fully backward compatible**
- Existing CUDA systems continue to use hardware acceleration
- No configuration changes needed for existing setups
- Automatic detection means no user intervention required

## Error Handling

The system gracefully handles:
- Missing CUDA drivers
- Unavailable hardware encoders
- FFmpeg encoder detection failures
- Automatic fallback to software encoding

## Alternative Solutions Considered

1. **Conditional compilation**: Too complex, not portable
2. **Runtime flags**: User burden to configure correctly
3. **Platform detection**: Less reliable than capability testing
4. **Configuration files**: Additional complexity

The implemented solution using environment variable detection (`USE_QAI_HUB_WHISPER`) combined with runtime testing provides the best balance of simplicity and reliability.

## Files Modified

- `reelsfy.py`: Added hardware detection and updated all FFmpeg commands
- `test_hardware_detection.py`: Comprehensive testing suite
- `SNAPDRAGON_CUDA_FIX.md`: This documentation

## Dependencies

No additional dependencies required. Uses existing:
- `subprocess` for FFmpeg testing
- `os.getenv()` for environment variable detection
- Standard Python libraries