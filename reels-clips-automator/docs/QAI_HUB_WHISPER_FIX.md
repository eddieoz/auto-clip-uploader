# QAI Hub Whisper Sample Rate Fix

## Issue Description

When running reelsfy.py on Snapdragon with QAI Hub WhisperSmallV2, users encountered this error:

```
Loading Whisper model...
Attempting to load QAI Hub WhisperSmallV2...
QAI Hub WhisperSmallV2 loaded successfully
Using QAI Hub WhisperSmallV2 (Snapdragon optimized)
Transcribing audio...
Error in QAI Hub transcription: 
...
AssertionError
```

The error occurred in `/qai_hub_models/models/_shared/hf_whisper/app.py` at line 77:
```python
assert audio_sample_rate is not None
```

## Root Cause

The QAI Hub `HfWhisperApp.transcribe()` method expects two parameters when transcribing a numpy array:
1. `audio`: The audio data as numpy array
2. `audio_sample_rate`: The sample rate as an integer

Our original implementation was only passing the audio data without the sample rate parameter.

## Fix Applied

### 1. Updated `_transcribe_qai_hub()` method

**Before:**
```python
audio = self._load_audio_for_qai_hub(audio_path)
transcription = self.qai_hub_app.transcribe(audio)
```

**After:**
```python
target_sr = 16000  # Standard sample rate for Whisper models
audio = self._load_audio_for_qai_hub(audio_path, target_sr)
transcription = self.qai_hub_app.transcribe(audio, audio_sample_rate=target_sr)
```

### 2. Enhanced `_load_audio_for_qai_hub()` method

Added comprehensive validation and debugging:
- File existence and size validation
- Explicit sample rate handling with librosa
- Audio format validation (float32)
- Better error reporting and debugging output

### 3. Improved Duration Calculation

**Before:**
```python
"end": len(audio) / 16000,  # Hard-coded sample rate
```

**After:**
```python
"end": len(audio) / target_sr,  # Use actual sample rate
```

### 4. Added Debug Output

The fix includes detailed logging to help troubleshoot issues:
- Audio file loading progress
- Audio data characteristics (shape, dtype, sample rate)
- QAI Hub transcription call parameters
- Success/failure notifications

## Changes Made

### File: `whisper_qai_hub.py`

1. **Lines 109-120**: Updated `_transcribe_qai_hub()` to pass `audio_sample_rate` parameter
2. **Lines 57-89**: Enhanced `_load_audio_for_qai_hub()` with validation and debugging
3. **Line 127**: Fixed duration calculation to use actual sample rate
4. Added comprehensive error handling and debug output

## Testing

The fix addresses the specific AssertionError by ensuring:
1. ✅ Audio sample rate is explicitly provided to QAI Hub transcribe method
2. ✅ Audio is loaded with correct target sample rate (16000 Hz)
3. ✅ Audio data is validated and formatted correctly (float32)
4. ✅ Debug output helps identify any remaining issues

## Usage

The fix is backward compatible. Users on Snapdragon can now:

```bash
# Enable QAI Hub Whisper
export USE_QAI_HUB_WHISPER=true
python reelsfy.py --file video.mp4
```

Expected output:
```
Loading Whisper model...
Attempting to load QAI Hub WhisperSmallV2...
QAI Hub WhisperSmallV2 loaded successfully
Using QAI Hub WhisperSmallV2 (Snapdragon optimized)
Transcribing audio...
Loading audio file: /tmp/audio.wav (target_sr=16000)
Audio loaded: shape=(48000,), dtype=float32, sample_rate=16000
Calling QAI Hub transcribe with audio_sample_rate=16000
QAI Hub transcription completed successfully
```

## Fallback Behavior

If QAI Hub transcription still fails for any reason, the system will automatically fall back to standard Whisper:

```python
except Exception as e:
    print(f"QAI Hub transcription failed: {e}")
    print("Falling back to standard Whisper...")
    use_qai_hub = False
```

This ensures the processing continues even if there are unforeseen compatibility issues.

## Environment Setup

For Snapdragon devices, ensure these packages are installed:

```bash
pip install qai_hub_models[whisper_small_v2]
# or
pip install qai_hub_models
```

Then set the environment variable:
```bash
export USE_QAI_HUB_WHISPER=true
```