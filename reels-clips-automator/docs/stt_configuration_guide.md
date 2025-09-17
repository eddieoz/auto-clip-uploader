# Speech-to-Text (STT) Configuration Guide

This guide provides comprehensive instructions for configuring and using the Speech-to-Text functionality in the auto-clip-uploader system.

## Overview

The system supports multiple STT models with flexible configuration options:

- **Whisper**: Standard OpenAI Whisper implementation (default)
- **Faster Whisper**: Optimized implementation for improved performance
- **Voxtral**: Alternative local STT model (legacy support)

## Quick Start

### Basic Configuration

1. **Default Setup (Whisper)**:
   ```bash
   # No configuration needed - Whisper is the default
   # Optional: specify model size
   echo "WHISPER_MODEL=small" >> reels-clips-automator/.env
   ```

2. **Enable Faster Whisper**:
   ```bash
   # Enable Faster Whisper for improved performance
   echo "STT_MODEL=faster-whisper" >> reels-clips-automator/.env
   echo "FASTER_WHISPER_MODEL_SIZE=small" >> reels-clips-automator/.env
   ```

3. **Install Dependencies**:
   ```bash
   # For Whisper (default)
   pip install openai-whisper

   # For Faster Whisper
   pip install faster-whisper

   # For CUDA support (optional, for GPU acceleration)
   pip install faster-whisper[cuda]
   ```

## Environment Variables Reference

### Core STT Configuration

#### `STT_MODEL`
- **Purpose**: Selects which STT engine to use
- **Values**:
  - `"whisper"` (default): Standard OpenAI Whisper
  - `"faster-whisper"`: Optimized Faster Whisper
  - `"voxtral"`: Legacy Voxtral model
- **Example**: `STT_MODEL=faster-whisper`

### Whisper Configuration

#### `WHISPER_MODEL`
- **Purpose**: Specifies the Whisper model size
- **Values**: `tiny`, `base`, `small`, `medium`, `large`
- **Default**: `small`
- **Trade-offs**:
  - `tiny`: Fastest, least accurate, ~39 MB
  - `base`: Fast, basic accuracy, ~74 MB
  - `small`: Balanced, good accuracy, ~244 MB
  - `medium`: Slower, better accuracy, ~769 MB
  - `large`: Slowest, best accuracy, ~1550 MB
- **Example**: `WHISPER_MODEL=medium`

### Faster Whisper Configuration

#### `FASTER_WHISPER_MODEL_SIZE`
- **Purpose**: Specifies the Faster Whisper model size
- **Values**: `tiny`, `base`, `small`, `medium`, `large`
- **Default**: `small`
- **Example**: `FASTER_WHISPER_MODEL_SIZE=medium`

#### `FASTER_WHISPER_DEVICE`
- **Purpose**: Specifies the computing device
- **Values**:
  - `"auto"` (default): Automatically select best available device
  - `"cpu"`: Force CPU processing
  - `"cuda"`: Force CUDA/GPU processing
- **Example**: `FASTER_WHISPER_DEVICE=cuda`

#### `FASTER_WHISPER_COMPUTE_TYPE`
- **Purpose**: Specifies the computation precision
- **Values**:
  - `"default"` (default): Auto-select based on device
  - `"int8"`: 8-bit integer (faster, less memory, slight quality loss)
  - `"float16"`: 16-bit float (CUDA only, good balance)
  - `"float32"`: 32-bit float (highest quality, more memory)
- **Example**: `FASTER_WHISPER_COMPUTE_TYPE=float16`

## Configuration Examples

### Development Environment

```bash
# reels-clips-automator/.env
# Balanced configuration for development

# Use Faster Whisper for improved performance
STT_MODEL=faster-whisper

# Small model for faster processing during development
FASTER_WHISPER_MODEL_SIZE=small

# Auto-detect best device (CPU/GPU)
FASTER_WHISPER_DEVICE=auto

# Default compute type
FASTER_WHISPER_COMPUTE_TYPE=default

# OpenAI API for viral detection
OPENAI_API_KEY=your_api_key_here
```

### Production Environment (GPU Server)

```bash
# reels-clips-automator/.env
# High-performance configuration for production

# Use Faster Whisper for optimal performance
STT_MODEL=faster-whisper

# Medium model for good accuracy/speed balance
FASTER_WHISPER_MODEL_SIZE=medium

# Use GPU acceleration
FASTER_WHISPER_DEVICE=cuda

# 16-bit precision for GPU efficiency
FASTER_WHISPER_COMPUTE_TYPE=float16

# Production API key
OPENAI_API_KEY=your_production_api_key
```

### CPU-Only Environment

```bash
# reels-clips-automator/.env
# Configuration for CPU-only environments

# Use Faster Whisper
STT_MODEL=faster-whisper

# Small model for CPU processing
FASTER_WHISPER_MODEL_SIZE=small

# Force CPU processing
FASTER_WHISPER_DEVICE=cpu

# 8-bit for CPU efficiency
FASTER_WHISPER_COMPUTE_TYPE=int8

# API configuration
OPENAI_API_KEY=your_api_key_here
```

### High-Accuracy Environment

```bash
# reels-clips-automator/.env
# Configuration for maximum transcription accuracy

# Use standard Whisper for proven accuracy
STT_MODEL=whisper

# Large model for best accuracy
WHISPER_MODEL=large

# API configuration
OPENAI_API_KEY=your_api_key_here
```

### Legacy Environment

```bash
# reels-clips-automator/.env
# Fallback to legacy Voxtral if needed

# Use Voxtral (legacy)
STT_MODEL=voxtral

# Voxtral-specific configuration
VOXTRAL_QUANTIZATION=Q4_K_M
VOXTRAL_DEVICE=cuda

# API configuration
OPENAI_API_KEY=your_api_key_here
```

## Hardware Considerations

### GPU Requirements

For CUDA acceleration with Faster Whisper:

| Model Size | Minimum VRAM | Recommended VRAM | Performance Gain |
|-----------|--------------|-----------------|------------------|
| tiny | 1 GB | 2 GB | 2-3x faster |
| base | 1 GB | 2 GB | 2-3x faster |
| small | 2 GB | 4 GB | 3-4x faster |
| medium | 3 GB | 6 GB | 3-5x faster |
| large | 5 GB | 8 GB | 4-6x faster |

### CPU Requirements

For CPU-only processing:

| Model Size | Minimum RAM | Recommended RAM | Cores |
|-----------|-------------|----------------|-------|
| tiny | 2 GB | 4 GB | 2+ |
| base | 2 GB | 4 GB | 2+ |
| small | 4 GB | 8 GB | 4+ |
| medium | 6 GB | 12 GB | 6+ |
| large | 8 GB | 16 GB | 8+ |

## Performance Optimization

### For Speed

1. **Use Faster Whisper**:
   ```bash
   STT_MODEL=faster-whisper
   ```

2. **Enable GPU acceleration**:
   ```bash
   FASTER_WHISPER_DEVICE=cuda
   FASTER_WHISPER_COMPUTE_TYPE=float16
   ```

3. **Choose smaller models**:
   ```bash
   FASTER_WHISPER_MODEL_SIZE=small
   ```

### For Accuracy

1. **Use larger models**:
   ```bash
   WHISPER_MODEL=large
   # or
   FASTER_WHISPER_MODEL_SIZE=large
   ```

2. **Use higher precision**:
   ```bash
   FASTER_WHISPER_COMPUTE_TYPE=float32
   ```

### For Memory Efficiency

1. **Use quantized models**:
   ```bash
   FASTER_WHISPER_COMPUTE_TYPE=int8
   ```

2. **Use smaller models**:
   ```bash
   FASTER_WHISPER_MODEL_SIZE=tiny
   ```

3. **Force CPU processing** (if GPU memory is limited):
   ```bash
   FASTER_WHISPER_DEVICE=cpu
   ```

## Troubleshooting

### Common Issues

#### 1. "faster_whisper module not found"

**Problem**: Faster Whisper package not installed.

**Solution**:
```bash
pip install faster-whisper
```

#### 2. "CUDA out of memory"

**Problem**: GPU doesn't have enough VRAM for the selected model.

**Solutions**:
- Use a smaller model: `FASTER_WHISPER_MODEL_SIZE=small`
- Use lower precision: `FASTER_WHISPER_COMPUTE_TYPE=int8`
- Force CPU processing: `FASTER_WHISPER_DEVICE=cpu`

#### 3. "No transcription service available"

**Problem**: No STT engine is properly installed or configured.

**Solution**:
```bash
# Install at least one STT engine
pip install openai-whisper
# or
pip install faster-whisper
```

#### 4. Poor transcription quality

**Problem**: Transcription results are inaccurate.

**Solutions**:
- Use a larger model: `FASTER_WHISPER_MODEL_SIZE=medium`
- Improve audio quality (preprocess audio)
- Use standard Whisper for comparison: `STT_MODEL=whisper`

#### 5. Slow processing speed

**Problem**: Transcription takes too long.

**Solutions**:
- Use Faster Whisper: `STT_MODEL=faster-whisper`
- Enable GPU: `FASTER_WHISPER_DEVICE=cuda`
- Use smaller model: `FASTER_WHISPER_MODEL_SIZE=small`

### Debug Mode

Enable debug logging to troubleshoot issues:

```bash
# Add to your environment
export PYTHONPATH="."
export LOG_LEVEL=DEBUG

# Run with debug output
python reelsfy.py --file input.mp4 --debug
```

### Validation Commands

Test your configuration:

```bash
# Test STT service availability
cd reels-clips-automator
python -c "
from transcription_service import get_transcriber
transcriber = get_transcriber()
print(f'Using: {transcriber.name}')
print(f'Available: {transcriber.is_available()}')
"

# Test basic transcription
python -c "
from transcription_service import get_transcriber
transcriber = get_transcriber()
# Use a short audio file for testing
result = transcriber.transcribe('path/to/short_audio.wav')
print(f'Transcribed {len(result)} segments')
"
```

## Migration Guide

### From Whisper to Faster Whisper

1. **Install Faster Whisper**:
   ```bash
   pip install faster-whisper
   ```

2. **Update configuration**:
   ```bash
   # Change from:
   # WHISPER_MODEL=small

   # To:
   STT_MODEL=faster-whisper
   FASTER_WHISPER_MODEL_SIZE=small
   ```

3. **Test configuration**:
   ```bash
   python reelsfy.py --file test_video.mp4
   ```

### From Voxtral to Faster Whisper

1. **Install Faster Whisper**:
   ```bash
   pip install faster-whisper
   ```

2. **Update configuration**:
   ```bash
   # Change from:
   # STT_MODEL=voxtral

   # To:
   STT_MODEL=faster-whisper
   FASTER_WHISPER_MODEL_SIZE=small
   FASTER_WHISPER_DEVICE=auto
   ```

## Best Practices

### Development

- Use `small` models for faster iteration
- Enable debug logging for troubleshooting
- Test with short audio clips first

### Production

- Use `medium` or `large` models for better accuracy
- Enable GPU acceleration when available
- Monitor memory usage and processing times
- Set up fallback mechanisms

### Testing

- Validate configuration changes with known audio files
- Compare transcription quality between models
- Benchmark performance before deploying
- Test error handling scenarios

## Support

For additional help:

1. **Check logs**: Review application logs for error messages
2. **Validate hardware**: Ensure your system meets minimum requirements
3. **Test configuration**: Use validation commands to verify setup
4. **Consult documentation**: Review performance benchmarks and troubleshooting guides

## Future Considerations

### Upcoming Features

- Additional STT model support
- Automatic model selection based on content
- Dynamic quality/speed optimization
- Multi-language support improvements

### Performance Improvements

- Model quantization options
- Batch processing capabilities
- Streaming transcription support
- Custom model fine-tuning