# STT Performance Benchmarks

This document provides performance benchmarks and comparisons between different Speech-to-Text (STT) models available in the auto-clip-uploader system.

## Overview

The system supports two STT models:
- **Whisper**: Standard OpenAI Whisper implementation
- **Faster Whisper**: Optimized implementation using the faster-whisper library

## Performance Targets

Based on the faster-whisper epic requirements:

| Metric | Whisper (Baseline) | Faster Whisper (Target) | Improvement Target |
|--------|-------------------|-------------------------|-------------------|
| **Speed** | 1.0x | 2-4x faster | 2-4x improvement |
| **Memory** | Baseline | 50% reduction | 50% less peak memory |
| **Quality** | 100% | >95% similarity | Maintain quality |
| **Compatibility** | 100% | 100% | No breaking changes |

## Benchmark Test Suite

### Test Audio Characteristics

The benchmark suite should test with various audio characteristics:

1. **Duration Tests**:
   - Short clips (5-30 seconds)
   - Medium clips (1-5 minutes)
   - Long clips (5+ minutes)

2. **Audio Quality Tests**:
   - High quality (44.1kHz, 16-bit)
   - Medium quality (22kHz, 16-bit)
   - Low quality (8kHz, 16-bit)

3. **Content Tests**:
   - Clear speech
   - Speech with background noise
   - Multiple speakers
   - Technical content
   - Casual conversation

### Performance Metrics

#### 1. Processing Speed

```python
# Example benchmark code structure
import time
from transcription_service import get_transcriber

def benchmark_speed(audio_path, model_type):
    transcriber = get_transcriber(model_type)

    start_time = time.time()
    result = transcriber.transcribe(audio_path)
    end_time = time.time()

    processing_time = end_time - start_time
    audio_duration = get_audio_duration(audio_path)
    real_time_factor = processing_time / audio_duration

    return {
        'model': model_type,
        'audio_duration': audio_duration,
        'processing_time': processing_time,
        'real_time_factor': real_time_factor,
        'segments_count': len(result)
    }
```

#### 2. Memory Usage

```python
# Example memory monitoring structure
import psutil
import os

def benchmark_memory(audio_path, model_type):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    transcriber = get_transcriber(model_type)
    model_loaded_memory = process.memory_info().rss / 1024 / 1024  # MB

    result = transcriber.transcribe(audio_path)
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB

    return {
        'model': model_type,
        'initial_memory_mb': initial_memory,
        'model_loaded_memory_mb': model_loaded_memory,
        'peak_memory_mb': peak_memory,
        'memory_increase_mb': peak_memory - initial_memory
    }
```

#### 3. Transcription Quality

```python
# Example quality comparison structure
from difflib import SequenceMatcher

def benchmark_quality(audio_path, reference_transcription=None):
    whisper_result = get_transcriber('whisper').transcribe(audio_path)
    faster_whisper_result = get_transcriber('faster-whisper').transcribe(audio_path)

    whisper_text = ' '.join([seg['text'] for seg in whisper_result])
    faster_whisper_text = ' '.join([seg['text'] for seg in faster_whisper_result])

    similarity = SequenceMatcher(None, whisper_text, faster_whisper_text).ratio()

    quality_metrics = {
        'text_similarity': similarity,
        'whisper_segments': len(whisper_result),
        'faster_whisper_segments': len(faster_whisper_result),
        'segment_count_diff': abs(len(whisper_result) - len(faster_whisper_result))
    }

    if reference_transcription:
        whisper_accuracy = SequenceMatcher(None, reference_transcription, whisper_text).ratio()
        faster_whisper_accuracy = SequenceMatcher(None, reference_transcription, faster_whisper_text).ratio()
        quality_metrics.update({
            'whisper_accuracy': whisper_accuracy,
            'faster_whisper_accuracy': faster_whisper_accuracy
        })

    return quality_metrics
```

## Expected Benchmark Results

### Speed Benchmarks

| Audio Duration | Whisper Time | Faster Whisper Time | Speed Improvement |
|---------------|--------------|-------------------|------------------|
| 30 seconds | 8.5s | 3.2s | 2.7x faster |
| 2 minutes | 32s | 11s | 2.9x faster |
| 5 minutes | 78s | 24s | 3.3x faster |
| 10 minutes | 165s | 45s | 3.7x faster |

*Note: Actual results will vary based on hardware configuration and audio content.*

### Memory Benchmarks

| Model Size | Whisper Peak Memory | Faster Whisper Peak Memory | Memory Reduction |
|-----------|-------------------|----------------------------|-----------------|
| Small | 1.2 GB | 640 MB | 47% reduction |
| Medium | 2.1 GB | 1.1 GB | 48% reduction |
| Large | 3.8 GB | 2.0 GB | 47% reduction |

### Quality Benchmarks

| Test Category | Text Similarity | Timestamp Accuracy | Quality Score |
|--------------|----------------|-------------------|---------------|
| Clear Speech | >98% | <0.1s difference | Excellent |
| Background Noise | >95% | <0.2s difference | Very Good |
| Multiple Speakers | >93% | <0.3s difference | Good |
| Technical Content | >96% | <0.1s difference | Very Good |

## Hardware Configurations

### Recommended for Faster Whisper

1. **GPU Configuration**:
   - CUDA-compatible GPU with 4GB+ VRAM
   - Environment: `FASTER_WHISPER_DEVICE=cuda`
   - Compute type: `FASTER_WHISPER_COMPUTE_TYPE=float16`

2. **CPU Configuration**:
   - Multi-core CPU (8+ cores recommended)
   - Environment: `FASTER_WHISPER_DEVICE=cpu`
   - Compute type: `FASTER_WHISPER_COMPUTE_TYPE=int8`

3. **Memory Requirements**:
   - Minimum: 8GB RAM
   - Recommended: 16GB+ RAM for large models

## Running Benchmarks

### Setup

1. Install benchmark dependencies:
   ```bash
   pip install psutil
   ```

2. Prepare test audio files:
   ```bash
   # Create test audio directory
   mkdir -p reels-clips-automator/benchmark_audio
   # Add various audio files for testing
   ```

### Execution

```bash
# Run comprehensive benchmarks
cd reels-clips-automator
python benchmark_stt.py

# Run specific benchmark
python benchmark_stt.py --test speed --model faster-whisper
python benchmark_stt.py --test memory --model whisper
python benchmark_stt.py --test quality --audio test_audio.wav
```

### Example Benchmark Script

```bash
#!/bin/bash
# benchmark_runner.sh

echo "=== STT Performance Benchmarks ==="

# Test different audio durations
for duration in "30s" "2min" "5min"; do
    echo "Testing ${duration} audio..."
    python benchmark_stt.py --duration $duration --output results_${duration}.json
done

# Test different model sizes
for model_size in "small" "medium" "large"; do
    echo "Testing ${model_size} model..."
    WHISPER_MODEL=$model_size FASTER_WHISPER_MODEL_SIZE=$model_size \
    python benchmark_stt.py --compare --output results_${model_size}.json
done

echo "Benchmarks complete. Check results_*.json files."
```

## Interpreting Results

### Speed Metrics

- **Real-time Factor < 1.0**: Faster than real-time (ideal)
- **Real-time Factor = 1.0**: Real-time processing
- **Real-time Factor > 1.0**: Slower than real-time

### Memory Metrics

- Monitor peak memory usage during transcription
- Consider model loading overhead vs. processing overhead
- GPU memory usage for CUDA configurations

### Quality Metrics

- **Text Similarity >95%**: Acceptable quality preservation
- **Timestamp Accuracy <0.2s**: Good timing preservation
- **Segment Count Similarity**: Structure preservation

## Troubleshooting Performance

### Common Issues

1. **Slower than expected performance**:
   - Check hardware configuration
   - Verify CUDA availability for GPU acceleration
   - Monitor CPU/GPU utilization

2. **High memory usage**:
   - Use smaller model sizes
   - Enable memory optimization flags
   - Monitor for memory leaks

3. **Quality degradation**:
   - Compare with reference transcriptions
   - Test with different audio qualities
   - Adjust model parameters

### Optimization Tips

1. **For Speed**:
   - Use GPU acceleration when available
   - Choose appropriate model size for use case
   - Batch process multiple files

2. **For Memory**:
   - Use quantized models (int8)
   - Process shorter segments
   - Clear model cache between runs

3. **For Quality**:
   - Use larger models for better accuracy
   - Preprocess audio (noise reduction)
   - Fine-tune language settings

## Continuous Monitoring

### Automated Benchmarks

Set up automated benchmarking to monitor performance regression:

```yaml
# .github/workflows/benchmark.yml (example)
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install psutil

    - name: Run benchmarks
      run: |
        cd reels-clips-automator
        python benchmark_stt.py --quick

    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: reels-clips-automator/benchmark_results.json
```

### Performance Monitoring

Track performance metrics over time to detect regressions:

- Processing speed trends
- Memory usage patterns
- Quality consistency
- Error rates

## Conclusion

Regular performance benchmarking ensures that:

1. **Performance targets are met**: 2-4x speed improvement, 50% memory reduction
2. **Quality is maintained**: >95% transcription similarity
3. **Regressions are detected**: Automated monitoring catches issues
4. **Optimizations are validated**: Changes can be measured objectively

The benchmark suite provides a comprehensive framework for validating the performance improvements delivered by the Faster Whisper integration while ensuring that quality and compatibility requirements are maintained.