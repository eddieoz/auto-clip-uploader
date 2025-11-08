# Talking Face Detection Architecture Guide

## System Overview

The talking face detection system intelligently selects the best-speaking face in videos through multi-component analysis and scoring. It consists of three main modules working together.

### Architecture Diagram

```
                        Video Frame
                            |
                            v
                    ┌──────────────────┐
                    │  Face Detection  │ (OpenCV Haar Cascade)
                    └────────┬─────────┘
                             |
                             v
            ┌────────────────────────────────────┐
            |   Talking Face Detector            |
            |  (Main orchestration module)       |
            └────┬──────────────────────┬────────┘
                 |                      |
          ┌──────v──────┐       ┌──────v──────┐
          │   Mouth     │       │    Face     │
          │  Detector   │       │   Quality   │
          │(Movement)   │       │   Scorer    │
          └─────┬───────┘       └──────┬──────┘
                |                      |
                v                      v
          Movement Score         Quality Score
            (0.0-1.0)              (0.0-1.0)
                |                      |
                └──────────┬───────────┘
                           v
                  ┌────────────────────┐
                  │ Combined Scoring   │
                  │ (Movement:Quality) │
                  │ (60%:40%)          │
                  └────────┬───────────┘
                           v
                    Combined Score
                      (0.0-1.0)
                           |
                           v
                  ┌────────────────────┐
                  │ Hysteresis Filter  │
                  │ (Prevent Jitter)   │
                  └────────┬───────────┘
                           v
                    Selected Face Index
                    or None (fallback)
```

## Core Modules

### 1. TalkingFaceDetector (Main Orchestrator)

**File**: `talking_face_detector.py` (340 lines)

**Purpose**: Orchestrates the entire face selection pipeline

**Key Responsibilities**:
- Initialize sub-modules (MouthDetector, FaceQualityScorer)
- Evaluate all detected faces
- Apply hysteresis to prevent jitter
- Cache results for performance
- Track face history

**Key Methods**:

#### `__init__(...)`
Initialize detector with configuration parameters.

```python
detector = TalkingFaceDetector(
    movement_weight=0.6,          # Weight for mouth movement
    quality_weight=0.4,           # Weight for face quality
    min_score_threshold=0.3,      # Minimum score to accept
    hysteresis_threshold=0.2,     # Stability threshold
    cache_ttl=1.0,                # Cache duration (seconds)
    performance_profiler=None     # Optional profiler
)
```

#### `get_best_talking_face(frame, faces, previous_frame, current_face_index)`
Select the best talking face from detected faces.

```python
best_idx = detector.get_best_talking_face(
    frame=current_frame,           # Current video frame
    faces=[(x, y, w, h), ...],    # Detected face bounding boxes
    previous_frame=prev_frame,     # Previous frame for movement detection
    current_face_index=0           # Currently tracked face (for hysteresis)
)

if best_idx is not None:
    # Use the selected face
    x, y, w, h = faces[best_idx]
else:
    # Fallback to center crop
    pass
```

**Return Value**:
- `int`: Index of best face (0-based)
- `None`: No suitable face found, use fallback

#### `get_face_scores()`
Get scores from last evaluation.

```python
scores = detector.get_face_scores()
# Returns: {
#   'movement': 0.967,    # Mouth movement score
#   'quality': 0.780,     # Face quality score
#   'combined': 0.896     # Weighted combination
# }
```

#### `get_face_history()`
Get history of face selections.

```python
history = detector.get_face_history()
# Returns: [
#   {
#     'timestamp': 1699431234.567,
#     'face_index': 0,
#     'score': 0.896,
#     'movement': 0.967,
#     'quality': 0.780
#   },
#   ...
# ]
```

### 2. MouthDetector (Movement Detection)

**File**: `mouth_detector.py` (380 lines)

**Purpose**: Detect active mouth movement using optical flow

**Algorithm**:
1. Extract mouth region from face (lower 40%)
2. Calculate optical flow (Farneback algorithm)
3. Analyze flow vectors in mouth region
4. Return movement score 0.0-1.0

**Key Methods**:

#### `detect_mouth_movement(frame1, frame2, face_bbox)`
Detect mouth movement between two frames.

```python
detector = MouthDetector()

# Between two consecutive frames
movement_score = detector.detect_mouth_movement(
    previous_frame,
    current_frame,
    (x, y, w, h)  # Face bounding box
)
# Returns: float (0.0-1.0)
#   0.0  = No movement (static face)
#   0.5  = Some movement (talking)
#   1.0  = Maximum movement (shouting)
```

**Scoring Components**:
- **Mean magnitude** (40%): Average flow magnitude
- **Movement percentage** (30%): % of pixels with significant flow
- **Max magnitude** (20%): Peak flow magnitude
- **Std magnitude** (10%): Variation in flow

**Performance**:
- Time: ~10-15ms per face
- Optimized: Skipped for low-quality faces (quality < 0.3)
- Cached: Results reused for same frame/bbox combo

### 3. FaceQualityScorer (Quality Assessment)

**File**: `face_quality_scorer.py` (430 lines)

**Purpose**: Score face quality and clarity

**Scoring Dimensions**:

#### Size Score (25% weight)
Evaluates if face size is appropriate.

- **Optimal**: 5-25% of frame area
- **Small faces** (<2%): Score ~0.3 (may be blurry)
- **Large faces** (>40%): Score ~0.5 (may be extreme close-up)
- **Ideal** (10-20%): Score ~1.0

#### Position Score (15% weight)
Evaluates if face is well-positioned.

- **Centered**: Score ~1.0 (good framing)
- **Slightly off-center**: Score ~0.7 (acceptable)
- **At edge**: Score ~0.3 (poor framing)
- **Extreme edge**: Score ~0.1 (barely visible)

#### Variance Score (45% weight - MOST IMPORTANT)
Detects facial features and rejects false positives.

Uses 4 sub-metrics:
- **Laplacian variance** (40%): Edge detection (sharpness)
- **Gradient magnitude** (30%): Texture analysis
- **Edge percentage** (20%): Feature boundaries
- **Intensity variance** (10%): Overall variation

This component effectively rejects:
- Walls (uniform color)
- Posters (flat texture)
- Shadows (low variance)

#### Aspect Ratio Score (15% weight)
Evaluates if face shape looks natural.

- **Optimal**: 0.8-1.2 (nearly square, natural face)
- **Acceptable**: 0.6-1.5 (slightly stretched)
- **Extreme**: <0.4 or >2.0 (heavily penalized)

**Key Methods**:

#### `score_face_quality(frame, face_bbox)`
Score quality of a detected face.

```python
scorer = FaceQualityScorer()

quality_score = scorer.score_face_quality(frame, (x, y, w, h))
# Returns: float (0.0-1.0)
#   0.0-0.3 = Very low (wall, poster, extreme close-up)
#   0.3-0.6 = Low (poor quality, edge position)
#   0.6-0.8 = Medium (acceptable quality)
#   0.8-1.0 = High (excellent quality)
```

#### `get_score_breakdown(frame, face_bbox)`
Get detailed component scores.

```python
breakdown = scorer.get_score_breakdown(frame, (x, y, w, h))
# Returns: {
#   'size': 0.750,          # Size component
#   'position': 0.850,      # Position component
#   'variance': 0.780,      # Clarity component
#   'aspect_ratio': 0.950   # Shape component
# }
```

## Data Flow

### Single Frame Processing

```
Input: frame, faces=[(x1,y1,w1,h1), (x2,y2,w2,h2), ...]

For each face:
    1. Check cache (fast path)
       If cached score exists: return cached

    2. Evaluate quality
       quality_score = FaceQualityScorer.score(frame, bbox)

    3. Detect movement (optimization: skip if quality < 0.3)
       movement_score = MouthDetector.detect(prev_frame, frame, bbox)

    4. Calculate combined score
       If prev_frame available:
           combined = 0.6 * movement + 0.4 * quality
       Else:
           combined = quality (first frame)

    4. Cache result
       cache[(frame_hash, bbox)] = (combined, movement, quality)

All faces scored. Filter by threshold (0.3).
Sort by combined score (descending).
Apply hysteresis: compare best_face vs current_face.

Return: best_face_index or None
```

### Video Processing Integration

In `reelsfy.py`, integrated every 5 seconds (switch_interval):

```
Every 5 seconds (frame_count % switch_interval == 0):
    1. Detect faces with OpenCV
    2. Call detector.get_best_talking_face(...)
    3. If result is not None:
        - Use selected face for next 5 seconds
        - Update tracker
        - Log selection metrics
    4. If result is None:
        - Clear face_positions (triggers center crop)
        - Log that no suitable face was found
    5. Store current frame as prev_frame for next cycle
```

## Performance Optimization

### Caching Strategy

**Problem**: Repeatedly evaluating the same face in consecutive frames is expensive

**Solution**: Cache face scores with TTL (time-to-live)

```python
cache_key = hash(frame_data_subsample, bbox)
if cache_key in cache and not expired(cache[cache_key]):
    # Fast path: use cached score (~1ms)
    return cache[cache_key]
else:
    # Slow path: compute score (~13ms)
    score = evaluate(frame, bbox)
    cache[cache_key] = score
    return score
```

**Expected Performance**:
- **First evaluation**: 13-15ms (uncached)
- **Cached evaluation**: <1ms (10-15x faster)
- **Cache hit rate**: 80-90% for typical videos

### Early Exit Optimization

**Skip optical flow for low-quality faces**:

```python
# Quality score calculated first (fast: 2-3ms)
quality_score = score_quality(frame, bbox)  # 2ms

# If quality is too low, face won't pass threshold anyway
if quality_score >= 0.3 and previous_frame is not None:
    # Only do expensive optical flow if quality is reasonable
    movement_score = detect_mouth_movement(...)  # 10-13ms
else:
    movement_score = 0.0  # Skip expensive computation
```

**Result**: 25% performance improvement on realistic videos

### Cache Size Management

```python
if len(cache) > 100:
    # Remove 20 oldest entries (LRU-style)
    remove_oldest_20_entries()
```

Prevents unbounded memory growth while maintaining hit rate.

## Hysteresis (Anti-Jitter)

**Problem**: Without hysteresis, face can switch erratically

**Solution**: Require 20% score improvement to switch

```python
# Hysteresis threshold: 0.2 (20% improvement required)

if current_face is best_face:
    # Already tracking best face, no switch needed
    return current_face
elif best_face.score >= current_face.score + 0.2:
    # New face is 20%+ better, switch
    return best_face
else:
    # Keep current face (hysteresis prevents jitter)
    return current_face
```

**Effect**:
- Prevents rapid switching (more stable video)
- Still responds to major changes (when real speaker changes)
- Zero switches in typical single-speaker videos

## Configuration

### Environment Variables

Located in `reels-clips-automator/.env`:

```bash
TALKING_FACE_MOVEMENT_WEIGHT=0.6      # Mouth movement weight (0.0-1.0)
TALKING_FACE_QUALITY_WEIGHT=0.4       # Face quality weight (0.0-1.0)
TALKING_FACE_MIN_SCORE=0.3            # Minimum score threshold (0.0-1.0)
TALKING_FACE_HYSTERESIS=0.2           # Hysteresis threshold (0.0-1.0)
```

**Constraint**: `MOVEMENT_WEIGHT + QUALITY_WEIGHT = 1.0`

### Tuning Guidelines

| Scenario | Movement Weight | Quality Weight | Min Score | Hysteresis |
|----------|-----------------|----------------|-----------|------------|
| Single speaker (clear video) | 0.5 | 0.5 | 0.5 | 0.3 |
| Multi-speaker | 0.8 | 0.2 | 0.2 | 0.1 |
| Poor video quality | 0.4 | 0.6 | 0.2 | 0.25 |
| Noisy detection | 0.4 | 0.6 | 0.5 | 0.3 |

## Testing

### Unit Tests (78 tests, 100% passing)

Located in `/tests`:
- `test_mouth_detector.py` (16 tests)
- `test_face_quality_scorer.py` (18 tests)
- `test_talking_face_detector.py` (14 tests)
- `test_reelsfy_integration.py` (11 tests)
- `test_performance.py` (12 tests)
- `test_e2e_talking_face.py` (7 tests)

### E2E Testing

Real video validation with `input_tmp/input8.mkv`:
- 7 scenarios tested
- 100% face focus accuracy achieved
- 0 unnecessary face switches
- 0.78 average quality score
- 0.97 average movement score (talking detected)

## Extension Points

Developers can extend the system:

### 1. Custom Movement Detector

```python
class AudioAnalyzerMovementDetector:
    """Use audio energy instead of optical flow"""

    def detect_mouth_movement(self, frame1, frame2, face_bbox, audio_chunk):
        # Implement audio-based detection
        pass
```

### 2. Custom Quality Scorer

```python
class DeepLearningQualityScorer:
    """Use deep learning for quality assessment"""

    def score_face_quality(self, frame, face_bbox):
        # Use CNN to score face
        pass
```

### 3. Custom Selection Strategy

```python
class SpeakerDiarizationDetector:
    """Use speech diarization for speaker tracking"""

    def get_best_talking_face(self, frame, faces, speaker_id):
        # Track specific speaker across faces
        pass
```

## Performance Characteristics

### Processing Time

| Operation | Time | Notes |
|-----------|------|-------|
| Face quality score | 2-3ms | Fast (numpy operations) |
| Mouth movement detect | 10-15ms | Expensive (optical flow) |
| Cache lookup (hit) | <1ms | Very fast |
| Combined scoring | <1ms | Simple arithmetic |
| Hysteresis check | <0.1ms | Trivial |
| **Total (uncached)** | 13-15ms | Per face |
| **Total (cached)** | <1ms | Per face |

### Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| Cache (100 entries) | ~20KB | Auto-limited |
| History (30 entries) | ~5KB | Fixed-size deque |
| Detector object | ~10KB | Small overhead |
| **Total overhead** | ~35KB | Minimal |

### Accuracy Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Face focus accuracy | 100% | ≥90% |
| False positive rate | 0% | <5% |
| Face switching per minute | 0 | <2 |
| Processing overhead | 0.26% | <10% |

## Troubleshooting Development

### Debug Mode

Enable detailed logging in your code:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

detector = TalkingFaceDetector()

# Get detailed evaluation
detailed = detector.get_detailed_evaluation(frame, faces, prev_frame)
for result in detailed:
    logger.debug(f"Face {result['index']}: "
                f"quality={result['quality_score']:.3f}, "
                f"movement={result['movement_score']:.3f}, "
                f"combined={result['combined_score']:.3f}")
```

### Performance Profiling

```python
from performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
detector = TalkingFaceDetector(performance_profiler=profiler)

# Process frames...

profiler.print_summary()
profiler.save_report("report.json")
```

## References

### Algorithm Papers

- **Optical Flow**: Farneback, G. (2003). Two-frame motion estimation based on polynomial expansion
- **Haar Cascades**: Viola & Jones (2001). Rapid object detection using a boosted cascade
- **Variance-based Detection**: Laplacian edge detection for sharp feature detection

### Related Work

- Face detection: OpenCV Haar Cascades
- Mouth detection: Cascade classifier on mouth ROI
- Movement tracking: KCF tracker for temporal continuity

## Contributing

When extending the system:

1. Follow the module interface (input frame, output score)
2. Return scores in [0, 1] range
3. Include comprehensive docstrings
4. Add unit tests (minimum 80% coverage)
5. Test with real video samples
6. Document your changes

---

For user-facing documentation, see `docs/talking-face-detection.md`
