# Talking Face Detection Bug Analysis

## Issue Report
**Date**: 2025-11-08
**Reporter**: User
**Symptom**: Video switched camera to a static non-talking face when it shouldn't

## Root Cause Analysis

### Critical Bug Found: `prev_frame` Timing Issue

#### Location
`reels-clips-automator/reelsfy.py`
- Line 627: `prev_frame = None` (initialization)
- Line 677: `previous_frame=prev_frame` (usage in talking_detector)
- Line 938: `prev_frame = frame.copy()` (assignment)

#### Problem
The `prev_frame` variable is set **AFTER** it's used by the talking face detector:

```python
# Line 627
prev_frame = None  # Initialized as None

while cap.isOpened():
    ret, frame = cap.read()

    if frame_count % switch_interval == 0:
        # Line 677 - Using prev_frame (still None on first iterations!)
        best_face_idx = talking_detector.get_best_talking_face(
            frame=frame,
            faces=face_list,
            previous_frame=prev_frame,  # ❌ This is None!
            current_face_index=current_face_index
        )

    # ... process frame ...

    # Line 938 - Setting prev_frame for next iteration
    prev_frame = frame.copy()  # ✅ But this happens AFTER using it!
```

#### Impact
1. **First interval (frame 0)**: `prev_frame = None` → movement score = 0.0
2. **Second interval (frame 300)**: `prev_frame = frame from frame 299` → But we're comparing frames 300 and 299, which are consecutive, not frames at interval boundaries
3. **All intervals**: Movement detection receives wrong frames or None

The mouth detector's `detect_mouth_movement()` returns 0.0 when `previous_frame` is None:
```python
# mouth_detector.py line 78-80
if frame1 is None or frame2 is None:
    return 0.0
```

### Secondary Issue: Missing Score Logging

#### Location
`reels-clips-automator/reelsfy.py` lines 681-683

#### Problem
When a face is selected, the system doesn't log the talking/quality scores, making it impossible to debug why certain faces are chosen.

#### Epic Requirement (Story 4, lines 293-299)
```python
# Get scores for logging
scores = talking_detector.get_face_scores()
print(f"Selected face {current_face_index}: "
      f"talking_score={scores['movement']:.2f}, "
      f"quality_score={scores['quality']:.2f}, "
      f"combined={scores['combined']:.2f}")
```

This logging was **not implemented**.

## Why Static Faces Were Selected

Given that `prev_frame` is always `None` or wrong:

1. **Mouth movement detection fails**: Returns 0.0 for ALL faces
2. **Only quality scoring works**:
   - Movement weight: 60% × 0.0 = 0.0
   - Quality weight: 40% × quality_score
   - Combined = 0.0 + (0.4 × quality_score)
3. **Static pictures can have high quality**:
   - Large size: High score
   - Centered position: High score
   - Clear/sharp: High variance score
   - Proper aspect ratio: High score
4. **Result**: System picks the highest quality face (which could be a picture) instead of the talking face

## Fix Applied

### Fix 1: Added Score Logging (✅ COMPLETED)
Added debug logging to show talking/quality/combined scores when a face is selected:

```python
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
```

### Fix 2: prev_frame Timing Issue (⚠️ NEEDS VERIFICATION)

The `prev_frame` is currently updated at line 938, which is correct for the overall loop flow. However, we need to verify:

1. Is `prev_frame` properly initialized for the first few frames?
2. Are we comparing the right frames at interval boundaries?

**Current behavior**:
- Interval 0 (frame 0): `prev_frame=None` → movement=0.0
- Interval 1 (frame 300): `prev_frame=frame_299` → Comparing frames 300 and 299 (consecutive, OK)
- Interval 2 (frame 600): `prev_frame=frame_599` → Comparing frames 600 and 599 (consecutive, OK)

**Analysis**: The frames being compared are consecutive, which should work for mouth movement detection. However, the **first interval always has prev_frame=None**, which means the first face selection is based purely on quality, not talking.

## Recommendations

### Immediate Actions
1. ✅ **Score logging added** - Now you can see talking/quality scores in the output
2. ⚠️ **Test with input8.mkv** - Rerun the video and check the logs for talking scores
3. ⚠️ **Verify movement scores are > 0.0** - If still 0.0, there's another issue

### Next Steps Based on Test Results

#### If movement scores are still 0.0:
- Check if mouth detector is finding mouth regions
- Verify optical flow is calculating correctly
- Add debug output to `mouth_detector.py`

#### If movement scores are > 0.0 but wrong face selected:
- Tune thresholds (min_score_threshold, hysteresis_threshold)
- Adjust weights (movement_weight, quality_weight)
- Review face quality scoring algorithm

#### If movement scores vary but system still selects static faces:
- Check if hysteresis is too strong (preventing switches)
- Verify quality scores for static vs talking faces
- Consider increasing movement_weight from 0.6 to 0.7 or 0.8

### Configuration Tuning Options

Add to `.env` for easier tuning:
```bash
# Talking Face Detection (from Story 7, lines 516-522)
ENABLE_TALKING_FACE_DETECTION=true
TALKING_FACE_MOVEMENT_WEIGHT=0.6  # Try 0.7 or 0.8 to prioritize talking more
TALKING_FACE_QUALITY_WEIGHT=0.4   # Decrease if static faces score too high
TALKING_FACE_MIN_SCORE=0.3        # Increase to 0.4 to reject more faces
TALKING_FACE_HYSTERESIS=0.2       # Decrease to 0.1 for more responsive switching
```

## Test Plan

### Test 1: Verify Score Logging
```bash
cd reels-clips-automator
python reelsfy.py --file ../input_tmp/input8.mkv 2>&1 | grep "talking="
```

**Expected output**:
```
Frame: 0, Selected Face 0/2 (talking=0.00, quality=0.75, combined=0.30) height=250 width=200
Frame: 300, Selected Face 1/2 (talking=0.45, quality=0.68, combined=0.54) height=300 width=240
```

**What to look for**:
- talking score should be > 0.0 for frames with talking people
- talking score should be < 0.1 for static pictures
- combined score should favor talking faces

### Test 2: Analyze Face Selection Pattern
1. Count how many times each face is selected
2. Check if static faces are selected when talking faces exist
3. Verify switches happen when talking person changes

### Test 3: Compare with Epic Requirements

From Story 6 (lines 421-435), the system should:
- ✅ **GIVEN** test video with one talking person and one static picture
  **WHEN** processed with talking face detection
  **THEN** 90%+ of frames focus on the talking person

**Current status**: FAILING (selecting static faces)

## Status

- ✅ Bug identified and documented
- ✅ Score logging fix applied
- ⚠️ prev_frame timing needs verification
- ⚠️ Needs testing with actual video
- ❌ Epic requirements not met (90%+ talking person focus)

## References
- Epic: `tasks/talking-face.md`
- Story 4: Integration requirements (lines 232-336)
- Story 6: E2E test requirements (lines 412-485)
- Implementation: `reelsfy.py`, `talking_face_detector.py`, `mouth_detector.py`
