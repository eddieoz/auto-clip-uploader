"""
Talking Face Detector Module

This module combines mouth movement detection and face quality scoring
to identify the best actively talking face in video frames.

Usage:
    detector = TalkingFaceDetector()
    best_face_idx = detector.get_best_talking_face(frame, faces, prev_frame, current_idx)

    if best_face_idx is not None:
        print(f"Best talking face: {best_face_idx}")
"""

import cv2
import numpy as np
import time
from collections import deque

from mouth_detector import MouthDetector
from face_quality_scorer import FaceQualityScorer


class TalkingFaceDetector:
    """
    Detects the best talking face by combining movement and quality scores.

    This class integrates mouth movement detection and face quality scoring
    to intelligently select the most suitable face for video processing.

    Features:
    - Weighted scoring: movement (60%) + quality (40%)
    - Hysteresis to prevent rapid switching
    - Face history tracking (last 30 frames)
    - Result caching for performance
    - Minimum score threshold filtering

    Attributes:
        movement_weight: Weight for movement score (default: 0.7)
        quality_weight: Weight for quality score (default: 0.3)
        min_score_threshold: Minimum combined score to accept face (default: 0.3)
        hysteresis_threshold: Score improvement needed to switch faces (default: 0.2)
        min_movement_threshold: Minimum movement required to consider face (default: 0.15)
        static_penalty_threshold: Movement below this triggers static penalty (default: 0.1)
        static_frame_count: Frames of static movement before penalty (default: 10)
        cache_ttl: Cache time-to-live in seconds (default: 1.0)
    """

    def __init__(self,
                 movement_weight=0.7,
                 quality_weight=0.3,
                 min_score_threshold=0.3,
                 hysteresis_threshold=0.2,
                 min_movement_threshold=0.15,
                 static_penalty_threshold=0.1,
                 static_frame_count=10,
                 cache_ttl=1.0,
                 verbose=False,
                 performance_profiler=None):
        """
        Initialize the TalkingFaceDetector.

        Args:
            movement_weight: Weight for mouth movement score (default: 0.7)
            quality_weight: Weight for face quality score (default: 0.3)
            min_score_threshold: Minimum score to accept a face (default: 0.3)
            hysteresis_threshold: Score improvement needed to switch (default: 0.2)
            min_movement_threshold: Minimum movement to consider face (default: 0.15)
            static_penalty_threshold: Movement below this triggers penalty (default: 0.1)
            static_frame_count: Frames of static movement before penalty (default: 10)
            cache_ttl: Cache validity duration in seconds (default: 1.0)
            verbose: Enable debug logging (default: False)
            performance_profiler: Optional performance profiler for metrics
        """
        self.movement_weight = movement_weight
        self.quality_weight = quality_weight
        self.min_score_threshold = min_score_threshold
        self.hysteresis_threshold = hysteresis_threshold
        self.min_movement_threshold = min_movement_threshold
        self.static_penalty_threshold = static_penalty_threshold
        self.static_frame_count = static_frame_count
        self.cache_ttl = cache_ttl
        self.verbose = verbose
        self.profiler = performance_profiler

        # Validate weights sum to 1.0
        total = movement_weight + quality_weight
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total}, expected 1.0. Normalizing...")
            self.movement_weight /= total
            self.quality_weight /= total

        # Initialize sub-modules
        self.mouth_detector = MouthDetector()
        self.face_scorer = FaceQualityScorer()

        # Face history (last 30 evaluations)
        self.face_history = deque(maxlen=30)

        # Static face tracking: face_index -> consecutive static frame count
        self.static_face_counters = {}

        # Cache for face evaluations
        self.cache = {}
        self.cache_timestamps = {}

        # Store last evaluation scores for debugging
        self.last_scores = {}

        # Store last rejection reasons for debugging
        self.last_rejection_reason = {}

    def get_best_talking_face(self, frame, faces, previous_frame=None, current_face_index=None):
        """
        Select the best talking face from detected faces.

        Args:
            frame (numpy.ndarray): Current frame
            faces (list): List of face bounding boxes [(x, y, w, h), ...]
            previous_frame (numpy.ndarray): Previous frame for movement detection
            current_face_index (int): Currently tracked face index (for hysteresis)

        Returns:
            int: Index of best face, or None if no suitable face found

        Example:
            >>> detector = TalkingFaceDetector()
            >>> best_idx = detector.get_best_talking_face(frame, faces, prev_frame, 0)
            >>> if best_idx is not None:
            ...     x, y, w, h = faces[best_idx]
        """
        # Validate inputs
        if frame is None or faces is None or len(faces) == 0:
            return None

        # Evaluate all faces
        face_scores = []

        for i, face_bbox in enumerate(faces):
            try:
                # Check cache first
                cache_key = self._get_cache_key(frame, face_bbox)
                cached_score = self._get_cached_score(cache_key)

                if cached_score is not None:
                    combined_score, movement_score, quality_score = cached_score
                    if self.profiler:
                        self.profiler.record_cache_hit()
                else:
                    if self.profiler:
                        self.profiler.record_cache_miss()

                    # Calculate quality score
                    quality_score = self.face_scorer.score_face_quality(frame, face_bbox)

                    # OPTIMIZATION: Skip expensive mouth detection for low-quality faces
                    # If quality is very low, the face won't pass threshold anyway
                    movement_score = 0.0
                    if previous_frame is not None and quality_score >= 0.3:
                        # Only do expensive optical flow if quality is reasonable
                        movement_score = self.mouth_detector.detect_mouth_movement(
                            previous_frame, frame, face_bbox
                        )

                    # Calculate combined score
                    # If no previous frame, use quality-only scoring
                    if previous_frame is None:
                        combined_score = quality_score
                    else:
                        combined_score = (
                            movement_score * self.movement_weight +
                            quality_score * self.quality_weight
                        )

                    # FIX: Apply minimum movement threshold to reject static faces
                    # This prevents high-quality static images (thumbnails, posters) from being selected
                    # NOTE: Only applies when we have previous_frame (can evaluate movement)
                    # For frame 0 scenarios, use get_best_talking_face_with_lookahead() instead
                    rejection_reason = None
                    if previous_frame is not None and movement_score < self.min_movement_threshold:
                        rejection_reason = f"movement too low ({movement_score:.3f} < {self.min_movement_threshold})"
                        combined_score = 0.0  # Force rejection

                    # FIX: Apply static face penalty for consistently static faces
                    # Track consecutive frames with very low movement
                    if previous_frame is not None:
                        if movement_score < self.static_penalty_threshold:
                            # Increment static counter for this face
                            self.static_face_counters[i] = self.static_face_counters.get(i, 0) + 1

                            # Apply penalty if face has been static for too many frames
                            if self.static_face_counters[i] >= self.static_frame_count:
                                penalty_multiplier = 0.5
                                combined_score *= penalty_multiplier
                                if rejection_reason is None:
                                    rejection_reason = f"consistently static for {self.static_face_counters[i]} frames (penalty applied)"
                        else:
                            # Reset counter if face shows movement
                            self.static_face_counters[i] = 0

                    # Store rejection reason for debugging
                    if rejection_reason:
                        self.last_rejection_reason[i] = rejection_reason

                    # Cache the result
                    self._cache_score(cache_key, (combined_score, movement_score, quality_score))

                face_scores.append({
                    'index': i,
                    'combined': combined_score,
                    'movement': movement_score,
                    'quality': quality_score,
                    'bbox': face_bbox
                })

            except Exception as e:
                print(f"Warning: Error evaluating face {i}: {str(e)}")
                continue

        # Debug logging: Show evaluation results for all faces
        if self.verbose and len(face_scores) > 0:
            print(f"\n[TalkingFaceDetector] Evaluated {len(face_scores)} faces:")
            for face in face_scores:
                idx = face['index']
                rejection = self.last_rejection_reason.get(idx, "")
                status = f"REJECTED ({rejection})" if rejection else "OK"
                print(f"  Face {idx}: combined={face['combined']:.3f} "
                      f"(movement={face['movement']:.3f}, quality={face['quality']:.3f}) - {status}")

        # Filter faces below threshold
        suitable_faces = [f for f in face_scores if f['combined'] >= self.min_score_threshold]

        if not suitable_faces:
            if self.verbose:
                rejected_count = len([f for f in face_scores if f['combined'] < self.min_score_threshold])
                print(f"[TalkingFaceDetector] No suitable faces found. "
                      f"{rejected_count} face(s) below threshold {self.min_score_threshold}")
            return None

        # Sort by combined score (descending)
        suitable_faces.sort(key=lambda x: x['combined'], reverse=True)

        best_face = suitable_faces[0]

        # Apply hysteresis if we have a current face
        if current_face_index is not None and current_face_index < len(faces):
            # Find score for current face
            current_face_score = next(
                (f for f in face_scores if f['index'] == current_face_index),
                None
            )

            if current_face_score is not None:
                # Only switch if new face is significantly better
                score_improvement = best_face['combined'] - current_face_score['combined']

                if score_improvement < self.hysteresis_threshold:
                    # Keep current face (hysteresis prevents switching)
                    if self.verbose:
                        print(f"[TalkingFaceDetector] Hysteresis: Keeping current face {current_face_index} "
                              f"(improvement {score_improvement:.3f} < threshold {self.hysteresis_threshold})")
                    best_face = current_face_score
                else:
                    if self.verbose and best_face['index'] != current_face_index:
                        print(f"[TalkingFaceDetector] Switching from face {current_face_index} to face {best_face['index']} "
                              f"(improvement {score_improvement:.3f} >= threshold {self.hysteresis_threshold})")
            else:
                if self.verbose:
                    print(f"[TalkingFaceDetector] Current face {current_face_index} no longer detected")

        if self.verbose:
            print(f"[TalkingFaceDetector] Selected face {best_face['index']}: "
                  f"combined={best_face['combined']:.3f} "
                  f"(movement={best_face['movement']:.3f}, quality={best_face['quality']:.3f})")

        # Store scores for debugging
        self.last_scores = {
            'movement': best_face['movement'],
            'quality': best_face['quality'],
            'combined': best_face['combined']
        }

        # Add to history
        self.face_history.append({
            'timestamp': time.time(),
            'face_index': best_face['index'],
            'score': best_face['combined'],
            'movement': best_face['movement'],
            'quality': best_face['quality']
        })

        return best_face['index']

    def get_best_talking_face_with_lookahead(self, frames, faces, current_face_index=None):
        """
        Select the best talking face by analyzing multiple frames ahead.

        This method provides more robust face selection by looking at movement
        across multiple frames instead of just a single frame comparison.
        Particularly useful at the start of a video or when switching faces.

        Args:
            frames (list): List of frames to analyze (uses self.static_frame_count frames)
            faces (list): List of face bounding boxes [(x, y, w, h), ...]
            current_face_index (int): Currently tracked face index (for hysteresis)

        Returns:
            int: Index of best face, or None if no suitable face found

        Example:
            >>> detector = TalkingFaceDetector()
            >>> # Read next 10 frames from video
            >>> lookahead_frames = [cap.read()[1] for _ in range(10)]
            >>> best_idx = detector.get_best_talking_face_with_lookahead(
            ...     lookahead_frames, faces, current_face_index=0
            ... )
        """
        if not frames or len(frames) == 0:
            if self.verbose:
                print("[TalkingFaceDetector] No lookahead frames provided")
            return None

        if not faces or len(faces) == 0:
            if self.verbose:
                print("[TalkingFaceDetector] No faces to evaluate")
            return None

        # Limit frames to static_frame_count
        lookahead_frames = frames[:self.static_frame_count]

        if self.verbose:
            print(f"\n[TalkingFaceDetector] Look-ahead evaluation with {len(lookahead_frames)} frames")

        # Aggregate scores for each face across all frames
        face_aggregate_scores = []

        for face_idx, face_bbox in enumerate(faces):
            movement_scores = []
            quality_scores = []

            # Evaluate this face across all lookahead frames
            for frame_idx in range(len(lookahead_frames) - 1):
                prev_frame = lookahead_frames[frame_idx]
                curr_frame = lookahead_frames[frame_idx + 1]

                try:
                    # Calculate quality score (only once, using first frame)
                    if frame_idx == 0:
                        quality_score = self.face_scorer.score_face_quality(curr_frame, face_bbox)
                        quality_scores.append(quality_score)

                    # Calculate movement score between consecutive frames
                    if quality_score >= 0.3:  # Only if quality is reasonable
                        movement_score = self.mouth_detector.detect_mouth_movement(
                            prev_frame, curr_frame, face_bbox
                        )
                        movement_scores.append(movement_score)

                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Error evaluating face {face_idx} at frame {frame_idx}: {e}")
                    continue

            # Aggregate scores
            if not movement_scores:
                avg_movement = 0.0
                max_movement = 0.0
            else:
                avg_movement = np.mean(movement_scores)
                max_movement = np.max(movement_scores)

            if not quality_scores:
                avg_quality = 0.0
            else:
                avg_quality = quality_scores[0]  # Quality is constant across frames

            # Use average movement for combined score
            # This represents typical talking activity across the window
            combined_score = (
                avg_movement * self.movement_weight +
                avg_quality * self.quality_weight
            )

            # Apply minimum movement threshold
            rejection_reason = None
            if avg_movement < self.min_movement_threshold:
                rejection_reason = f"avg movement too low ({avg_movement:.3f} < {self.min_movement_threshold})"
                combined_score = 0.0

            face_aggregate_scores.append({
                'index': face_idx,
                'combined': combined_score,
                'avg_movement': avg_movement,
                'max_movement': max_movement,
                'quality': avg_quality,
                'bbox': face_bbox,
                'rejection_reason': rejection_reason
            })

            if self.verbose:
                status = f"REJECTED ({rejection_reason})" if rejection_reason else "OK"
                print(f"  Face {face_idx}: combined={combined_score:.3f} "
                      f"(avg_movement={avg_movement:.3f}, max_movement={max_movement:.3f}, "
                      f"quality={avg_quality:.3f}) - {status}")

        # Filter faces below threshold
        suitable_faces = [f for f in face_aggregate_scores if f['combined'] >= self.min_score_threshold]

        if not suitable_faces:
            if self.verbose:
                rejected_count = len([f for f in face_aggregate_scores if f['combined'] < self.min_score_threshold])
                print(f"[TalkingFaceDetector] No suitable faces found in lookahead. "
                      f"{rejected_count} face(s) below threshold {self.min_score_threshold}")
            return None

        # Sort by combined score (descending)
        suitable_faces.sort(key=lambda x: x['combined'], reverse=True)

        best_face = suitable_faces[0]

        # Apply hysteresis if we have a current face
        if current_face_index is not None and current_face_index < len(faces):
            current_face_score = next(
                (f for f in face_aggregate_scores if f['index'] == current_face_index),
                None
            )

            if current_face_score is not None:
                score_improvement = best_face['combined'] - current_face_score['combined']

                if score_improvement < self.hysteresis_threshold:
                    if self.verbose:
                        print(f"[TalkingFaceDetector] Hysteresis: Keeping current face {current_face_index} "
                              f"(improvement {score_improvement:.3f} < threshold {self.hysteresis_threshold})")
                    best_face = current_face_score
                else:
                    if self.verbose and best_face['index'] != current_face_index:
                        print(f"[TalkingFaceDetector] Switching from face {current_face_index} to face {best_face['index']} "
                              f"(improvement {score_improvement:.3f} >= threshold {self.hysteresis_threshold})")
            else:
                if self.verbose:
                    print(f"[TalkingFaceDetector] Current face {current_face_index} no longer detected")

        if self.verbose:
            print(f"[TalkingFaceDetector] Selected face {best_face['index']} via lookahead: "
                  f"combined={best_face['combined']:.3f} "
                  f"(avg_movement={best_face['avg_movement']:.3f}, quality={best_face['quality']:.3f})")

        # Store scores for debugging
        self.last_scores = {
            'movement': best_face['avg_movement'],
            'quality': best_face['quality'],
            'combined': best_face['combined']
        }

        # Add to history
        self.face_history.append({
            'timestamp': time.time(),
            'face_index': best_face['index'],
            'score': best_face['combined'],
            'movement': best_face['avg_movement'],
            'quality': best_face['quality'],
            'method': 'lookahead'
        })

        return best_face['index']

    def get_face_scores(self):
        """
        Get the scores from the last evaluation.

        Returns:
            dict: Dictionary with 'movement', 'quality', and 'combined' scores
        """
        return self.last_scores.copy() if self.last_scores else {
            'movement': 0.0,
            'quality': 0.0,
            'combined': 0.0
        }

    def get_face_history(self):
        """
        Get the face evaluation history.

        Returns:
            list: List of historical evaluations with timestamps and scores
        """
        return list(self.face_history)

    def clear_cache(self):
        """Clear the evaluation cache."""
        self.cache.clear()
        self.cache_timestamps.clear()

    def clear_history(self):
        """Clear the face evaluation history."""
        self.face_history.clear()

    def _get_cache_key(self, frame, face_bbox):
        """
        Generate a cache key for a frame and face bbox.

        Uses frame hash and bbox coordinates.

        Args:
            frame: Input frame
            face_bbox: Face bounding box

        Returns:
            str: Cache key
        """
        # Use a simple hash of frame data + bbox
        # For performance, only hash a small region
        x, y, w, h = face_bbox
        region = frame[y:y+h:10, x:x+w:10]  # Subsample for speed
        frame_hash = hash(region.tobytes())

        return f"{frame_hash}_{x}_{y}_{w}_{h}"

    def _get_cached_score(self, cache_key):
        """
        Retrieve a cached score if still valid.

        Args:
            cache_key: Cache key

        Returns:
            tuple: (combined_score, movement_score, quality_score) or None
        """
        if cache_key not in self.cache:
            return None

        # Check if cache is still valid
        cache_time = self.cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > self.cache_ttl:
            # Cache expired
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return None

        return self.cache[cache_key]

    def _cache_score(self, cache_key, scores):
        """
        Cache a score evaluation.

        Args:
            cache_key: Cache key
            scores: Tuple of (combined, movement, quality) scores
        """
        self.cache[cache_key] = scores
        self.cache_timestamps[cache_key] = time.time()

        # Limit cache size to prevent memory issues
        if len(self.cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )[:20]

            for key in oldest_keys:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]

    def get_detailed_evaluation(self, frame, faces, previous_frame=None):
        """
        Get detailed evaluation of all faces for debugging/analysis.

        Args:
            frame: Current frame
            faces: List of face bounding boxes
            previous_frame: Previous frame (optional)

        Returns:
            list: List of dictionaries with detailed scores for each face
        """
        detailed_results = []

        for i, face_bbox in enumerate(faces):
            try:
                quality_score = self.face_scorer.score_face_quality(frame, face_bbox)
                quality_breakdown = self.face_scorer.get_score_breakdown(frame, face_bbox)

                movement_score = 0.0
                if previous_frame is not None:
                    movement_score = self.mouth_detector.detect_mouth_movement(
                        previous_frame, frame, face_bbox
                    )

                combined_score = (
                    movement_score * self.movement_weight +
                    quality_score * self.quality_weight
                )

                detailed_results.append({
                    'index': i,
                    'bbox': face_bbox,
                    'combined_score': combined_score,
                    'movement_score': movement_score,
                    'quality_score': quality_score,
                    'quality_breakdown': quality_breakdown,
                    'meets_threshold': combined_score >= self.min_score_threshold
                })

            except Exception as e:
                print(f"Error evaluating face {i}: {str(e)}")
                continue

        return detailed_results


# Example usage and testing
if __name__ == "__main__":
    print("TalkingFaceDetector Module")
    print("=" * 50)

    detector = TalkingFaceDetector()
    print(f"Weights: Movement={detector.movement_weight}, Quality={detector.quality_weight}")
    print(f"Thresholds: Min={detector.min_score_threshold}, Hysteresis={detector.hysteresis_threshold}")

    # Create test frames
    print("\nCreating test frames...")

    frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Face 1: High quality, talking
    cv2.circle(frame1, (960, 540), 150, (180, 180, 180), -1)
    cv2.circle(frame1, (910, 510), 15, (50, 50, 50), -1)
    cv2.circle(frame1, (1010, 510), 15, (50, 50, 50), -1)
    cv2.ellipse(frame1, (960, 590), (40, 25), 0, 0, 180, (100, 100, 100), -1)

    cv2.circle(frame2, (960, 540), 150, (180, 180, 180), -1)
    cv2.circle(frame2, (910, 510), 15, (50, 50, 50), -1)
    cv2.circle(frame2, (1010, 510), 15, (50, 50, 50), -1)
    cv2.ellipse(frame2, (960, 590), (40, 15), 0, 0, 180, (100, 100, 100), -1)

    # Face 2: Low quality, static
    for frame in [frame1, frame2]:
        cv2.circle(frame, (100, 100), 50, (150, 150, 150), -1)
        cv2.circle(frame, (85, 90), 5, (50, 50, 50), -1)
        cv2.circle(frame, (115, 90), 5, (50, 50, 50), -1)

    faces = [(810, 390, 300, 300), (50, 50, 100, 100)]

    print("\nTesting face selection...")
    best_idx = detector.get_best_talking_face(frame2, faces, frame1, None)

    if best_idx is not None:
        print(f"✅ Selected face: {best_idx}")
        scores = detector.get_face_scores()
        print(f"   Movement: {scores['movement']:.3f}")
        print(f"   Quality: {scores['quality']:.3f}")
        print(f"   Combined: {scores['combined']:.3f}")
    else:
        print("❌ No suitable face found")

    # Test detailed evaluation
    print("\nDetailed evaluation of all faces:")
    detailed = detector.get_detailed_evaluation(frame2, faces, frame1)

    for result in detailed:
        print(f"\nFace {result['index']}:")
        print(f"  Combined: {result['combined_score']:.3f}")
        print(f"  Movement: {result['movement_score']:.3f}")
        print(f"  Quality: {result['quality_score']:.3f}")
        print(f"  Meets threshold: {result['meets_threshold']}")

    print("\n" + "=" * 50)
    print("Module test complete!")
