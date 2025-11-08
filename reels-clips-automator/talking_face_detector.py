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
        movement_weight: Weight for movement score (default: 0.6)
        quality_weight: Weight for quality score (default: 0.4)
        min_score_threshold: Minimum combined score to accept face (default: 0.3)
        hysteresis_threshold: Score improvement needed to switch faces (default: 0.2)
        cache_ttl: Cache time-to-live in seconds (default: 1.0)
    """

    def __init__(self,
                 movement_weight=0.6,
                 quality_weight=0.4,
                 min_score_threshold=0.3,
                 hysteresis_threshold=0.2,
                 cache_ttl=1.0,
                 performance_profiler=None):
        """
        Initialize the TalkingFaceDetector.

        Args:
            movement_weight: Weight for mouth movement score (default: 0.6)
            quality_weight: Weight for face quality score (default: 0.4)
            min_score_threshold: Minimum score to accept a face (default: 0.3)
            hysteresis_threshold: Score improvement needed to switch (default: 0.2)
            cache_ttl: Cache validity duration in seconds (default: 1.0)
            performance_profiler: Optional performance profiler for metrics
        """
        self.movement_weight = movement_weight
        self.quality_weight = quality_weight
        self.min_score_threshold = min_score_threshold
        self.hysteresis_threshold = hysteresis_threshold
        self.cache_ttl = cache_ttl
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

        # Cache for face evaluations
        self.cache = {}
        self.cache_timestamps = {}

        # Store last evaluation scores for debugging
        self.last_scores = {}

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

        # Filter faces below threshold
        suitable_faces = [f for f in face_scores if f['combined'] >= self.min_score_threshold]

        if not suitable_faces:
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
                    best_face = current_face_score

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
