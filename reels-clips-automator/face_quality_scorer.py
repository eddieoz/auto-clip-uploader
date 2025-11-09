"""
Face Quality Scorer Module

This module provides functionality to score face detection quality
to distinguish real faces from false positives (walls, cabinets, pictures).

Usage:
    scorer = FaceQualityScorer()
    quality_score = scorer.score_face_quality(frame, face_bbox)

    if quality_score > 0.7:
        print("High quality real face")
    elif quality_score < 0.3:
        print("Likely false positive")
"""

import cv2
import numpy as np


class FaceQualityScorer:
    """
    Scores face detection quality based on multiple criteria.

    This class evaluates detected faces to distinguish real faces from
    false positives like walls, posters, or patterns.

    Scoring components:
    - Size score (25% weight): Larger faces score higher
    - Position score (15% weight): Centered faces score higher
    - Variance score (45% weight): High local variance indicates real features
    - Aspect ratio score (15% weight): Face-like proportions score higher

    Attributes:
        size_weight: Weight for size component (default: 0.25)
        position_weight: Weight for position component (default: 0.15)
        variance_weight: Weight for variance component (default: 0.45)
        aspect_ratio_weight: Weight for aspect ratio component (default: 0.15)
    """

    def __init__(self,
                 size_weight=0.25,
                 position_weight=0.15,
                 variance_weight=0.45,
                 aspect_ratio_weight=0.15):
        """
        Initialize the FaceQualityScorer with scoring weights.

        Args:
            size_weight: Weight for size score (default: 0.25)
            position_weight: Weight for position score (default: 0.15)
            variance_weight: Weight for variance score (default: 0.45)
            aspect_ratio_weight: Weight for aspect ratio score (default: 0.15)
        """
        self.size_weight = size_weight
        self.position_weight = position_weight
        self.variance_weight = variance_weight
        self.aspect_ratio_weight = aspect_ratio_weight

        # Validate weights sum to 1.0
        total = size_weight + position_weight + variance_weight + aspect_ratio_weight
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Weights sum to {total}, expected 1.0. Normalizing...")
            # Normalize weights
            self.size_weight /= total
            self.position_weight /= total
            self.variance_weight /= total
            self.aspect_ratio_weight /= total

    def score_face_quality(self, frame, face_bbox):
        """
        Calculate a comprehensive quality score for a detected face.

        Args:
            frame (numpy.ndarray): Input frame (BGR or grayscale)
            face_bbox (tuple): Face bounding box as (x, y, width, height)

        Returns:
            float: Quality score between 0.0 (poor/false positive) and 1.0 (high quality)
                   Returns 0.0 if inputs are invalid

        Example:
            >>> scorer = FaceQualityScorer()
            >>> score = scorer.score_face_quality(frame, (100, 100, 200, 200))
            >>> if score > 0.7:
            ...     print("High quality face")
        """
        # Validate inputs
        if frame is None:
            return 0.0

        if not isinstance(face_bbox, (tuple, list)) or len(face_bbox) != 4:
            return 0.0

        x, y, w, h = face_bbox

        # Validate bounding box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return 0.0

        # Get frame dimensions
        if len(frame.shape) == 3:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_h, frame_w = frame.shape

        # Validate bounding box is within frame bounds
        if x >= frame_w or y >= frame_h or (x + w) > frame_w or (y + h) > frame_h:
            return 0.0

        try:
            # Calculate individual score components
            size_score = self._calculate_size_score(frame, face_bbox)
            position_score = self._calculate_position_score(frame, face_bbox)
            variance_score = self._calculate_variance_score(frame, face_bbox)
            aspect_ratio_score = self._calculate_aspect_ratio_score(frame, face_bbox)

            # Calculate weighted average
            total_score = (
                size_score * self.size_weight +
                position_score * self.position_weight +
                variance_score * self.variance_weight +
                aspect_ratio_score * self.aspect_ratio_weight
            )

            # Clamp to [0.0, 1.0]
            total_score = max(0.0, min(1.0, total_score))

            return total_score

        except Exception as e:
            print(f"Warning: Face quality scoring failed: {str(e)}")
            return 0.0

    def _calculate_size_score(self, frame, face_bbox):
        """
        Calculate score based on face size relative to frame.

        Larger faces are generally better for video content.

        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            float: Size score (0.0 to 1.0)
        """
        x, y, w, h = face_bbox

        # Get frame dimensions
        if len(frame.shape) == 3:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_h, frame_w = frame.shape

        # Calculate face area as percentage of frame area
        face_area = w * h
        frame_area = frame_w * frame_h
        area_ratio = face_area / frame_area

        # Optimal face size is 5-25% of frame
        # Below 5%: score decreases linearly
        # 5-25%: optimal range (score 0.8-1.0)
        # Above 25%: score decreases (too close to camera)

        if area_ratio < 0.001:  # Very small (< 0.1%)
            score = area_ratio / 0.001  # 0.0 to 1.0 as it approaches 0.1%
        elif area_ratio < 0.05:  # Small (0.1% to 5%)
            score = 0.3 + (area_ratio - 0.001) / (0.05 - 0.001) * 0.5  # 0.3 to 0.8
        elif area_ratio <= 0.25:  # Optimal (5% to 25%)
            score = 0.8 + (area_ratio - 0.05) / (0.25 - 0.05) * 0.2  # 0.8 to 1.0
        else:  # Too large (> 25%)
            score = max(0.5, 1.0 - (area_ratio - 0.25) / 0.5)  # 1.0 down to 0.5

        return max(0.0, min(1.0, score))

    def _calculate_position_score(self, frame, face_bbox):
        """
        Calculate score based on face position in frame.

        Centered faces are generally better for composition.

        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            float: Position score (0.0 to 1.0)
        """
        x, y, w, h = face_bbox

        # Get frame dimensions
        if len(frame.shape) == 3:
            frame_h, frame_w = frame.shape[:2]
        else:
            frame_h, frame_w = frame.shape

        # Calculate face center
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        # Calculate frame center
        frame_center_x = frame_w / 2
        frame_center_y = frame_h / 2

        # Calculate distance from center (normalized)
        dx = abs(face_center_x - frame_center_x) / frame_center_x
        dy = abs(face_center_y - frame_center_y) / frame_center_y

        # Euclidean distance from center
        distance_from_center = np.sqrt(dx**2 + dy**2)

        # Score: 1.0 at center, decreasing with distance
        # At edges (distance ~1.414), score should be ~0.3
        score = max(0.0, 1.0 - distance_from_center * 0.5)

        return max(0.0, min(1.0, score))

    def _calculate_variance_score(self, frame, face_bbox):
        """
        Calculate score based on local variance in face region.

        Real faces have high local variance (eyes, nose, mouth features).
        Walls/flat surfaces have low variance.

        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            float: Variance score (0.0 to 1.0)
        """
        x, y, w, h = face_bbox

        # Extract face region
        face_region = frame[y:y+h, x:x+w]

        if face_region.size == 0:
            return 0.0

        # Convert to grayscale if needed
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region

        # Calculate Laplacian variance (measure of sharpness/detail)
        laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
        variance = laplacian.var()

        # Calculate intensity variance
        intensity_variance = np.var(gray_face)

        # Edge detection for feature detection
        edges = cv2.Canny(gray_face, 50, 150)
        edge_percentage = np.sum(edges > 0) / edges.size

        # Calculate gradient magnitude for texture analysis
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(gradient_magnitude)

        # Combine metrics
        # High variance = real face with features
        # Low variance = uniform surface (wall)

        # Normalize Laplacian variance (typical range: 100-2000 for faces, 0-10 for walls)
        # Use a threshold to filter out low-variance regions
        laplacian_score = min(max(variance - 10, 0) / 1000.0, 1.0)

        # Normalize intensity variance (typical: 500-5000 for faces, 0-50 for walls)
        intensity_score = min(max(intensity_variance - 50, 0) / 2000.0, 1.0)

        # Edge score (typical: 5-20% for faces, <2% for walls)
        edge_score = min(max(edge_percentage - 0.02, 0) / 0.15, 1.0)

        # Gradient score (typical: 10-50 for faces, <5 for walls)
        gradient_score = min(max(gradient_mean - 5, 0) / 40.0, 1.0)

        # Weighted combination - all metrics should agree for high score
        total_variance_score = (
            laplacian_score * 0.4 +
            gradient_score * 0.3 +
            edge_score * 0.2 +
            intensity_score * 0.1
        )

        return max(0.0, min(1.0, total_variance_score))

    def _calculate_aspect_ratio_score(self, frame, face_bbox):
        """
        Calculate score based on face aspect ratio.

        Human faces typically have aspect ratio between 1:1 and 1:1.3 (width:height).
        Extreme aspect ratios (very wide or very tall) are likely false positives.

        Args:
            frame: Input frame
            face_bbox: Face bounding box (x, y, w, h)

        Returns:
            float: Aspect ratio score (0.0 to 1.0)
        """
        x, y, w, h = face_bbox

        if h == 0:
            return 0.0

        # Calculate aspect ratio (width / height)
        aspect_ratio = w / h

        # Ideal face aspect ratio: 0.8 to 1.2 (slightly taller to slightly wider)
        # Acceptable range: 0.6 to 1.5
        # Outside this range: likely not a face

        if 0.8 <= aspect_ratio <= 1.2:
            # Optimal range
            score = 1.0
        elif 0.6 <= aspect_ratio < 0.8:
            # Slightly too tall
            score = 0.5 + (aspect_ratio - 0.6) / (0.8 - 0.6) * 0.5  # 0.5 to 1.0
        elif 1.2 < aspect_ratio <= 1.5:
            # Slightly too wide
            score = 1.0 - (aspect_ratio - 1.2) / (1.5 - 1.2) * 0.5  # 1.0 to 0.5
        elif 0.4 <= aspect_ratio < 0.6:
            # Very tall
            score = 0.2 + (aspect_ratio - 0.4) / (0.6 - 0.4) * 0.3  # 0.2 to 0.5
        elif 1.5 < aspect_ratio <= 2.0:
            # Very wide
            score = 0.5 - (aspect_ratio - 1.5) / (2.0 - 1.5) * 0.3  # 0.5 to 0.2
        else:
            # Extremely non-face-like
            score = 0.1

        return max(0.0, min(1.0, score))

    def get_score_breakdown(self, frame, face_bbox):
        """
        Get detailed breakdown of all score components.

        Useful for debugging and understanding why a face scored a certain way.

        Args:
            frame: Input frame
            face_bbox: Face bounding box

        Returns:
            dict: Score breakdown with individual components
        """
        if frame is None or not isinstance(face_bbox, (tuple, list)):
            return {
                'total': 0.0,
                'size': 0.0,
                'position': 0.0,
                'variance': 0.0,
                'aspect_ratio': 0.0
            }

        try:
            size_score = self._calculate_size_score(frame, face_bbox)
            position_score = self._calculate_position_score(frame, face_bbox)
            variance_score = self._calculate_variance_score(frame, face_bbox)
            aspect_ratio_score = self._calculate_aspect_ratio_score(frame, face_bbox)

            total_score = (
                size_score * self.size_weight +
                position_score * self.position_weight +
                variance_score * self.variance_weight +
                aspect_ratio_score * self.aspect_ratio_weight
            )

            return {
                'total': max(0.0, min(1.0, total_score)),
                'size': size_score,
                'position': position_score,
                'variance': variance_score,
                'aspect_ratio': aspect_ratio_score,
                'weights': {
                    'size': self.size_weight,
                    'position': self.position_weight,
                    'variance': self.variance_weight,
                    'aspect_ratio': self.aspect_ratio_weight
                }
            }

        except Exception as e:
            print(f"Error getting score breakdown: {str(e)}")
            return {
                'total': 0.0,
                'size': 0.0,
                'position': 0.0,
                'variance': 0.0,
                'aspect_ratio': 0.0
            }


# Example usage and testing
if __name__ == "__main__":
    print("FaceQualityScorer Module")
    print("=" * 50)

    scorer = FaceQualityScorer()
    print(f"Weights: Size={scorer.size_weight}, Position={scorer.position_weight}, "
          f"Variance={scorer.variance_weight}, AspectRatio={scorer.aspect_ratio_weight}")

    # Create test frames
    print("\nCreating test frames...")

    # High quality face (large, centered, detailed)
    frame1 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(frame1, (960, 540), 150, (180, 180, 180), -1)
    cv2.circle(frame1, (910, 510), 15, (50, 50, 50), -1)  # Eyes
    cv2.circle(frame1, (1010, 510), 15, (50, 50, 50), -1)
    cv2.ellipse(frame1, (960, 590), (40, 20), 0, 0, 180, (100, 100, 100), -1)  # Mouth

    high_quality_bbox = (810, 390, 300, 300)

    print("\nTesting high quality face...")
    high_quality_score = scorer.score_face_quality(frame1, high_quality_bbox)
    breakdown1 = scorer.get_score_breakdown(frame1, high_quality_bbox)
    print(f"High quality face score: {high_quality_score:.4f} (should be >= 0.7)")
    print(f"Breakdown: Size={breakdown1['size']:.3f}, Position={breakdown1['position']:.3f}, "
          f"Variance={breakdown1['variance']:.3f}, AspectRatio={breakdown1['aspect_ratio']:.3f}")

    # Low quality face (small, edge, blurry)
    frame2 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.circle(frame2, (100, 100), 25, (150, 150, 150), -1)
    face_region = frame2[75:125, 75:125]
    if face_region.size > 0:
        face_region = cv2.GaussianBlur(face_region, (15, 15), 5)
        frame2[75:125, 75:125] = face_region

    low_quality_bbox = (75, 75, 50, 50)

    print("\nTesting low quality face...")
    low_quality_score = scorer.score_face_quality(frame2, low_quality_bbox)
    breakdown2 = scorer.get_score_breakdown(frame2, low_quality_bbox)
    print(f"Low quality face score: {low_quality_score:.4f} (should be <= 0.4)")
    print(f"Breakdown: Size={breakdown2['size']:.3f}, Position={breakdown2['position']:.3f}, "
          f"Variance={breakdown2['variance']:.3f}, AspectRatio={breakdown2['aspect_ratio']:.3f}")

    # False positive (wall)
    frame3 = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame3[400:600, 800:1000] = (120, 120, 120)  # Uniform wall

    wall_bbox = (800, 400, 200, 200)

    print("\nTesting false positive (wall)...")
    wall_score = scorer.score_face_quality(frame3, wall_bbox)
    breakdown3 = scorer.get_score_breakdown(frame3, wall_bbox)
    print(f"Wall score: {wall_score:.4f} (should be <= 0.3)")
    print(f"Breakdown: Size={breakdown3['size']:.3f}, Position={breakdown3['position']:.3f}, "
          f"Variance={breakdown3['variance']:.3f}, AspectRatio={breakdown3['aspect_ratio']:.3f}")

    print("\n" + "=" * 50)
    print("Module test complete!")
