"""
Mouth Movement Detector Module

This module provides functionality to detect mouth movement in video frames
to identify actively talking faces vs static faces or pictures.

Usage:
    detector = MouthDetector()
    score = detector.detect_mouth_movement(frame1, frame2, face_bbox)

    if score > 0.4:
        print("Person is talking")
    elif score < 0.1:
        print("Static face or picture")
"""

import cv2
import numpy as np
import os


class MouthDetector:
    """
    Detects mouth movement in video frames using optical flow analysis.

    This class analyzes the mouth region of detected faces to determine
    if the person is actively speaking or if it's a static face/picture.

    Attributes:
        mouth_cascade: OpenCV cascade classifier for mouth detection
        flow_params: Parameters for optical flow calculation
    """

    def __init__(self):
        """
        Initialize the MouthDetector with cascade classifiers and flow parameters.
        """
        # Try to load mouth cascade classifier
        # OpenCV provides haarcascade_mcs_mouth.xml for mouth detection
        cascade_path = cv2.data.haarcascades + "haarcascade_mcs_mouth.xml"

        if os.path.exists(cascade_path):
            self.mouth_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            # Fallback: use alternative mouth cascade or None
            self.mouth_cascade = None

        # Parameters for optical flow calculation (Farneback method)
        self.flow_params = {
            'pyr_scale': 0.5,      # Pyramid scale
            'levels': 3,           # Number of pyramid layers
            'winsize': 15,         # Averaging window size
            'iterations': 3,       # Number of iterations
            'poly_n': 5,           # Size of pixel neighborhood
            'poly_sigma': 1.2,     # Gaussian sigma for poly_n
            'flags': 0             # Operation flags
        }

    def detect_mouth_movement(self, frame1, frame2, face_bbox):
        """
        Detect mouth movement between two consecutive frames.

        Args:
            frame1 (numpy.ndarray): First frame (previous frame)
            frame2 (numpy.ndarray): Second frame (current frame)
            face_bbox (tuple): Face bounding box as (x, y, width, height)

        Returns:
            float: Movement score between 0.0 (no movement) and 1.0 (high movement)
                   Returns 0.0 if detection fails or inputs are invalid

        Example:
            >>> detector = MouthDetector()
            >>> score = detector.detect_mouth_movement(prev_frame, curr_frame, (100, 100, 200, 200))
            >>> if score > 0.4:
            ...     print("Person is talking")
        """
        # Validate inputs
        if frame1 is None or frame2 is None:
            return 0.0

        if not isinstance(face_bbox, (tuple, list)) or len(face_bbox) != 4:
            return 0.0

        x, y, w, h = face_bbox

        # Validate bounding box
        if w <= 0 or h <= 0 or x < 0 or y < 0:
            return 0.0

        # Check if frames are the same shape
        if frame1.shape != frame2.shape:
            return 0.0

        # Get frame dimensions
        frame_h, frame_w = frame1.shape[:2]

        # Validate bounding box is within frame bounds
        if x >= frame_w or y >= frame_h or (x + w) > frame_w or (y + h) > frame_h:
            return 0.0

        # Handle very small faces
        if w < 20 or h < 20:
            return 0.0

        try:
            # Calculate optical flow in the mouth region
            flow = self._calculate_optical_flow(frame1, frame2, face_bbox)

            if flow is None:
                return 0.0

            # Calculate movement score from optical flow
            score = self._calculate_movement_score(flow)

            # Normalize and clamp score to [0.0, 1.0]
            score = max(0.0, min(1.0, score))

            return score

        except Exception as e:
            # Handle any unexpected errors gracefully
            print(f"Warning: Mouth movement detection failed: {str(e)}")
            return 0.0

    def _get_mouth_region(self, face_bbox):
        """
        Calculate the mouth region within a face bounding box.

        The mouth region is typically in the lower third of the face.

        Args:
            face_bbox (tuple): Face bounding box as (x, y, width, height)

        Returns:
            tuple: Mouth region as (x, y, width, height)
        """
        x, y, w, h = face_bbox

        # Mouth is typically in the lower third of the face
        # Start from 60% down the face height
        mouth_y = int(y + h * 0.6)
        mouth_h = int(h * 0.4)  # Cover bottom 40% of face

        # Mouth width is typically 60% of face width, centered
        mouth_w = int(w * 0.6)
        mouth_x = int(x + w * 0.2)  # Center the mouth region

        return (mouth_x, mouth_y, mouth_w, mouth_h)

    def _calculate_optical_flow(self, frame1, frame2, face_bbox):
        """
        Calculate optical flow in the mouth region between two frames.

        Args:
            frame1 (numpy.ndarray): Previous frame
            frame2 (numpy.ndarray): Current frame
            face_bbox (tuple): Face bounding box

        Returns:
            numpy.ndarray: Optical flow vectors (H, W, 2) or None if calculation fails
        """
        try:
            # Get mouth region
            mouth_region = self._get_mouth_region(face_bbox)
            mx, my, mw, mh = mouth_region

            # Validate mouth region bounds
            frame_h, frame_w = frame1.shape[:2]
            if mx < 0 or my < 0 or (mx + mw) > frame_w or (my + mh) > frame_h:
                return None

            if mw <= 0 or mh <= 0:
                return None

            # Convert frames to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = frame1.copy()

            if len(frame2.shape) == 3:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = frame2.copy()

            # Extract mouth regions
            mouth1 = gray1[my:my+mh, mx:mx+mw]
            mouth2 = gray2[my:my+mh, mx:mx+mw]

            # Ensure regions are valid
            if mouth1.size == 0 or mouth2.size == 0:
                return None

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(
                mouth1,
                mouth2,
                None,
                pyr_scale=self.flow_params['pyr_scale'],
                levels=self.flow_params['levels'],
                winsize=self.flow_params['winsize'],
                iterations=self.flow_params['iterations'],
                poly_n=self.flow_params['poly_n'],
                poly_sigma=self.flow_params['poly_sigma'],
                flags=self.flow_params['flags']
            )

            return flow

        except Exception as e:
            print(f"Warning: Optical flow calculation failed: {str(e)}")
            return None

    def _calculate_movement_score(self, flow):
        """
        Calculate a movement score from optical flow vectors.

        Args:
            flow (numpy.ndarray): Optical flow vectors (H, W, 2)

        Returns:
            float: Movement score (0.0 to 1.0+, will be clamped later)
        """
        if flow is None or flow.size == 0:
            return 0.0

        # Calculate flow magnitude for each pixel
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

        # Calculate statistics
        mean_magnitude = np.mean(flow_magnitude)
        max_magnitude = np.max(flow_magnitude)
        std_magnitude = np.std(flow_magnitude)

        # Calculate percentage of pixels with significant movement
        # Threshold: movement > 0.5 pixels
        significant_movement_mask = flow_magnitude > 0.5
        movement_percentage = np.sum(significant_movement_mask) / flow_magnitude.size

        # Combine metrics into a single score
        # - Mean magnitude: average movement across all pixels
        # - Movement percentage: how much of the region is moving
        # - Max magnitude: peak movement (indicates strong motion)
        # - Std magnitude: variation in movement (talking has varied movement)

        # Normalize components
        mean_score = min(mean_magnitude / 2.0, 1.0)  # Normalize by expected max
        max_score = min(max_magnitude / 5.0, 1.0)
        percentage_score = movement_percentage
        std_score = min(std_magnitude / 1.5, 1.0)

        # Weighted combination
        # Mean and percentage are most important for talking detection
        score = (
            mean_score * 0.4 +        # Average movement
            percentage_score * 0.3 +  # Spatial distribution
            max_score * 0.2 +         # Peak movement
            std_score * 0.1           # Movement variation
        )

        return score

    def detect_mouth_region_with_cascade(self, frame, face_bbox):
        """
        Attempt to detect mouth region using Haar cascade classifier.

        This is an alternative/complementary method to the geometric approach.

        Args:
            frame (numpy.ndarray): Input frame
            face_bbox (tuple): Face bounding box

        Returns:
            tuple: Mouth bounding box (x, y, w, h) or None if not detected
        """
        if self.mouth_cascade is None:
            return None

        x, y, w, h = face_bbox

        # Extract face region
        face_region = frame[y:y+h, x:x+w]

        if face_region.size == 0:
            return None

        # Convert to grayscale if needed
        if len(face_region.shape) == 3:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_region

        # Detect mouth in face region
        # Focus on lower half of face for better accuracy
        lower_face = gray_face[int(h*0.5):, :]

        mouths = self.mouth_cascade.detectMultiScale(
            lower_face,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(w*0.2), int(h*0.1))
        )

        if len(mouths) > 0:
            # Take the first detected mouth
            mx, my, mw, mh = mouths[0]

            # Convert coordinates back to full frame
            abs_mx = x + mx
            abs_my = y + int(h*0.5) + my

            return (abs_mx, abs_my, mw, mh)

        return None


# Example usage and testing
if __name__ == "__main__":
    print("MouthDetector Module")
    print("=" * 50)

    detector = MouthDetector()
    print(f"Mouth cascade loaded: {detector.mouth_cascade is not None}")

    # Create test frames
    print("\nCreating test frames...")

    # Static frames (no movement)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame1, (320, 240), 80, (180, 180, 180), -1)
    cv2.ellipse(frame1, (320, 270), (30, 15), 0, 0, 180, (100, 100, 100), -1)

    frame2 = frame1.copy()

    face_bbox = (240, 160, 160, 160)

    print("Testing static frames...")
    static_score = detector.detect_mouth_movement(frame1, frame2, face_bbox)
    print(f"Static face score: {static_score:.4f} (should be <= 0.1)")

    # Moving frames
    frame3 = frame1.copy()
    cv2.ellipse(frame3, (320, 275), (35, 25), 0, 0, 180, (100, 100, 100), -1)

    print("\nTesting moving frames...")
    moving_score = detector.detect_mouth_movement(frame1, frame3, face_bbox)
    print(f"Moving mouth score: {moving_score:.4f} (should be >= 0.3)")

    print("\n" + "=" * 50)
    print("Module test complete!")
