"""Detect one hand and draw its landmarks on a camera preview.

Owner: Triet (Hand tracking).
Provides: HandLandmarks and HandTracker to the collector and live worker.
Consumes: mirrored BGR camera frames and MediaPipe Hands.
Contract: docs/interfaces.md section 4.2. One tracker belongs to one thread.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np

from src.config import (
    MP_MAX_NUM_HANDS,
    MP_MIN_DETECTION_CONFIDENCE,
    MP_MIN_TRACKING_CONFIDENCE,
    MP_MODEL_COMPLEXITY,
    NUM_COORDS,
    NUM_LANDMARKS,
)


@dataclass(frozen=True)
class HandLandmarks:
    """Store raw coordinates and handedness together for shared preprocessing.

    Attributes:
        points: Floating array (21, 3), ordered as MediaPipe's hand landmarks.
            x/y are normalized by image width/height; z is wrist-relative depth
            on approximately x's scale, not depth measured in metres.
        handedness: "Left" or "Right", assuming the input is a mirrored image.
        score: Handedness confidence in [0, 1], not gesture-classifier confidence.

    Frozen fields do not freeze the NumPy pixels/coordinates themselves.
    """

    points: np.ndarray
    handedness: str
    score: float


class HandTracker:
    """Wrap MediaPipe Hands; create and use one instance per processing thread."""

    def __init__(
        self,
        max_num_hands: int = MP_MAX_NUM_HANDS,
        model_complexity: int = MP_MODEL_COMPLEXITY,
        min_detection_confidence: float = MP_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = MP_MIN_TRACKING_CONFIDENCE,
    ) -> None:
        """Create a video-mode detector using the project's configured thresholds.

        Args:
            max_num_hands: Positive maximum hand count; detect returns the first.
            model_complexity: MediaPipe model choice, 0 or 1.
            min_detection_confidence: Detection threshold in [0, 1].
            min_tracking_confidence: Tracking threshold in [0, 1].

        Raises:
            ValueError: A model option is outside its supported range.
        """
        if not isinstance(max_num_hands, int) or max_num_hands < 1:
            raise ValueError("max_num_hands must be a positive integer.")
        if model_complexity not in (0, 1):
            raise ValueError("model_complexity must be 0 or 1.")
        for threshold in (min_detection_confidence, min_tracking_confidence):
            if not 0 <= threshold <= 1:
                raise ValueError("MediaPipe confidence thresholds must be in [0, 1].")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._closed = False

    def detect(self, image_bgr: np.ndarray) -> HandLandmarks | None:
        """Find the first hand without changing the supplied image.

        Args:
            image_bgr: Mirrored BGR uint8 array, shape (height, width, 3).

        Returns:
            Raw landmarks with handedness/score, or None if no hand is detected.

        Raises:
            ValueError: The image has the wrong dtype/shape or is empty.
            RuntimeError: The tracker is closed or MediaPipe returns incomplete data.
        """
        self._validate_image(image_bgr)
        if self._closed:
            raise RuntimeError("HandTracker is closed; create a new tracker.")
        # OpenCV stores BGR; MediaPipe expects RGB. Conversion creates a separate
        # array, preserving the source frame for other consumers and drawing.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self._hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return None
        if not result.multi_handedness or not result.multi_handedness[0].classification:
            raise RuntimeError("MediaPipe returned landmarks without handedness.")
        label = result.multi_handedness[0].classification[0]
        points = np.array(
            [(point.x, point.y, point.z) for point in result.multi_hand_landmarks[0].landmark],
            dtype=np.float32,
        )
        if points.shape != (NUM_LANDMARKS, NUM_COORDS):
            raise RuntimeError("MediaPipe returned an unexpected number of hand landmarks.")
        return HandLandmarks(points=points, handedness=label.label, score=float(label.score))

    def draw(self, image_bgr: np.ndarray, hand: HandLandmarks) -> None:
        """Draw landmark points and connections onto an image in place.

        Args:
            image_bgr: Writable BGR uint8 preview image, shape (height, width, 3).
                Pass frame.image.copy() when the source Frame is shared.
            hand: Raw normalized landmarks, not the preprocessed feature vector.

        Raises:
            ValueError: The image or coordinate shape/values are invalid.
        """
        self._validate_image(image_bgr)
        points = np.asarray(hand.points)
        if points.shape != (NUM_LANDMARKS, NUM_COORDS) or not np.isfinite(points).all():
            raise ValueError("Drawing requires 21 finite (x, y, z) landmark coordinates.")
        # MediaPipe's drawing utility accepts its protobuf container. Rebuilding
        # that container keeps the public interface independent of protobuf.
        proto = landmark_pb2.NormalizedLandmarkList(
            landmark=[
                landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z))
                for x, y, z in points
            ]
        )
        mp.solutions.drawing_utils.draw_landmarks(
            image_bgr, proto, mp.solutions.hands.HAND_CONNECTIONS
        )

    def close(self) -> None:
        """Release the MediaPipe graph once; call from the tracker's owner thread."""
        if not self._closed:
            self._hands.close()
            self._closed = True

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Reject inputs that cannot represent a nonempty BGR uint8 camera image."""
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or 0 in image.shape
        ):
            raise ValueError("Expected a nonempty BGR uint8 image of shape (height, width, 3).")
