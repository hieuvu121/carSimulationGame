"""Prepare the same landmark features for model training and live prediction.

Owner: Triet (Hand tracking).
Provides: preprocess(), row_to_landmarks() and PREPROCESS_VERSION.
Consumes: HandLandmarks from detection or raw CSV rows; never accesses a camera.
Contract: docs/interfaces.md section 4.2 and docs/game-logic.md section 6.1.
"""

from collections.abc import Sequence

import numpy as np

from src.capture.landmarks import HandLandmarks
from src.config import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MIDDLE_MCP_IDX,
    MIN_PALM_SIZE,
    NUM_COORDS,
    NUM_FEATURES,
    NUM_LANDMARKS,
    PREPROCESS_VERSION,
    WRIST_IDX,
)


def preprocess(landmarks: HandLandmarks) -> np.ndarray:
    """Make raw hand coordinates independent of position, scale and handedness.

    Args:
        landmarks: One hand with points of shape (21, 3) and handedness
            "Left" or "Right". Coordinates must use the configured frame aspect.

    Returns:
        A new float32 array of shape (63,), ordered [x0, y0, z0, ..., x20, y20, z20].
        The wrist is the origin and the x-y palm length is one. Input is unchanged.

    Raises:
        ValueError: Coordinates are invalid/non-finite, handedness is unknown, or
            the wrist-to-middle-MCP palm length is below MIN_PALM_SIZE.

    This function has no I/O, mutable global state or camera/library processing.
    Training and live prediction must both call it to avoid mismatched features.
    """
    if landmarks.handedness not in ("Left", "Right"):
        raise ValueError('handedness must be "Left" or "Right".')
    try:
        points = np.array(landmarks.points, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("Landmark coordinates must be numeric.") from exc
    if points.shape != (NUM_LANDMARKS, NUM_COORDS):
        raise ValueError(f"Expected landmark shape (21, 3), got {points.shape}.")
    if not np.isfinite(points).all():
        raise ValueError("Landmark coordinates must be finite; remove NaN/inf samples.")

    # MediaPipe normalizes x by width and y by height. z uses x's scale too.
    # Converting x/z to height units avoids distorting a 640x480 hand shape.
    with np.errstate(over="ignore", invalid="ignore"):
        points[:, [0, 2]] *= FRAME_WIDTH / FRAME_HEIGHT
        points -= points[WRIST_IDX].copy()
        if landmarks.handedness == "Left":
            points[:, 0] *= -1

        # Use x-y palm length: finger extension varies by gesture and z is noisy.
        palm = float(np.linalg.norm(points[MIDDLE_MCP_IDX, :2]))
        if not np.isfinite(points).all() or not np.isfinite(palm):
            raise ValueError("Landmark magnitude is too large to normalize safely.")
        if palm < MIN_PALM_SIZE:
            raise ValueError("Degenerate hand: wrist and middle-finger MCP are too close.")
        points /= palm
        features = points.reshape(NUM_FEATURES).astype(np.float32)

    if not np.isfinite(features).all():
        raise ValueError("Normalized landmark values exceed the float32 range.")
    # Rotation is deliberately retained (spec section 6.1). Changing this
    # algorithm later requires a PREPROCESS_VERSION bump and model retraining.
    return features


def row_to_landmarks(features_raw: Sequence[float], handedness: str) -> HandLandmarks:
    """Rebuild raw HandLandmarks from a CSV row's f0..f62 fields.

    Args:
        features_raw: Exactly 63 raw numeric values in landmark x/y/z order.
        handedness: Recorded "Left" or "Right" label.

    Returns:
        Independent (21, 3) coordinates for preprocess(). score is a 1.0
        placeholder because CSVs do not store handedness confidence; preprocessing
        never uses that field. This helper does not normalize the coordinates.

    Raises:
        ValueError: Wrong feature count, nonnumeric/non-finite values or unknown hand.
    """
    if handedness not in ("Left", "Right"):
        raise ValueError('handedness must be "Left" or "Right".')
    try:
        features = np.array(features_raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV landmark fields must be numeric.") from exc
    if features.shape != (NUM_FEATURES,) or not np.isfinite(features).all():
        raise ValueError("Expected exactly 63 finite raw CSV landmark values.")
    return HandLandmarks(
        points=features.reshape(NUM_LANDMARKS, NUM_COORDS),
        handedness=handedness,
        score=1.0,
    )
