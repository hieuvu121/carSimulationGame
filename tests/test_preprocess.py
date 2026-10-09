"""Check HT.3/I3 against numerical invariants, without using a webcam or dataset.

Owner: Triet. Covers docs/workflow.md section 9 and game-logic.md section 6.1.
"""

import numpy as np
import pytest

from src.capture.landmarks import HandLandmarks
from src.capture.preprocess import PREPROCESS_VERSION, preprocess, row_to_landmarks
from src.config import MIDDLE_MCP_IDX, PREPROCESS_VERSION as CONFIG_VERSION


@pytest.fixture
def hand() -> HandLandmarks:
    """Use a repeatable nondegenerate raw hand, independent of the implementation."""
    points = np.random.default_rng(218).uniform(0.2, 0.8, size=(21, 3))
    points[0] = (0.4, 0.7, 0.0)
    points[9] = (0.5, 0.4, -0.03)
    return HandLandmarks(points, "Right", 0.95)


def test_shape_dtype_wrist_unit_palm_and_version(hand: HandLandmarks) -> None:
    """The model receives 63 float32 features with the promised normalization."""
    features = preprocess(hand)
    points = features.reshape(21, 3)
    assert features.shape == (63,)
    assert features.dtype == np.float32
    np.testing.assert_array_equal(points[0], np.zeros(3))
    assert np.linalg.norm(points[MIDDLE_MCP_IDX, :2]) == pytest.approx(1.0)
    assert PREPROCESS_VERSION == CONFIG_VERSION


def test_translation_invariance(hand: HandLandmarks) -> None:
    """Moving the whole hand across the image must not change model features."""
    shifted = HandLandmarks(hand.points + (0.2, -0.1, 0.4), "Right", 0.95)
    np.testing.assert_allclose(preprocess(shifted), preprocess(hand), atol=1e-6)


def test_positive_scale_invariance_about_wrist(hand: HandLandmarks) -> None:
    """Changing apparent hand size must not change normalized features."""
    scaled = hand.points[0] + 3.75 * (hand.points - hand.points[0])
    np.testing.assert_allclose(
        preprocess(HandLandmarks(scaled, "Right", 0.95)), preprocess(hand), atol=1e-6
    )


def test_mirrored_left_matches_right(hand: HandLandmarks) -> None:
    """One shared feature space must handle a reflected left hand."""
    left = hand.points.copy()
    left[:, 0] = 1.0 - left[:, 0]
    np.testing.assert_allclose(
        preprocess(HandLandmarks(left, "Left", 0.95)), preprocess(hand), atol=1e-6
    )


def test_aspect_correction_applies_to_x_and_z() -> None:
    """An analytic example catches omission of the less obvious z correction."""
    points = np.zeros((21, 3))
    points[9] = (0.0, 0.5, 0.0)
    points[1] = (0.3, 0.0, 0.15)
    result = preprocess(HandLandmarks(points, "Right", 1.0)).reshape(21, 3)
    np.testing.assert_allclose(result[1], (0.8, 0.0, 0.4), atol=1e-6)


def test_no_mutation_no_alias_and_deterministic(hand: HandLandmarks) -> None:
    """Read-only source data remains reusable by drawing and other consumers."""
    original = hand.points.copy()
    hand.points.flags.writeable = False
    first = preprocess(hand)
    np.testing.assert_array_equal(preprocess(hand), first)
    np.testing.assert_array_equal(hand.points, original)
    assert not np.shares_memory(first, hand.points)


@pytest.mark.parametrize("shape", [(20, 3), (21, 2), (63,), (0, 3)])
def test_rejects_bad_shape(shape: tuple[int, ...]) -> None:
    """A malformed detection must not reach the classifier as a feature vector."""
    with pytest.raises(ValueError, match="shape"):
        preprocess(HandLandmarks(np.zeros(shape), "Right", 1.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_points(hand: HandLandmarks, bad: float) -> None:
    """NaN and infinity must be rejected before normalization spreads them."""
    points = hand.points.copy()
    points[4, 2] = bad
    with pytest.raises(ValueError, match="finite"):
        preprocess(HandLandmarks(points, "Right", 1.0))


@pytest.mark.parametrize("label", ["right", "Unknown", ""])
def test_rejects_unknown_handedness(hand: HandLandmarks, label: str) -> None:
    """Unknown labels cannot silently select the wrong mirroring rule."""
    with pytest.raises(ValueError, match="handedness"):
        preprocess(HandLandmarks(hand.points, label, 1.0))


def test_rejects_degenerate_xy_palm_even_with_large_z() -> None:
    """Depth alone must not rescue a zero-length x-y palm."""
    points = np.zeros((21, 3))
    points[9, 2] = 10.0
    with pytest.raises(ValueError, match="Degenerate"):
        preprocess(HandLandmarks(points, "Right", 1.0))


def test_row_reconstruction_preserves_raw_data_and_features(hand: HandLandmarks) -> None:
    """CSV and live detections must enter the identical preprocessing path."""
    flat = hand.points.reshape(-1)
    rebuilt = row_to_landmarks(flat, "Right")
    np.testing.assert_array_equal(rebuilt.points, hand.points)
    np.testing.assert_array_equal(preprocess(rebuilt), preprocess(hand))
    assert rebuilt.handedness == "Right"
    assert not np.shares_memory(rebuilt.points, flat)


@pytest.mark.parametrize("row", [[0.0] * 62, [np.nan] * 63, ["bad"] * 63])
def test_rejects_invalid_csv_features(row: list[object]) -> None:
    """CSV reconstruction refuses missing, non-finite and nonnumeric coordinates."""
    with pytest.raises(ValueError):
        row_to_landmarks(row, "Right")


def test_csv_rejects_unknown_handedness() -> None:
    """CSV labels use the same explicit handedness vocabulary as live input."""
    with pytest.raises(ValueError, match="handedness"):
        row_to_landmarks([0.0] * 63, "unknown")
