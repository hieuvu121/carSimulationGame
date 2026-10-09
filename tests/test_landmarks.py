"""Check HT.2 with controlled detector output plus a real MediaPipe blank image.

Owner: Triet. Covers docs/workflow.md section 9 without requiring a webcam.
"""

from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

import src.capture.landmarks as module
from src.capture.landmarks import HandLandmarks, HandTracker


class FakeHands:
    """Observe the RGB input and supply deterministic model output."""

    def __init__(self, **kwargs: object) -> None:
        """Remember constructor options without allocating a model."""
        self.options = kwargs
        self.result = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
        self.seen = None
        self.close_count = 0

    def process(self, image: np.ndarray) -> SimpleNamespace:
        """Record exactly what the wrapper supplies to the model."""
        self.seen = image
        return self.result

    def close(self) -> None:
        """Count releases so repeatable cleanup can be checked."""
        self.close_count += 1


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> Iterator[HandTracker]:
    """Keep model inference deterministic except in the explicitly real-model test."""
    monkeypatch.setattr(module.mp.solutions.hands, "Hands", FakeHands)
    tracker = HandTracker()
    yield tracker
    tracker.close()


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((20, 20), dtype=np.uint8),
        np.zeros((20, 20, 3), dtype=np.float32),
        np.zeros((20, 20, 4), dtype=np.uint8),
        np.zeros((0, 20, 3), dtype=np.uint8),
        [[1, 2, 3]],
    ],
)
def test_detect_rejects_invalid_images(tracker: HandTracker, image: object) -> None:
    """Bad inputs must fail before entering native model inference."""
    with pytest.raises(ValueError, match="BGR uint8"):
        tracker.detect(image)


def test_detect_converts_rgb_returns_first_hand_and_preserves_source(tracker: HandTracker) -> None:
    """Check colour conversion, output metadata and the first-hand contract."""
    model = tracker._hands
    coordinates = [SimpleNamespace(x=i / 30, y=0.5, z=-i / 100) for i in range(21)]
    label = SimpleNamespace(label="Left", score=0.91)
    model.result = SimpleNamespace(
        multi_hand_landmarks=[
            SimpleNamespace(landmark=coordinates),
            SimpleNamespace(landmark=[]),  # The contract chooses only the first hand.
        ],
        multi_handedness=[SimpleNamespace(classification=[label])],
    )
    image = np.full((10, 12, 3), (10, 20, 30), dtype=np.uint8)
    before = image.copy()
    hand = tracker.detect(image)
    assert hand is not None
    assert hand.points.shape == (21, 3)
    assert hand.points.dtype == np.float32
    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.91)
    np.testing.assert_allclose(hand.points[20], (20 / 30, 0.5, -0.2))
    np.testing.assert_array_equal(model.seen[0, 0], (30, 20, 10))
    np.testing.assert_array_equal(image, before)
    assert not np.shares_memory(image, model.seen)
    assert model.options["static_image_mode"] is False


def test_no_detection_returns_none(tracker: HandTracker) -> None:
    """No-hand detection is a normal result, not an exception."""
    assert tracker.detect(np.zeros((20, 20, 3), dtype=np.uint8)) is None


def test_draw_changes_pixels_but_not_shape_or_landmarks(tracker: HandTracker) -> None:
    """Drawing annotates only the preview pixels and preserves the raw geometry."""
    points = np.zeros((21, 3))
    points[:, 0] = np.linspace(0.2, 0.8, 21)
    points[:, 1] = np.linspace(0.3, 0.7, 21)
    before = points.copy()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.draw(image, HandLandmarks(points, "Right", 0.9))
    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8
    assert np.count_nonzero(image) > 0
    np.testing.assert_array_equal(points, before)


def test_close_is_repeatable_and_detect_after_close_fails(tracker: HandTracker) -> None:
    """The native model graph is released once and cannot be reused after closing."""
    tracker.close()
    tracker.close()
    assert tracker._hands.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        tracker.detect(np.zeros((20, 20, 3), dtype=np.uint8))


def test_real_mediapipe_returns_none_on_black_frame() -> None:
    """Exercise the installed real model, not a fake, without opening a webcam."""
    tracker = HandTracker()
    try:
        assert tracker.detect(np.zeros((480, 640, 3), dtype=np.uint8)) is None
    finally:
        tracker.close()
