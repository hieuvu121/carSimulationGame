"""Run Triet's Iteration 2 live preview without saving images or video.

Owner: Triet. Provides the manual HT.1/HT.2 demonstration required by workflow.md.
Run from the repository root: python -m src.capture.check.preview_tracking
Press Q or Escape to quit. Each processed frame index is printed for PR evidence.
"""

import argparse
import time

import cv2

from src.capture.camera import Camera
from src.capture.landmarks import HandTracker
from src.capture.preprocess import preprocess
from src.config import CAMERA_INDEX


def main() -> None:
    """Show mirrored landmarks/FPS and print indices; always clean up resources.

    Optional CLI arguments choose the device and stop after a duration in seconds.
    This needs a real webcam/display and is deliberately separate from pytest.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive.")

    tracker = HandTracker()
    try:
        with Camera(index=args.camera) as camera:
            started = time.perf_counter()
            while args.seconds is None or time.perf_counter() - started < args.seconds:
                frame = camera.read()
                hand = tracker.detect(frame.image)
                # Drawing changes pixels: copy the shared image before annotation.
                preview = frame.image.copy()
                status = "NO HAND"
                if hand is not None:
                    tracker.draw(preview, hand)
                    try:
                        features = preprocess(hand)
                        status = f"{hand.handedness}: {features.size} features"
                    except ValueError as exc:
                        # A bad detection should be visible in the demo, not
                        # silently passed to a future classifier.
                        status = f"Invalid landmarks: {exc}"
                fps = camera.fps
                cv2.putText(
                    preview, f"Capture {fps:.1f} FPS | {status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
                cv2.imshow("Triet - hand tracking (Q/Escape to quit)", preview)
                print(f"frame={frame.index} t_capture={frame.t_capture:.6f} fps={fps:.2f}")
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
