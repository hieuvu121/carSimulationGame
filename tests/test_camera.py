"""Verify HT.1 with a controlled fake device; no physical webcam is opened.

Owner: Triet. Covers the camera acceptance cases in docs/workflow.md section 9.
"""

from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import src.capture.camera as module
from src.capture.camera import Camera, Frame
from src.config import CAMERA_MAX_FAILED_READS


class FakeCapture:
    """Supply frames only when the test asks, simulating a driver waiting on I/O."""

    def __init__(self, opened: bool = True) -> None:
        """Create a device with manually controlled frames and shutdown events."""
        self.opened = opened
        self.items = Queue()
        self.stop = threading.Event()
        self.entered = threading.Event()
        self.released = threading.Event()
        self.release_count = 0
        self.settings = {}
        self.read_threads = []

    def isOpened(self) -> bool:
        """Match OpenCV's device-open query."""
        return self.opened

    def set(self, key: int, value: float) -> bool:
        """Record driver requests without changing the supplied image dimensions."""
        self.settings[key] = value
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Wait for a supplied frame/failure or for the simulated driver to stop."""
        self.entered.set()
        self.read_threads.append(threading.get_ident())
        while not self.stop.is_set():
            try:
                item = self.items.get(timeout=0.01)
            except Empty:
                continue
            if isinstance(item, Exception):
                raise item
            if item is None:
                return False, None
            return True, item.copy()
        return False, None

    def release(self) -> None:
        """Count device releases and unblock any simulated I/O."""
        self.release_count += 1
        self.opened = False
        self.stop.set()
        self.released.set()

    def push(self, image: np.ndarray | None = None) -> None:
        """A failed read is None; arrays represent successful captures."""
        self.items.put(image)


@pytest.fixture
def make_camera(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable]:
    """Own cleanup even when an assertion fails while the fake driver is waiting."""
    devices = []

    def make(**options: int | bool) -> tuple[Camera, FakeCapture]:
        """Connect one Camera instance to its own controlled fake device."""
        fake = FakeCapture()
        monkeypatch.setattr(module.cv2, "VideoCapture", lambda index: fake)
        camera = Camera(**options)
        devices.append((camera, fake))
        return camera, fake

    yield make
    for camera, fake in devices:
        fake.stop.set()
        camera.close()


def wait_for_frame(camera: Camera, index: int) -> Frame:
    """Wait for the public snapshot to reach an index without consuming read()."""
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        frame = camera.latest()
        if frame is not None and frame.index >= index:
            return frame
        time.sleep(0.001)
    pytest.fail(f"Capture worker did not publish frame {index}.")


def test_latest_before_capture_is_immediate(make_camera: Callable) -> None:
    """Preview access must not wait for the fake driver to produce its first image."""
    camera, fake = make_camera()
    assert fake.entered.wait(1.0)
    started = time.perf_counter()
    assert camera.latest() is None
    assert time.perf_counter() - started < 0.1


def test_threaded_indices_times_and_read_never_repeats(make_camera: Callable) -> None:
    """Fresh reads have increasing IDs/times and execute capture in the worker."""
    camera, fake = make_camera()
    frames = []
    for _ in range(3):
        fake.push(np.zeros((8, 8, 3), dtype=np.uint8))
        frames.append(camera.read(timeout=1.0))
    assert [frame.index for frame in frames] == [0, 1, 2]
    assert all(b.t_capture > a.t_capture for a, b in zip(frames, frames[1:]))
    assert all(ident != threading.get_ident() for ident in fake.read_threads)
    assert camera.latest() is frames[-1]
    with pytest.raises(RuntimeError, match="Timed out"):
        camera.read(timeout=0.03)  # No new frame: must not return frame 2 again.


def test_slow_consumer_gets_newest_and_latest_does_not_consume(make_camera: Callable) -> None:
    """A processing backlog skips frames instead of replaying stale images."""
    camera, fake = make_camera()
    for level in range(5):
        fake.push(np.full((8, 8, 3), level, dtype=np.uint8))
    snapshot = wait_for_frame(camera, 4)
    assert snapshot.index == 4
    assert camera.latest() is snapshot
    delivered = camera.read()
    assert delivered is snapshot
    assert int(delivered.image[0, 0, 0]) == 4


@pytest.mark.parametrize("mirror", [True, False])
def test_resize_and_mirroring(make_camera: Callable, mirror: bool) -> None:
    """Even a driver ignoring size settings must produce the contracted image."""
    camera, fake = make_camera(threaded=False, mirror=mirror)
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, 0] = (10, 20, 30)
    image[:, 1] = (70, 80, 90)
    fake.push(image)
    frame = camera.read()
    assert frame.image.shape == (480, 640, 3)
    assert frame.image.dtype == np.uint8
    np.testing.assert_array_equal(frame.image[0, 0], image[0, -1 if mirror else 0])
    np.testing.assert_array_equal(image[0, 0], (10, 20, 30))


@pytest.mark.parametrize("threaded", [True, False])
def test_five_consecutive_failures_reach_caller(make_camera: Callable, threaded: bool) -> None:
    """Hardware failures must become useful caller exceptions in either mode."""
    camera, fake = make_camera(threaded=threaded)
    for _ in range(CAMERA_MAX_FAILED_READS):
        fake.push()
    with pytest.raises(RuntimeError, match="Repeated camera read failures"):
        camera.read()


@pytest.mark.parametrize("threaded", [True, False])
def test_success_resets_failure_streak(make_camera: Callable, threaded: bool) -> None:
    """Occasional failed reads must not accumulate across successful captures."""
    camera, fake = make_camera(threaded=threaded)
    for expected in (0, 1):
        for _ in range(CAMERA_MAX_FAILED_READS - 1):
            fake.push()
        fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
        assert camera.read().index == expected


def test_timeout_without_new_frames(make_camera: Callable) -> None:
    """The caller has a bounded wait even while the capture device is blocked."""
    camera, _ = make_camera()
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="Timed out"):
        camera.read(timeout=0.03)
    assert 0.02 <= time.perf_counter() - started < 0.5


def test_background_exception_reaches_reader(make_camera: Callable) -> None:
    """Unexpected worker exceptions must not disappear into the background."""
    camera, fake = make_camera()
    fake.items.put(ValueError("simulated driver failure"))
    with pytest.raises(RuntimeError, match="simulated driver failure"):
        camera.read()


def test_close_joins_quickly_and_releases_once(make_camera: Callable) -> None:
    """Normal shutdown finishes within one second and repeated close is harmless."""
    camera, fake = make_camera()
    assert fake.entered.wait(1.0)
    # A real working driver returns its next frame shortly. Let this blocked
    # fake return shortly after close() requests shutdown.
    timer = threading.Timer(0.01, fake.stop.set)
    timer.start()
    try:
        started = time.perf_counter()
        camera.close()
        assert time.perf_counter() - started < 1.0
        camera.close()
        assert fake.released.is_set()
        assert fake.release_count == 1
        with pytest.raises(RuntimeError, match="closed"):
            camera.read()
    finally:
        timer.join()


def test_stalled_driver_close_is_bounded_and_eventually_releases(make_camera: Callable) -> None:
    """A hung driver is reported without releasing hardware underneath its read."""
    camera, fake = make_camera()
    assert fake.entered.wait(1.0)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="did not stop"):
        camera.close()
    assert time.perf_counter() - started < 1.5
    assert fake.release_count == 0  # Do not release underneath a native read.
    fake.stop.set()
    assert fake.released.wait(1.0)
    camera.close()
    assert fake.release_count == 1


def test_concurrent_readers_receive_distinct_frames(make_camera: Callable) -> None:
    """The last-delivered index is shared safely across simultaneous callers."""
    camera, fake = make_camera()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(camera.read, 1.0) for _ in range(2)]
        fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
        done, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
        assert len(done) == 1
        fake.push(np.ones((4, 4, 3), dtype=np.uint8))
        assert sorted(future.result(timeout=1.0).index for future in futures) == [0, 1]


def test_synchronous_mode_captures_in_caller_and_context_closes(make_camera: Callable) -> None:
    """Debug mode performs I/O in the caller and with-block exit releases it."""
    camera, fake = make_camera(threaded=False)
    with camera:
        for expected in (0, 1):
            fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
            assert camera.read().index == expected
    assert set(fake.read_threads) == {threading.get_ident()}
    assert fake.release_count == 1


def test_fps_is_measured_and_stale_history_expires(
    make_camera: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FPS follows known intervals and drops to zero after the history expires."""
    clock = SimpleNamespace(now=10.0)
    # Replace only this module's clock, not Python's global time module.
    monkeypatch.setattr(module, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    camera, fake = make_camera(threaded=False)
    assert camera.fps == 0.0
    for timestamp in (10.0, 10.1, 10.2):
        clock.now = timestamp
        fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
        camera.read()
    assert camera.fps == pytest.approx(10.0)  # Two intervals / 0.2 seconds.
    clock.now = 12.0
    assert camera.fps == 0.0


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout(make_camera: Callable, timeout: float) -> None:
    """Reject durations that would make waiting meaningless or unbounded."""
    camera, _ = make_camera(threaded=False)
    with pytest.raises(ValueError, match="timeout"):
        camera.read(timeout)


def test_failed_open_is_actionable_and_releases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failure during construction still releases the unsuccessful device handle."""
    fake = FakeCapture(opened=False)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda index: fake)
    with pytest.raises(RuntimeError, match="Cannot open camera index 0"):
        Camera()
    assert fake.release_count == 1


def test_close_wakes_a_reader_waiting_for_a_frame(make_camera: Callable) -> None:
    """Shutdown notifications must interrupt a reader's longer frame timeout."""
    camera, fake = make_camera()
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(camera.read, 5.0)
        timer = threading.Timer(0.02, fake.stop.set)
        timer.start()
        try:
            camera.close()
            with pytest.raises(RuntimeError, match="closed"):
                reader.result(timeout=0.5)
        finally:
            timer.join()


def test_latest_does_not_wait_for_synchronous_device_io(make_camera: Callable) -> None:
    """The separate I/O lock keeps preview access responsive in debug mode too."""
    camera, fake = make_camera(threaded=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reader = pool.submit(camera.read, 1.0)
        assert fake.entered.wait(0.5)
        started = time.perf_counter()
        assert camera.latest() is None
        assert time.perf_counter() - started < 0.1
        fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
        assert reader.result(timeout=1.0).index == 0


def test_synchronous_capture_rejects_a_frame_arriving_after_deadline(make_camera: Callable) -> None:
    """A completed native call must not turn an expired deadline into success."""
    camera, fake = make_camera(threaded=False)
    timer = threading.Timer(0.04, lambda: fake.push(np.zeros((4, 4, 3), dtype=np.uint8)))
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="Timed out"):
            camera.read(timeout=0.01)
    finally:
        timer.join()


def test_synchronous_capture_serializes_two_readers(make_camera: Callable) -> None:
    """Concurrent debug reads cannot corrupt counters or overlap device ownership."""
    camera, fake = make_camera(threaded=False)
    fake.push(np.zeros((4, 4, 3), dtype=np.uint8))
    fake.push(np.ones((4, 4, 3), dtype=np.uint8))
    with ThreadPoolExecutor(max_workers=2) as pool:
        readers = [pool.submit(camera.read, 1.0) for _ in range(2)]
        assert sorted(reader.result(timeout=1.0).index for reader in readers) == [0, 1]
