import cv2  # Capture images from the webcam.
import time  # Measure how much time has passed.

DURATION_SECONDS = 10  # Measure the camera over a 10-second period.


camera = cv2.VideoCapture(0)  # Open your default webcam.

if not camera.isOpened():
    raise RuntimeError("Could not open the webcam.")

# Request the same resolution for both bright and dim measurements.
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #Request image 640 pixel wide
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) #Request image 480 pixel height


try:
    # Discard the first 30 frames so the camera can adjust to the lighting.
    for _ in range(30):
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Could not read a camera frame.")

    frame_count = 0  # Count only the frames captured during our measurement.
    start_time = time.perf_counter()  # Record when measurement begins.

    #keeps capturing until approximately 10 seconds have elapsed.
    while time.perf_counter() - start_time < DURATION_SECONDS:
        success, frame = camera.read()
        if not success:
            raise RuntimeError("Could not read a camera frame.")
        frame_count += 1  # One more frame was successfully captured.

    elapsed_seconds = time.perf_counter() - start_time
    fps = frame_count / elapsed_seconds

    print(f"Frames captured: {frame_count}")
    print(f"Time measured: {elapsed_seconds:.2f} seconds")
    print(f"Camera FPS: {fps:.2f}")
finally:
    camera.release()  # Close the camera even if an error occurs.

    