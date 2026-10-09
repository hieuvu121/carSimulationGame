'''
Verify webcam + MediaPipe on own laptop with a 10-line script (landmarks printed).
'''
import cv2  # Read images from the webcam.
import mediapipe as mp  # Detect hand landmarks.
camera = cv2.VideoCapture(0)  # Open the default camera.
success, frame = camera.read()  # Capture one image.
camera.release()  # We have the image, so we can close the camera.


if not success: raise RuntimeError("Could not capture an image.")

#Static image mode = True treats this as an individual photo
#max_num_hands = 1 limits detection to one hand
with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert colours for MediaPipe ti BGR.
    result = hands.process(rgb_frame)  # Look for a hand in the image.
    print(result.multi_hand_landmarks)  # Print detected hand-point coordinates.
