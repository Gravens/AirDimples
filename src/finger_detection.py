import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

import cursorman
from utils import denormalize_coordinates, log


def launch_detection_on_capture(capture):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while capture.isOpened():
            ret, image = capture.read()
            if not ret:
                log.warning("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)

            # Prepare for drawing on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_size = (image.shape[0], image.shape[1])

            # If hands were found
            if results.multi_hand_landmarks:

                # Get landmark of the first hand
                landmark = results.multi_hand_landmarks[0]

                # Draw the hand annotations on the image.
                mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS)

                finger_indecies = {'thumb': 4, 'index': 8, 'middle': 12}

                # Get normalized coordinates for specified fingers
                normalized_finger_coords = {}
                for finger, index in finger_indecies.items():
                    try:
                        normalized_finger_coords[finger] = landmark.landmark[index]
                    except IndexError:
                        normalized_finger_coords[finger] = None

                # Convert normalized coordinates to correspond to the image
                image_finger_coords = {}
                for finger, coords in normalized_finger_coords.items():
                    image_finger_coords[finger] = (
                        None if coords is None
                        else denormalize_coordinates((coords.x, coords.y), image_size)
                    )

                if all(image_finger_coords.values()):
                    cursorman.move_cursor_on_screen(normalized_finger_coords)

            # Show image on the screen
            cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(30) == ord("q"):
                break


def launch_detection_on_webcam():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    launch_detection_on_capture(capture)

    capture.release()


if __name__ == '__main__':
    launch_detection_on_webcam()
