import cv2
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing

import cursorman
from utils import denormalize_coordinates, lag_check, draw_connection, log


def launch_detection_on_capture(capture):
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        smooth_ratio = 7
        last_screen_x, last_screen_y = 0, 0
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

            # If hands were found
            if results.multi_hand_landmarks:

                # Get landmark of the first hand
                landmark = results.multi_hand_landmarks[0]

                # Draw the hand annotations on the image.
                mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS)

                finger_indecies = {'THUMB_TIP': 4,
                                   'THUMB_IP': 3,
                                   'INDEX_FINGER_TIP': 8,
                                   'INDEX_FINGER_MCP': 5,
                                   'MIDDLE_FINGER_TIP': 12,
                                   'MIDDLE_FINGER_MCP': 9,
                                   'RING_FINGER_TIP': 16,
                                   'RING_FINGER_MCP': 13,
                                   'PINKY_TIP': 20,
                                   'PINKY_MCP': 17,
                                   'WRIST': 0}

                # Get normalized coordinates for specified points
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
                        else denormalize_coordinates((coords.x, coords.y), (image.shape[1], image.shape[0]))
                    )

                if all(image_finger_coords.values()):
                    lag = lag_check(image_finger_coords)
                    if not lag:
                        last_screen_x, last_screen_y, click_prepare, click,  = cursorman.process(normalized_finger_coords,
                                                                                                 last_screen_x,
                                                                                                 last_screen_y,
                                                                                                 smooth_ratio)

                        if click_prepare:
                            draw_connection(image,
                                            image_finger_coords['INDEX_FINGER_TIP'],
                                            image_finger_coords['MIDDLE_FINGER_TIP'],
                                            click)
                        else:
                            cv2.circle(image,
                                       image_finger_coords['INDEX_FINGER_TIP'],
                                       10,
                                       (255, 196, 0),
                                       -1)

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
