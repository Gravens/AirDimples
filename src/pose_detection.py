import cv2
from mediapipe.python.solutions import pose
import mediapipe.python.solutions.drawing_utils as mp_drawing


def launch_detection_on_capture(capture):
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    pose_instance = pose.Pose()
    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose_instance.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            print(results.pose_landmarks, '\n\n\n')
            mp_drawing.draw_landmarks(image, results.pose_landmarks, pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe body', image)

        if cv2.waitKey(30) == ord("q"):
            break


def launch_detection_on_webcam():
    capture = cv2.VideoCapture(0)
    launch_detection_on_capture(capture)

    capture.release()


if __name__ == "__main__":
    launch_detection_on_webcam()