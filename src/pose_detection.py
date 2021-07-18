import cv2
from mediapipe.python.solutions import pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
from gameplay import SoloIntensiveFastAim, SoloClassic, GameWithFriend


def launch_detection_on_capture(capture):
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    pose_instance = pose.Pose()
    # pose_instance_2 for 1x1 game
    pose_instance_2 = pose.Pose()
    ret, frame = capture.read()
    game = GameWithFriend(frame.shape,
                          SoloClassic((frame.shape[0], frame.shape[1] // 2), circle_radius=50, life_time=1, max_items=10),
                          SoloClassic((frame.shape[0], frame.shape[1] // 2), circle_radius=50, life_time=1, max_items=10))

    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        if type(game) != GameWithFriend:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose_instance.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, pose.POSE_CONNECTIONS)

            game_status = game.process(image, results=results)
        else:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            p1_area = image[:, :image.shape[1] // 2]
            p2_area = image[:, image.shape[1] // 2:]

            results1 = pose_instance.process(p1_area)
            results2 = pose_instance_2.process(p2_area)

            if results1.pose_landmarks:
                mp_drawing.draw_landmarks(p1_area,
                                          results1.pose_landmarks,
                                          pose.POSE_CONNECTIONS)

            if results2.pose_landmarks:
                mp_drawing.draw_landmarks(p2_area,
                                          results2.pose_landmarks,
                                          pose.POSE_CONNECTIONS)

            game_status = game.process(image, results=(results1, results2))

        cv2.imshow("Show", image)

        if cv2.waitKey(1) == ord("q") or not game_status:
            break


def launch_detection_on_webcam():
    capture = cv2.VideoCapture(0)
    launch_detection_on_capture(capture)

    capture.release()


if __name__ == "__main__":
    launch_detection_on_webcam()
