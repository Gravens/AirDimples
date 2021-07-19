import cv2
from mediapipe.python.solutions import pose

import utils
from gameplay import SoloClassic, GameWithFriend
from models.mediapipe_pose import MediapipePoseModel
from utils import log


def launch_detection_on_capture(capture):
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    pose_instance = pose.Pose()
    # pose_instance_2 for 1x1 game
    pose_instance_2 = pose.Pose()
    ret, frame = capture.read()
    model = MediapipePoseModel()
    game = SoloClassic(
        frame.shape,
        circle_radius=50,
        life_time=1,
        max_items=10,
        body_part_indexes=model.body_part_indexes
    )
    # game = GameWithFriend(
    #     frame.shape,
    #     SoloClassic(
    #         (frame.shape[0], frame.shape[1] // 2), circle_radius=50,
    #         life_time=1, max_items=10
    #     ),
    #     SoloClassic(
    #         (frame.shape[0], frame.shape[1] // 2), circle_radius=50,
    #         life_time=1, max_items=10
    #     )
    # )

    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            log.warning("Ignoring empty camera frame.")
            continue

        if type(game) != GameWithFriend:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose_instance.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            joints = model.get_joints_from_result(results)

            utils.draw_joints(image, joints, model.SKELETON)

            game_status = game.process(image, landmarks=joints)
        else:
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            p1_area = image[:, :image.shape[1] // 2]
            p2_area = image[:, image.shape[1] // 2:]

            results1 = pose_instance.process(p1_area)
            results2 = pose_instance_2.process(p2_area)

            joints1 = model.get_joints_from_result(results1)
            joints2 = model.get_joints_from_result(results2)

            utils.draw_joints(p1_area, joints1, model.SKELETON)
            utils.draw_joints(p2_area, joints2, model.SKELETON)

            game_status = game.process(image, results=(joints1, joints2))

        cv2.imshow("Show", image)

        if cv2.waitKey(1) == ord("q") or not game_status:
            break


def launch_detection_on_webcam():
    capture = cv2.VideoCapture(0)
    launch_detection_on_capture(capture)

    capture.release()


def main():
    launch_detection_on_webcam()


if __name__ == "__main__":
    main()
