import time
from threading import Thread

import cv2

import utils
from models.intel_pose import IntelPoseModel
from utils import log


class VideoDisplay(Thread):
    def __init__(self, frame_deque, joints_deque, game, fps=24, window_name='Video'):
        super().__init__()
        self._keep_running = False

        self.frame_deque = frame_deque
        self.joints_deque = joints_deque
        self.game = game
        self.fps = fps
        self.window_name = window_name

    def __del__(self):
        cv2.destroyAllWindows()

    def display_last(self):
        if not self.frame_deque:
            log.warning('No frames to display; Output fps may be set too high')
            return
        frame = self.frame_deque[-1]

        if self.joints_deque:
            joints = self.joints_deque[-1]
            utils.draw_joints(frame, joints, skeleton=IntelPoseModel.SKELETON)
        else:
            joints = None

        frame = cv2.flip(frame, 1)
        # TODO Implement gameplay drawing without flip
        flipped_joints = utils.flip_joints(joints)

        game_status = self.game.process(frame, flipped_joints)

        cv2.imshow(self.window_name, frame)
        cv2.waitKey(1)

    def run(self):
        self._keep_running = True
        while self._keep_running:
            self.display_last()
            # TODO Match fps more accurately
            time.sleep(1 / self.fps)

    def stop(self):
        self._keep_running = False
