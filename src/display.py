import time
from threading import Thread

import cv2

import utils
from models.intel_pose import IntelPoseModel
from models.mediapipe_pose import MediapipePoseModel
from utils import log
from gameplay import GameWithFriendOpenVINO


class DisplayThread(Thread):
    def __init__(self, frame_deque, joints_deque, fps=24, window_name='Video', input_thread=None, inference_thread=None, gui=None):
        super().__init__()
        self._keep_running = False

        self.input_thread = input_thread
        self.inference_thread = inference_thread
        self.gui = gui
        self.frame_deque = frame_deque
        self.joints_deque = joints_deque
        self.game = None
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
            joints = []

        frame = cv2.flip(frame, 1)
        # TODO Implement gameplay drawing without flip
        flipped_joints = []

        for item in joints:
            flipped_joints.append(utils.flip_joints(item))

        if self.gui.start_status:
            game_status = True
            if self.gui.countdown != 0:
                self.gui.start_prepare(frame)
            elif type(self.gui.game_mode) != GameWithFriendOpenVINO:
                game_status = self.gui.game_mode.process(frame, flipped_joints[0] if len(flipped_joints) != 0 else [])
            else:
                game_status = self.gui.game_mode.process(frame, flipped_joints)

            if not game_status:
                self.gui.reset()
        else:
            q = self.gui.process(frame, flipped_joints)
            if q:
                self.inference_thread.stop()
                self.input_thread.stop()
                self.stop()
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
