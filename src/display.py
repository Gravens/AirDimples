import time
from threading import Thread

import cv2
import keyboard

import drawing
import utils
from config import config
from gameplay import GameWithFriendOpenVINO
from utils import log


class DisplayThread(Thread):
    def __init__(self, frame_deque, joints_deque, fps=24, gui=None):
        super().__init__()
        self._keep_running = False

        self.gui = gui
        self.frame_deque = frame_deque
        self.joints_deque = joints_deque
        self.game = None
        self.fps = fps

    def __del__(self):
        cv2.destroyAllWindows()

    def quit_app(self):
        log.info('Exiting...')
        self.stop()

    def display_last(self):
        if keyboard.is_pressed(config.app.quit_key):
            self.quit_app()

        if not self.frame_deque:
            log.warning('No frames to display; Output fps may be set too high')
            return
        frame = self.frame_deque[-1]

        if self.joints_deque:
            joints = self.joints_deque[-1]
            drawing.draw_joints(frame, joints, skeleton=config.app.model.SKELETON)
            for person_joints in joints:
                drawing.draw_limb_circles(frame, person_joints, config.app.model.BODY_PART_INDEXES)
        else:
            joints = []

        if config.app.flip_image:
            frame = cv2.flip(frame, 1)
            flipped_joints = [utils.flip_joints(item) for item in joints]
        else:
            flipped_joints = joints

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
                self.quit_app()
        cv2.imshow(config.app.window_name, frame)
        cv2.waitKey(1)

    def run(self):
        self._keep_running = True
        while self._keep_running:
            self.display_last()
            # TODO Match fps more accurately
            time.sleep(1 / self.fps)

    def stop(self):
        self._keep_running = False
