from threading import Thread

import cv2
from mediapipe.python.solutions import pose

from models.mediapipe_pose import MediapipePoseModel
from utils import log


class MediapipeInferenceThread(Thread):
    def __init__(self, frame_deque, joints_deque):
        super().__init__()
        self._keep_running = False

        self.frame_deque = frame_deque
        self.joints_deque = joints_deque

        # Initialize Mediapipe engine
        self.pose_instance = pose.Pose()

    def process_last(self):
        if not self.frame_deque:
            log.warning('No frames to process; Input fps may be too low')
            return
        frame = self.frame_deque[-1]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        result = self.pose_instance.process(frame)
        joints = MediapipePoseModel.get_joints_from_result(result)

        self.joints_deque.append(joints)

    def run(self):
        self._keep_running = True
        while self._keep_running:
            self.process_last()

    def stop(self):
        self._keep_running = False
