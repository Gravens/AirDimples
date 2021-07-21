import math
from threading import Thread

import cv2
from openvino.inference_engine import IECore

from models.intel_pose import IntelPoseModel
from pose_utils import models
from pose_utils.pipelines import get_user_config, AsyncPipeline
from utils import log


class OpenvinoInferenceThread(Thread):
    def __init__(self, frame_deque, joints_deque, model_path, net_input_width, capture_shape, device='CPU'):
        super().__init__()
        self._keep_running = False

        self.frame_deque = frame_deque
        self.joints_deque = joints_deque

        # Initialize Inference Engine
        self.ie = IECore()
        plugin_config = get_user_config(device, '', None)
        self.model = IntelPoseModel()

        # Prepare model parameters
        cap_height, cap_width, _ = capture_shape
        aspect_ratio = cap_width / cap_height
        if aspect_ratio >= 1:
            target_size = math.floor(cap_height * net_input_width / cap_width)
        else:
            target_size = net_input_width

        model_embedding = models.HpeAssociativeEmbedding(
            self.ie, model_path, aspect_ratio=aspect_ratio,
            target_size=target_size, prob_threshold=0.1,
        )

        # Initialize pipeline
        self.hpe_pipeline = AsyncPipeline(
            self.ie,
            model_embedding,
            plugin_config,
            device=device,
            max_num_requests=1,
        )
        self.net_input_size = (model_embedding.w, model_embedding.h)

    def process_last(self):
        if not self.frame_deque:
            log.warning('No frames to process; Input fps may be too low')
            return
        frame = self.frame_deque[-1]
        resized_frame = cv2.resize(frame, self.net_input_size, interpolation=cv2.INTER_AREA)

        self.hpe_pipeline.submit_data(resized_frame, 0, {'frame': resized_frame, 'start_time': 0})
        self.hpe_pipeline.await_any()

        results = self.hpe_pipeline.get_result(0)

        joints = self.model.get_joints_from_result(results)
        self.joints_deque.append(joints)

    def run(self):
        self._keep_running = True
        while self._keep_running:
            self.process_last()

    def stop(self):
        self._keep_running = False
