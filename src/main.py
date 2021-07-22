import collections
import time

from gameplay import SoloClassic, GameWithFriendOpenVINO, SoloIntensiveFastAim
from models.mediapipe_pose import MediapipePoseModel
from models.intel_pose import IntelPoseModel
from display import DisplayThread
from openvino_inference import OpenvinoInferenceThread
from mediapipe_inference import MediapipeInferenceThread
from capture import CaptureThread
from GUI import GUI


def main():
    model = IntelPoseModel()

    frame_deque = collections.deque(maxlen=5)
    joints_deque = collections.deque(maxlen=5)

    input_thread = CaptureThread(frame_deque)

    input_fps = input_thread.benchmark_fps(num_frames=80)
    input_shape = input_thread.get_input_shape()

    game = SoloClassic(
        input_shape,
        circle_radius=50,
        life_time=2,
        max_items=100,
        body_part_indexes=model.body_part_indexes
    )

    """
    inference_thread = MediapipeInferenceThread(
        frame_deque,
        joints_deque
    )
    """

    inference_thread = OpenvinoInferenceThread(
        frame_deque, joints_deque,
        model_path=f'{__file__}/../../models/intel/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml',
        net_input_width=256,
        capture_shape=input_shape,
    )

    gui = GUI(input_shape, model.body_part_indexes)
    display_thread = DisplayThread(frame_deque, joints_deque, fps=input_fps, window_name='Just Dance', gui=gui)

    input_thread.start()
    inference_thread.start()
    display_thread.start()

    input_thread.join()
    inference_thread.join()
    display_thread.join()


if __name__ == '__main__':
    main()
