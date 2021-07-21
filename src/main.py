import collections
import time

from gameplay import SoloClassic
from models.intel_pose import IntelPoseModel
from display import DisplayThread
from openvino_inference import OpenvinoInferenceThread
from capture import CaptureThread


def main():
    frame_deque = collections.deque(maxlen=5)
    joints_deque = collections.deque(maxlen=5)

    input_thread = CaptureThread(frame_deque)

    input_fps = input_thread.benchmark_fps(num_frames=80)
    input_shape = input_thread.get_input_shape()

    game = SoloClassic(
        input_shape,
        circle_radius=50,
        life_time=2,
        max_items=10,
        body_part_indexes=IntelPoseModel().body_part_indexes
    )

    inference_thread = OpenvinoInferenceThread(
        frame_deque, joints_deque,
        model_path=f'{__file__}/../../models/intel/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml',
        net_input_width=256,
        capture_shape=input_shape,
    )
    display_thread = DisplayThread(frame_deque, joints_deque, game, fps=input_fps, window_name='Just Dance')

    input_thread.start()
    inference_thread.start()
    display_thread.start()

    time.sleep(30)
    input_thread.stop()
    inference_thread.stop()
    display_thread.stop()

    input_thread.join()
    inference_thread.join()
    display_thread.join()


if __name__ == '__main__':
    main()
