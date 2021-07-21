import collections
import time

from gameplay import SoloClassic
from models.intel_pose import IntelPoseModel
from video_display import VideoDisplay
from video_inference import VideoInference
from video_reader import VideoReader


def main():
    frame_deque = collections.deque(maxlen=5)
    joints_deque = collections.deque(maxlen=5)

    video_reader = VideoReader(frame_deque)

    input_fps = video_reader.benchmark_fps(num_frames=80)
    input_shape = video_reader.get_input_shape()

    game = SoloClassic(
        input_shape,
        circle_radius=50,
        life_time=2,
        max_items=10,
        body_part_indexes=IntelPoseModel().body_part_indexes
    )

    video_inference = VideoInference(
        frame_deque, joints_deque,
        model_path=f'{__file__}/../../models/intel/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml',
        net_input_width=256,
        capture_shape=input_shape,
    )
    video_display = VideoDisplay(frame_deque, joints_deque, game, fps=input_fps, window_name='Just Dance')

    video_reader.start()
    video_inference.start()
    video_display.start()

    time.sleep(30)
    video_reader.stop()
    video_inference.stop()
    video_display.stop()

    video_reader.join()
    video_inference.join()
    video_display.join()


if __name__ == '__main__':
    main()
