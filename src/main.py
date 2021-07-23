import collections

from GUI import GUI
from capture import CaptureThread
from config import config
from display import DisplayThread
from gameplay import SoloIntensiveFastAim
from openvino_inference import OpenvinoInferenceThread


def main():
    frame_deque = collections.deque(maxlen=config.app.max_frames_stored)
    joints_deque = collections.deque(maxlen=config.app.max_joints_stored)

    input_thread = CaptureThread(frame_deque)

    if config.input_benchmarking.enabled:
        input_fps = input_thread.benchmark_fps(config.input_benchmarking.frame_count)
    else:
        input_fps = config.input_benchmarking.default_fps

    input_shape = input_thread.get_input_shape()

    inference_thread = OpenvinoInferenceThread(frame_deque, joints_deque, capture_shape=input_shape)

    gui = GUI(input_shape)
    display_thread = DisplayThread(frame_deque, joints_deque, fps=input_fps, gui=gui)

    # Start all threads
    input_thread.start()
    inference_thread.start()
    display_thread.start()

    # Wait for the display thread to finish
    display_thread.join()

    # Stop other threads
    input_thread.stop()
    inference_thread.stop()

    # Wait for other threads to finish
    input_thread.join()
    inference_thread.join()


if __name__ == '__main__':
    main()
