import collections
import time

from video_display import VideoDisplay
from video_reader import VideoReader


def main():
    frame_deque = collections.deque(maxlen=5)

    video_reader = VideoReader(frame_deque)

    input_fps = video_reader.benchmark_fps(num_frames=80)
    video_display = VideoDisplay(frame_deque, fps=input_fps, window_name='Just Dance')

    video_reader.start()
    video_display.start()

    time.sleep(15)
    video_display.stop()
    video_reader.stop()

    video_display.join()
    video_reader.join()


if __name__ == '__main__':
    main()
