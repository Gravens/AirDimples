import time
from threading import Thread

import cv2

from utils import log


class VideoDisplay(Thread):
    def __init__(self, deque, fps=24, window_name='Video'):
        super().__init__()
        self._keep_running = False

        self.deque = deque
        self.fps = fps
        self.window_name = window_name

    def __del__(self):
        cv2.destroyAllWindows()

    def display_last(self):
        if not self.deque:
            log.warning('No frames to display; Output fps may be set too high')
            return
        frame = self.deque.pop()
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
