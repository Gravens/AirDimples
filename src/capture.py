import time
from threading import Thread

import cv2

from utils import log


class CaptureThread(Thread):
    def __init__(self, deque, cap_source=0):
        super().__init__()
        self._keep_running = False

        self.deque = deque
        self.cap_source = cap_source
        self._init_capture()

    def __del__(self):
        self._release_capture()

    def _init_capture(self):
        log.debug(f'Initializing capture from {self.cap_source}')
        self.capture = cv2.VideoCapture(self.cap_source)
        if not self.capture.isOpened():
            log.error('Video input is not accessible')
            raise IOError

    def _release_capture(self):
        log.debug('Releasing capture...')
        self.capture.release()
        if self.capture.isOpened():
            log.warning('Capture has failed to be released')
        else:
            log.debug('Capture was released successfully')

    def get_input_shape(self):
        self.read_next()
        return self.deque[-1].shape

    def benchmark_fps(self, num_frames):
        log.info('Benchmarking input fps...')

        start = time.time()
        for i in range(num_frames):
            self.capture.read()
        end = time.time()

        elapsed = end - start
        fps = num_frames / elapsed
        fps = int(fps)

        log.info(f'Read {num_frames} frames, took {elapsed:.3f} seconds [{fps} FPS]')
        return fps

    def read_next(self):
        ret, frame = self.capture.read()
        if not ret:
            log.warning('Received empty camera frame')
            return
        self.deque.append(frame)

    def run(self):
        self._keep_running = True
        while self._keep_running:
            self.read_next()

    def stop(self):
        self._keep_running = False
