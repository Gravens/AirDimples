import cv2
from time import time
from random import randint


class SoloPlay:
    def __init__(self, w_size, circle_radius=20, interval=1):
        self.w_size = w_size
        self.circle_radius = circle_radius
        self.interval = interval
        self.last_draw_timestamp = time()
        self.circles = []

    def process(self, frame, results=None):
        if results:
            pass

        cur_time = time()
        if cur_time - self.last_draw_timestamp > self.interval:
            self.add_new_circle()

        self.draw_circles(frame)
        cv2.imshow("Show", frame)

    def add_new_circle(self):
        center = (randint(self.circle_radius, self.w_size[1] - self.circle_radius),
                  randint(self.circle_radius, self.w_size[0] - self.circle_radius))
        self.circles.append(center)
        self.last_draw_timestamp = time()

    def draw_circles(self, frame):
        print(self.circles)
        for center in self.circles:
            cv2.circle(frame, center, self.circle_radius, (141, 227, 30), 2)



