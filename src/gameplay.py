import logging

import cv2
from time import time
from random import randint


class SoloGame:
    def __init__(self, w_size, circle_radius=20, interval=1, max_items=4):
        self.w_size = w_size
        self.circle_radius = circle_radius
        self.interval = interval
        self.max_items = max_items
        self.last_draw_timestamp = time()
        self.sides = ["L", "R"]
        self.body_part_indexes = {
            "L_hand": (20, 22, 18, 16),
            "R_hand": (21, 19, 17, 15),
            "L_foot": (28, 32, 30),
            "R_foot": (27, 31, 29)
        }
        # colors[0] - hand, colors[1] - leg
        self.colors = [(122, 36, 27), (15, 255, 235)]
        self.circles = []
        self.score = 0

    def process(self, frame, results=None):
        logging.basicConfig(level='DEBUG', filename='logger.log')
        logger = logging.getLogger()
        if results and results.pose_landmarks:
            self.pop_out_circles(results.pose_landmarks.landmark)

        cur_time = time()
        if cur_time - self.last_draw_timestamp > self.interval:
            self.add_new_circle()

        if len(self.circles) == self.max_items:
            logger.debug("Max items on the screen! You lost!")
            return False

        self.draw_circles(frame)
        self.draw_score(frame)
        cv2.imshow("Show", frame)
        return True

    def circle_includes(self, circle, body_part, landmarks):
        for index in self.body_part_indexes[body_part]:
            # TODO Rename variables
            lxs = (landmarks[index].x * self.w_size[1] - circle["center"][0])**2
            lys = (landmarks[index].y * self.w_size[0] - circle["center"][1])**2
            if body_part == "R_hand":
                print(lxs + lys)
            if lxs + lys <= self.circle_radius**2:
                side, part = body_part.split("_")
                need_part = "hand" if circle["color"] == self.colors[0] else "foot"
                if side == circle["side"] and part == need_part:
                    return True
        return False

    def pop_out_circles(self, landmarks):
        for item in self.circles:
            for body_part in self.body_part_indexes:
                if self.circle_includes(item, body_part, landmarks):
                    self.score += 1
                    self.circles.remove(item)

    def add_new_circle(self):
        center = (randint(self.circle_radius, self.w_size[1] - self.circle_radius),
                  randint(self.circle_radius, self.w_size[0] - self.circle_radius))
        color = self.colors[randint(0, 1)]
        side = self.sides[randint(0, 1)]
        self.circles.append({"center": center,
                             "color": color,
                             "side": side})
        self.last_draw_timestamp = time()

    def draw_score(self, frame):
        cv2.putText(frame, "Score " + str(self.score), (10, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)

    def draw_circles(self, frame):
        for item in self.circles:
            cv2.circle(frame, item["center"], self.circle_radius, item["color"], 2)
            cv2.putText(
                frame,
                item["side"],
                (item["center"][0] - 4, item["center"][1] + 5),
                cv2.FONT_ITALIC, 0.55,
                item["color"],
                2
            )
