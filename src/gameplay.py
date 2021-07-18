import cv2
from time import time
from random import randint, shuffle
from math import floor
from object_manager import DefaultCircleManager, PackmanManager, MoovingCircleManager


class SoloGame:
    def __init__(self, w_size, circle_radius=20, interval=1, max_items=4):
        self.w_size = w_size
        self.circle_radius = circle_radius
        self.interval = interval
        self.max_items = max_items
        self.last_draw_timestamp = time()
        self.body_part_indexes = {
            "L_hand": (20, 22, 18, 16),
            "R_hand": (21, 19, 17, 15),
            "L_foot": (28, 32, 30),
            "R_foot": (27, 31, 29)
        }

        self.DCM = DefaultCircleManager(w_size)
        self.PM = PackmanManager(w_size)
        self.MCM = MoovingCircleManager(w_size)

        self.score = 0

    def process(self, frame, results=None):
        if results and results.pose_landmarks:
            self.pop_out_circles(results.pose_landmarks.landmark)
            self.pop_out_packmans(results.pose_landmarks.landmark)
            self.pop_out_ellipse_curves(results.pose_landmarks.landmark)

        cur_time = time()
        if cur_time - self.last_draw_timestamp > self.interval:
            chance = randint(1, 10)

            if chance > 2:
                self.add_new_circle()
            else:
                if chance == 1: self.add_packman()
                else: self.add_new_ellipse_curve()

        if len(self.DCM.circles) + len(self.PM.packmans) + len(self.MCM.ellipse_curves) == self.max_items:
            print("Max items on the screen! You lost!")
            return False

        self.draw_objects(frame)
        self.draw_score(frame)
        cv2.imshow("Show", frame)
        return True

    def pop_out_ellipse_curves(self, landmarks):
        score_bonus = self.MCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus

    def add_new_ellipse_curve(self):
        self.MCM.add(self.circle_radius)
        self.last_draw_timestamp = time()

    def pop_out_packmans(self, landmarks):
        score_bonus = self.PM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus

    def add_packman(self):
        self.PM.add(self.circle_radius)
        self.last_draw_timestamp = time()

    def pop_out_circles(self, landmarks):
        score_bonus = self.DCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus

    def add_new_circle(self):
        self.DCM.add(self.circle_radius)
        self.last_draw_timestamp = time()

    def draw_score(self, frame):
        cv2.putText(frame, "Score " + str(self.score), (10, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)

    def draw_objects(self, frame):
        for item in self.DCM.circles:
            cv2.circle(frame, item.center, self.circle_radius, item.color, 2)
            cv2.putText(
                frame,
                item.side,
                (item.center[0] - 4, item.center[1] + 5),
                cv2.FONT_ITALIC, 0.55,
                item.color,
                2
            )

        for item in self.PM.packmans:
            center = tuple(map(floor, item.center))
            cv2.circle(frame, center, self.circle_radius, item.color, 2)
            cv2.line(
                frame,
                (center[0], center[1]),
                (center[0] + self.circle_radius * self.PM.vectors[item.last_vector][0],
                 center[1] + self.circle_radius * self.PM.vectors[item.last_vector][1]),
                item.color,
                2
            )

        for item in self.MCM.ellipse_curves:
            center = tuple(map(floor, item.center))
            cv2.circle(frame, center, self.circle_radius, item.color, 2)