import cv2
from time import time
from random import randint, shuffle
from math import floor


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

        self.vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.speed = 7
        self.curves = []
        self.max_curve_progress = max(w_size)/16

        self.score = 0

    def process(self, frame, results=None):
        if results and results.pose_landmarks:
            self.pop_out_circles(results.pose_landmarks.landmark)
            self.pop_out_curves(results.pose_landmarks.landmark)

        cur_time = time()
        if cur_time - self.last_draw_timestamp > self.interval:
            chance = randint(1, 10)

            if chance > 2:
                self.add_new_circle()
            else:
                self.add_curve()

        if len(self.circles) + len(self.curves) == self.max_items:
            print("Max items on the screen! You lost!")
            return False

        self.draw_objects(frame)
        self.draw_score(frame)
        cv2.imshow("Show", frame)
        return True

    def circle_in_area(self, center):
        x_valid = self.circle_radius < center[0] < self.w_size[1] - self.circle_radius
        y_valid = self.circle_radius < center[1] < self.w_size[0] - self.circle_radius
        return x_valid and y_valid

    def pop_out_curves(self, landmarks):
        for index, item in enumerate(self.curves):
            include = False
            for body_part in self.body_part_indexes:
                if self.circle_includes(item,
                                        body_part,
                                        landmarks,
                                        body_part_importance=False,
                                        side_importance=False):
                    include = True
                    break

            if include:
                item["color"] = (0, 255, 0)
                item["earned_progress"] += 1
            else:
                item["color"] = (0, 0, 255)

            if include or item["progress"] > 0:
                item["progress"] += 1

            if self.max_curve_progress > item["progress"] > 0:
                chance = randint(1, 10)
                vector_priority = (self.vectors[item["last_vector"]],
                                   self.vectors[item["last_vector"] - 1],
                                   self.vectors[(item["last_vector"] + 1) % 4]) if chance <= 9 else \
                                  (self.vectors[item["last_vector"] - 1],
                                   self.vectors[(item["last_vector"] + 1) % 4])
                for dx, dy in vector_priority:
                    new_center = (item["center"][0] + dx * self.speed, item["center"][1] + dy * self.speed)
                    valid_center = self.circle_in_area(new_center)
                    if valid_center:
                        item["center"] = new_center
                        item["last_vector"] = self.vectors.index((dx, dy))
                        break

            if item["progress"] == self.max_curve_progress:
                # print(item["progress"], center_out_of_range, dy)
                accuracy = item["earned_progress"] / item["progress"]
                if accuracy >= 0.6:
                    self.score += 3
                self.curves.remove(item)

    def add_curve(self):
        if len(self.curves) == 1:
            return
        center = (randint(self.circle_radius, self.w_size[1] - self.circle_radius),
                  randint(self.circle_radius, self.w_size[0] - self.circle_radius))
        color = (0, 0, 255)

        copy_vectors = self.vectors.copy()
        shuffle(copy_vectors)
        for dx, dy in copy_vectors:
            future_center = (center[0] + dx * self.speed, center[1] + dy * self.speed)
            valid_center = self.circle_in_area(future_center)
            if valid_center:
                last_vector = self.vectors.index((dx, dy))
                break

        self.curves.append({"center": center,
                            "color": color,
                            "progress": 0,
                            "earned_progress": 0,
                            "last_vector": last_vector})
        self.last_draw_timestamp = time()

    def circle_includes(self,
                        circle,
                        body_part,
                        landmarks,
                        side_importance=True,
                        body_part_importance=True):

        for index in self.body_part_indexes[body_part]:
            # TODO Rename variables
            lxs = (landmarks[index].x * self.w_size[1] - circle["center"][0])**2
            lys = (landmarks[index].y * self.w_size[0] - circle["center"][1])**2

            if lxs + lys <= self.circle_radius**2:
                side, part = body_part.split("_")
                need_part = "hand" if circle["color"] == self.colors[0] else "foot"
                if (side == circle.get("side") or not side_importance) and (part == need_part or not body_part_importance):
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

    def draw_objects(self, frame):
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

        for item in self.curves:
            center = tuple(map(floor, item["center"]))
            cv2.circle(frame, center, self.circle_radius, item["color"], 2)
            cv2.line(
                frame,
                (center[0], center[1]),
                (center[0] + self.circle_radius * self.vectors[item["last_vector"]][0],
                 center[1] + self.circle_radius * self.vectors[item["last_vector"]][1]),
                item["color"],
                2
            )
