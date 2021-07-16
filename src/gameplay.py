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

        # packman is fast but trajectory is easy
        self.vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.p_speed = 7
        self.packmans = []
        self.max_packman_progress = 300

        # curve is slow but trajectory is more complex
        self.c_speed = 5
        self.ellipse_curves = []
        self.ellipse_amax = w_size[1] / 8
        self.ellipse_bmax = w_size[0] / 8
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
                else: self.add_ellipse_curve()

        if len(self.circles) + len(self.packmans) == self.max_items:
            print("Max items on the screen! You lost!")
            return False

        self.draw_objects(frame)
        self.draw_score(frame)
        cv2.imshow("Show", frame)
        return True

    def pop_out_ellipse_curves(self, landmarks):
        for index, item in enumerate(self.ellipse_curves):
            include = False
            for body_part in self.body_part_indexes:
                if self.area_includes(item,
                                      body_part,
                                      landmarks,
                                      threshold=self.circle_radius**2,
                                      body_part_required=False,
                                      side_required=False):
                    include = True
                    break

            dy = item["equation"]((item["progress"] + self.c_speed) % (2 * item["a"])) -\
                 item["equation"](item["progress"] % (2 * item["a"]))

            item["color"] = (0, 255, 0) if include else (0, 0, 255)
            item["earned_progress"] += include * self.c_speed
            item["progress"] += (include or item["progress"] != 0) * self.c_speed

            if item["progress"] != 0:
                item["center"] = (item["center"][0] + item["vector"] * (self.c_speed * (-1 if item["progress"] >= item["a"] * 2 else 1)),
                                  item["center"][1] + item["vector"] * (dy * (-1 if item["progress"] >= item["a"] * 2 else 1)))

            if item["progress"] >= item["a"] * 4:
                accuracy = item["earned_progress"] / item["progress"]
                if accuracy >= 0.7:
                    self.score += 3
                self.ellipse_curves.remove(item)

    def add_ellipse_curve(self):

        a = randint(self.ellipse_amax // 2, self.ellipse_amax)
        b = randint(self.ellipse_amax // 2, self.ellipse_amax)
        print(self.circle_radius + a * 2, self.w_size[1] - a * 2 - self.circle_radius, self.circle_radius + b * 2, self.w_size[0] - b * 2 - self.circle_radius)
        center = (randint(self.circle_radius + a * 2, self.w_size[1] - a * 2 - self.circle_radius),
                  randint(self.circle_radius + b * 2, self.w_size[0] - b * 2 - self.circle_radius))

        equation = lambda x: b * (1 - ((x - a)**2)/(a**2))**(1/2)
        color = (0, 0, 255)
        # 1 - right 2 - left
        vector = [1, -1][randint(0, 1)]

        self.ellipse_curves.append({
            "a": a,
            "b": b,
            "center": center,
            "equation": equation,
            "color": color,
            "progress": 0,
            "earned_progress": 0,
            "vector": vector
        })

        self.last_draw_timestamp = time()

    def circle_in_area(self, center):
        x_valid = self.circle_radius < center[0] < self.w_size[1] - self.circle_radius
        y_valid = self.circle_radius < center[1] < self.w_size[0] - self.circle_radius
        return x_valid and y_valid

    def pop_out_packmans(self, landmarks):
        for index, item in enumerate(self.packmans):
            include = False
            for body_part in self.body_part_indexes:
                if self.area_includes(item,
                                      body_part,
                                      landmarks,
                                      self.circle_radius**2,
                                      body_part_required=False,
                                      side_required=False):
                    include = True
                    break

            item["color"] = (0, 255, 0) if include else (0, 0, 255)
            item["earned_progress"] += include * self.p_speed
            item["progress"] += (include or item["progress"] != 0) * self.p_speed

            if self.max_packman_progress > item["progress"] > 0:
                chance = randint(1, 10)
                vector_priority = (self.vectors[item["last_vector"]],
                                   self.vectors[item["last_vector"] - 1],
                                   self.vectors[(item["last_vector"] + 1) % 4]) if chance <= 9 else \
                                  (self.vectors[item["last_vector"] - 1],
                                   self.vectors[(item["last_vector"] + 1) % 4])
                for dx, dy in vector_priority:
                    new_center = (item["center"][0] + dx * self.p_speed, item["center"][1] + dy * self.p_speed)
                    valid_center = self.circle_in_area(new_center)
                    if valid_center:
                        item["center"] = new_center
                        item["last_vector"] = self.vectors.index((dx, dy))
                        break

            if item["progress"] >= self.max_packman_progress:
                accuracy = item["earned_progress"] / item["progress"]
                if accuracy >= 0.8:
                    self.score += 3
                self.packmans.remove(item)

    def add_packman(self):
        if len(self.packmans) == 1:
            return
        center = (randint(self.circle_radius, self.w_size[1] - self.circle_radius),
                  randint(self.circle_radius, self.w_size[0] - self.circle_radius))
        color = (0, 0, 255)

        copy_vectors = self.vectors.copy()
        shuffle(copy_vectors)
        for dx, dy in copy_vectors:
            future_center = (center[0] + dx * self.p_speed, center[1] + dy * self.p_speed)
            valid_center = self.circle_in_area(future_center)
            if valid_center:
                last_vector = self.vectors.index((dx, dy))
                break

        self.packmans.append({"center": center,
                              "color": color,
                              "progress": 0,
                              "earned_progress": 0,
                              "last_vector": last_vector})
        self.last_draw_timestamp = time()

    def area_includes(self,
                      circle,
                      body_part,
                      landmarks,
                      threshold,
                      equation=lambda x, y, a, b: (x - a)**2 + (y - b)**2,
                      side_required=True,
                      body_part_required=True):

        for index in self.body_part_indexes[body_part]:

            if equation(landmarks[index].x * self.w_size[1],
                        landmarks[index].y * self.w_size[0],
                        circle["center"][0],
                        circle["center"][1]) <= threshold:
                side, part = body_part.split("_")
                need_part = "hand" if circle["color"] == self.colors[0] else "foot"
                if (side == circle.get("side") or not side_required) and (part == need_part or not body_part_required):
                    return True
        return False

    def pop_out_circles(self, landmarks):
        for item in self.circles:
            for body_part in self.body_part_indexes:
                if self.area_includes(item,
                                      body_part,
                                      landmarks,
                                      threshold=self.circle_radius**2,
                                      ):
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

        for item in self.packmans:
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

        for item in self.ellipse_curves:
            center = tuple(map(floor, item["center"]))
            cv2.circle(frame, center, self.circle_radius, item["color"], 2)