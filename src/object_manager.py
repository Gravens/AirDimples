from objects import DefaultCircle, Packman, MoovingCircle
from random import randint, shuffle


def circle_includes(circle,
                    body_part,
                    landmarks,
                    radius,
                    body_part_indexes,
                    w_size,
                    hand_color=(122, 36, 27),
                    side_required=True,
                    body_part_required=True):

    for index in body_part_indexes[body_part]:
        if landmarks[index] is None:
            continue
        lxs = (landmarks[index].x * w_size[1] - circle.center[0]) ** 2
        lys = (landmarks[index].y * w_size[0] - circle.center[1]) ** 2

        if lxs + lys <= radius ** 2:
            side, part = body_part.split("_")
            need_part = "hand" if circle.color == hand_color else "foot"
            if (not side_required or side == circle.side) and (part == need_part or not body_part_required):
                return True
    return False


class DefaultCircleManager:
    def __init__(self, w_size):
        # colors[0] - hand, colors[1] - leg
        self.w_size = w_size
        self.colors = [(122, 36, 27), (15, 255, 235)]
        self.sides = ["L", "R"]
        self.circles = []
        self.last = None

    def add(self, circle_radius, hands_only=True, follow_last=False):
        
        if follow_last:
            center = (randint(circle_radius, min(self.last.center[0] + circle_radius, self.w_size[1] - circle_radius)),
                      randint(circle_radius, min(self.last.center[1] + circle_radius, self.w_size[0] - circle_radius)))
            color = self.last.color
            side = self.last.side
        else:
            center = (randint(circle_radius, self.w_size[1] - circle_radius),
                      randint(circle_radius, self.w_size[0] - circle_radius))
            color = self.colors[0 if hands_only else randint(0, 1)]
            side = self.sides[randint(0, 1)]

        self.last = DefaultCircle(center, color, side)
        self.circles.append(self.last)

    def pop_out(self, landmarks, body_part_indexes, radius):
        score_count = 0
        for item in self.circles:
            for body_part in body_part_indexes:
                if circle_includes(item,
                                   body_part,
                                   landmarks,
                                   radius,
                                   body_part_indexes,
                                   self.w_size
                                   ):
                    score_count += 1
                    self.circles.remove(item)

        return score_count


class PackmanManager:
    def __init__(self, w_size):
        # packman is fast but trajectory is easy
        self.w_size = w_size
        self.vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.p_speed = 5
        self.packmans = []
        self.max_packman_progress = 300

    def circle_in_area(self, center, circle_radius):
        x_valid = circle_radius < center[0] < self.w_size[1] - circle_radius
        y_valid = circle_radius < center[1] < self.w_size[0] - circle_radius
        return x_valid and y_valid

    def add(self, circle_radius):
        center = (randint(circle_radius, self.w_size[1] - circle_radius),
                  randint(circle_radius, self.w_size[0] - circle_radius))
        color = (0, 0, 255)

        copy_vectors = self.vectors.copy()
        shuffle(copy_vectors)
        for dx, dy in copy_vectors:
            future_center = (center[0] + dx * self.p_speed, center[1] + dy * self.p_speed)
            valid_center = self.circle_in_area(future_center, circle_radius)
            if valid_center:
                last_vector = self.vectors.index((dx, dy))
                break

        self.packmans.append(Packman(center, color, last_vector))

    def pop_out(self, landmarks, body_part_indexes, circle_radius):
        score_bonus = 0
        for index, item in enumerate(self.packmans):
            include = False
            for body_part in body_part_indexes:
                if circle_includes(item,
                                   body_part,
                                   landmarks,
                                   circle_radius,
                                   body_part_indexes,
                                   self.w_size,
                                   body_part_required=False,
                                   side_required=False):
                    include = True
                    break

            item.color = (0, 255, 0) if include else (0, 0, 255)
            item.earned_progress += include * self.p_speed
            item.progress += (include or item.progress != 0) * self.p_speed

            if self.max_packman_progress > item.progress > 0:
                chance = randint(1, 10)
                vector_priority = (self.vectors[item.last_vector],
                                   self.vectors[item.last_vector - 1],
                                   self.vectors[(item.last_vector + 1) % 4]) if chance <= 9 else \
                                  (self.vectors[item.last_vector - 1],
                                   self.vectors[(item.last_vector + 1) % 4])
                for dx, dy in vector_priority:
                    new_center = (item.center[0] + dx * self.p_speed, item.center[1] + dy * self.p_speed)
                    valid_center = self.circle_in_area(new_center, circle_radius)
                    if valid_center:
                        item.center = new_center
                        item.last_vector = self.vectors.index((dx, dy))
                        break

            if item.progress >= self.max_packman_progress:
                accuracy = item.earned_progress / item.progress
                if accuracy >= 0.8:
                    score_bonus += 3
                self.packmans.remove(item)

        return score_bonus


class MoovingCircleManager:
    def __init__(self, w_size):
        # curve is slow but trajectory is more complex
        self.w_size = w_size
        self.c_speed = 4
        self.ellipse_curves = []
        self.ellipse_amax = w_size[1] / 8
        self.ellipse_bmax = w_size[0] / 8

    def add(self, circle_radius):
        a = randint(self.ellipse_amax // 2, self.ellipse_amax)
        b = randint(self.ellipse_amax // 2, self.ellipse_amax)
        center = (randint(circle_radius + a * 2, self.w_size[1] - a * 2 - circle_radius),
                  randint(circle_radius + b * 2, self.w_size[0] - b * 2 - circle_radius))
        equation = lambda x: b * (1 - ((x - a) ** 2) / (a ** 2)) ** (1 / 2)
        color = (0, 0, 255)
        # 1 - right 2 - left
        vector = [1, -1][randint(0, 1)]

        self.ellipse_curves.append(MoovingCircle(a, b, center, equation, color, vector))

    def pop_out(self, landmarks, body_part_indexes, circle_radius):
        score_bonus = 0
        for index, item in enumerate(self.ellipse_curves):
            include = False
            for body_part in body_part_indexes:
                if circle_includes(item,
                                   body_part,
                                   landmarks,
                                   circle_radius,
                                   body_part_indexes,
                                   self.w_size,
                                   body_part_required=False,
                                   side_required=False):
                    include = True
                    break

            dy = item.equation((item.progress + self.c_speed) % (2 * item.a)) - \
                 item.equation(item.progress % (2 * item.a))

            item.color = (0, 255, 0) if include else (0, 0, 255)
            item.earned_progress += include * self.c_speed
            item.progress += (include or item.progress != 0) * self.c_speed

            if item.progress != 0:
                item.center = (item.center[0] + item.vector * (self.c_speed * (-1 if item.progress >= item.a * 2 else 1)),
                               item.center[1] + item.vector * (dy * (-1 if item.progress >= item.a * 2 else 1)))

            if item.progress >= item.a * 4:
                accuracy = item.earned_progress / item.progress
                if accuracy >= 0.7:
                    score_bonus += 3
                self.ellipse_curves.remove(item)

        return score_bonus
