import cv2
from time import time
from random import randint
from object_manager import DefaultCircleManager, PackmanManager, MoovingCircleManager
from utils import log, Joint
from drawing import draw_objects
from config import config


class Game:
    def __init__(self, w_size):
        self.w_size = w_size
        self.body_part_indexes = config.app.model.BODY_PART_INDEXES
        self.hands_only = not config.gameplay.foot_circles_enabled
        self.circle_radius = config.gameplay.circle_radius

        self.last_draw_timestamp = time()
        self.DCM = DefaultCircleManager(w_size)
        self.PM = PackmanManager(w_size)
        self.MCM = MoovingCircleManager(w_size)

        self.score = 0

    def add_new_ellipse_curve(self):
        self.MCM.add(self.circle_radius)
        self.last_draw_timestamp = time()

    def add_packman(self):
        self.PM.add(self.circle_radius)
        self.last_draw_timestamp = time()

    def draw_score(self, frame):
        cv2.putText(frame, "Score " + str(self.score), (10, 50), cv2.FONT_ITALIC, 2, (255, 0, 0), 3)

    def add_new_circle(self):
        self.DCM.add(self.circle_radius, hands_only=self.hands_only)
        self.last_draw_timestamp = time()


class SoloIntensiveFastAim(Game):
    def __init__(self, w_size):
        super().__init__(w_size)
        self.max_items = config.gameplay.intensive_max_circles_on_screen
        self.interval = config.gameplay.intensive_interval

    def process(self, frame, landmarks=None):
        if landmarks:
            self.pop_out_circles(landmarks)
            self.pop_out_packmans(landmarks)
            self.pop_out_ellipse_curves(landmarks)

        cur_time = time()
        if cur_time - self.last_draw_timestamp > self.interval:
            chance = randint(1, 10)

            if chance > 2:
                self.add_new_circle()
            else:
                if chance == 1:
                    self.add_packman()
                else:
                    self.add_new_ellipse_curve()

        if len(self.DCM.circles) + len(self.PM.packmans) + len(self.MCM.ellipse_curves) == self.max_items:
            log.info("Max items on the screen! You lost!")
            return False

        draw_objects(
            frame, self.DCM.circles, self.PM.packmans, self.MCM.ellipse_curves, self.circle_radius,
            self.PM.vectors, self.body_part_indexes, landmarks
        )
        self.draw_score(frame)
        return True

    def pop_out_ellipse_curves(self, landmarks):
        score_bonus = self.MCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus

    def pop_out_packmans(self, landmarks):
        score_bonus = self.PM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus

    def pop_out_circles(self, landmarks):
        score_bonus = self.DCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        self.score += score_bonus


class SoloClassic(Game):
    def __init__(self, w_size):
        super().__init__(w_size)
        self.max_items = config.gameplay.classic_max_circles_destroyed
        self.obj_life_time = config.gameplay.classic_circle_life_time
        self.death_count = -1
        self.obj_live_status = {
            "circle": False,
            "packman": False,
            "mooving_circle": False
        }

    def process(self, frame, landmarks=None):
        if landmarks:
            cur_time = time()
            self.pop_out_circles(landmarks, cur_time)
            self.pop_out_packmans(landmarks, cur_time)
            self.pop_out_ellipse_curves(landmarks, cur_time)

        if not any(self.obj_live_status.values()):
            chance = randint(1, 10)
            self.death_count += 1
            if chance > 2:
                self.add_new_circle()
                self.obj_live_status["circle"] = True
            else:
                if chance == 1:
                    self.add_packman()
                    self.obj_live_status["packman"] = True
                else:
                    self.add_new_ellipse_curve()
                    self.obj_live_status["mooving_circle"] = True

        if self.death_count == self.max_items:
            log.info(f"Game over, your score: {self.score}")
            return False

        draw_objects(
            frame, self.DCM.circles, self.PM.packmans, self.MCM.ellipse_curves, self.circle_radius,
            self.PM.vectors, self.body_part_indexes, landmarks
        )
        self.draw_score(frame)
        return True

    def pop_out_ellipse_curves(self, landmarks, cur_time):
        score_bonus = self.MCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        if score_bonus or cur_time - self.last_draw_timestamp >= self.obj_life_time:
            if len(self.MCM.ellipse_curves) \
                    and 0 < self.MCM.ellipse_curves[0].progress < self.MCM.ellipse_curves[0].a * 4:
                return
            self.obj_live_status["mooving_circle"] = False
            self.MCM.ellipse_curves = []
        self.score += score_bonus

    def pop_out_packmans(self, landmarks, cur_time):
        score_bonus = self.PM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        if score_bonus or cur_time - self.last_draw_timestamp >= self.obj_life_time:
            if len(self.PM.packmans) and 0 < self.PM.packmans[0].progress < self.PM.max_packman_progress:
                return
            self.obj_live_status["packman"] = False
            self.PM.packmans = []
        self.score += score_bonus

    def pop_out_circles(self, landmarks, cur_time):
        score_bonus = self.DCM.pop_out(landmarks, self.body_part_indexes, self.circle_radius)
        if score_bonus or cur_time - self.last_draw_timestamp >= self.obj_life_time:
            self.obj_live_status["circle"] = False
            self.DCM.circles = []
        self.score += score_bonus


class GameWithFriendOpenVINO:
    def __init__(self, w_size, mode1, mode2):
        self.w_size = w_size
        self.p1 = mode1
        self.p2 = mode2
        self.p1_game_status = True
        self.p2_game_status = True

    def get_side(self, joints):
        left_count = 0
        right_count = 0
        for joint in joints:
            if joint.x <= 1/2:
                left_count += 1
            else:
                right_count += 1
        print(left_count, right_count)
        return left_count > right_count

    def validate_joints(self, joints, side):
        for index, joint in enumerate(joints):
            if side == 1:
                if joint.x >= 1/2:
                    joints[index] = None
                else:
                    joints[index] = Joint(joint.x * 2, joint.y, joint.score)
            elif side == 0:
                if joint.x <= 1/2:
                    joints[index] = None
                else:
                    joints[index] = Joint((joint.x - 0.5) * 2, joint.y, joint.score)

    def process(self, image, results):
        for item in results:
            if self.get_side(item):
                self.validate_joints(item, 1)
                if self.p1_game_status:
                    self.p1_game_status = self.p1.process(image[:, :self.w_size[1] // 2], item)
            else:
                self.validate_joints(item, 0)
                if self.p2_game_status:
                    self.p2_game_status = self.p2.process(image[:, self.w_size[1] // 2:], item)

        return self.p1_game_status or self.p2_game_status
