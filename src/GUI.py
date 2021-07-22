from gameplay import GameWithFriendOpenVINO, SoloIntensiveFastAim, SoloClassic
from models.intel_pose import IntelPoseModel
from models.mediapipe_pose import MediapipePoseModel
from time import  time
import cv2

CIRCLE_RADIUS = 50
LIFE_TIME = 2
INTERVAL = 3
MAX_ITEM_DEATH = 10
MAX_ITEMS_ON_SCREEN = 4
BODY_PART_INDEXES = IntelPoseModel().body_part_indexes


class Button:
    def __init__(self, tl_point, br_point, text, w_size):
        self.tl_point = tl_point
        self.br_point = br_point
        self.text = text
        self.blue = (255, 0, 0)
        self.green = (0, 255, 0)
        self.thickness = 2

        self.text_x = (self.br_point[0] + self.tl_point[0]) // 2 - int((self.br_point[0] - self.tl_point[0]) / 3.5)
        self.text_y = (self.br_point[1] + self.tl_point[1]) // 2 + 8

        self.click_interval = 2
        self.last_click_timestamp = time()

        self.w_size = w_size

        self.clicked = False

    def click(self):
        c_time = time()
        if c_time - self.last_click_timestamp >= self.click_interval:
            self.clicked = not self.clicked
            self.last_click_timestamp = c_time

    def include(self, joint, normalized=True):
        x_valid = self.tl_point[0] <= (joint.x * self.w_size[1] if normalized else joint.x) <= self.br_point[0]
        y_valid = self.tl_point[1] <= (joint.y * self.w_size[0] if normalized else joint.y) <= self.br_point[1]
        return x_valid and y_valid

    def draw(self, image):
        cv2.rectangle(image,
                      self.tl_point,
                      self.br_point,
                      self.green if self.clicked else self.blue,
                      self.thickness)

        cv2.putText(
            image,
            self.text,
            (self.text_x, self.text_y),
            cv2.FONT_ITALIC,
            0.7,
            self.green if self.clicked else self.blue,
        )


class GUI:
    def __init__(self, w_size, body_part_indexes, add_quit_button=False):
        self.start_status = False
        self.quit_status = False
        self.player_count = None
        self.game_mode = None
        self.body_part_indexes = body_part_indexes
        self.w_size = w_size

        self.margin_top = int(w_size[0] * 0.05)
        self.margin_left = int(w_size[1] * 0.05)

        self.menu_item_width = int(w_size[1] * 0.23)
        self.menu_item_height = int(w_size[1] * 0.14)

        self.countdown = 5
        self.last_countdown_timestamp = time()

        self.buttons = {
            'one_player_btn': Button((self.margin_left, self.margin_top),
                                     (self.margin_left + self.menu_item_width,
                                      self.margin_top + self.menu_item_height),
                                     "1 Player",
                                     self.w_size),
            'two_player_btn': Button((self.margin_left, self.margin_top * 2 + self.menu_item_height),
                                     (self.margin_left + self.menu_item_width,
                                      self.margin_top * 2 + self.menu_item_height * 2),
                                     "2 Player",
                                     self.w_size),
            'classic_mode_btn': Button((self.margin_left, self.margin_top * 3 + self.menu_item_height * 2),
                                       (self.margin_left + self.menu_item_width,
                                        self.margin_top * 3 + self.menu_item_height * 3),
                                       "Classic",
                                       self.w_size),
            'intensive_mode_btn': Button((self.margin_left, self.margin_top * 4 + self.menu_item_height * 3),
                                         (self.margin_left + self.menu_item_width,
                                          self.margin_top * 4 + self.menu_item_height * 4),
                                         "Intensive Aim",
                                         self.w_size),
            'start_btn': Button((self.w_size[1] - self.margin_left - self.menu_item_width, self.margin_top),
                                (self.w_size[1] - self.margin_left, self.margin_top + self.menu_item_height),
                                "Start",
                                self.w_size),
            'quit_btn': Button((self.w_size[1] - self.margin_left - self.menu_item_width,
                                self.w_size[0] - self.margin_top - self.menu_item_height),
                               (self.w_size[1] - self.margin_left, self.w_size[0] - self.margin_top),
                               "Quit",
                               self.w_size)
        }
        # To hit center
        self.buttons['intensive_mode_btn'].text_x -= self.margin_left
        self.buttons['start_btn'].text_x += self.margin_left // 2
        self.buttons['quit_btn'].text_x += self.margin_left // 2
        if not add_quit_button:
            self.buttons.pop('quit_btn')

    def process(self, image, joints):

        if len(joints) != 0:
            self.update_buttons(joints)

        self.update_game_params()

        self.check_start()

        self.draw_menu(image)

        return self.quit_status

    def start_prepare(self, image):
        cur_t = time()
        if cur_t - self.last_countdown_timestamp >= 1:
            self.countdown -= 1
            self.last_countdown_timestamp = cur_t

        cv2.putText(image,
                    str(self.countdown),
                    ((self.w_size[1] // 2) - 35, (self.w_size[0] // 2) + 30),
                    cv2.FONT_ITALIC,
                    2,
                    (0, 255, 255),
                    4)

    def reset(self):
        self.start_status = False
        self.game_mode = None
        self.player_count = None
        self.countdown = 5
        for button in self.buttons.values():
            button.clicked = False

    def check_start(self):
        if self.buttons["start_btn"].clicked:
            if self.player_count == 2:
                p_area_size = (self.w_size[0], self.w_size[1] // 2, self.w_size[2])
                if self.game_mode == 0:
                    self.game_mode = GameWithFriendOpenVINO(self.w_size,
                                                            SoloClassic(p_area_size,
                                                                        circle_radius=CIRCLE_RADIUS,
                                                                        life_time=LIFE_TIME,
                                                                        max_items=MAX_ITEM_DEATH,
                                                                        body_part_indexes=BODY_PART_INDEXES),
                                                            SoloClassic(p_area_size,
                                                                        circle_radius=CIRCLE_RADIUS,
                                                                        life_time=LIFE_TIME,
                                                                        max_items=MAX_ITEM_DEATH,
                                                                        body_part_indexes=BODY_PART_INDEXES))
                else:
                    self.game_mode = GameWithFriendOpenVINO(self.w_size,
                                                            SoloIntensiveFastAim(p_area_size,
                                                                                 circle_radius=CIRCLE_RADIUS,
                                                                                 interval=INTERVAL,
                                                                                 max_items=MAX_ITEMS_ON_SCREEN,
                                                                                 body_part_indexes=BODY_PART_INDEXES),
                                                            SoloIntensiveFastAim(p_area_size,
                                                                                 circle_radius=CIRCLE_RADIUS,
                                                                                 interval=INTERVAL,
                                                                                 max_items=MAX_ITEMS_ON_SCREEN,
                                                                                 body_part_indexes=BODY_PART_INDEXES))
            else:
                if self.game_mode == 0:
                    self.game_mode = SoloClassic(self.w_size,
                                                 circle_radius=CIRCLE_RADIUS,
                                                 life_time=LIFE_TIME,
                                                 max_items=MAX_ITEM_DEATH,
                                                 body_part_indexes=BODY_PART_INDEXES)
                else:
                    self.game_mode = SoloIntensiveFastAim(self.w_size,
                                                          circle_radius=CIRCLE_RADIUS,
                                                          interval=INTERVAL,
                                                          max_items=MAX_ITEMS_ON_SCREEN,
                                                          body_part_indexes=BODY_PART_INDEXES)

    def update_game_params(self):
        if self.buttons['one_player_btn'].clicked:
            self.player_count = 1

        if self.buttons['two_player_btn'].clicked:
            self.player_count = 2

        if self.buttons['classic_mode_btn'].clicked:
            self.game_mode = 0

        if self.buttons['intensive_mode_btn'].clicked:
            self.game_mode = 1

        if self.buttons['start_btn'].clicked:
            self.start_status = True

        if 'quit_btn' in self.buttons and self.buttons['quit_btn'].clicked:
            self.quit_status = True

    def toggle_buttons(self, clicked_name):
        connections = [("one_player_btn", "two_player_btn"), ("classic_mode_btn", "intensive_mode_btn")]
        for connection in connections:
            if clicked_name in connection:
                self.buttons[connection[not connection.index(clicked_name)]].clicked = False

    def update_buttons(self, joints):
        for item in joints:
            for body_part in self.body_part_indexes:
                for index in self.body_part_indexes[body_part]:
                    for name, button in self.buttons.items():
                        if button.include(item[index]):
                            if name == "start_btn" and (self.player_count is None or self.game_mode is None):
                                continue
                            button.click()
                            self.toggle_buttons(name)

    def draw_menu(self, image):
        for button in self.buttons.values():
            button.draw(image)
