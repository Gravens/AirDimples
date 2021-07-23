from time import time

import cv2

import utils
from config import config
from gameplay import GameWithFriendOpenVINO, SoloIntensiveFastAim, SoloClassic


class Label:
    def __init__(
            self,
            text,
            pos=(0, 0),
            font_face=cv2.FONT_HERSHEY_COMPLEX,
            font_scale=1.0,
            color=(255, 255, 255),
            thickness=1,
    ):
        self.text = text
        self.pos = pos
        self.font_face = font_face
        self.font_scale = font_scale
        self.color = color
        self.thickness = thickness

    def get_size(self):
        text_size, _ = cv2.getTextSize(self.text, self.font_face, self.font_scale, self.thickness)
        return text_size

    def center_on_point(self, point):
        center_x, center_y = point
        text_w, text_h = self.get_size()
        pos_x = center_x - (text_w // 2)
        pos_y = center_y + (text_h // 2)
        self.pos = (pos_x, pos_y)

    def draw(self, image):
        cv2.putText(
            image,
            self.text,
            self.pos,
            self.font_face,
            self.font_scale,
            self.color,
            self.thickness,
            lineType=cv2.LINE_AA,
        )


class Button:
    def __init__(self, tl_point, br_point, text, w_size):
        self.tl_point = tl_point
        self.br_point = br_point
        self.thickness = 2

        center = utils.get_int_middle_point(tl_point, br_point)
        self.label = Label(text, font_scale=0.8, color=config.graphics.label_default_color)
        self.label.center_on_point(center)

        self.click_interval = 2
        self.last_click_timestamp = time()

        self.w_size = w_size

        self.clicked = False

    def click(self, state=None):
        c_time = time()
        if c_time - self.last_click_timestamp >= self.click_interval:
            self.clicked = not self.clicked if state is None else state
            self.last_click_timestamp = c_time

    def include(self, joint, normalized=True):
        x_valid = self.tl_point[0] <= (joint.x * self.w_size[1] if normalized else joint.x) <= self.br_point[0]
        y_valid = self.tl_point[1] <= (joint.y * self.w_size[0] if normalized else joint.y) <= self.br_point[1]
        return x_valid and y_valid

    def draw(self, image):
        if self.clicked:
            color = config.graphics.button_clicked_color
            self.label.color = config.graphics.label_clicked_color
        else:
            color = config.graphics.button_default_color
            self.label.color = config.graphics.label_default_color
        cv2.rectangle(image,
                      self.tl_point,
                      self.br_point,
                      color,
                      self.thickness)
        self.label.draw(image)


class StartButton(Button):
    def __init__(self, tl_point, br_point, text, w_size):
        super().__init__(tl_point, br_point, text, w_size)
        self.click_interval = 0.4

    def draw(self, image):
        if self.clicked:
            color = config.graphics.button_clicked_color
            self.label.color = config.graphics.label_clicked_color
        else:
            color = config.graphics.button_default_color
            self.label.color = config.graphics.label_default_color
        cv2.circle(
            image,
            utils.get_int_middle_point(self.tl_point, self.br_point),
            (self.br_point[1] - self.tl_point[1]) // 2,
            color,
            self.thickness
        )
        self.label.draw(image)


class GUI:
    def __init__(self,  w_size):
        self.start_status = False
        self.quit_status = False
        self.player_count = None
        self.game_mode = None
        self.body_part_indexes = config.app.model.BODY_PART_INDEXES
        self.w_size = w_size

        self.countdown = 5
        self.last_countdown_timestamp = time()

        image_h, image_w, _ = w_size

        top_row_margin = 40
        top_row_btn_count = 4
        button_width = (image_w - top_row_margin * (top_row_btn_count + 1)) // top_row_btn_count
        button_height = button_width * 9 // 16

        top_row_right_xs = [(top_row_margin + button_width)*(i+1) for i in range(top_row_btn_count)]
        top_row_left_xs = [right_x - button_width for right_x in top_row_right_xs]
        top_row_top_y = top_row_margin
        top_row_bottom_y = top_row_margin + button_height

        top_row_tls = [(top_row_left_x, top_row_top_y) for top_row_left_x in top_row_left_xs]
        top_row_brs = [(top_row_right_x, top_row_bottom_y) for top_row_right_x in top_row_right_xs]

        start_btn_height = int(button_height*1.3)

        left_start_tl = (top_row_left_xs[0], top_row_bottom_y + top_row_margin)
        left_start_br = (left_start_tl[0] + start_btn_height, left_start_tl[1] + start_btn_height)
        right_start_tl = (top_row_left_xs[-1] + button_width - start_btn_height, top_row_bottom_y + top_row_margin)
        right_start_br = (right_start_tl[0] + start_btn_height, right_start_tl[1] + start_btn_height)

        self.buttons = {
            'one_player': Button(top_row_tls[0], top_row_brs[0], "1 Player", self.w_size),
            'two_players': Button(top_row_tls[1], top_row_brs[1], "2 Players", self.w_size),
            'classic_mode': Button(top_row_tls[2], top_row_brs[2], "Classic", self.w_size),
            'intensive_mode': Button(top_row_tls[3], top_row_brs[3], "Intensive Aim", self.w_size),
            'left_start': StartButton(left_start_tl, left_start_br, "Start", self.w_size),
            'right_start': StartButton(right_start_tl, right_start_br, "Start", self.w_size),
        }

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

        image_h, image_w, _ = image.shape

        lbl_countdown = Label(
            str(self.countdown),
            font_scale=config.graphics.countdown_label_font_scale,
            thickness=config.graphics.countdown_label_thickness,
            color=config.graphics.countdown_label_color
        )
        lbl_countdown.center_on_point((image_w//2, image_h//2))
        lbl_countdown.draw(image)

    def reset(self):
        self.start_status = False
        self.game_mode = None
        self.player_count = None
        self.countdown = 5
        for button in self.buttons.values():
            button.clicked = False

    def check_start(self):
        if self.buttons['left_start'].clicked and self.buttons['right_start'].clicked:
            if self.player_count == 2:
                p_area_size = (self.w_size[0], self.w_size[1] // 2, self.w_size[2])
                if self.game_mode == 0:
                    self.game_mode = GameWithFriendOpenVINO(
                        self.w_size,
                        SoloClassic(p_area_size),
                        SoloClassic(p_area_size),
                    )
                else:
                    self.game_mode = GameWithFriendOpenVINO(
                        self.w_size,
                        SoloIntensiveFastAim(p_area_size),
                        SoloIntensiveFastAim(p_area_size),
                    )
            else:
                if self.game_mode == 0:
                    self.game_mode = SoloClassic(self.w_size)
                else:
                    self.game_mode = SoloIntensiveFastAim(self.w_size)

    def update_game_params(self):
        if self.buttons['one_player'].clicked:
            self.player_count = 1

        if self.buttons['two_players'].clicked:
            self.player_count = 2

        if self.buttons['classic_mode'].clicked:
            self.game_mode = 0

        if self.buttons['intensive_mode'].clicked:
            self.game_mode = 1

        if self.buttons['left_start'].clicked and self.buttons['right_start'].clicked:
            self.start_status = True

    def toggle_buttons(self, clicked_name):
        connections = [("one_player", "two_players"), ("classic_mode", "intensive_mode")]
        for connection in connections:
            if clicked_name in connection:
                self.buttons[connection[not connection.index(clicked_name)]].clicked = False

    def update_buttons(self, joints):
        for name, button in self.buttons.items():
            indexes_inside = []
            for item in joints:
                for body_part in self.body_part_indexes:
                    for index in self.body_part_indexes[body_part]:
                        if button.include(item[index]):
                            indexes_inside.append(index)
            if type(button) is StartButton:
                if self.player_count is None or self.game_mode is None:
                    continue
                button.click(state=indexes_inside)
            elif indexes_inside:
                button.click()
                self.toggle_buttons(name)

    def draw_menu(self, image):
        for button in self.buttons.values():
            button.draw(image)

        image_size = image.shape[:2]
        image_h, image_w = image_size

        lbl_quit = Label(f'Press {config.app.quit_key} to quit', font_face=cv2.FONT_HERSHEY_COMPLEX_SMALL)
        lbl_quit.center_on_point((image_w//2, 18))
        lbl_quit.draw(image)
