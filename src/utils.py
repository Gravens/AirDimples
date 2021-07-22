import logging
import sys
from typing import NamedTuple
from math import floor

import cv2


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format='[%(asctime)s] [%(module)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger()


class Point(NamedTuple):
    x: float
    y: float


class Joint(NamedTuple):
    x: float
    y: float
    score: float


def get_int_middle_point(point1, point2):
    p1x, p1y = point1
    p2x, p2y = point2
    p3x = (p1x + p2x) // 2
    p3y = (p1y + p2y) // 2
    return p3x, p3y


def flip_joints(joints):
    if joints is None:
        return None
    ret = []
    for joint in joints:
        ret.append(Joint(1-joint.x, joint.y, joint.score))
    return ret


def normalize(coordinate: int, length: int) -> float:
    """Convert a pixel coordinate to normalized float coordinate between 0 and 1"""
    if not (0 <= coordinate <= length):
        raise ValueError('Coordinate exceeds bounds')
    return coordinate / length


def denormalize(coordinate: float, length: int) -> int:
    """Convert a normalized float coordinate between 0 and 1 to a pixel coordinate"""
    if not (0 <= coordinate <= 1):
        raise ValueError('Coordinate exceeds bounds')
    return int(coordinate * length)


def denormalize_coordinates(coordinates, size):
    """
    Convert normalized coordinates to integer coordinates that correspond to plane size.

    Take (xn, yn) where (0 <= xn, yn <= 1) and (width, height).

    Return False if normalized coordinates are not valid.

    Return (x, y) where (0 <= x <= width) and (0 <= y <= height).
    """
    # Unpack arguments
    xn, yn = coordinates
    width, height = size

    # Check validity
    if not (0 <= xn <= 1 and 0 <= yn <= 1):
        return False

    # Denormalize and round to int
    x = int(xn * width)
    y = int(yn * height)

    return x, y


JOINT_COLOR = (0, 0, 255)
JOINT_RADIUS = 2
JOINT_THICKNESS = 2
CONNECTION_COLOR = (0, 255, 0)
CONNECTION_THICKNESS = 2
THRESHOLD = 0.1


def draw_joints(image, joints, skeleton=None):
    """Draw joints and optionally the skeleton on the image"""
    img_rows, img_cols, _ = image.shape

    # Denormalize joints coordinates and only select valid ones
    for item in joints:
        idx_to_coordinates = {}
        for idx, joint in enumerate(item):
            if joint.score < THRESHOLD:
                continue
            joint_px = denormalize_coordinates((joint.x, joint.y), (img_cols, img_rows))
            if joint_px is False:
                continue
            idx_to_coordinates[idx] = joint_px

        # Draw skeleton connections
        if skeleton:
            for connection in skeleton:
                st_idx, en_idx = connection
                if st_idx in idx_to_coordinates and en_idx in idx_to_coordinates:
                    cv2.line(
                        image,
                        idx_to_coordinates[st_idx],
                        idx_to_coordinates[en_idx],
                        CONNECTION_COLOR,
                        CONNECTION_THICKNESS
                    )

        # Draw joints above the skeleton
        for joint_px in idx_to_coordinates.values():
            cv2.circle(image, joint_px, JOINT_RADIUS, JOINT_COLOR, JOINT_THICKNESS)


def draw_circle(image, center, circle_radius, color, thickness=2):
    cv2.circle(image, center, circle_radius, color, thickness, lineType=cv2.LINE_AA)


HAND_CIRCLE_COLOR = (122, 36, 27)
FOOT_CIRCLE_COLOR = (15, 255, 235)


def draw_limb_circles(image, joints, body_part_indexes, threshold=0.3, radius=20):
    if not joints:
        return

    image_h, image_w, _ = image.shape

    def draw_circles_of_indexes(indexes, color):
        for idx in indexes:
            if joints[idx] is None or joints[idx].score < threshold:
                continue
            center = (denormalize(joints[idx].x, image_w), denormalize(joints[idx].y, image_h))
            draw_circle(image, center, radius, color)

    hand_indexes = (body_part_indexes["R_hand"][0], body_part_indexes["L_hand"][0])
    foot_indexes = (body_part_indexes["R_foot"][0], body_part_indexes["L_foot"][0])

    draw_circles_of_indexes(hand_indexes, HAND_CIRCLE_COLOR)
    draw_circles_of_indexes(foot_indexes, FOOT_CIRCLE_COLOR)


def draw_objects(frame, circles, packmans, ellipse_curves, circle_radius, vectors, body_part_indexes, joints):
    for item in circles:
        draw_circle(frame, item.center, circle_radius, item.color)
        cv2.putText(
            frame,
            item.side,
            (item.center[0] - 4, item.center[1] + 5),
            cv2.FONT_ITALIC, 0.55,
            item.color,
            2
        )

    for item in packmans:
        center = tuple(map(floor, item.center))
        draw_circle(frame, center, circle_radius, item.color)
        cv2.line(
            frame,
            (center[0], center[1]),
            (center[0] + circle_radius * vectors[item.last_vector][0],
             center[1] + circle_radius * vectors[item.last_vector][1]),
            item.color,
            2
        )

    for item in ellipse_curves:
        center = tuple(map(floor, item.center))
        draw_circle(frame, center, circle_radius, item.color)
