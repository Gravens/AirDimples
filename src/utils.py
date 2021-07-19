from typing import NamedTuple

import cv2


class Point(NamedTuple):
    x: float
    y: float


class Joint(NamedTuple):
    x: float
    y: float
    score: float


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
    x = round(xn * width)
    y = round(yn * height)

    return x, y


JOINT_COLOR = (0, 0, 255)
JOINT_RADIUS = 2
JOINT_THICKNESS = 2
CONNECTION_COLOR = (0, 255, 0)
CONNECTION_THICKNESS = 2


def draw_joints(image, joints, skeleton=None):
    """Draw joints and optionally the skeleton on the image"""
    img_rows, img_cols, _ = image.shape

    # Denormalize joints coordinates and only select valid ones
    idx_to_coordinates = {}
    for idx, joint in enumerate(joints):
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
