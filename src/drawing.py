from math import floor

import cv2

from config import config
from utils import denormalize


def draw_joints(image, joints, skeleton=None):
    """Draw joints and optionally the skeleton on the image"""
    image_h, image_w, _ = image.shape

    # Denormalize joints coordinates and only select valid ones
    for item in joints:
        idx_to_coordinates = {}
        for idx, joint in enumerate(item):
            if joint.score < config.app.detection_threshold:
                continue
            joint_px = denormalize(joint.x, image_w), denormalize(joint.y, image_h)
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
                        config.graphics.connection_color,
                        config.graphics.connection_thickness,
                    )

        # Draw joints above the skeleton
        for joint_px in idx_to_coordinates.values():
            cv2.circle(
                image,
                joint_px,
                config.graphics.joint_radius,
                config.graphics.joint_color,
                config.graphics.joint_thickness,
            )


def draw_circle(image, center, circle_radius, color, thickness=2):
    cv2.circle(image, center, circle_radius, color, thickness, lineType=cv2.LINE_AA)


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

    draw_circles_of_indexes(hand_indexes, config.graphics.hand_color)
    draw_circles_of_indexes(foot_indexes, config.graphics.foot_color)


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
