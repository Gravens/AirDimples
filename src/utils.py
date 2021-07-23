import logging
import sys
from typing import NamedTuple

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
