import enum

from utils import Joint, log


class MediapipePoseModel:

    def __init__(self):
        self.body_part_indexes = {
            "L_hand": (21, 19, 17, 15),
            "R_hand": (20, 22, 18, 16),
            "L_foot": (27, 31, 29),
            "R_foot": (28, 32, 30),
        }

    @staticmethod
    def get_joints_from_result(result=None):
        """
        Get list of joints from neural net output.

        If result is None, returns empty list.
        """
        if result is None or result.pose_landmarks is None:
            return []

        try:
            joints = [Joint(landmark.x, landmark.y, landmark.visibility) for landmark in result.pose_landmarks.landmark]
            return joints
        except Exception:
            log.error("Unable to convert result to joints")
            raise
    
    class Landmark(enum.IntEnum):
        NOSE = 0
        LEFT_EYE_INNER = 1
        LEFT_EYE = 2
        LEFT_EYE_OUTER = 3
        RIGHT_EYE_INNER = 4
        RIGHT_EYE = 5
        RIGHT_EYE_OUTER = 6
        LEFT_EAR = 7
        RIGHT_EAR = 8
        MOUTH_LEFT = 9
        MOUTH_RIGHT = 10
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_PINKY = 17
        RIGHT_PINKY = 18
        LEFT_INDEX = 19
        RIGHT_INDEX = 20
        LEFT_THUMB = 21
        RIGHT_THUMB = 22
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_HEEL = 29
        RIGHT_HEEL = 30
        LEFT_FOOT_INDEX = 31
        RIGHT_FOOT_INDEX = 32

    SKELETON = frozenset([
        (Landmark.NOSE, Landmark.RIGHT_EYE_INNER),
        (Landmark.RIGHT_EYE_INNER, Landmark.RIGHT_EYE),
        (Landmark.RIGHT_EYE, Landmark.RIGHT_EYE_OUTER),
        (Landmark.RIGHT_EYE_OUTER, Landmark.RIGHT_EAR),
        (Landmark.NOSE, Landmark.LEFT_EYE_INNER),
        (Landmark.LEFT_EYE_INNER, Landmark.LEFT_EYE),
        (Landmark.LEFT_EYE, Landmark.LEFT_EYE_OUTER),
        (Landmark.LEFT_EYE_OUTER, Landmark.LEFT_EAR),
        (Landmark.MOUTH_RIGHT, Landmark.MOUTH_LEFT),
        (Landmark.RIGHT_SHOULDER, Landmark.LEFT_SHOULDER),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
        (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
        (Landmark.RIGHT_WRIST, Landmark.RIGHT_PINKY),
        (Landmark.RIGHT_WRIST, Landmark.RIGHT_INDEX),
        (Landmark.RIGHT_WRIST, Landmark.RIGHT_THUMB),
        (Landmark.RIGHT_PINKY, Landmark.RIGHT_INDEX),
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
        (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
        (Landmark.LEFT_WRIST, Landmark.LEFT_PINKY),
        (Landmark.LEFT_WRIST, Landmark.LEFT_INDEX),
        (Landmark.LEFT_WRIST, Landmark.LEFT_THUMB),
        (Landmark.LEFT_PINKY, Landmark.LEFT_INDEX),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
        (Landmark.RIGHT_HIP, Landmark.LEFT_HIP),
        (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
        (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
        (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
        (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
        (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
        (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
        (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
        (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
        (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
        (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    ])
