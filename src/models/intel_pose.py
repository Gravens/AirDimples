import enum

from utils import Point, Joint, normalize


class IntelPoseModel:

    def __init__(self):
        self.body_part_indexes = {
            "L_hand": (10,),
            "R_hand": (9,),
            "L_foot": (16,),
            "R_foot": (15,)
        }

    @staticmethod
    def get_joints_from_result(result=None):
        """
        Get list of joints from neural net output.

        If result is None, returns empty list.
        """
        if result is None:
            return []

        try:
            (poses, scores), frame_meta = result
            if len(poses) < 1:
                return []
            img_rows, img_cols, _ = frame_meta['frame'].shape
            pose = poses[0]
            joints = [Joint(normalize(x, img_cols), normalize(y, img_rows), score) for x, y, score, _ in pose]
            return joints
        except Exception:
            print("Unable to convert result to joints")
            raise

    class Landmark(enum.IntEnum):
        NOSE = 0
        LEFT_EYE = 1
        RIGHT_EYE = 2
        LEFT_EAR = 3
        RIGHT_EAR = 4
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_ELBOW = 7
        RIGHT_ELBOW = 8
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16

    SKELETON = frozenset([
        (Landmark.LEFT_ANKLE, Landmark.LEFT_KNEE),
        (Landmark.RIGHT_ANKLE, Landmark.RIGHT_KNEE),
        (Landmark.LEFT_KNEE, Landmark.LEFT_HIP),
        (Landmark.RIGHT_KNEE, Landmark.RIGHT_HIP),
        (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
        (Landmark.LEFT_SHOULDER, Landmark.RIGHT_HIP),
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
        (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
        (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
        (Landmark.LEFT_EYE, Landmark.RIGHT_EYE),
        (Landmark.NOSE, Landmark.LEFT_EYE),
        (Landmark.NOSE, Landmark.RIGHT_EYE),
        (Landmark.LEFT_EYE, Landmark.LEFT_EAR),
        (Landmark.RIGHT_EYE, Landmark.RIGHT_EAR),
        (Landmark.LEFT_EAR, Landmark.LEFT_SHOULDER),
        (Landmark.RIGHT_EAR, Landmark.RIGHT_SHOULDER),
    ])
