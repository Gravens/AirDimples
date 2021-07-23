import enum

from models.base_pose import PoseModel
from utils import Joint, normalize, log


def distance(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**(1/2)


def get_sin(elbow_p, wrist_p, c):
    return (wrist_p.y - elbow_p.y) / c


def get_cos(elbow_p, wrist_p, c):
    return (wrist_p.x - elbow_p.x) / c


def get_additional_joint(divider, elbow_p, wrist_p):
    cos_a = get_cos(elbow_p, wrist_p, divider)
    sin_a = get_sin(elbow_p, wrist_p, divider)

    a = sin_a / divider
    b = cos_a / divider
    if not (0 <= wrist_p.x + b <= 1 and 0 <= wrist_p.y + a <= 1):
        return wrist_p
    return Joint(wrist_p.x + b, wrist_p.y + a, wrist_p.score)


class IntelPoseModel(PoseModel):

    @staticmethod
    def get_joints_from_result(result=None):
        if result is None:
            return []

        try:
            (poses, scores), frame_meta = result
            if len(poses) < 1:
                return []
            img_rows, img_cols, _ = frame_meta['frame'].shape
            joints = []
            for ind, pose in enumerate(poses):
                if ind == 2:
                    break
                joints.append([Joint(normalize(x, img_cols), normalize(y, img_rows), score) for x, y, score, _ in pose])
                # Replace wrist on hand
                joints[ind][9] = get_additional_joint(2, joints[ind][7], joints[ind][9])
                joints[ind][10] = get_additional_joint(2, joints[ind][8], joints[ind][10])

            return joints
        except Exception:
            log.error("Unable to convert result to joints")
            raise

    BODY_PART_INDEXES = {
        "L_hand": (9,),
        "R_hand": (10,),
        "L_foot": (15,),
        "R_foot": (16,)
    }
# (7, 9) (8, 10)
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
        (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
        (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
        (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
        (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
        (Landmark.NOSE, Landmark.LEFT_EYE),
        (Landmark.NOSE, Landmark.RIGHT_EYE),
        (Landmark.LEFT_EYE, Landmark.LEFT_EAR),
        (Landmark.RIGHT_EYE, Landmark.RIGHT_EAR),
        (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    ])
