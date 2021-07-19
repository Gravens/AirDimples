import enum

from utils import Point, Joint, normalize


class IntelPoseModel:

    def __init__(self):
        pass

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

    SKELETON = frozenset([
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ])
