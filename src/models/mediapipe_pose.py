from utils import Point


class MediapipePoseModel:

    def __init__(self):
        self.body_part_indexes = {
            "L_hand": (20, 22, 18, 16),
            "R_hand": (21, 19, 17, 15),
            "L_foot": (28, 32, 30),
            "R_foot": (27, 31, 29)
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
            joints = [Point(landmark.x, landmark.y) for landmark in result.pose_landmarks.landmark]
            return joints
        except Exception:
            print("Unable to convert result to joints")
            raise
