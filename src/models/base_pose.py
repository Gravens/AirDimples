import enum
from typing import Optional, Any, List

from utils import Joint


class PoseModel:

    @staticmethod
    def get_joints_from_result(result: Optional[Any]) -> List[Joint]:
        """
        Get list of joints from neural net output.

        If result is None, returns empty list.
        """
        raise NotImplementedError

    BODY_PART_INDEXES: dict = None
    SKELETON: frozenset = None

    class Landmark(enum.IntEnum):
        pass
