

class MediapipePoseModel:

    def __init__(self):
        self.body_part_indexes = {
            "L_hand": (20, 22, 18, 16),
            "R_hand": (21, 19, 17, 15),
            "L_foot": (28, 32, 30),
            "R_foot": (27, 31, 29)
        }
