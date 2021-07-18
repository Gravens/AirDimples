class DefaultCircle:
    def __init__(self, center, color, side):
        self.center = center
        self.color = color
        self.side = side


class Packman:
    def __init__(self, center, color, last_vector):
        self.center = center
        self.color = color
        self.last_vector = last_vector
        self.progress = 0
        self.earned_progress = 0


class MoovingCircle:
    def __init__(self, a, b, center, equation, color, vector):
        self.a = a
        self.b = b
        self.center = center
        self.equation = equation
        self.color = color
        self.vector = vector
        self.progress = 0
        self.earned_progress = 0
