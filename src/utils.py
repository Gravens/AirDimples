from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


def denormalize_coordinates(coordinates, size):
    """
    Convert normalized coordinates to integer coordinates that correspond to plane size.

    Take (xn, yn) where (0 <= xn, yn <= 1) and (width, height).

    Return False if normalized coordinates are not valid.

    Return (x, y) where (0 <= x <= width) and (0 <= y <= height).
    """
    # Unpack arguments
    xn, yn = coordinates
    width, height = size

    # Check validity
    if not (0 <= xn <= 1 and 0 <= yn <= 1):
        return False

    # Denormalize and round to int
    x = round(xn * width)
    y = round(yn * height)

    return x, y
