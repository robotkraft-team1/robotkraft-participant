from enum import Enum

class TubeColor(Enum):
    RED = 0
    YELLOW = 1
    BLUE = 2

def string_to_color(string: str) -> TubeColor:
    if string.startswith("r"):
        return TubeColor.RED
    if string.startswith("b"):
        return TubeColor.BLUE
    return TubeColor.YELLOW