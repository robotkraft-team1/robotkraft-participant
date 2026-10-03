from .colors import string_to_color, TubeColor
from .sort_algo import Tube, quicksort

def list_seen_tubes() -> list[Tube]:
    return [Tube(TubeColor.BLUE), Tube(TubeColor.RED), Tube(TubeColor.YELLOW), Tube(TubeColor.YELLOW), Tube(TubeColor.YELLOW), Tube(TubeColor.RED), Tube(TubeColor.YELLOW), Tube(TubeColor.BLUE)]

def get_empty_tube_place() -> int:
    return 25

def main_task(order_str: list[str]):
    order = [string_to_color(s) for s in order_str]

    tubes = list_seen_tubes()
    quicksort(tubes, order, empty=get_empty_tube_place())
    print(tubes)