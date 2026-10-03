from enum import Enum
import typing as T



class TubeColor(Enum):
    RED = 0
    YELLOW = 1
    BLUE = 2

class Tube:
    def __init__(self, color: TubeColor) -> None:
        self.color: TubeColor = color
    
    def compare(self, other: T.Self, order_list: list[TubeColor]) -> int:
        for color in order_list:
            if color == self.color and color == other.color:
                return 0
            if color == self.color:
                return 1
            if color == other.color:
                return -1
        raise ValueError("order_list is not complete")
    
    @T.override
    def __str__(self) -> str:
        return f"<Tube color={self.color}>"
    
    @T.override
    def __repr__(self) -> str:
        return str(self)

def partition(arr: list[Tube], low: int, high: int, order_list: list[TubeColor], empty: int):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j].compare(pivot, order_list) == 1:
            i += 1
            print(f"Move {j} into {empty}")
            print(f"Move {i} into {j}")
            print(f"Move {empty} into {i}")
            arr[i], arr[j] = arr[j], arr[i]

    if arr[high].compare(arr[i + 1], order_list) != 0:
        print(f"Move {high} into {empty}")
        print(f"Move {i + 1} into {high}")
        print(f"Move {empty} into {i + 1}")
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def _quicksort(arr: list[Tube], low: int, high: int, order_list: list[TubeColor], empty: int):
    if low < high:
        p = partition(arr, low, high, order_list, empty)
        _quicksort(arr, low, p - 1, order_list, empty)
        _quicksort(arr, p + 1, high, order_list, empty)

def quicksort(arr: list[Tube], order_list: list[TubeColor], empty: int):
    return _quicksort(arr, 0, len(arr)-1, order_list, empty)


if __name__ == "__main__":
    arr = [Tube(TubeColor.BLUE), Tube(TubeColor.RED), Tube(TubeColor.YELLOW), Tube(TubeColor.YELLOW), Tube(TubeColor.YELLOW), Tube(TubeColor.RED), Tube(TubeColor.YELLOW), Tube(TubeColor.BLUE)]
    quicksort(arr, [TubeColor.RED, TubeColor.BLUE, TubeColor.YELLOW], empty=25)
    print(arr)