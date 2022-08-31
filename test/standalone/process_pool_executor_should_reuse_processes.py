# See also https://stackoverflow.com/q/50970376

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

class VeryLargeObject:
    def __init__(self) -> None:
        print('Initialised')

    def __call__(self) -> None:
        pass

def heavy_computation(i: int) -> int:
    time.sleep(1)
    return i + 1

def sub_proc(obj: int) -> int:
    global vlo
    if 'vlo' not in globals():
        vlo = VeryLargeObject()
    vlo()  # if the object is callable, we can confirm that the object is there

    return heavy_computation(obj)

def main_proc():
    objs = list(range(8))
    ctx = multiprocessing.get_context('spawn')
    max_workers = 2
    print(f"'Initialised' should only be printed for {max_workers} times.")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for i, result in enumerate(executor.map(sub_proc, objs)):
            expected_result = i + 1
            assert result == expected_result

if __name__ == '__main__':
    main_proc()
