from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import multiprocessing
import time

from lib.utils import ProcessPoolExecutorWithQueueSizeLimit

def f(i, j):
    print(f'c {i:02d} {j:02d}')
    return i

def main():
    ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutorWithQueueSizeLimit(queue_size=4, max_workers=2, mp_context=ctx) as p:
        for i in p.map(f, range(16), range(16)):
            time.sleep(0.15)
            print(f'm {i:02d} {i:02d}')

if __name__ == '__main__':
    main()
