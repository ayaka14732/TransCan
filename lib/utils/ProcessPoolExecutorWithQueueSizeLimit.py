from concurrent.futures import ProcessPoolExecutor
from itertools import tee
from queue import Queue
from threading import Thread

class ProcessPoolExecutorWithQueueSizeLimit(ProcessPoolExecutor):
    def __init__(self, queue_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_futures = Queue(maxsize=queue_size)

    def map(self, fn, *iterables):
        queue_futures = self.queue_futures

        it0, it1 = tee(zip(*iterables))

        def usher():
            for args in it0:
                future = self.submit(fn, *args)
                queue_futures.put(future)

        thread = Thread(target=usher)
        thread.start()

        for _ in it1:
            item = queue_futures.get()
            yield item.result()  # item is a future
