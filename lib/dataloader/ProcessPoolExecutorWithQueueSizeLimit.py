from concurrent.futures import ProcessPoolExecutor
from queue import Queue
from threading import Thread

class ProcessPoolExecutorWithQueueSizeLimit(ProcessPoolExecutor):
    def __init__(self, queue_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.done_marker = object()
        self.queue_futures = Queue(maxsize=queue_size)

    def map(self, fn, *iterables):
        def usher():
            for args in zip(*iterables):
                future = self.submit(fn, *args)
                self.queue_futures.put(future)
            self.queue_futures.put(self.done_marker)

        thread = Thread(target=usher)
        thread.start()

        while True:
            item = self.queue_futures.get()
            if item is self.done_marker:
                break
            yield item.result()  # item is a future
