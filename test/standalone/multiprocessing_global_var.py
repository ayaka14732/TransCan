import multiprocessing
import time

obj = [1, 2]

def f():
    time.sleep(2)
    assert obj == [1, 2]

def main():
    global obj

    ctx = multiprocessing.get_context('spawn')

    assert obj == [1, 2]

    process = ctx.Process(target=f)
    process.start()

    obj = [1, 2, 3]
    assert obj == [1, 2, 3]

    process.join()

if __name__ == '__main__':
    main()
