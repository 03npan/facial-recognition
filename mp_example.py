from multiprocessing import Process, Queue
import time


def pull_from_queue(q, frame_length):
    target_time = time.monotonic()
    while True:
        target_time += frame_length
        item = q.get()
        if item is None:
            break
        print(item)
        end = time.monotonic()
        time.sleep(max(0, target_time - time.monotonic()))

def add_to_queue(q, frame_length):
    while True:
        start = time.monotonic()
        for i in range(5):
            q.put(i)
        end = time.monotonic()
        time.sleep(frame_length * 5 - (end - start))


if __name__ == '__main__':
    q = Queue()
    frame_length = 0.2
    pull_proc = Process(target=pull_from_queue, args=(q, frame_length))
    pull_proc.start()
    add_proc = Process(target=add_to_queue, args=(q, frame_length))
    add_proc.start()

    pull_proc.join()
    add_proc.join()

