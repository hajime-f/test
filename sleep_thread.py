import time
import threading

def worker(num):
    print(time.time(), num)

def schedule(interval, f):
    base_time = time.time()
    next_time = 0
    while True:
        t = threading.Thread(target=f, args=(1,))
        t.start()
        next_time = ((base_time - time.time()) % interval) or interval
        time.sleep(next_time)

if __name__ == '__main__':

    schedule(1, worker)


