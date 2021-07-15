import time
import threading

class Dog:

    def __init__(self, n):
        self.n = n

    def bow(self):
        print('bow-wow', self.n)

        
def dog_bow(num):

    dog = Dog(num)
    
    base_time = time.time()
    next_time = 0
    n_itr = 0

    while True:
        t = threading.Thread(target=dog.bow)
        t.start()
        next_time = ((base_time - time.time()) % num) or num
        time.sleep(next_time)
        
        
if __name__ == '__main__':
    
    for i in range(1, 4):
        t = threading.Thread(target=dog_bow, args=(i,))
        t.start()
        
