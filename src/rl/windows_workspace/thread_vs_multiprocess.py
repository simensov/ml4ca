
import time
import threading
import multiprocessing

def fibonacci(number):
    arg_prev = 0
    arg_now = 1
    for i in range(number):
        tmparg = arg_now
        arg_now += arg_prev 
        arg_prev = tmparg

    return arg_now

if __name__ == '__main__':
    n1, n2 = 1000000,2000000
    t1 = threading.Thread(target=fibonacci,args=(n1,))
    t2 = threading.Thread(target=fibonacci,args=(n2,))
    p1 = multiprocessing.Process(target=fibonacci,args=(n1,))
    p2 = multiprocessing.Process(target=fibonacci,args=(n2,))

    t = time.time()
    t1.start(); t2.start()
    t1.join(); t2.join()
    print(time.time() - t) # 97 s with n1,n2 = 1000000,2000000

    t = time.time()
    p1.start(); p2.start()
    p1.join(); p2.join()
    print(time.time() - t) # 43 s with n1,n2 = 1000000,2000000