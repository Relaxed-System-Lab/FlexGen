from queue import Queue
d2c_task_queue = Queue() 

def d2c_func():
    def process_task(x):
        print('processing:', x)

    while True:
        # if not d2c_task_queue.empty():
        task = d2c_task_queue.get() 
        print('get task:', task) 
        process_task(task)
        d2c_task_queue.task_done()

        if task == 'q':
            break 

from threading import Thread 
d2c_thread = Thread(target=d2c_func) 
d2c_thread.start()

while True:
    x = input()
    d2c_task_queue.put(x)

    if x == 'q':
        break 


d2c_task_queue.join()
d2c_thread.join() 
