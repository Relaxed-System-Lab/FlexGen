import torch
import torch.nn as nn

import numpy as np
from numpy.lib.format import open_memmap
from threading import Thread
from queue import Queue 

device = torch.device(0)

b, s, h = 64, 2048, 4096 # bs >> h
x1 = torch.rand(size=(b,s,h), pin_memory=True)
x2 = torch.rand(size=(b,s,h), pin_memory=True)
x3 = torch.rand(size=(b,s,h), pin_memory=True)
x4 = torch.rand(size=(b,s//6,h), pin_memory=True)
w1 = torch.rand(size=(h,h), pin_memory=True)
w2 = torch.rand(size=(h,h), pin_memory=True)
w3 = torch.rand(size=(h,h), pin_memory=True)
w4 = torch.rand(size=(h,h//6), pin_memory=True)
s1 = torch.cuda.Stream(device=device)
s2 = torch.cuda.Stream(device=device)
s3 = torch.cuda.Stream(device=1)

# d2c task queue thread
d2c_task_queue = Queue() 

def d2c_func():
    def process_task(file_name):
        torch.cuda.nvtx.range_push(f'd2c-{file_name}')
        x4_mmap = torch.from_numpy(np.lib.format.open_memmap(file_name))
        x4.copy_(x4_mmap)
        torch.cuda.nvtx.range_pop() 

    while True:
        task = d2c_task_queue.get()  

        if task == 'q':
            d2c_task_queue.task_done()
            break 

        process_task(task)
        d2c_task_queue.task_done()



d2c_thread = Thread(target=d2c_func) 
d2c_thread.start()


iters = 10

import os 

np_x4 = x4.detach().numpy()
np_w4 = w4.detach().numpy()
for i in range(iters):
    open_memmap(f'x4-{i}', mode="w+", shape=np_x4.shape, dtype=np_x4.dtype)
    open_memmap(f'w4-{i}', mode="w+", shape=np_w4.shape, dtype=np_w4.dtype)

x1 = x1.to(device)
x2_gpu = x2.to(device)
x3_gpu = x3.to(1)
w1 = w1.to(device)
w2_gpu = w2.to(device)
w3_gpu = w3.to(1)

# sudo sh -c 'echo 3 >  /proc/sys/vm/drop_caches'
os.system('sudo -S sh -c \'echo 3 >  /proc/sys/vm/drop_caches\'')





def run(iters=iters):

    for i in range(iters):
        torch.cuda.nvtx.range_push('iter{}'.format(i))

        # d2c thread
        # def d2c():
        #     torch.cuda.nvtx.range_push(f'd2c-{i}')
        #     x4_mmap = torch.from_numpy(np.lib.format.open_memmap(f'x4-{i}'))
        #     w4_mmap = torch.from_numpy(np.lib.format.open_memmap(f'w4-{i}'))
        #     x4.copy_(x4_mmap)
        #     w4.copy_(w4_mmap)
        #     # x4.copy_(x4_mmap, non_blocking=True)
        #     # w4.copy_(w4_mmap, non_blocking=True)
        #     torch.cuda.nvtx.range_pop()

        # thread = Thread(target=d2c) 
        # thread.start()
        # d2c()
        d2c_task_queue.put(f'x4-{i}')
        # d2c_task_queue.put(f'w4-{i}')

        with torch.cuda.stream(s1):
            x1 @ w1
            x2_gpu @ w2_gpu
    
        with torch.cuda.stream(s2): 
            x2_gpu.copy_(x2, non_blocking=True)
            w2_gpu.copy_(w2, non_blocking=True)

        with torch.cuda.stream(s3): # another gpu
            x3_gpu.copy_(x3, non_blocking=True)
            w3_gpu.copy_(w3, non_blocking=True)

        torch.cuda.synchronize()
        # thread.join()

        torch.cuda.nvtx.range_pop()
    
    d2c_task_queue.put('q') 
    d2c_thread.join()
    d2c_task_queue.join()
    
        

if __name__=='__main__':
    # warmup
    # run()
    import os 
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    from time import time 
    start = time()
    torch.cuda.cudart().cudaProfilerStart()
    run()
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()
    end = time()
    print(end - start)