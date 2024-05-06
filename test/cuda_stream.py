import torch
import torch.nn as nn


device = torch.device(0)

s1 = torch.cuda.Stream(device=device)
s2 = torch.cuda.Stream(device=device)


x1 = torch.rand(size=(1024*4, 1024*4), pin_memory=True)
x2 = torch.rand(size=(1024*4, 1024*4), pin_memory=True)
w1 = torch.rand(size=(1024*4, 1024*4), pin_memory=True)
w2 = torch.rand(size=(1024*4, 1024*4), pin_memory=True)

x1 = x1.to(device)
x2_gpu = x2.to(device)
w1 = w1.to(device)
w2_gpu = w2.to(device)

def run(iters=10):
    
    for i in range(iters):
        torch.cuda.nvtx.mark('iter{}'.format(i))

        with torch.cuda.stream(s1):
            out1 = x1.matmul(w1)
            out2 = x2_gpu.matmul(w2_gpu)
    
        with torch.cuda.stream(s2): # better than to set as s1
            x2_gpu.copy_(x2)
            w2_gpu.copy_(w2)
            
        

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