import torch
import torch.nn as nn
from transformers import OPTForCausalLM

import numpy as np
from numpy.lib.format import open_memmap

from threading import Thread
from queue import Queue 

from dataclasses import dataclass

### Tasks 
@dataclass
class D2C:
    d_file_name: str
    d_indices: None 
    c_tensor: None 
    c_indices: None 

C2D = D2C 

@dataclass
class G2C:
    g_tensor: None
    g_indices: None
    c_tensor: None
    c_indices: None

C2G = G2C

@dataclass
class G2G:
    src_tensor: None
    src_indices: None
    dst_tensor: None
    dst_indices: None

@dataclass(frozen=True)
class Task:
    C2D = C2D
    D2C = D2C
    G2C = G2C
    C2G = C2G 
    G2G = G2G


### DM Engine
class DataMovementEngine:
    """
    asynchronously copy data between GPU/CPU & CPU/Disk
    1) dst.copy_(src)
    TODO: 2) vector.push & pop
    
    """
    def __init__(self, comp_device=0, single_device=True) -> None:
        
        assert torch.cuda.is_available() 

        self.single_device = single_device
        
        # task streams
        if self.single_device:
            self.comp_device = comp_device
            self.comp_stream = torch.cuda.Stream(self.comp_device)
            self.c2g_stream = torch.cuda.Stream(self.comp_device)
            self.g2c_stream = torch.cuda.Stream(self.comp_device)
            self.g2g_stream = torch.cuda.Stream(self.comp_device)
        else:
            # multi devices
            raise NotImplementedError('only single device is supported, for now')
        
        self.d2c_queue = Queue()
        self.c2d_queue = Queue()
        self.d2c_thread = Thread(target=self.d2c_runtime) 
        self.c2d_thread = Thread(target=self.c2d_runtime) 

    def start(self):
        self.d2c_thread.start()
        self.c2d_thread.start()

    def sync(self) -> None:
        self.d2c_queue.join()
        self.c2d_queue.join()
        self.g2c_stream.synchronize()
        self.c2g_stream.synchronize()

    def submit_c2g_task(self, task: C2G):
        with torch.cuda.stream(self.c2g_stream):
            task.g_tensor[task.g_indices].copy_(task.c_tensor[task.c_indices])

    def submit_g2c_task(self, task: C2G):
        with torch.cuda.stream(self.g2c_stream):
            task.c_tensor[task.c_indices].copy_(task.g_tensor[task.g_indices])

    def submit_g2g_task(self, task: G2G):
        with torch.cuda.stream(self.g2g_stream):
            task.dst_tensor[task.dst_indices].copy_(task.src_tensor[task.src_indices])

    def submit_d2c_task(self, task):
        self.d2c_queue.put(task)

    def submit_c2d_task(self, task):
        self.c2d_queue.put(task)

    def d2c_runtime(self):
        def process_task(task: D2C):
            torch.cuda.nvtx.range_push(f'd2c-{task.d_file_name}')
            torch.cuda.nvtx.range_push(f'1')
            d_tensor = torch.from_numpy(open_memmap(task.d_file_name))
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_push(f'2')
            task.c_tensor[task.c_indices].copy_(d_tensor[task.d_indices])
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_pop() 

        while True:
            task = self.d2c_queue.get()  
            if task is None:
                break 
            process_task(task)
            self.d2c_queue.task_done()

    def c2d_runtime(self):
        def process_task(task: C2D):
            torch.cuda.nvtx.range_push(f'c2d-{task.d_file_name}')

            torch.cuda.nvtx.range_push(f'1')
            np_memmap = np.lib.format.open_memmap(task.d_file_name)
            d_tensor = torch.from_numpy(np_memmap)
            torch.cuda.nvtx.range_pop() 

            torch.cuda.nvtx.range_push(f'2')
            d_tensor[task.d_indices].copy_(task.c_tensor[task.c_indices]) 
            torch.cuda.nvtx.range_pop() 

            torch.cuda.nvtx.range_push(f'3')
            np_memmap.flush() 
            torch.cuda.nvtx.range_pop() 

            torch.cuda.nvtx.range_push(f'4')
            del np_memmap
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_pop() 

        while True:
            task = self.c2d_queue.get()  
            if task is None:
                break
            process_task(task)
            self.c2d_queue.task_done()

    def stop(self):
        self.d2c_queue.put(None)
        self.c2d_queue.put(None)
        self.d2c_queue.join()
        self.c2d_queue.join()
        self.d2c_thread.join()
        self.c2d_thread.join()
    
    def __del__(self):
        self.stop()
