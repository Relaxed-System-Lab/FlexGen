import torch
import torch.nn as nn
from transformers import OPTForCausalLM

import numpy as np
from numpy.lib.format import open_memmap

from threading import Thread
from queue import Queue 

from dataclasses import dataclass


# project name: 
class DataMovementEngine:
    """asynchronously copy data between GPU/CPU & CPU/Disk"""
    def __init__(self, comp_device, single_device=True, debug_mode=True) -> None:
        
        assert torch.cuda.is_available() 

        self.comp_device = comp_device
        self.single_device = single_device
        self.debug_mode = debug_mode
        
        # task streams
        if self.single_device:
            self.comp_stream = torch.cuda.Stream(self.comp_device)
            self.c2g_stream = torch.cuda.Stream(self.comp_device)
            self.g2c_stream = torch.cuda.Stream(self.comp_device)
        else:
            # multi devices
            raise NotImplementedError('only single device is supported, for now')
        
        self.d2c_queue = Queue()
        self.c2d_queue = Queue()
        self.d2c_thread = Thread(target=self.d2c_runtime) 
        self.c2d_thread = Thread(target=self.c2d_runtime) 
        self.d2c_thread.start()
        self.c2d_thread.start()

    # TODO: submit g2c, c2g tasks

    def submit_d2c_task(self, task):
        self.d2c_queue.put(task)

    def submit_c2d_task(self, task):
        self.c2d_queue.put(task)

    def d2c_runtime(self):
        def process_task(task):
            d_file_name, d_indices, c_tensor, c_indices = task
            torch.cuda.nvtx.range_push(f'd2c-{d_file_name}')
            torch.cuda.nvtx.range_push(f'1')
            d_tensor = torch.from_numpy(open_memmap(d_file_name))
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_push(f'2')
            c_tensor[c_indices].copy_(d_tensor[d_indices])
            torch.cuda.nvtx.range_pop() 
            torch.cuda.nvtx.range_pop() 

        while True:
            task = self.d2c_queue.get()  
            if task is None:
                break 
            process_task(task)
            self.d2c_queue.task_done()

    def c2d_runtime(self):
        def process_task(task):
            c_tensor, c_indices, d_file_name, d_indices = task
            torch.cuda.nvtx.range_push(f'c2d-{d_file_name}')

            torch.cuda.nvtx.range_push(f'1')
            np_memmap = np.lib.format.open_memmap(d_file_name)
            d_tensor = torch.from_numpy(np_memmap)
            torch.cuda.nvtx.range_pop() 

            torch.cuda.nvtx.range_push(f'2')
            d_tensor[d_indices].copy_(c_tensor[c_indices]) 
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

    def close(self):
        self.d2c_queue.put(None)
        self.c2d_queue.put(None)
        self.d2c_queue.join()
        self.c2d_queue.join()
        self.d2c_thread.join()
        self.c2d_thread.join()
    
    def __del__(self):
        self.close()

@dataclass
class TaskD2C:
    d_file_name = None
    d_indices = None 
    c_tensor = None
    c_indices = None

TaskC2D = TaskD2C 

@dataclass
class TaskG2C:
    g_tensor = None
    g_indices = None
    c_tensor = None
    c_indices = None

TaskC2G = TaskG2C