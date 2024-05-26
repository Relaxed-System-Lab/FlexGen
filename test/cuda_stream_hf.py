import torch
import torch.nn as nn
from transformers import OPTForCausalLM

import numpy as np
from numpy.lib.format import open_memmap

from threading import Thread
from queue import Queue 

import functools 

device = torch.device(0)
device2 = torch.device(1)

# project name: 
class XEngine:
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

def find_module_list(module: nn.Module):
    def _find_module_list(module: nn.Module):
        if isinstance(module, nn.ModuleList):
            yield module
        else:
            for child in module.children():
                yield from _find_module_list(child)
    
    g = _find_module_list(module)
    try:
        return next(iter(g))
    except:
        raise ValueError(f'{module.__class__.__name__} does not have a nn.ModuleList structure')


class ToXModel:
    def __init__(self, hf_model, ) -> None:
        self.hf_model = hf_model 

        self.layers = self.get_layers()

    def get_layers(self):
        if isinstance(self.hf_model, (OPTForCausalLM, )):
            return find_module_list(self.hf_model)
        else:
            raise NotImplementedError()
    
    def override_layer_forward(self, i: int):
        layer = self.layers[i]
        old_forward = layer.forward

        @functools.wraps(old_forward)
        def new_forward(*args, **kwargs):
            print(f'{args = }, {kwargs = }')
            return old_forward(*args, **kwargs)
        
        layer.forward = new_forward
        return layer

    def to_x_layers(self):
        for i, layer in enumerate(self.layers):
            self.override_layer_forward(i)

