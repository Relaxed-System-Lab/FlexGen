# import torch
# import torch.nn as nn
# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, OPTForCausalLM, MistralForCausalLM
# from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
# from accelerate.hooks import remove_hook_from_module
# from accelerate.utils import named_module_tensors, find_tied_parameters
# from accelerate.utils import honor_type
# from typing import Mapping

# from math import ceil 
# import numpy as np
# from numpy.lib.format import open_memmap

# import os
# import sys
# import json
# from copy import deepcopy
# from dataclasses import dataclass

# from threading import Thread
# from queue import Queue 

# import functools 
# checkpoint = 'facebook/opt-125m'
# # checkpoint = 'facebook/opt-13B'
# # checkpoint = 'mistralai/Mistral-7B-v0.1'

# comp_device = 0
# torch_dtype = torch.float16
# weights_offload_dir = f'_weights_offload/{checkpoint}/{torch_dtype}'


# def find_module_list(module: nn.Module) -> tuple[nn.Module, str]:
#     def _find_module_list(module: nn.Module, prefix=''):
#         if isinstance(module, nn.ModuleList):
#             yield module, prefix
#         else:
#             for name, child in module.named_children():
#                 yield from _find_module_list(child, prefix=prefix+'.'+name if prefix else name)
    
#     g = _find_module_list(module)
#     try:
#         return next(iter(g))
#     except:
#         raise ValueError(f'{module.__class__.__name__} does not have a nn.ModuleList structure')

# def get_info(obj, debug=False):
#     if isinstance(obj, (tuple, list)):
#         ret = honor_type(obj, (get_info(o) for o in obj))
#         if len(set(ret)) == 1 and len(ret) > 1:
#             return f"{len(ret)} * {ret[0]}"
#         else:
#             return ret 
#     elif isinstance(obj, Mapping):
#         return type(obj)({k: get_info(v) for k, v in obj.items()})
#     elif isinstance(obj, (torch.Tensor)):
#         if debug:
#             return f"{obj.__class__.__name__}(shape={tuple(obj.size())}, dtype={obj.dtype}, device={obj.device}, mem/elem/dtype={sys.getsizeof(obj.storage()) / obj.numel() / obj.element_size():.3f})"
#         else:
#             return f"{obj.__class__.__name__}(shape={tuple(obj.size())}, mem/elem/dtype={sys.getsizeof(obj.storage()) / obj.numel() / obj.element_size():.3f})"
#     elif isinstance(obj, (int, bool, type(None))):
#         return f"{obj}"
#     else:
#         return f"{obj.__class__.__name__}: {obj}"

# """
# 1. get model parameter & buffer names
# 2. find the transformer block module
# 3. get a device map
# 4. get offloaded weights np.memmap files
# """
# class ModelPrepare:
#     def __init__(self, **kwargs) -> None:
#         self.checkpoint = kwargs.get('checkpoint')
#         self.torch_dtype = kwargs.get('torch_dtype')
#         self.comp_device = kwargs.get('comp_device')
#         self.weights_offload_dir = kwargs.get('weights_offload_dir') 

#         self.empty_model = self.get_empty_model()
#         self.layers, self.layers_name = self.parse_model_architecture()
#         self.device_map = self.get_device_map()
#         self.prepare_weights_memmap()

#         self.model = self.init_model_weights()

#     def get_empty_model(self):
#         self.config = AutoConfig.from_pretrained(checkpoint)
#         with init_empty_weights(): 
#             e = AutoModelForCausalLM.from_config(self.config,)
#         # don't run e.tie_weights() or the tied weights will not be in the device map
#         # e.tie_weights()            
#         return e

#     def parse_model_architecture(self):
#         layers_module, layers_name = find_module_list(self.empty_model)
#         return layers_module, layers_name

#     def get_device_map(self):
#         """
#         give the found transformer block list, set it to the `meta` or `disk` device; 
#         send the device map to AutoModelForCausalLM.from_pretrained() and set the weights_offload_dir, the code from huggingface will automatically prepare the np.memmap files in the offload folder
#         """
#         res = {}
#         for n, t in named_module_tensors(self.empty_model, recurse=True):
#             if isinstance(t, nn.Parameter) and t.dim() > 1 and self.layers_name in n:
#                 res[n] = 'disk'
#             else: # bias/norm/buffer/not transformer block
#                 res[n] = self.comp_device
#         return res

#     def prepare_weights_memmap(self):
#         """init all nn.Parameter in model's transformer blocks to meta device , and others to compute device. (based on the device map)"""
#         # all parameters of the model will be offloaded as memory-mapped array in a given folder.
#         if not os.path.exists(self.weights_offload_dir):
#             try:
#                 AutoModelForCausalLM.from_pretrained(
#                     self.checkpoint, 
#                     device_map={'':'disk'},  
#                     torch_dtype=self.torch_dtype, 
#                     offload_folder=self.weights_offload_dir, 
#                     use_safetensors=False # use pytorch *.bin, as accelerate disk_offload have some bugs for safetensors
#                 )
#             except:
#                 pass 
        
#     def init_model_weights(self):
#         model = AutoModelForCausalLM.from_pretrained(
#             self.checkpoint, 
#             device_map={k:v if v != 'disk' else 'meta' for k, v in self.device_map.items()}, # use 'meta' for no behavior 
#             torch_dtype=self.torch_dtype, 
#             offload_folder=None, 
#             use_safetensors=False 
#         )

#         # remove accelerate disk_offload hooks (if has)
#         model = remove_hook_from_module(model, recurse=True) 
#         return model


# class DiskWeightsLoader:
#     def __init__(self, weights_offload_dir) -> None:
#         self.weights_offload_folder = weights_offload_dir

#         with open(os.path.join(weights_offload_dir, "index.json"), "r") as f: 
#             self.index = json.load(f)  

#     def open_memmap(self, key: str) -> np.memmap:
#         metadata = self.index[key]

#         f_name = os.path.join(weights_offload_dir, key + '.dat')

#         shape = tuple(metadata["shape"])
#         if shape == ():
#             # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
#             shape = (1,)

#         dtype = metadata["dtype"]
#         if dtype == "bfloat16":
#             # NumPy does not support bfloat16 so this was saved as a int16
#             dtype = "int16"

#         weight = np.memmap(f_name, dtype=dtype, shape=shape, mode="r") # no data movement

#         if len(metadata["shape"]) == 0:
#             weight = weight[0]

#         # weight = torch.from_numpy(weight) # no data movement

#         if metadata["dtype"] == "bfloat16":
#             weight = weight.view(torch.bfloat16)

#         return weight
    
# # mp
# mp = ModelPrepare(
#     checkpoint=checkpoint,
#     comp_device=comp_device,
#     torch_dtype=torch_dtype, 
#     weights_offload_dir=weights_offload_dir
# )
# model = mp.model


# # dl
# dl = DiskWeightsLoader(weights_offload_dir)
# mmap = dl.open_memmap(key="model.decoder.layers.0.fc1.bias")
# d_tensor = torch.from_numpy(mmap)

# g_tensor = torch.zeros(*mmap.shape, device = 0, dtype = d_tensor.dtype, pin_memory=False)

# mmap, g_tensor.copy_(d_tensor) # d2g

# class Policy:
#     _comp_device = 0 # 'cuda:0'
    
#     @classmethod
#     def set_comp_device(cls, device):
#         cls._comp_device = device

#     def __init__(self, **kwargs):
#         self.kwargs = kwargs 
#         self.x, self.y, self.z = self.get_vars(['x', 'y', 'z'])
#         self.g, self.c, self.d = self.get_vars(['g', 'c', 'd'])

#     def get_vars(self, vars: list[str]):
#         values = [self.kwargs.get(var) for var in vars]
#         assert all(val is None or 0 <= val <= 1 for val in values) 
#         assert len([val for val in values if val is None]) <= 1 or (1 in values)
#         assert sum([val for val in values if val is not None]) <= 1
        
#         for i, val in enumerate(values):
#             if val is None:
#                 values[i] = 1 - sum([_val for _val in values if _val is not None])
#         return values 
    
#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(x, y, z, g, c, d) = {self.x, self.y, self.z, self.g, self.c, self.d}'
    
#     def get_id_to_gcd_dict(self, num_ids) -> dict:
#         # by policy: self.g, self.c, self.d
#         res = {}

#         cs = np.cumsum([1 / num_ids for _ in range(num_ids)])
#         gc_cut = self.g
#         cd_cut = self.g + self.c
#         for i, x in enumerate(cs):
#             if x <= gc_cut: 
#                 res[i] = self._comp_device # gpu
#             elif gc_cut < x <= cd_cut:
#                 res[i] = 'cpu'
#             elif cd_cut < x <= 1:
#                 res[i] = 'disk' 
        
#         return res 


# policy = Policy(g=0.2, c=0.2, x=1, )
# policy.get_id_to_gcd_dict(10)

# import torch
# import numpy as np
# from numpy.lib.format import open_memmap

# # Dict of NumPy dtype -> torch dtype (when the correspondence exists)
# numpy_to_torch_dtype_dict = {
#     # np.bool       : torch.bool,
#     np.uint8      : torch.uint8,
#     np.int8       : torch.int8,
#     np.int16      : torch.int16,
#     np.int32      : torch.int32,
#     np.int64      : torch.int64,
#     np.float16    : torch.float16,
#     np.float32    : torch.float32,
#     np.float64    : torch.float64,
#     np.complex64  : torch.complex64,
#     np.complex128 : torch.complex128
# }

# # Dict of torch dtype -> NumPy dtype
# torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}


# class Vector:
#     _max_len = None

#     @classmethod
#     def set_disk_vector_max_len(cls, max_len: int):
#         # if we use disk vector, we want a fixed vector length 
#         # to avoid copying between memmap file storages
#         cls._max_len = max_len

#     def __init__(self, 
#         data_shape: list[int], 
#         dtype: torch.dtype | np.dtype, 
#         device: torch.device | str | int, 
#         dim: int, 
#         cap: int | None = None,
#         **kwargs
#     ):
#         self.data_shape = data_shape # mutable
#         self.dtype = dtype
#         self.device = device 

#         # push and pop dim
#         if 0 <= dim <= len(self.data_shape) - 1:
#             self.dim = dim
#         elif -len(self.data_shape) <= dim <= -1:
#             self.dim = len(self.data_shape) + dim
#         else:
#             raise ValueError('dim error')

#         # capacity of storage
#         if device == 'disk' and cap is None:
#             assert self._max_len is not None, \
#                 'try to call Vector.set_disk_vector_max_len(max_len) in advance.'
#             self.cap = self._max_len
#         elif cap is not None:
#             self.cap = cap
#         else: 
#             # default: 1.5 x data_length
#             self.cap = data_shape[dim] * 3 // 2
        
#         # init storage
#         self.storage_shape = [s if d != self.dim else self.cap for d, s in enumerate(self.data_shape)] # mutable

#         if self.device != 'disk':
#             # cpu | gpu
#             self.pin_memory = self.device in ['cpu', torch.device('cpu')]

#             self.storage = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory)
#         else:
#             # disk
#             self.file_name = kwargs.get("file_name")
#             self.mmap = open_memmap(self.file_name, shape=tuple(self.storage_shape), dtype=self.dtype, mode='w+')

#             self.storage = torch.from_numpy(self.mmap)

#         # w infos
#         self.chunk_id = kwargs.get('chunk_id')
#         self.w_name = kwargs.get('w_name')
        
#     @property 
#     def rear(self):
#         return self.data_shape[self.dim]  
    
#     def length(self):
#         return self.data_shape[self.dim]  
    
#     def empty(self):
#         return self.rear == 0
    
#     @classmethod
#     def from_tensor(cls, tensor: torch.Tensor, dim: int, device=None, **kwargs):
#         device = device if device is not None else tensor.device # default to tensor.device
#         dtype = tensor.dtype if device != 'disk' else torch_to_numpy_dtype_dict[tensor.dtype] # torch.dtype | np.dtype
        
#         vec = cls(data_shape=list(tensor.shape), dtype=dtype, device=device, dim=dim, **kwargs)
#         indices = [slice(0, s) for s in tensor.shape]
#         vec.storage[*indices].copy_(tensor[*indices]) 
        
#         if device == 'disk':
#             vec.mmap.flush()

#         return vec

#     def move_to_device(self, device, **kwargs):
#         tmp = self.from_tensor(self.data, self.dim, device, **kwargs)
#         self.__dict__ = tmp.__dict__
#         return self

#     @property
#     def shape(self):
#         return tuple(self.data_shape)
    
#     def size(self):
#         return tuple(self.data_shape)

#     def storage_size(self):
#         return tuple(self.storage.shape)
    
#     @property
#     def data(self):
#         data_slices = [slice(0, s) for s in self.data_shape]
#         return self.storage[*data_slices]

#     def check_copyable(self, x: torch.Tensor):
#         assert len(x.shape) == len(self.storage_shape), "dimension number mismatch"

#         for d, (x, s) in enumerate(zip(x.shape, self.storage_shape)):
#             if d != self.dim and x != s:
#                 return False 
#         return True

#     def assign(self, x):
#         assert self.check_copyable(x)
        

#     def increase_storage(self, push_len):
#         # change storage_shape, reallocate & copy storage
#         self.cap = min((self.rear + push_len) * 3 // 2, self._max_len)
#         self.storage_shape[self.dim] = self.cap
#         tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory) 
#         data_indices = [slice(0, s) for s in self.data_shape]
#         tmp[*data_indices].copy_(self.storage[*data_indices])
#         self.storage = tmp

#     def shrink_storage(self):
#         # change storage_shape, reallocate & copy storage(data)
#         self.cap = self.rear * 3 // 2
#         self.storage_shape[self.dim] = self.cap
#         tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory) 
#         data_indices = [slice(0, s) for s in self.data_shape]
#         tmp[*data_indices].copy_(self.storage[*data_indices])
#         self.storage = tmp

#     def push_back(self, x: torch.Tensor):
#         # change data_shape (& self.rear simultaneously), copy x to storage, optional change storage_shape
#         assert self.check_copyable(x)

#         push_len = x.shape[self.dim]
#         if self.rear + push_len > self.cap:
#             if self.device == 'disk':
#                 raise RuntimeError("disk vector oom")
#             self.increase_storage(push_len)
            
#         push_slice = [slice(None) for _ in range(len(self.storage_shape))]
#         push_slice[self.dim] = slice(self.rear, self.rear + push_len)
#         self.storage[push_slice].copy_(x)
#         # print(f"{push_slice, self.storage[push_slice].shape, x.shape = }\n")

#         self.data_shape[self.dim] += push_len 

#     def pop_back(self, pop_len: int = 1, return_popped_vector=True):
#         # rear -= pop_len (by modifying self.data_shape)
#         pop_slice = [slice(None) for _ in range(len(self.storage_shape))]
#         assert self.rear - pop_len >= 0
#         pop_slice[self.dim] = slice(self.rear - pop_len, self.rear)
        
#         if return_popped_vector:
#             ret = self.from_tensor(self.storage[pop_slice], dim=self.dim)  
#         else:
#             ret = None 

#         self.data_shape[self.dim] -= pop_len  

#         if self.rear < self.cap // 2:
#             if self.device != 'disk':
#                 self.shrink_storage() 

#         return ret

#     def can_do_pop_and_push(self, vec_to_push):
#         if self.dim != vec_to_push.dim:
#             return False
#         return self.check_copyable(vec_to_push.storage)

#     def pop_and_push(self, pop_len: int, vec_to_push):
#         assert self.can_do_pop_and_push(vec_to_push)
#         pop_slice = [slice(None) for _ in range(len(self.storage_shape))]
#         assert self.rear - pop_len >= 0
#         pop_slice[self.dim] = slice(self.rear - pop_len, self.rear)
#         pop_data = self.storage[pop_slice]

#         vec_to_push.push_back(pop_data) # push
#         self.pop_back(pop_len, return_popped_vector=False) # pop 

#     def push_to_other_vec(self, vec_to_push):
#         assert self.can_do_pop_and_push(vec_to_push)
#         push_data = self.data
#         vec_to_push.push_back(push_data) 

#     def __repr__(self) -> str:
#         return f"Vector(data={self.data.shape}, device={self.device})"




# ### Tasks 
# @dataclass
# class D2C:
#     d_file_name: str
#     d_indices: None 
#     c_tensor: None 
#     c_indices: None 

# C2D = D2C 

# @dataclass
# class G2C:
#     g_tensor: None
#     g_indices: None
#     c_tensor: None
#     c_indices: None

# C2G = G2C

# @dataclass
# class G2G:
#     src_tensor: None
#     src_indices: None
#     dst_tensor: None
#     dst_indices: None

# @dataclass(frozen=True)
# class Task:
#     C2D = C2D
#     D2C = D2C
#     G2C = G2C
#     C2G = C2G 
#     G2G = G2G


# ### DM Engine
# class DataMovementEngine:
#     """
#     asynchronously copy data between GPU/CPU & CPU/Disk
#     1) dst.copy_(src)
#     TODO: 2) vector.push & pop
    
#     """
#     def __init__(self, **kwargs) -> None:
        
#         assert torch.cuda.is_available() 

#         self.single_device = kwargs.get('single_device', True)
        
#         # task streams
#         if self.single_device:
#             self.comp_device = kwargs.get('comp_device', 0)
#             self.comp_stream = torch.cuda.Stream(self.comp_device)
#             self.c2g_stream = torch.cuda.Stream(self.comp_device)
#             self.g2c_stream = torch.cuda.Stream(self.comp_device)
#             self.g2g_stream = torch.cuda.Stream(self.comp_device)
#         else:
#             # multi devices
#             raise NotImplementedError('only single device is supported, for now')
        
#         self.d2c_queue = Queue()
#         self.c2d_queue = Queue()
#         self.d2c_thread = Thread(target=self.d2c_runtime) 
#         self.c2d_thread = Thread(target=self.c2d_runtime) 

#     def start(self):
#         self.d2c_thread.start()
#         self.c2d_thread.start()

#     def sync(self) -> None:
#         self.d2c_queue.join()
#         self.c2d_queue.join()
#         self.g2c_stream.synchronize()
#         self.c2g_stream.synchronize()

#     def submit_c2g_task(self, task: C2G):
#         with torch.cuda.stream(self.c2g_stream):
#             task.g_tensor[task.g_indices].copy_(task.c_tensor[task.c_indices])

#     def submit_g2c_task(self, task: C2G):
#         with torch.cuda.stream(self.g2c_stream):
#             task.c_tensor[task.c_indices].copy_(task.g_tensor[task.g_indices])

#     def submit_g2g_task(self, task: G2G):
#         with torch.cuda.stream(self.g2g_stream):
#             task.dst_tensor[task.dst_indices].copy_(task.src_tensor[task.src_indices])

#     def submit_d2c_task(self, task):
#         self.d2c_queue.put(task)

#     def submit_c2d_task(self, task):
#         self.c2d_queue.put(task)

#     def d2c_runtime(self):
#         def process_task(task: D2C):
#             torch.cuda.nvtx.range_push(f'd2c-{task.d_file_name}')
#             torch.cuda.nvtx.range_push(f'1')
#             d_tensor = torch.from_numpy(open_memmap(task.d_file_name))
#             torch.cuda.nvtx.range_pop() 
#             torch.cuda.nvtx.range_push(f'2')
#             task.c_tensor[task.c_indices].copy_(d_tensor[task.d_indices])
#             torch.cuda.nvtx.range_pop() 
#             torch.cuda.nvtx.range_pop() 

#         while True:
#             task = self.d2c_queue.get()  
#             if task is None:
#                 break 
#             process_task(task)
#             self.d2c_queue.task_done()

#     def c2d_runtime(self):
#         def process_task(task: C2D):
#             torch.cuda.nvtx.range_push(f'c2d-{task.d_file_name}')

#             torch.cuda.nvtx.range_push(f'1')
#             np_memmap = np.lib.format.open_memmap(task.d_file_name)
#             d_tensor = torch.from_numpy(np_memmap)
#             torch.cuda.nvtx.range_pop() 

#             torch.cuda.nvtx.range_push(f'2')
#             d_tensor[task.d_indices].copy_(task.c_tensor[task.c_indices]) 
#             torch.cuda.nvtx.range_pop() 

#             torch.cuda.nvtx.range_push(f'3')
#             np_memmap.flush() 
#             torch.cuda.nvtx.range_pop() 

#             torch.cuda.nvtx.range_push(f'4')
#             del np_memmap
#             torch.cuda.nvtx.range_pop() 
#             torch.cuda.nvtx.range_pop() 

#         while True:
#             task = self.c2d_queue.get()  
#             if task is None:
#                 break
#             process_task(task)
#             self.c2d_queue.task_done()

#     def stop(self):
#         self.d2c_queue.put(None)
#         self.c2d_queue.put(None)
#         self.d2c_queue.join()
#         self.c2d_queue.join()
#         self.d2c_thread.join()
#         self.c2d_thread.join()
    
#     def __del__(self):
#         self.stop()


# class Buffer:
#     def __init__(self, buff_device: dict[str, nn.Parameter | torch.Tensor], buff_host, loaded=False):
#         self.buff_device = buff_device 
#         self.buff_host = buff_host
#         self.loaded = loaded

#     # def __getitem__(self, key):
#     #     return self.buff[key]
    
#     # def __setitem__(self, key, value):
#     #     self.buff[key] = value


# class RunningBuffer:
    
#     def __init__(self, **kwargs) -> None:
#         self.num_minibatches = kwargs.get('num_minibatches')
#         self.comp_device = kwargs.get("comp_device", 0)

#         # policy & hf_model
#         self.policy: Policy = kwargs.get('policy') 
#         self.mp: ModelPrepare = kwargs.get('mp')
#         self.hf_model = mp.model
#         self.config = mp.config 
#         self.h_d = self.config.hidden_size

#         # layers
#         self.layers, self.layers_name = find_module_list(self.hf_model)  
#         self.l = len(self.layers) 

#         # curr layer & next layer
#         self.w_buffer_curr = self.init_w_buffer()
#         self.w_buffer_next = deepcopy(self.w_buffer_curr)

#         # curr batch & next batch
#         self.x_buffer_curr = ...
#         self.x_buffer_next = ...
#         self.y_buffer_curr = ...
#         self.y_buffer_curr = ...
    
#     def w_generator(self):
#         return (
#             (n, t) for n, t in named_module_tensors(self.layers[0], recurse=True) 
#             if isinstance(t, nn.Parameter) and t.dim() > 1
#         )
    
#     def init_w_buffer(self):
#         # for a running layer
#         buff_device = {
#             n: torch.zeros(*t.shape, dtype=t.dtype, device=self.comp_device) 
#             for n, t in self.w_generator()
#         }

#         buff_host = {
#             n: torch.zeros(*t.shape, dtype=t.dtype, device='cpu', pin_memory=True) 
#             for n, t in self.w_generator()
#         }

#         return Buffer(buff_device=buff_device, buff_host=buff_host, loaded=False)
        


# class Home:
#     """ 
#     home of w/x/y (weights & caches) on g/c/d 
    
#     memory view:
#     W: 12 l h^2, X: 2 l b s h_kv, Y: l b s h_d

#     (l, ) *                 (l, b) *
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    + - s - 
#     + -  W(GPU) arrs  - +   + - X(GPU) vecs - - +   + - Y(GPU) vecs - - +    |
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +    h
#     + -  W(CPU) arrs  - +   + - X(CPU) vecs - - +   + - Y(CPU) vecs - - +    |
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +      
#     + -  W(Disk) arrs - +   + - X(Disk) vecs  - +   + - Y(Disk) vecs  - +
#     + - - - - - - - - - +   + - - - - - - - - - +   + - - - - - - - - - +
    
#     where an arr(array) is a vec(vector) with a fixed length in the dim of `s'.
#     More importantly, each arr | vec is with a fixed chunk_size in the dim of `h', 
#     a bunch of arrs | vecs comprises the whole memory view.

#     the gpu arr | vec is based on cuda tensor
#     the cpu arr | vec is based on pinned memory
#     the disk arr | vec is based on np.memmap
#     """
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs 

#         # policy & hf_model
#         self.policy: Policy = kwargs.get('policy') 
#         self.mp: ModelPrepare = kwargs.get('mp')
#         self.hf_model = mp.model
#         self.config = mp.config 
#         self.h_d = self.config.hidden_size

#         # layers
#         self.layers, self.layers_name = find_module_list(self.hf_model)  
#         self.l = len(self.layers) 

#         # weight files
#         self.weights_offload_dir = kwargs.get("weights_offload_dir")
#         self.disk_weights_loader = DiskWeightsLoader(self.weights_offload_dir)

#         # chunks in h dim
#         # self.num_chunks = kwargs.get("num_chunks", 16) # default to 16
#         self.default_num_chunks = kwargs.get("num_chunks", 16) # default to 16 for (h, h) shaped tensors
#         assert self.h_d % self.default_num_chunks == 0
#         self.chunk_size = self.h_d // self.default_num_chunks 
#         self.w_name_to_chunk_dim_dict = self.get_w_name_to_chunk_dim_dict()
#         self.w_name_to_num_chunks_dict = self.get_w_name_to_num_chunks_dict()
        

#         # name: vector(l, *, h // num_chunks) x num_chunks
#         self.w_home: dict[str, list[Vector]] = self.init_w_home() 
#         self.x_home: list[Vector] | None = None # vector(l, 2, b, s_x, h_kv // num_chunks) x num_chunks
#         self.y_home: list[Vector] | None = None # vector(l, b, s_y, h_d // num_chunks) x num_chunks
    
#     def w_generator(self):
#         return (
#             (n, t) for n, t in named_module_tensors(self.layers[0], recurse=True) 
#             if isinstance(t, nn.Parameter) and t.dim() > 1
#         )

#     def get_w_name_to_chunk_dim_dict(self):
#         res = {}

#         w_tensor_list = [(n, t.to('meta')) for n, t in self.w_generator()]

#         def get_chunk_dim(t):
#             # get the longest dim
#             assert t.dim() == 2

#             s = t.shape 
#             if len(set(s)) == 1:
#                 return len(s) - 1 # -1
#             else:
#                 return np.argmax(s)

#         for w_name, t_meta in w_tensor_list:
#             chunk_dim = get_chunk_dim(t_meta)
#             res[w_name] = chunk_dim

#         return res 
    
#     def get_w_name_to_num_chunks_dict(self):
#         res = {}

#         w_tensor_list = [(n, t.to('meta')) for n, t in self.w_generator()]

#         def get_chunk_dim(t):
#             # get the longest dim
#             assert t.dim() == 2
#             s = t.shape 
#             if len(set(s)) == 1:
#                 return len(s) - 1 # -1
#             else:
#                 return np.argmax(s)

#         for w_name, t_meta in w_tensor_list:
#             chunk_dim = get_chunk_dim(t_meta)
#             num_chunks = ceil(t_meta.shape[chunk_dim] / self.chunk_size)
#             res[w_name] = num_chunks
            
#         return res 
    
#     def init_w_home(self):
#         # load w files -> g/c/d
#         # return: {w_name: (l, *w_shape) splitted to g/c/d}
#         self.w_home = {}

#         w_tensor_list = [
#             (n, t.to('meta')) for n, t in named_module_tensors(self.layers[0], recurse=True) 
#             if isinstance(t, nn.Parameter) and t.dim() > 1
#         ]

#         for w_name, t_meta in w_tensor_list:
#             chunk_dim = self.w_name_to_chunk_dim_dict[w_name] 
#             num_chunks = self.w_name_to_num_chunks_dict[w_name]

#             # init self.w_home[w_name] as list of vectors
#             self.w_home[w_name] = [None for _ in range(num_chunks)]
            
#             # data_shape
#             data_shape = list(deepcopy(t_meta.shape))
#             data_shape[chunk_dim] = self.chunk_size
#             data_shape = [self.l] + data_shape 
#             # print(f'{w_name, data_shape, t_meta.shape, chunk_dim = }')

#             # device
#             chunk_id_to_vector_device_dict = self.policy.get_id_to_gcd_dict(num_ids=num_chunks)

#             for chunk_id in range(num_chunks):
#                 # device
#                 device = chunk_id_to_vector_device_dict[chunk_id]

#                 # dtype, file_name
#                 if device == 'disk':
#                     dtype = torch_to_numpy_dtype_dict[t_meta.dtype]
#                     file_name = f'{self.weights_offload_dir}/{w_name}-chunk-{chunk_id}' # of l layers aggregated
#                 else:
#                     dtype = t_meta.dtype 
#                     file_name = None 

#                 # dim, cap; randomly select one, as w vec length never changes
#                 dim = -2 
#                 cap = data_shape[-2]

#                 # init vector storage
#                 self.w_home[w_name][chunk_id] = Vector(
#                     data_shape=data_shape,
#                     device=device,
#                     dtype=dtype,
#                     dim=dim,
#                     cap=cap,
#                     file_name=file_name, 

#                     # w infos
#                     chunk_id=chunk_id,
#                     w_name=w_name,
#                 )

#             # assign real weight data to vector storage
#             for layer_id in range(self.l):
#                 mmap = self.disk_weights_loader.open_memmap(
#                     key=self.layers_name + f'.{layer_id}.' + w_name
#                 ) 
                
#                 # to n chunks, and assign chunked data to the initialized home
#                 for chunk_id in range(num_chunks):
#                     chunk_indices = [slice(None) for _ in range(len(mmap.shape))]
#                     chunk_indices[chunk_dim] = slice(
#                         self.chunk_size * chunk_id, 
#                         min(self.chunk_size * (chunk_id + 1), mmap.shape[chunk_dim])
#                     )

#                     data = mmap[*chunk_indices]  # (*, chunk_size) or (chunk_size, *)

#                     # assign
#                     chunk_vector = self.w_home[w_name][chunk_id] # (l, *, chunk_size) or (l, chunk_size, *)
#                     chunk_vector.data[layer_id].copy_(torch.from_numpy(data)) 
        
#         return self.w_home 

#     def init_x_home(self):
#         ...

#     def init_y_home(self):
#         ...

# class MemoryManagementEngine:
#     def __init__(self, **kwargs) -> None:
#         self.home = Home(**kwargs)
#         self.running_buffer = RunningBuffer(**kwargs)
#         self.dm_engine = DataMovementEngine(**kwargs) 

#         self.w_generator = self.home.w_generator

#     def get_num_vecs_on_devices(self, ):
#         hm = self.home.w_home 

#         num_vecs = len([v for _, vectors in hm.items() for v in vectors])
#         num_cpu_vecs = 0
#         num_disk_vecs = 0
#         for _, vectors in hm.items():
#             for v in vectors:
#                 if v.device == 'disk':
#                     num_disk_vecs += 1
#                 elif v.device in ['cpu', torch.device("cpu")]:
#                     num_cpu_vecs += 1

#         num_gpu_vecs = num_vecs - num_cpu_vecs - num_disk_vecs

#         return num_gpu_vecs, num_cpu_vecs, num_disk_vecs
        
#     def get_vecs_on_device(self, device):
#         if device == 'disk':
#             device = ['disk']
#         elif device in ['cpu', torch.device('cpu')]:
#             device = ['cpu', torch.device('cpu')]
#         else:
#             device = [self.running_buffer.comp_device, torch.device(device), device]

#         hm = self.home.w_home 
#         res = []
#         for _, vectors in hm.items():
#             for v in vectors:
#                 if v.device in device:
#                     res.append(v) 
#         return res 

#     def w_home_to_running_buffer(self, layer_id, buffer: RunningBuffer, minibatch_id):
#         nmb = self.running_buffer.num_minibatches 
#         ngv, ncv, ndv = self.get_num_vecs_on_devices()
#         ngv_per_mb, ncv_per_mb, ndv_per_mb = ceil(ngv / nmb), ceil(ncv / nmb), ceil(ndv / nmb)

#         gvecs = self.get_vecs_on_device(self.running_buffer.comp_device)
#         cvecs = self.get_vecs_on_device('cpu')
#         dvecs = self.get_vecs_on_device('disk')

#         # get vecs (of all layers originally) to copy (g->g & c->g & d->c / d->g) for the minibatch
#         # list[Vector(l, *, chunk_size) or Vector(l, chunk_size, *)]
#         gvecs_mb = gvecs[ngv_per_mb * minibatch_id: min(ngv, ngv_per_mb * (minibatch_id + 1))]
#         cvecs_mb = cvecs[ncv_per_mb * minibatch_id: min(ncv, ncv_per_mb * (minibatch_id + 1))]
#         dvecs_mb = dvecs[ndv_per_mb * minibatch_id: min(ndv, ndv_per_mb * (minibatch_id + 1))]
        
#         # copy layer_id's weights from home (vecs) to buffer (tensors)
#         if not buffer.w_buffer_curr.loaded:
#             # copy to w_buffer_curr.buff
#             buff_to_copy = buffer.w_buffer_curr
#         else:
#             # copy to w_buffer_next.buff
#             buff_to_copy = buffer.w_buffer_next 

#         # curr, next
#         for vec in gvecs_mb:
#             w_name, chunk_id = vec.w_name, vec.chunk_id

#         for vec in cvecs_mb:
#             w_name, chunk_id = vec.w_name, vec.chunk_id

#         for vec in dvecs_mb:
#             w_name, chunk_id = vec.w_name, vec.chunk_id
            
        
        



# # home = Home(policy=policy, mp=mp, weights_offload_dir=weights_offload_dir) 

# num_minibatches = 8
# # buffer = RunningBuffer(policy=policy, mp=mp, num_minibatches=num_minibatches, comp_device=0)

# mme = MemoryManagementEngine(
#     policy=policy, 
#     mp=mp, 
#     weights_offload_dir=weights_offload_dir,
#     num_minibatches=num_minibatches, 
#     comp_device=0
# )

# get_info(mme.home.w_home), get_info(mme.running_buffer.w_buffer_curr.buff)
# home = mme.home.w_home
# # for n, vs in home.items():
# #     print(n, vs)
# #     break
# set([repr(v) for n, vs in home.items() for v in vs])



# def kv_cache_kwarg_name(hf_model):
#     if isinstance(hf_model, OPTForCausalLM | MistralForCausalLM):
#         return 'past_key_value'
#     else:
#         raise NotImplementedError() 

# class Buffer:
#     def __init__(self, buff, loaded_flag=False):
#         self.buff = buff 
#         self.loaded_flag = loaded_flag

#     def __getitem__(self, key):
#         return self.buff[key]
    
#     def __setitem__(self, key, value):
#         self.buff[key] = value

# class Model:
#     def __init__(self, **kwargs) -> None:
#         # ModelPrepare 
#         self.checkpoint = kwargs.get('checkpoint')
#         self.torch_dtype = kwargs.get('torch_dtype')
#         self.comp_device = kwargs.get('comp_device')
#         self.weights_offload_dir = kwargs.get('weights_offload_dir') 
#         self.mp = ModelPrepare(**kwargs)

#         self.device_map = self.mp.device_map 
#         self.hf_model = self.mp.model  
#         self.layers, self.layers_name = find_module_list(self.hf_model)  
#         self.kv_cache_kwarg_name = kv_cache_kwarg_name(self.hf_model)
#         self.weight_keys = self.device_map.keys() 

#         # Data Movement 
#         self.disk_weight_loader = DiskWeightsLoader(self.weights_offload_dir)
#         self.dm_engine = DataMovementEngine(self.comp_device)
#         self.dm_engine.start()

#         # flexgen
#         self.policy = kwargs.get('policy')
#         self.m = kwargs.get('m') # minibatches
#         self.max_gmem = kwargs.get('max_gmem') 
#         self.max_cmem = kwargs.get('max_cmem') 
#         self.max_dmem = kwargs.get('max_dmem') 

#         # w/x/y/z home & layer running buffers
#         self.w_home = ... # g/c/d
#         self.x_home = ... # g/c/d
#         self.y_home = ... # g/c/d

#         # weights
#         self.w_buff_curr = Buffer(
#             {
#                 n: torch.zeros(*t.shape, dtype=t.dtype, device=self.comp_device) 
#                 for n, t in self.w_generator()
#             }, 
#             loaded_flag=False
#         )
#         self.w_buff_next = deepcopy(self.w_buff_curr)

#         # kv cache: 2x (b, s, h_kv)
#         self.x_buff_curr = Buffer(None, False) # Vector
#         self.x_buff_next = Buffer(None, False)

#         # actv: 1x (b, s, h_a)
#         self.y_buff_curr = Buffer(None, False)
#         self.y_buff_next = Buffer(None, False)

    
#     def w_generator(self):
#         return (
#             (n, t) for n, t in named_module_tensors(self.layers[0], recurse=True) 
#             if isinstance(t, nn.Parameter) and t.dim() > 1
#         )

#     def override_layer_forward(self, i: int):
#         layer = self.layers[i]
#         old_forward = layer.forward

#         # reference layer weights to layer running buffer
#         def set_reference(module: nn.Module, name: str, reference: torch.Tensor | nn.Parameter):
#             splits = name.split(".")
#             for s in splits[:-1]:
#                 module = getattr(module, s)
                
#             if isinstance(reference, torch.Tensor):
#                 reference = nn.Parameter(reference)
            
#             setattr(module, splits[-1], reference)    
#             # print(name, getattr(module, splits[-1]).device)

#         for name, _ in self.w_generator():
#             set_reference(layer, name, self.w_buff_curr[name])

#         @functools.wraps(old_forward)
#         def new_forward(*args, **kwargs):
#             print(f'\t{i = }, {get_info(args) = }, \n\t{i = }, {get_info(kwargs) = }')

#             # load 1 / ngb of next layer
#             #   copy from home buffers at g/c/d to layer running buffer

#             if self.kv_cache_kwarg_name not in kwargs:
#                 ## PREFILL PHASE 
#                 # 1. offload kv & actv cache of prev batch
#                 ...
#             else:
#                 ## DECODING PHASE
#                 # 1. load kv & actv caches of next batch 

#                 # 2. offload kv | actv caches of prev batch 

#                 # 3. compute curr batch
#                 ...

#             if isinstance(self.hf_model, (OPTForCausalLM, )):
#                 actv_recomp = args[0] # b,1,h / bzh
#                 kv_cache = kwargs.get('past_key_value') # b,n_kv_heads,s_cache,h_kv    x2
#                 attn_mask = kwargs.get('attention_mask') # b,1,1,s_all  (bsz, 1, tgt_len, src_len)

#             # prepare args, kwargs for hf api's
#             args_for_old = args
#             kwargs_for_old = kwargs

#             # hf execution
#             old_output = old_forward(*args_for_old, **kwargs_for_old) # h'=(b,z,h), kv=(b,n,s_all,h) x2
            
#             # prepare our output from hf's output
#             output = old_output
#             print(f'\t{i = }, {get_info(output) = }\n')

#             # swap: curr buff & next buff (for batch & layer levels)
            
#             return output
        
#         layer.forward = new_forward
#         return layer

#     def override_hf_model_forward(self):
#         old_forward = self.hf_model.forward
#         @functools.wraps(old_forward)
#         def new_forward(*args, **kwargs):
#             print(f'hf_model {get_info(args) = }, \nhf_model {get_info(kwargs) = }\n')

#             # new to hf: args, kwargs
#             args_for_old = args
#             kwargs_for_old = kwargs

#             # hf execution
#             old_output = old_forward(*args_for_old, **kwargs_for_old) 

#             # hf to new: output
#             output = old_output 
#             print(f'hf_model {get_info(output) = }\n')
            
#             return output
        
#         self.hf_model.forward = new_forward
#         return self.hf_model

#     def override_forward_functions(self):
#         for i, _ in enumerate(self.layers):
#             self.override_layer_forward(i)
#         self.override_hf_model_forward()
#         return self.hf_model 


# num_prompts = 16
# prompts = None
# prompt_len = 50
# comp_device = 0
# gen_len = 20

# Vector.set_disk_vector_max_len(prompt_len + gen_len + 10)

# # hf_model= OPTForCausalLM.from_pretrained(checkpoint)
# model = Model(
#     checkpoint=checkpoint,
#     comp_device=comp_device,
#     torch_dtype=torch_dtype, 
#     weights_offload_dir=weights_offload_dir
# ).override_forward_functions()

# # test
# if True:
#     if prompts is None:  # get default prompts
#         prompts = [
#             "for i in range(10): ",
#             "Who are you? Are you conscious?",
#             "Where is Deutschland?",
#             "How is Huawei Mate 60 Pro?",
#         ]
#     prompts = (
#         prompts * (num_prompts // len(prompts))
#         + prompts[: (num_prompts % len(prompts))]
#     )

#     # tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(checkpoint) # , padding_side="left"
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token  # eos padding

#     # inputs
#     inputs = tokenizer(
#         prompts,
#         padding="max_length",
#         max_length=prompt_len,
#         return_tensors="pt",
#         # padding=True,
#     ).to(comp_device)

#     # generate
#     generate_ids = model.generate(
#         inputs.input_ids,
#         max_new_tokens=gen_len,  # max_lengths
        
#         # num_beams=6, 

#         # num_beam_groups=2, 
#         # diversity_penalty=0.1, 
        
#         # do_sample=True, 
#     )

#     # outputs
#     output_texts = tokenizer.batch_decode(
#         generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     print(output_texts)