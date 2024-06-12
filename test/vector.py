import torch
import numpy as np
from numpy.lib.format import open_memmap

# Dict of NumPy dtype -> torch dtype (when the correspondence exists)
numpy_to_torch_dtype_dict = {
    # np.bool       : torch.bool,
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}


class Vector:
    _max_len = None

    @classmethod
    def set_disk_vector_max_len(cls, max_len: int):
        # if we use disk vector, we want a fixed vector length 
        # to avoid copying between memmap file storages
        cls._max_len = max_len

    def __init__(self, 
        data_shape: list[int], 
        dtype: torch.dtype | np.dtype, 
        device: torch.device | str | int, 
        dim: int, 
        cap: int | None = None,
        **kwargs
    ):
        self.data_shape = data_shape # mutable
        self.dtype = dtype
        self.device = device 

        # push and pop dim
        if 0 <= dim <= len(self.data_shape) - 1:
            self.dim = dim
        elif -len(self.data_shape) <= dim <= -1:
            self.dim = len(self.data_shape) + dim
        else:
            raise ValueError('dim error')

        # capacity of storage
        if device == 'disk' and cap is None:
            assert self._max_len is not None, \
                'try to call Vector.set_disk_vector_max_len(max_len) in advance.'
            self.cap = self._max_len
        elif cap is not None:
            self.cap = cap
        else: 
            # default: 1.5 x data_length
            self.cap = data_shape[dim] * 3 // 2
        
        # init storage
        self.storage_shape = [s if d != self.dim else self.cap for d, s in enumerate(self.data_shape)] # mutable

        if self.device != 'disk':
            # cpu | gpu
            self.pin_memory = self.device in ['cpu', torch.device('cpu')]

            self.storage = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory)
        else:
            # disk
            self.file_name = kwargs.get("file_name")
            self.mmap = open_memmap(self.file_name, shape=tuple(self.storage_shape), dtype=self.dtype, mode='w+')

            self.storage = torch.from_numpy(self.mmap)
        
    @property 
    def rear(self):
        return self.data_shape[self.dim]  
    
    def length(self):
        return self.data_shape[self.dim]  
    
    def empty(self):
        return self.rear == 0
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, dim: int, device=None, **kwargs):
        device = device if device is not None else tensor.device # default to tensor.dtype
        dtype = tensor.dtype if device != 'disk' else torch_to_numpy_dtype_dict[tensor.dtype] # torch.dtype | np.dtype
        
        vec = cls(data_shape=list(tensor.shape), dtype=dtype, device=device, dim=dim, **kwargs)
        indices = [slice(0, s) for s in tensor.shape]
        vec.storage[*indices].copy_(tensor[*indices]) 
        
        if device == 'disk':
            vec.mmap.flush()

        return vec

    def move_to_device(self, device, **kwargs):
        tmp = self.from_tensor(self.data, self.dim, device, **kwargs)
        self.__dict__ = tmp.__dict__
        return self

    @property
    def shape(self):
        return tuple(self.data_shape)
    
    def size(self):
        return tuple(self.data_shape)

    def storage_size(self):
        return tuple(self.storage.shape)
    
    @property
    def data(self):
        data_slices = [slice(0, s) for s in self.data_shape]
        return self.storage[*data_slices]

    def check_copyable(self, x: torch.Tensor):
        assert len(x.shape) == len(self.storage_shape), "dimension number mismatch"

        for d, (x, s) in enumerate(zip(x.shape, self.storage_shape)):
            if d != self.dim and x != s:
                return False 
        return True

    def increase_storage(self, push_len):
        # change storage_shape, reallocate & copy storage
        self.cap = (self.rear + push_len) * 3 // 2
        self.storage_shape[self.dim] = self.cap
        tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory) 
        data_indices = [slice(0, s) for s in self.data_shape]
        tmp[*data_indices].copy_(self.storage[*data_indices])
        self.storage = tmp

    def shrink_storage(self):
        # change storage_shape, reallocate & copy storage(data)
        self.cap = self.rear * 3 // 2
        self.storage_shape[self.dim] = self.cap
        tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory) 
        data_indices = [slice(0, s) for s in self.data_shape]
        tmp[*data_indices].copy_(self.storage[*data_indices])
        self.storage = tmp

    def push_back(self, x: torch.Tensor):
        # change data_shape (& self.rear simultaneously), copy x to storage, optional change storage_shape
        assert self.check_copyable(x)

        push_len = x.shape[self.dim]
        if self.rear + push_len > self.cap:
            if self.device == 'disk':
                raise RuntimeError("disk vector oom")
            self.increase_storage(push_len)
            
        push_slice = [slice(None) for _ in range(len(self.storage_shape))]
        push_slice[self.dim] = slice(self.rear, self.rear + push_len)
        self.storage[push_slice].copy_(x)
        # print(f"{push_slice, self.storage[push_slice].shape, x.shape = }\n")

        self.data_shape[self.dim] += push_len 

    def pop_back(self, pop_len: int = 1, return_popped_vector=True):
        # rear -= pop_len (by modifying self.data_shape)
        pop_slice = [slice(None) for _ in range(len(self.storage_shape))]
        assert self.rear - pop_len >= 0
        pop_slice[self.dim] = slice(self.rear - pop_len, self.rear)
        
        if return_popped_vector:
            ret = self.from_tensor(self.storage[pop_slice], dim=self.dim)  
        else:
            ret = None 

        self.data_shape[self.dim] -= pop_len  

        if self.rear < self.cap // 2:
            if self.device != 'disk':
                self.shrink_storage() 

        return ret

    def __repr__(self) -> str:
        return f"{self.data.tolist()}"
