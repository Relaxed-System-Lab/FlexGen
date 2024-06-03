import torch 
from copy import deepcopy

class Vector:
    def __init__(self, 
        data_shape: list[int], 
        dtype: torch.dtype, 
        device: torch.device | str | int, 
        dim: int, 
        cap: int | None = None 
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
        if cap is None:
            self.cap = data_shape[dim] * 2
        else:
            self.cap = cap  
        
        # init storage
        self.storage_shape = [s if d != self.dim else self.cap for d, s in enumerate(self.data_shape)] # mutable
        self.storage = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device)
        
    @property 
    def rear(self):
        return self.data_shape[self.dim]  

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, dim: int):
        vec = cls(list(tensor.shape), tensor.dtype, tensor.device, dim)
        indices = [slice(0, s) for s in tensor.shape]
        vec.storage[*indices].copy_(tensor[*indices]) 
        return vec

    @property
    def shape(self):
        return tuple(self.data_shape)
    
    def size(self):
        return tuple(self.data_shape)

    def check_copyable(self, x: torch.Tensor):
        assert len(x.shape) == len(self.storage_shape), "dimension number mismatch"

        for d, (x, s) in enumerate(zip(x.shape, self.storage_shape)):
            if d != self.dim and x != s:
                return False 
        return True

    def double_storage(self):
        # change storage_shape, reallocate & copy storage
        self.cap *= 2
        self.storage_shape[self.dim] = self.cap
        tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device) 
        data_indices = [slice(0, s) for s in self.data_shape]
        tmp[*data_indices].copy_(self.storage[*data_indices])
        self.storage = tmp

    def half_storage(self):
        # change storage_shape, reallocate & copy storage(data)
        self.cap //= 2
        self.storage_shape[self.dim] = self.cap
        tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device) 
        data_indices = [slice(0, s) for s in self.data_shape]
        tmp[*data_indices].copy_(self.storage[*data_indices])
        self.storage = tmp

    def push_back(self, x: torch.Tensor):
        # change data_shape (& self.rear simultaneously), copy x to storage, optional change storage_shape
        assert self.check_copyable(x)

        push_len = x.shape[self.dim]
        if self.rear + push_len > self.cap:
            self.double_storage()
            
        push_slice = [None for _ in range(len(self.storage_shape))]
        push_slice[self.dim] = slice(self.rear, self.rear + push_len)
        self.storage[*push_slice].copy_(x)

        self.data_shape[self.dim] += push_len 

    def pop_back(self):
        # rear -= 1 (by modifying self.data_shape)
        pop_slice = [None for _ in range(len(self.storage_shape))]
        pop_slice[self.dim] = slice(self.rear - 1, self.rear)
        ret = deepcopy(self.storage[*pop_slice])
        self.data_shape[self.dim] -= 1  

        if self.rear < self.cap // 4:
            self.half_storage() 

        return ret

    def __repr__(self) -> str:
        return f"{self.storage}"

if __name__ == '__main__':
    dev = 0

    t = torch.tensor([1,2,3]).to(dev)
    v = Vector.from_tensor(t, -1)
    for i in range(4, 30):
        v.push_back(torch.tensor([i]).to(dev))
        print(len(v.storage), v.shape, v)


    while v.rear > 20:
        v.pop_back()
        print(len(v.storage), v.shape, v)

    for i in range(1, 11):
        v.push_back(torch.tensor([i]).to(dev))
        print(len(v.storage), v.shape, v)


    while v.rear > 0:
        v.pop_back()
        print(len(v.storage), v.shape, v)