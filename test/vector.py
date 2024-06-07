import torch 

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
            self.cap = data_shape[dim] * 3 // 2
        else:
            self.cap = cap  
        
        # init storage
        self.storage_shape = [s if d != self.dim else self.cap for d, s in enumerate(self.data_shape)] # mutable
        self.pin_memory = self.device in ['cpu', torch.device('cpu')]
        self.storage = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device, pin_memory=self.pin_memory)
        
    @property 
    def rear(self):
        return self.data_shape[self.dim]  
    
    def empty(self):
        return self.rear == 0
    
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
            self.shrink_storage() 

        return ret

    def can_do_pop_and_push(self, vec_to_push):
        if self.dim != vec_to_push.dim:
            return False
        return self.check_copyable(vec_to_push.storage)

    def pop_and_push(self, pop_len: int, vec_to_push):
        assert self.can_do_pop_and_push(vec_to_push)
        pop_slice = [slice(None) for _ in range(len(self.storage_shape))]
        assert self.rear - pop_len >= 0
        pop_slice[self.dim] = slice(self.rear - pop_len, self.rear)
        pop_data = self.storage[pop_slice]

        vec_to_push.push_back(pop_data) # pop and push

        self.data_shape[self.dim] -= pop_len  
        if self.rear < self.cap // 2:
            self.shrink_storage() 

    def push_to_other_vec(self, vec_to_push):
        assert self.can_do_pop_and_push(vec_to_push)
        push_data = self.data
        vec_to_push.push_back(push_data) 

    def __repr__(self) -> str:
        return f"{self.data.tolist()}"
