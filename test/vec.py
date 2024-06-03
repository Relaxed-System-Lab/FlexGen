import torch 

class Vector:
    def __init__(self, 
        data_shape: tuple[int], 
        dtype: torch.dtype, 
        device: torch.device | str | int, 
        s_dim: int, 
        cap: int | None = None 
    ):
        
        self.data_shape = data_shape
        self.dtype = dtype
        self.device = device 
        self.s_dim = s_dim
        self.cap = cap if cap else data_shape[s_dim] * 2

        self.rear = data_shape[s_dim]  # x[:rear] or x[:len(valid_data)]
        
        _storage_shape = list(data_shape)
        _storage_shape[s_dim] = self.cap
        _storage_shape = tuple(_storage_shape)
        self.storage_shape = _storage_shape
        self.storage = torch.zeros(self.storage_shape, dtype=dtype, device=device)
        

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, s_dim: int):
        return cls(tensor.shape, tensor.dtype, tensor.device, s_dim)

    @property
    def shape(self):
        return self.data_shape
    
    def size(self):
        return self.data_shape 

    def check_broadcastable(self, x: torch.Tensor):
        x_shape = list(x.shape)
        storage_shape = list(self.storage_shape)
        assert len(x_shape) == len(storage_shape)

        for dim, (x, s) in enumerate(zip(x_shape, storage_shape)):
            if dim != self.s_dim:
                if x != s and x != 1:
                    return False 
        return True

    def push_back(self, x: torch.Tensor):
        assert self.check_broadcastable(x)

        push_len = x.shape[self.s_dim]
        if self.rear + push_len > self.cap:
            self.cap *= 2
            self.storage_shape[self.s_dim] = self.cap
            tmp = torch.zeros(self.storage_shape, dtype=self.dtype, device=self.device) 
            tmp.copy_(self.storage)
            self.storage = tmp
            
        storage_slice = [None for _ in range(len(self.storage_shape))]
        storage_slice[self.s_dim] = slice(self.rear, self.rear + push_len)
        self.storage[*storage_slice].copy_(x)

        self.rear += push_len 

    def pop_back(self):
        storage_slice = [None for _ in range(len(self.storage_shape))]
        storage_slice[self.s_dim] = slice(self.rear - 1, self.rear)
        ret = self.storage[*storage_slice]
        self.rear -= 1 
        return ret

    @property
    def attn_mask(self):
        # (b, s, src_len, tgt_len)
        ...

    def log_sum_exp(self, q: torch.Tensor):
        # q: (b, n, s_q, h_d)
        ... 