from typing import Mapping, Tuple, Iterable
import numpy as np 
import os 
import torch
from math import floor

class MixTensor:
    def __init__(
        self, 
        mix_data: Tuple, 
        split_dim: int, 
        device: torch.device, 
        shape: torch.Size,
        percents: Mapping[str, float],
        file_path: str,
        dtype
    ):
        self.mix_data = mix_data
        self.split_dim = split_dim 
        self.device = device 
        self.shape = shape 
        self.percents = percents
        self.file_path = file_path
        self.dtype = dtype
    
    def size(self):
        return self.shape 
    
    @staticmethod
    def get_split_dim(tensor):
        dim_sizes = tensor.size()
        max_dim, max_size = -1, -1
        for dim, size in enumerate(dim_sizes):
            if size > max_size:
                max_size = size
                max_dim = dim 
        return max_dim 
    
    @staticmethod
    def tensor_dim_slice(tensor, dim, dim_slice):
        return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (dim_slice, )]
    
    @staticmethod
    def split_tensor(tensor, dim, percents):
        dim_size = tensor.size(dim)
        g_per, c_per, _ = [percents[dev] for dev in ['cuda', 'cpu', 'disk']]
        
        g_cut = floor(dim_size * g_per)
        c_cut = floor(dim_size * (g_per + c_per))

        g_data = MixTensor.tensor_dim_slice(tensor, dim, slice(0, g_cut))
        c_data = MixTensor.tensor_dim_slice(tensor, dim, slice(g_cut, c_cut))
        d_data = MixTensor.tensor_dim_slice(tensor, dim, slice(c_cut, dim_size))
        return g_data, c_data, d_data 

    @classmethod
    def from_tensor(
        cls, 
        tensor: torch.Tensor, 
        percents: Mapping[str, float],
        file_path: str 
    ):
        split_dim = cls.get_split_dim(tensor) 
        device = tensor.device 
        shape = tensor.shape
        dtype = tensor.dtype
        
        g_data, c_data, d_data = cls.split_tensor(tensor, split_dim, percents) 
        
        g_data = g_data.to('cuda' if torch.cuda.is_available() else 'cpu') if g_data.numel() else None
        c_data = c_data.to('cpu') if c_data.numel() else None
        if d_data.numel():
            d_data = d_data.cpu().numpy()
            shape = d_data.shape
            np_dtype = d_data.dtype 

            fp = np.memmap(file_path, mode="w+", shape=shape, dtype=np_dtype)
            fp[:] = d_data[:]
            d_data = (shape, np_dtype)
        else:
            d_data = None 
        mix_data = (g_data, c_data, d_data)

        return cls(
            mix_data=mix_data,
            split_dim=split_dim,
            device=device,
            shape=shape,
            percents=percents,
            file_path=file_path,
            dtype=dtype
        )

    @classmethod 
    def from_mixtensor(cls, mix_tensor):
        self = mix_tensor 
        return self 

    def to_tensor(self):
        g_data, c_data, d_data = self.mix_data 
        compute_device = self.device 

        tensor = []
        if g_data is not None:
            if g_data.device != torch.device(compute_device):
                g_data = g_data.to(compute_device) 
            tensor.append(g_data)
        if c_data is not None:
            if c_data.device != torch.device(compute_device):
                c_data = c_data.to(compute_device) 
            tensor.append(c_data)
        if d_data is not None:
            (shape, np_dtype) = d_data 
            d_data = np.memmap(self.file_path, shape=shape, dtype=np_dtype, mode='r')
            d_data = torch.from_numpy(d_data).to(compute_device)
            tensor.append(d_data)
            
        tensor = torch.cat(tensor, dim=self.split_dim) 

        return tensor        

    def __add__(self, mix_tensor):
        assert self.shape == mix_tensor.shape and type(self) == type(mix_tensor) # is same shape mix tensor
        res = self.to_tensor() + mix_tensor.to_tensor() 
        return self.from_tensor(res, self.percents, self.file_path)


class BatchMixTensor:
    def __init__(self, batches: Iterable[MixTensor]):
        self.dtype = batches[0].dtype
        self.device = batches[0].device
        self.batches = batches 

        self.shape = self.size()

    def __getitem__(self, i):
        return self.batches[i]
    
    def __setitem__(self, i, mt: MixTensor):
        self.batches[i] = mt

    def __len__(self):
        return len(self.batches)
    
    def size(self):
        shape = list(self[0].size()) 
        shape[0] *= len(self)
        return torch.Size(shape)

    def __add__(self, bmt):
        for k in range(len(self)): # K batches 
            # TODO flexgen: parallelly load k+1
            self_k = self[k].to_tensor()
            bmt_k = bmt[k].to_tensor()
            res = self_k + bmt_k 
            self[k] = MixTensor.from_tensor(res, self[k].percents, self[k].file_path)
        return self 

    def contiguous(self):
        tensor = []
        for mt in self:
            tensor.append(mt.to_tensor())
        return torch.cat(tensor)



if __name__ == '__main__':
    
    x = torch.tensor([1,2,3])
    m = MixTensor.from_tensor(x, percents={'cuda':0, 'cpu':0.5, 'disk':0.5}, file_path='test/m.dat')
    m2 = MixTensor.from_tensor(x, percents={'cuda':0, 'cpu':0.5, 'disk':0.5}, file_path='test/m2.dat')
    m = m + m2
    print(m.to_tensor())

