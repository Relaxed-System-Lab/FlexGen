from .utils import get_module_from_name, get_tied_target
from .logging import logging
import os 
import torch
import numpy as np
from accelerate.utils import set_module_tensor_to_device


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def flexgen_load_module_tensor(model, tensor_name, device, index, offload_folder, tied_params):
    tensor = get_module_from_name(model, tensor_name)
    if tensor.device == device:
        return 
    
    # else
    old_tensor_name = tensor_name
    
    dat_files = [f for f in os.listdir(offload_folder) if f.endswith('.dat')]
    tensor_name = get_tied_target(tensor_name, tied_params, dat_files) 
    metadata = index[tensor_name]

    # copied from accelerate.utils.offload
    shape = tuple(metadata["shape"])
    if shape == ():
        # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
        shape = (1,)

    dtype = metadata["dtype"]
    if dtype == "bfloat16":
        # NumPy does not support bfloat16 so this was saved as a int16
        dtype = "int16"
    
    # load .dat file
    save_path = os.path.join(offload_folder, tensor_name + '.dat')

    # to device 
    np_memmap = np.memmap(save_path, dtype=dtype, shape=shape, mode='r') 
    tmp = torch.from_numpy(np_memmap).to(device) 
    set_module_tensor_to_device(model, old_tensor_name, device, tmp)


def flexgen_offload_module_tensor(model, tensor_name, policy_device_map):
    tensor = get_module_from_name(model, tensor_name)
    device = policy_device_map[tensor_name]
    if device == 'disk': device = 'meta'

    if tensor.device != torch.device(device):
        set_module_tensor_to_device(model, tensor_name, device, tensor) # gtoc, ctog
