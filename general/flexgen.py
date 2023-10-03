import logging
logging.basicConfig(
    style='{',
    format='{asctime} [{filename}:{lineno} in {funcName}] {levelname} - {message}',
    handlers=[
        logging.FileHandler(".log", 'w'),
        logging.StreamHandler()
    ],
    level=logging.INFO
)
logging.info('Importing...')
import os
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import Module, ModuleList
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from accelerate.utils import find_tied_parameters
logging.info('Done!')

checkpoint = "facebook/opt-13b" # 1.3b 6.7b 13b 30b 66b 

logging.info(f'Initializing CausalLM: \'{checkpoint}\'')
config = AutoConfig.from_pretrained(checkpoint)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
    
# model.base_model_prefix # -> 'model'

model.tie_weights()
tied_params = find_tied_parameters(model)
# tied_params

class AttrDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

@dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent of weights/cache/activations on GPU/CPU/Disk %
    weights_gpu_percent: float
    weights_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    @property
    def weights_disk_percent(self):
        return 1.0 - self.weights_gpu_percent - self.weights_cpu_percent

    @property
    def cache_disk_percent(self):
        return 1.0 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 1.0 - self.act_gpu_percent - self.act_cpu_percent

policy = Policy(
    gpu_batch_size=8, 
    num_gpu_batches=8, 
    weights_gpu_percent=0.0, 
    weights_cpu_percent=0.3, 
    cache_gpu_percent=0.0, 
    cache_cpu_percent=0.2, 
    act_gpu_percent=0.0, 
    act_cpu_percent=0.5, 
    overlap=True, 
    pin_weight=True,
)

def get_layers_dict(lm_model: Module, prefix: str='') -> dict:
    # return a dict of {layer_name : layer_module ('meta')} with only leaf nodes & transformer layers
    layers_dict = {}
    for name, module in lm_model.named_children():
        # leaf nodes
        if len(list(module.named_children())) == 0:
            layers_dict[prefix+name] = module
        # ModuleList: transformer  
        elif isinstance(module, ModuleList):
            for block_name, block_module in module.named_children():
                layers_dict[prefix+name+'.'+block_name] = block_module
        else:
            layers_dict.update(get_layers_dict(module, prefix+name+'.'))
    return layers_dict

def named_module_tensors(module: Module, include_buffers: bool = True, recurse: bool = True):
    for named_parameter in module.named_parameters(recurse=recurse):
        yield named_parameter

    if include_buffers:
        for named_buffer in module.named_buffers(recurse=recurse):
            yield named_buffer

def get_device(cur_percent, percents, choices):
    # choose a device (gpu / cpu / disk) for a weight tensor by its percent of size
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 1.0) < 1e-5, f'{percents}'

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]

def get_policy_weight_map(model: PreTrainedModel, policy: Policy):
    """{module_name: device}"""
    assert model.device == torch.device('meta'), 'model is not on device meta.'
    
    # to ensure the tied params are allocated to the same device in the weight_map
    model.tie_weights()
    tied_params = find_tied_parameters(model)

    # layers to be scheduled
    layers_dict = get_layers_dict(model)

    # device assignment for each tensor in the model
    weight_assign_dict = {}
    devices = ['cuda', 'cpu', 'disk']
    percents_target = np.array([
        policy.weights_gpu_percent, 
        policy.weights_cpu_percent, 
        policy.weights_disk_percent
    ])
    
    # model size (parameters + buffers), here we do not repeatly sum the tied paramters 
    size_total = sum(np.prod(tensor.shape) for _, tensor in named_module_tensors(model))
    size_done, size_todo = 0, size_total
    percents_done, percents_todo = 0 * percents_target, percents_target  

    for layer_name, layer_module in layers_dict.items():
        # current layer
        tensor_sizes = [np.prod(tensor.shape) for _, tensor in named_module_tensors(layer_module)]
        tensor_sizes_cumsum = np.cumsum(tensor_sizes)

        device_allo_size_dict = {device: 0 for device in devices} # to balance the percents
        for i, (tensor_name, tensor) in enumerate(named_module_tensors(layer_module)):
            abs_tensor_name = layer_name + '.' + tensor_name

            def find_processed_tied(abs_tensor_name, tied_params, weight_assign_dict):
                # find the processed parameter (in weight_assign_dict) of the tied parameters.
                for tp in tied_params:
                    if abs_tensor_name in tp:
                        for p in tp:
                            if p in weight_assign_dict:
                                return p, tuple(tp)
                return None
            
            processed_tied = find_processed_tied(abs_tensor_name, tied_params, weight_assign_dict) 
            if processed_tied: # this tensor is tied and processed.
                p, tp = processed_tied
                weight_assign_dict[abs_tensor_name] = {
                    # 'shape':  tensor.shape,
                    'assigned_device': weight_assign_dict[p]['assigned_device'],
                    'tied': tp
                }
            else:
                mid_percent = (tensor_sizes_cumsum[i] - tensor_sizes[i] / 2) / tensor_sizes_cumsum[-1] # tensor mid size percent 
                device = get_device(mid_percent, percents_todo, devices)
                weight_assign_dict[abs_tensor_name] = {
                    'shape':  tensor.shape,
                    'assigned_device': device
                }
                
                device_allo_size_dict[device] += tensor_sizes[i]

        # update percents_todo
        size_layer = sum(device_allo_size_dict.values())
        if size_layer > 0:
            device_allo_percents = np.array([device_allo_size_dict[device] * 1. for device in devices]) / size_layer
            percents_done = (percents_done * size_done + device_allo_percents * size_layer) / (size_done + size_layer)      
        size_done += size_layer
        size_todo -= size_layer
        if size_todo > 0:
            percents_todo = (size_total * percents_target - size_done * percents_done) / size_todo 
        
        logging.info(f'{layer_name}, {percents_done}, size_todo: {size_todo}')


    device_map = {k:v['assigned_device'] for k, v in weight_assign_dict.items()}
    logging.info('device_map is prepared!')

    mem_g = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if 'cuda' in v['assigned_device'] and 'shape' in v]) * 2 / (2 ** 30)
    mem_c = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if v['assigned_device'] == 'cpu' and 'shape' in v]) * 2 / (2 ** 30)
    mem_d = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if v['assigned_device'] == 'disk' and 'shape' in v]) * 2 / (2 ** 30)
    mem = mem_d + mem_c + mem_g
    logging.info(f'CausalLM {checkpoint} is to be loaded on: ' 
                 f'\nGPU Mem {mem_g:.2f} GiB ({mem_g / mem:.2%}), ' 
                 f'CPU Mem {mem_c:.2f} GiB ({mem_c / mem:.2%}), '
                 f'Disk Mem {mem_d:.2f} Gib ({mem_d / mem:.2%})')
    
    # prepare output
    output = {
        'model': model,
        'tied_params': tied_params,
        'layers_dict': layers_dict,
        'weight_assign_dict': weight_assign_dict,
        'device_map': device_map
    }
    output = AttrDict(output)
    return output

output = get_policy_weight_map(model, policy)

device_map = output.device_map
offload_folder = 'offload/' + checkpoint.replace('/', '.')

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    device_map=device_map, 
    offload_folder=offload_folder, 
    offload_state_dict=True
)

logging.info(f'Model initialized!')
