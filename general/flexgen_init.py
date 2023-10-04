import os
import numpy as np
import json
from tqdm import tqdm 

import torch 
from torch.nn import Module, ModuleList
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import find_tied_parameters, named_module_tensors, set_module_tensor_to_device

from flexgen_utils import logging, Policy, AttrDict
from flexgen_utils import get_device, get_module_from_name, get_tied_target
from flexgen_utils import flexgen_load_module_tensor, flexgen_offload_module_tensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def policy_init(
    checkpoint: str, 
    policy: Policy, 
    offload_dir: str = 'offload_dir'
) -> AttrDict:
    # return: model, weight_map, layers

    offload_folder = os.path.join(offload_dir, checkpoint.replace('/', '.'))

    # parse model on meta device
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    model.eval()
    logger.debug(f'Got empty CausalLM: \'{checkpoint}\' on meta device.')

    tied_params = find_tied_parameters(model)

    model_info = AttrDict({
        'empty_model': model,
        'checkpoint': checkpoint,
    })
    output = get_policy_weight_map(model_info, policy)

    # load model according to policy
    policy_device_map = output.device_map
    flexgen_layers = output.layers_dict

    # download and process to .dat files
    if not check_disk(checkpoint, offload_folder):
        disk_weight_map = {name:'disk' for name in policy_device_map}
        try:
            AutoModelForCausalLM.from_pretrained(
                checkpoint, 
                device_map=disk_weight_map, 
                offload_folder=offload_folder, 
                offload_state_dict=True
            )
        except:
            pass

    # check the model on disk
    if not check_disk(checkpoint, offload_folder):
        err_msg = 'Mismatch between offload folder and model'
        logger.error(err_msg)
        raise RuntimeError(err_msg)
    logger.info(f'The whole model has been downloaded an processed to offload_folder: \'{offload_folder}\'')

    # policy init
    dat_files = [f for f in os.listdir(offload_folder) if f.endswith('.dat')]
    with open(os.path.join(offload_folder, 'index.json'), 'r') as f:
        index = json.load(f) # {name: {dtype, shape}}

    for tensor_name, device in tqdm(policy_device_map.items(), desc='model init: loading by policy...'):
        if device != 'disk':
            flexgen_load_module_tensor(model, tensor_name, device, index, offload_folder, tied_params) 

    remove_hook_from_module(model, recurse=True) # rm hooks
    logger.info('model has been loaded by policy.')   

    layer_names = list(flexgen_layers.keys())
    return AttrDict({
        'model': model, # no_grad (TODO), eval
        'weight_map': policy_device_map, 
        'layer_names': layer_names,
        'tied_params': tied_params,
        'index': index,
        'offload_folder': offload_folder,
        'dat_files': dat_files,
    })


def check_disk(checkpoint, offload_folder):
    # check if the model has a complete copy on disk.
    if not os.path.isdir(offload_folder):
        return False 
    
    config = AutoConfig.from_pretrained(checkpoint)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()
    tensor_names = [n for n, _ in named_module_tensors(model, include_buffers=True, recurse=True)]
    dat_file_names = [file[:-4] for file in os.listdir(offload_folder) if file.endswith('.dat')]
    # logger.info(set(tensor_names) - set(dat_file_names), set(dat_file_names) - set(tensor_names))
    return len(set(tensor_names) - set(dat_file_names)) == 0


def get_layers_dict(lm_model: Module, prefix: str='') -> dict:
    # return a dict of {layer_name : layer_module ('meta')} of emb / norm layers & transformer layers
    layers_dict = {}
    for name, module in lm_model.named_children():
        # leaf nodes: emb / norm layers
        if len(list(module.named_children())) == 0:
            layers_dict[prefix+name] = module
        # ModuleList: transformer  
        elif isinstance(module, ModuleList):
            for block_name, block_module in module.named_children():
                layers_dict[prefix+name+'.'+block_name] = block_module
        else:
            layers_dict.update(get_layers_dict(module, prefix+name+'.'))
    return layers_dict


def get_policy_weight_map(model_info: AttrDict, policy: Policy):
    """{module_name: device}"""
    model = model_info.empty_model 
    checkpoint = model_info.checkpoint
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
    size_total = sum(np.prod(tensor.shape) for _, tensor in named_module_tensors(model, include_buffers=True, recurse=True))
    size_done, size_todo = 0, size_total
    percents_done, percents_todo = 0 * percents_target, percents_target  

    for layer_name, layer_module in layers_dict.items():
        # current layer
        tensor_sizes = [np.prod(tensor.shape) for _, tensor in named_module_tensors(layer_module, include_buffers=True, recurse=True)]
        tensor_sizes_cumsum = np.cumsum(tensor_sizes)

        device_allo_size_dict = {device: 0 for device in devices} # to balance the percents
        for i, (tensor_name, tensor) in enumerate(named_module_tensors(layer_module, include_buffers=True, recurse=True)):
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
        
        logger.debug(f'{layer_name}, {percents_done}, size_todo: {size_todo}')


    device_map = {k:v['assigned_device'] for k, v in weight_assign_dict.items()}
    logger.info('device_map is prepared!')

    mem_g = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if 'cuda' in v['assigned_device'] and 'shape' in v]) * 2 / (2 ** 30)
    mem_c = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if v['assigned_device'] == 'cpu' and 'shape' in v]) * 2 / (2 ** 30)
    mem_d = sum([np.prod(v['shape']) for _, v in weight_assign_dict.items() if v['assigned_device'] == 'disk' and 'shape' in v]) * 2 / (2 ** 30)
    mem = mem_d + mem_c + mem_g
    logger.info(f'CausalLM {checkpoint} is to be loaded on: ' 
                 f'\nGPU Mem {mem_g:.2f} GiB ({mem_g / mem:.2%}), ' 
                 f'CPU Mem {mem_c:.2f} GiB ({mem_c / mem:.2%}), '
                 f'Disk Mem {mem_d:.2f} Gib ({mem_d / mem:.2%})')
    
    # prepare output
    output = {
        'model': model,
        'tied_params': tied_params,
        'layers_dict': layers_dict,
        'weight_assign_dict': weight_assign_dict,
        'device_map': device_map,
    }
    output = AttrDict(output)
    return output
