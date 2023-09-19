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

import numpy as np
import torch
from torch.nn import Module, ModuleList
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import load_checkpoint_and_dispatch, init_empty_weights

logging.info('Done!')

checkpoint = "facebook/opt-13b"

logging.info(f'Initializing CausalLM: \'{checkpoint}\'')
config = AutoConfig.from_pretrained(checkpoint)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

from dataclasses import dataclass

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

# policy_device_map

def get_policy_weight_map(lm_model: PreTrainedModel, policy: Policy):
    assert lm_model.device == torch.device('meta')

    def get_layers_dict(lm_model: Module, prefix: str='') -> dict:
        layers_dict = {}
        for name, module in lm_model.named_children():
            if len(list(module.named_children())) == 0:
                layers_dict[prefix+name] = module
            # Assume only transformer blocks are stored in ModuleList
            elif isinstance(module, ModuleList):
                for block_name, block_module in module.named_children():
                    layers_dict[prefix+name+'.'+block_name] = block_module
            else:
                layers_dict.update(get_layers_dict(module, prefix+name+'.'))
        return layers_dict
    
    layers_dict = get_layers_dict(lm_model)

    weight_assign_dict = {}
    
    def get_choice(cur_percent, percents, choices):
        percents = np.cumsum(percents)
        assert np.abs(percents[-1] - 1.0) < 1e-5

        for i in range(len(percents)):
            if cur_percent < percents[i]:
                return choices[i]
        return choices[-1]
    
    percents = [policy.weights_gpu_percent, policy.weights_cpu_percent, policy.weights_disk_percent]
    choices = ['cuda', 'cpu', 'disk']
    
    for layer_name, layer_module in layers_dict.items():
        
        sizes = [np.prod(para.shape) for _, para in layer_module.named_parameters()]
        sizes_cumsum = np.cumsum(sizes)
        logging.debug(f"<compute_weight_assignment> block: {layer_name}: sizes: {sizes}, sizes_cumsum: {sizes_cumsum}")
        
        for i, (para_name, para) in enumerate(layer_module.named_parameters()):
            logging.debug(f"<compute_weight_assignment> para: {para_name}: {para.shape}")
            current_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
            weight_assign_dict[layer_name+'.'+para_name] = {'shape':  para.shape,
                'assigned_device': get_choice(current_percent, percents, choices)}

    return weight_assign_dict

weight_map = get_policy_weight_map(model, policy)

mem_c = sum([np.prod(v['shape']) for k, v in weight_map.items() if v['assigned_device'] == 'cpu']) * 2 / (2 ** 30)
mem_d = sum([np.prod(v['shape']) for k, v in weight_map.items() if v['assigned_device'] == 'disk']) * 2 / (2 ** 30)


logging.info(f'Loading weights of CausalLM: {checkpoint}, CPU Mem: {mem_c:.2f} GiB, Disk Mem: {mem_d:.2f} Gib')

device_map = {k:v['assigned_device'] for k, v in weight_map.items()}

model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map=device_map, offload_folder="offload", torch_dtype=torch.float16, offload_state_dict=True
)

logging.info(f'Model initialized!')