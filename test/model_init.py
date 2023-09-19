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
from dataclasses import dataclass
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

# TODO: named_buffers?
def get_policy_weight_map(lm_model: PreTrainedModel, policy: Policy):
    """{module_name: device}"""
    assert lm_model.device == torch.device('meta')

    def get_layers_dict(lm_model: Module, prefix: str='') -> dict:
        # {layer_name : layer_module ('meta')}
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

    
    def get_choice(cur_percent, percents, choices):
        percents = np.cumsum(percents)
        assert np.abs(percents[-1] - 1.0) < 1e-5

        for i in range(len(percents)):
            if cur_percent < percents[i]:
                return choices[i]
        return choices[-1]
    weight_assign_dict = {}

    choices = ['cuda', 'cpu', 'disk']
    percents_target = [policy.weights_gpu_percent, policy.weights_cpu_percent, policy.weights_disk_percent]
    percents_target = np.array(percents_target)
    
    size_total = sum([p.numel() for _, p in model.named_parameters()])
    size_past, size_future = 0, size_total
    percents_past, percents_future = 0 * percents_target, percents_target  

    for layer_name, layer_module in layers_dict.items():
        # current layer
        param_sizes = [np.prod(para.shape) for _, para in layer_module.named_parameters()]
        param_sizes_cumsum = np.cumsum(param_sizes)
        size_layer = param_sizes_cumsum[-1]

        size_layer_devices = {device: 0 for device in choices}
        for i, (param_name, param) in enumerate(layer_module.named_parameters()):
            param_mid = (param_sizes_cumsum[i] - param_sizes[i] / 2) / param_sizes_cumsum[-1]
            device = get_choice(param_mid, percents_future, choices)

            weight_assign_dict[layer_name+'.'+param_name] = {
                'shape':  param.shape,
                'assigned_device': device
            }
            size_layer_devices[device] += param_sizes[i]

        percents_layer = np.array([size_layer_devices[device] * 1. for device in choices]) / size_layer
        
        # update past & future
        percents_past = (percents_past * size_past + percents_layer * size_layer) / (size_past + size_layer)      
        size_past += param_sizes_cumsum[-1]
        size_future -= param_sizes_cumsum[-1]
        percents_future = (size_total * percents_target - size_past * percents_past) / size_future if size_future > 0 else 0

    return weight_assign_dict

weight_map = get_policy_weight_map(model, policy)

mem_g = sum([np.prod(v['shape']) for k, v in weight_map.items() if 'cuda' in v['assigned_device']]) * 2 / (2 ** 30)
mem_c = sum([np.prod(v['shape']) for k, v in weight_map.items() if v['assigned_device'] == 'cpu']) * 2 / (2 ** 30)
mem_d = sum([np.prod(v['shape']) for k, v in weight_map.items() if v['assigned_device'] == 'disk']) * 2 / (2 ** 30)
logging.info(f'Loading weights of CausalLM: {checkpoint}, GPU Mem: {mem_g:.2f} GiB, CPU Mem: {mem_c:.2f} GiB, Disk Mem: {mem_d:.2f} Gib')

device_map = {k:v['assigned_device'] for k, v in weight_map.items()}
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, device_map=device_map, offload_folder="offload", torch_dtype=torch.float16, offload_state_dict=True
)

logging.info(f'Model initialized!')