# split/merge layer input/output data structures to minibatch/batch

import torch 
from accelerate.utils import honor_type
from typing import Mapping
from utils import logging, Policy 
from tensor import MixTensor, BlockTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_info(obj): 
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (get_info(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k:get_info(v) for k, v in obj.items()})
    elif isinstance(obj, (torch.Tensor, MixTensor, BlockTensor)):
        return f'<{obj.__class__.__name__}>: {tuple(obj.size())}, {obj.dtype}'
    elif isinstance(obj, (int, bool, type(None))): 
        return f'{obj}'
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return f'<{obj.__class__.__name__}>: {obj}'

def to_compute_device(obj): 
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (to_compute_device(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k:to_compute_device(v) for k, v in obj.items()})
    elif isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, (MixTensor, BlockTensor)):
        return obj.to_tensor()
    elif isinstance(obj, (int, bool, type(None))): 
        return obj
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return obj

def to_mixed_device(obj, policy, prefix): 
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], torch.Tensor) and isinstance(obj[1], torch.Tensor): 
        # KV cache
        m0 = MixTensor.from_tensor(
            obj[0], 
            percents={
                'cuda':policy.cache_gpu_percent, 
                'cpu':policy.cache_cpu_percent, 
                'disk':policy.cache_disk_percent, 
            }, 
            file_path=f'{prefix}.key.dat'
        )
        m1 = MixTensor.from_tensor(
            obj[1], 
            percents={
                'cuda':policy.cache_gpu_percent, 
                'cpu':policy.cache_cpu_percent, 
                'disk':policy.cache_disk_percent, 
            }, 
            file_path=f'{prefix}.value.dat'
        )
        return (m0, m1)
    elif isinstance(obj, torch.Tensor):# and obj.dtype != torch.bool:
        # activations
        return MixTensor.from_tensor(
            obj, 
            percents={
                'cuda':policy.act_gpu_percent, 
                'cpu':policy.act_cpu_percent, 
                'disk':policy.act_disk_percent, 
            }, 
            file_path=f'{prefix}.dat'
        )
    elif isinstance(obj, tuple):
        return tuple(to_mixed_device(o, policy, f'{prefix}.{i}') for i, o in enumerate(obj))
    elif isinstance(obj, Mapping):
        return type(obj)({key:to_mixed_device(value, policy, f'{prefix}.{key}') for key, value in obj.items()}) 
    elif isinstance(obj, (int, bool, type(None))): 
        return obj
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return obj

def concat_outputs(outputs): # concatenate K outputs to one output
    assert len(outputs), 'empty outputs.'
    assert isinstance(outputs[0], (MixTensor, torch.Tensor, tuple)), f'not supported type: {type(outputs[0])}.'
    
    if isinstance(outputs[0], torch.Tensor | MixTensor):
    #     return torch.cat(outputs, dim=0)
    # elif isinstance(outputs[0], MixTensor):
        return BlockTensor(outputs)
    elif isinstance(outputs[0], tuple):
        def f(outputs):
            ans = []
            for elem in zip(*outputs):
                if isinstance(elem[0], torch.Tensor | MixTensor):
                #     ans.append(torch.cat(elem, dim=0))
                # elif isinstance(elem[0], MixTensor):
                    ans.append(BlockTensor(elem))
                elif isinstance(elem[0], tuple):
                    ans.append(f(elem))
                # else:
                #     logger.warning(f'outputs: {elem[0]} of type \'{type(elem[0])}\' is not implemented.')
                #     ans.append(elem[0])
            return tuple(ans)

        return f(outputs)


def get_kth_batch_inputs(inputs, k, ngb): 
    """ 
    for inputs with a nested structure of tuple/list/dict/Tensor/BatchMixTensor
    """
    if isinstance(inputs, (tuple, list)): # e.g. args
        return honor_type(inputs, (get_kth_batch_inputs(inp, k, ngb) for inp in inputs))
    elif isinstance(inputs, Mapping): # e.g. kwargs
        return type(inputs)({key:get_kth_batch_inputs(value, k, ngb) for key, value in inputs.items()})
    elif isinstance(inputs, torch.Tensor):
        mini_size = inputs.size(0) // ngb
        return inputs[k * mini_size:(k + 1) * mini_size]
    elif isinstance(inputs, BlockTensor):
        mini_batch = inputs.batches[k]
        return mini_batch#.to_tensor()
    elif isinstance(inputs, (int, bool, type(None))): 
        return inputs
    else:
        logger.warning(f'inputs: {inputs} of type \'{type(inputs)}\' is not implemented.')
        return inputs

class BlockPolicyLoader:
    """
    block refers to: a layer's input/output data block
    """
    def __init__(
        self, 
        policy: Policy,
        args_offload_dir = 'args_offload_dir'
    ):
        self.policy = policy 
        self.args_offload_dir = args_offload_dir 
        self.K = policy.num_gpu_batches

    def layer_init(self, inputs, layer_name):
        self.inputs = inputs 
        self.layer_name = layer_name
        self.input_batches = [get_kth_batch_inputs(self.inputs, k, self.K) for k in range(self.K)]
        self.output_batches = [None for _ in range(self.K)]

    # for input
    def get_kth_input(self, k):
        return self.input_batches[k]
    
    def load_kth_input(self, k):
        self.input_batches[k] = to_compute_device(self.input_batches[k])

    def offload_kth_input(self, k):
        self.input_batches[k] = to_mixed_device(
            self.input_batches[k],
            self.policy, 
            prefix=f'{self.args_offload_dir}/{self.layer_name}.batch.{k}.input'
        )

    # for output
    def set_kth_output(self, k, output):
        self.output_batches[k] = output
    
    def get_kth_output(self, k):
        return self.output_batches[k]
    
    def load_kth_output(self, k):
        self.output_batches[k] = to_compute_device(self.output_batches[k])

    def offload_kth_output(self, k):
        self.output_batches[k] = to_mixed_device(
            self.output_batches[k], 
            self.policy, 
            prefix=f'{self.args_offload_dir}/{self.layer_name}.batch.{k}.output'
        )

    def concat_outputs(self):
        return concat_outputs(self.output_batches) 
