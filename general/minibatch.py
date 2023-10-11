# split/merge layer input/output data structures to minibatch/batch

import torch 
from accelerate.utils import honor_type
from typing import Mapping
from utils import logging 
from mixtensor import MixTensor, BatchMixTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_type_size_info(obj): # recursive
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (get_type_size_info(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k:get_type_size_info(v) for k, v in obj.items()})
    
    elif isinstance(obj, (torch.Tensor, MixTensor, BatchMixTensor)):
        return f'{type(obj)}: {obj.size()}'

    elif isinstance(obj, (int, bool, type(None))): 
        return f'{type(obj)}: {obj}'
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return f'{type(obj)}: {obj}'


def to_mixed_device(obj, policy, prefix): 
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], torch.Tensor) and isinstance(obj[1], torch.Tensor): # KV cache
        m0 = MixTensor.from_tensor(
            obj[0], 
            percents={
                'cuda':policy.cache_gpu_percent, 
                'cpu':policy.cache_cpu_percent, 
                'disk':policy.cache_disk_percent, 
            }, 
            file_path=f'{prefix}_key.dat'
        )
        m1 = MixTensor.from_tensor(
            obj[1], 
            percents={
                'cuda':policy.cache_gpu_percent, 
                'cpu':policy.cache_cpu_percent, 
                'disk':policy.cache_disk_percent, 
            }, 
            file_path=f'{prefix}_value.dat'
        )
        return (m0, m1)
    elif isinstance(obj, torch.Tensor):
        return MixTensor.from_tensor(
            obj, percents={
                'cuda':policy.act_gpu_percent, 
                'cpu':policy.act_cpu_percent, 
                'disk':policy.act_disk_percent, 
            }, 
            file_path=f'{prefix}.dat'
        )
    elif isinstance(obj, tuple):
        return honor_type(obj, (to_mixed_device(o, policy, f'{prefix}[{i}]') for i, o in enumerate(obj)))
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return obj


def concat_outputs(outputs): # concatenate K outputs to one output
    assert len(outputs), 'empty outputs.'
    assert isinstance(outputs[0], (MixTensor, torch.Tensor, tuple)), f'not supported type: {type(outputs[0])}.'
    
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    elif isinstance(outputs[0], MixTensor):
        return BatchMixTensor(outputs)
    elif isinstance(outputs[0], tuple):
        def f(outputs):
            ans = []
            for elem in zip(*outputs):
                if isinstance(elem[0], torch.Tensor):
                    ans.append(torch.cat(elem, dim=0))
                elif isinstance(elem[0], MixTensor):
                    ans.append(BatchMixTensor(elem))
                elif isinstance(elem[0], tuple):
                    ans.append(f(elem))
                else:
                    logger.warning(f'outputs: {elem[0]} of type \'{type(elem[0])}\' is not implemented.')
                    ans.append(elem[0])
            return tuple(ans)

        return f(outputs)


def load_kth_batch_inputs(inputs, k, ngb): # for both args, kwargs, with a nested structure of tuple/list/dict/Tensor
    if isinstance(inputs, (tuple, list)): # e.g. args
        return honor_type(inputs, (load_kth_batch_inputs(inp, k, ngb) for inp in inputs))
    elif isinstance(inputs, Mapping): # e.g. kwargs
        return type(inputs)({key:load_kth_batch_inputs(value, k, ngb) for key, value in inputs.items()})
    elif isinstance(inputs, torch.Tensor):
        mini_size = inputs.size(0) // ngb
        return inputs[k * mini_size:(k + 1) * mini_size]
    elif isinstance(inputs, MixTensor):
        inputs = inputs.to_tensor()
        mini_size = inputs.size(0) // ngb
        return inputs[k * mini_size:(k + 1) * mini_size]
    elif isinstance(inputs, BatchMixTensor):
        mini_batch = inputs.batches[k]
        return mini_batch.to_tensor()
    elif isinstance(inputs, (int, bool, type(None))): 
        return inputs
    else:
        logger.warning(f'inputs: {inputs} of type \'{type(inputs)}\' is not implemented.')
        return inputs

