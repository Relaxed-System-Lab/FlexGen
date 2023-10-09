# split/merge layer input/output data structures to minibatch/batch

import torch 
from accelerate.utils import honor_type
from typing import Mapping
from utils import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_size_info(obj): # recursive
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (get_size_info(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k:get_size_info(v) for k, v in obj.items()})
    elif isinstance(obj, torch.Tensor):
        return obj.size()
    elif isinstance(obj, (int, bool, type(None))): 
        return obj
    else:
        logger.warning(f'inputs: {obj} of type \'{type(obj)}\' is not implemented.')
        return obj

def get_kth_batch_inputs(inputs, k, ngb): # for both args, kwargs
    if isinstance(inputs, (tuple, list)): # e.g. args
        return honor_type(inputs, (get_kth_batch_inputs(inp, k, ngb) for inp in inputs))
    elif isinstance(inputs, Mapping): # e.g. kwargs
        return type(inputs)({key:get_kth_batch_inputs(value, k, ngb) for key, value in inputs.items()})
    elif isinstance(inputs, torch.Tensor):
        mini_size = inputs.size(0) // ngb
        return inputs[k * mini_size:(k + 1) * mini_size]
    elif isinstance(inputs, (int, bool, type(None))): # None, int, bool
        return inputs
    else:
        logger.warning(f'inputs: {inputs} of type \'{type(inputs)}\' is not implemented.')
        return inputs

def concat_outputs(outputs): # concat K outputs to one output
    assert len(outputs), 'empty outputs.'
    assert isinstance(outputs[0], (torch.Tensor, tuple)), f'Only supports layer output type of torch.Tensor or tuple. However, we get a {type(outputs[0])}.'
    
    if isinstance(outputs[0], torch.Tensor):
        return torch.cat(outputs, dim=0)
    elif isinstance(outputs[0], tuple):
        def f(outputs):
            ans = []
            for elem in zip(*outputs):
                if isinstance(elem[0], torch.Tensor):
                    ans.append(torch.cat(elem, dim=0))
                elif isinstance(elem[0], tuple):
                    ans.append(f(elem))
                elif isinstance(elem[0], (int, bool, type(None))): 
                    ans.append(elem[0])
                else:
                    logger.warning(f'outputs: {elem[0]} of type \'{type(elem[0])}\' is not implemented.')
                    ans.append(elem[0])
            return tuple(ans)

        return f(outputs)
            
