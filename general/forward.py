# rewrite layer forward function
import os 
import shutil

import torch
import functools 
import contextlib

from minibatch import get_type_size_info, to_compute_device, to_mixed_device, load_kth_batch_inputs, concat_outputs
from model import ModelPolicyLoader 
from utils import logging, get_module_from_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def reset_forward(model, layer_name):        
    layer = get_module_from_name(model, layer_name) 

    if hasattr(layer, "_flexgen_old_forward"):
        layer.forward = layer._flexgen_old_forward
        delattr(layer, "_flexgen_old_forward")
        logger.debug(f'{layer_name} from flexgen to old.')

    if hasattr(layer, "_test_old_forward"):
        layer.forward = layer._test_old_forward
        delattr(layer, "_test_old_forward")
        logger.debug(f'{layer_name} from test to old.')

def to_flexgen_forward(mpl, j, compute_device, args_offload_dir):
    # rewrite the j-th layer's forward
    layer_name = mpl.layer_names[j]
    next_layer_name = mpl.layer_names[(j + 1) % len(mpl.layer_names)]

    policy = mpl.policy
    ngb = policy.num_gpu_batches

    layer = get_module_from_name(mpl.model, layer_name)  
    if hasattr(layer, "_flexgen_old_forward"): return  
    
    layer._flexgen_old_forward = old_forward = layer.forward 

    @functools.wraps(old_forward)
    def new_forward(*args, **kwargs):
        # pre fwd: load curr & next weights, TODO: cuda stream
        mpl.load_layer_weights(layer_name, compute_device) 
        mpl.load_layer_weights(next_layer_name, compute_device) 
        
        # loop forward pass of K minibatches, TODO: cuda stream
        with torch.no_grad():

            logger.debug(f'args: {get_type_size_info(args)}')
            logger.debug(f'kwargs: {get_type_size_info(kwargs)}')

            # args = to_compute_device(args)
            # kwargs = to_compute_device(kwargs)
            
            outputs = []
            for k in range(ngb):

                # 'pre' fwd: load curr & next inputs (activations, KV cache) to compute device
                args_k = load_kth_batch_inputs(args, k, ngb) # TODO: CUDA stream
                kwargs_k = load_kth_batch_inputs(kwargs, k, ngb) # TODO: CUDA stream 

                # the k-th fwd pass
                output = old_forward(*args_k, **kwargs_k)

                # post fwd: 1) output: to mix, 2) args_k, kwargs_k: free (TODO?)
                output = to_mixed_device(output, policy, prefix=f'{args_offload_dir}/{layer_name}.batch.{k}.output')
                # output = to_compute_device(output)

                logger.debug(f'layer: {layer_name}, '
                             f'batch: {k}, '
                             f'args: {get_type_size_info(args_k)}, '
                             f'kwargs: {get_type_size_info(kwargs_k)}, '
                             f'output: {get_type_size_info(output)}')
                outputs.append(output) 
                # output: (act: mixtensor, (k: mixtensor, v: mixtensor))
                # outputs: K * (0, (1, 2))

            output = concat_outputs(outputs) 
            # output = to_compute_device(output)

            # last layer
            if layer_name == mpl.layer_names[-1]: 
                output = to_compute_device(output)
                
            logger.debug(f'outputs after concat: {get_type_size_info(output)}')  

        # post fwd: free curr weights
        mpl.offload_layer_weights(layer_name)
        return output

    layer.forward = new_forward
    logger.debug(f'{layer_name} to flexgen forward')

@contextlib.contextmanager 
def flexgen(checkpoint, policy, args_offload_dir = 'args_offload_dir'):
    os.makedirs(args_offload_dir, exist_ok=True) 

    # init model 
    from model import ModelPolicyLoader
    mpl = ModelPolicyLoader(checkpoint, policy)

    # rewrite layer forward
    for j, _ in enumerate(mpl.layer_names):
        compute_device = 'cpu'
        to_flexgen_forward(mpl, j, compute_device, args_offload_dir)
    
    try:
        yield mpl.model 
    finally:
        for layer_name in mpl.layer_names:
            reset_forward(mpl.model, layer_name)
        shutil.rmtree(args_offload_dir)
        