# rewrite layer forward function

import torch
import functools 
import contextlib

from minibatch import get_size_info, load_kth_batch_inputs, concat_outputs
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

def to_test_forward(mpl, layer_name, call_layer_log):
    layer = get_module_from_name(mpl.model, layer_name) 
    compute_device = 'cpu' 
    layer._test_old_forward = old_forward = layer.forward 

    @functools.wraps(old_forward)
    def new_forward(*args, **kwargs):
        mpl.load_layer_weights(layer_name, compute_device) 

        call_layer_log.append(layer_name)  # 

        with torch.no_grad():
            output = old_forward(*args, **kwargs)

        mpl.offload_layer_weights(layer_name)
        return output

    layer.forward = new_forward
    logger.debug(f'{layer_name} to test forward') 
    
@contextlib.contextmanager
def test(mpl, call_layer_log):
    model = mpl.model
    layer_names = mpl.layer_names

    # test run to get layer calling order
    for layer_name in layer_names:
        to_test_forward(mpl, layer_name, call_layer_log)
    yield 
    for layer_name in layer_names:
        reset_forward(model, layer_name)



def to_flexgen_forward(mpl, j, compute_device):
    # rewrite the j-th layer's forward
    layer_name = mpl.layer_names[j]
    next_layer_name = mpl.layer_names[(j + 1) % len(mpl.layer_names)]

    ngb = mpl.policy.num_gpu_batches

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
            logger.debug(f'args: {get_size_info(args)}')
            logger.debug(f'kwargs: {get_size_info(kwargs)}')
            # output = old_forward(*args, **kwargs)
            # logger.debug(f'output: {get_size_info(output)}')

            args_0 = load_kth_batch_inputs(args, 0, ngb)
            kwargs_0 = load_kth_batch_inputs(kwargs, 0, ngb)
            logger.debug(f'args_0: {get_size_info(args_0)}')
            logger.debug(f'kwargs_0: {get_size_info(kwargs_0)}')
            # output_0 = old_forward(*args_0, **kwargs_0)
            # logger.debug(f'output0: {get_size_info(output_0)}')

            outputs = []
            for k in range(ngb):
                logger.debug(f'layer: {layer_name}, batch: {k}')

                # 'pre' fwd: load curr & next inputs (activations, KV cache), store & offload prev 
                args_k = load_kth_batch_inputs(args, k, ngb)
                kwargs_k = load_kth_batch_inputs(kwargs, k, ngb)

                # TODO: load args, kwargs to compute device

                # the k-th fwd pass
                output = old_forward(*args_k, **kwargs_k)

                # TODO: offload args, kwargs to mixed device
                # TODO: offload output to mixed device

                outputs.append(output) 

            logger.debug(f'outputs before concat: {ngb} x {get_size_info(outputs[0])}')
            output = concat_outputs(outputs)
            logger.debug(f'outputs after concat: {get_size_info(output)}')                

        # post fwd: free curr weights
        mpl.offload_layer_weights(layer_name)
        return output

    layer.forward = new_forward
    logger.debug(f'{layer_name} to flexgen forward')

@contextlib.contextmanager
def flexgen(checkpoint, policy):
    # init model 
    from model import ModelPolicyLoader
    mpl = ModelPolicyLoader(checkpoint, policy)
    mpl.init_all_weights() # init 

    # test run, get layer order
    call_layer_log = []
    with test(mpl, call_layer_log):
        from test import test_hf_gen
        test_hf_gen(mpl.checkpoint, mpl.model, 1,1, prompts=['0'])

    assert len(call_layer_log) == len(mpl.layer_names) and set(call_layer_log) == set(mpl.layer_names)
    mpl.layer_names = call_layer_log

    # rewrite layer forward
    for j, _ in enumerate(mpl.layer_names):
        compute_device = 'cpu'
        to_flexgen_forward(mpl, j, compute_device)
    yield mpl.model
    for layer_name in mpl.layer_names:
        reset_forward(mpl.model, layer_name)
        