import torch
import functools 
import contextlib

from flexgen_utils import logging, get_module_from_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


from flexgen_utils import load_layer_weights, offload_layer_weights
from flexgen_minibatch import get_size_info, get_kth_batch_inputs, concat_outputs
from flexgen_init import policy_init

def to_old_forward(model, layer_names, j):
    layer_name = layer_names[j]
    layer = get_module_from_name(model, layer_name) 

    if hasattr(layer, "_flexgen_old_forward"):
        layer.forward = layer._flexgen_old_forward
        delattr(layer, "_flexgen_old_forward")
        logger.debug(f'{layer_name} from flexgen to old.')

    if hasattr(layer, "_test_old_forward"):
        layer.forward = layer._test_old_forward
        delattr(layer, "_test_old_forward")
        logger.debug(f'{layer_name} from test to old.')
    
    return layer

def to_test_forward(model, layer_names, j, compute_device, weight_map, index, offload_folder, dat_files, tied_params, call_layer_log):
    # rewrite the j-th layer's forward, to get layer order
    layer_name = layer_names[j]
    layer = get_module_from_name(model, layer_name)  
    if hasattr(layer, "_test_old_forward"): # has been rewriten
        return layer 
    
    logger.debug(f'{layer_name} to test forward')
    layer._test_old_forward = old_forward = layer.forward 

    @functools.wraps(old_forward)
    def new_forward(*args, **kwargs):
        # pre fwd: load curr weights
        load_layer_weights(model, layer_name, compute_device, index, offload_folder, dat_files, tied_params)
        call_layer_log.append(layer_name) 
        with torch.no_grad():
            logger.debug(f'{layer_name} forward pass:')
            # logger.debug(f'\targs: {get_size_info(args)}')
            # logger.debug(f'\tkwargs: {get_size_info(kwargs)}')
            output = old_forward(*args, **kwargs)
            # logger.debug(f'\toutput: {get_size_info(output)}')

        # post fwd: free curr weights
        offload_layer_weights(model, layer_name, weight_map)
        return output

    layer.forward = new_forward
    return layer



@contextlib.contextmanager
def test(flexgen_model, call_layer_log):
    model = flexgen_model.model
    weight_map = flexgen_model.weight_map
    layer_names = flexgen_model.layer_names
    layer_nums = len(layer_names)
    index = flexgen_model.index
    dat_files = flexgen_model.dat_files
    tied_params = flexgen_model.tied_params
    offload_folder = flexgen_model.offload_folder

    # to test fwd
    for j in range(layer_nums):
        compute_device = 'cpu'
        to_test_forward(model, layer_names, j, compute_device, weight_map, index, offload_folder, dat_files, tied_params, call_layer_log)
    
    yield 

    # to old fwd
    for j in range(layer_nums):
        to_old_forward(model, layer_names, j)
       

def to_flexgen_forward(model, layer_names, j, compute_device, index, weight_map, offload_folder, ngb, gbs, dat_files, tied_params):
    # rewrite the j-th layer's forward
    
    layer_name = layer_names[j]
    next_layer_name = layer_names[(j + 1) % len(layer_names)]

    layer = get_module_from_name(model, layer_name)  
    if hasattr(layer, "_flexgen_old_forward"): # has been rewriten
        return layer 
    
    logger.debug(f'{layer_name} to flexgen forward')
    layer._flexgen_old_forward = old_forward = layer.forward 

    @functools.wraps(old_forward)
    def new_forward(*args, **kwargs):
        # pre fwd: load curr & next weights, TODO: cuda stream
        load_layer_weights(model, layer_name, compute_device, index, offload_folder, dat_files, tied_params)
        load_layer_weights(model, next_layer_name, compute_device, index, offload_folder, dat_files, tied_params)
        
        # loop forward pass of K minibatches, TODO: cuda stream
        with torch.no_grad():
            logger.debug(f'args: {get_size_info(args)}')
            logger.debug(f'kwargs: {get_size_info(kwargs)}')
            # output = old_forward(*args, **kwargs)
            # logger.debug(f'output: {get_size_info(output)}')

            args_0 = get_kth_batch_inputs(args, 0, ngb)
            kwargs_0 = get_kth_batch_inputs(kwargs, 0, ngb)
            logger.debug(f'args_0: {get_size_info(args_0)}')
            logger.debug(f'kwargs_0: {get_size_info(kwargs_0)}')
            # output_0 = old_forward(*args_0, **kwargs_0)
            # logger.debug(f'output0: {get_size_info(output_0)}')

            outputs = []
            for k in range(ngb):
                logger.debug(f'layer: {layer_name}, batch: {k}')

                # 'pre' fwd: load curr & next inputs (activations, KV cache), store & offload prev 
                args_k = get_kth_batch_inputs(args, k, ngb)
                kwargs_k = get_kth_batch_inputs(kwargs, k, ngb)

                # the k-th fwd pass
                output = old_forward(*args_k, **kwargs_k)
                outputs.append(output) 
                
                # 'post' fwd: offload curr inputs

            logger.debug(f'outputs before concat: {ngb} x {get_size_info(outputs[0])}')
            output = concat_outputs(outputs)
            logger.debug(f'outputs after concat: {get_size_info(output)}')                

        # post fwd: free curr weights
        offload_layer_weights(model, layer_name, weight_map)
        return output

    layer.forward = new_forward
    return layer

@contextlib.contextmanager
def flexgen(checkpoint, policy):
    policy_init_output = policy_init(checkpoint, policy)

    model = policy_init_output.model
    weight_map = policy_init_output.weight_map
    layer_names = policy_init_output.layer_names
    layer_nums = len(layer_names)
    index = policy_init_output.index
    dat_files = policy_init_output.dat_files
    tied_params = policy_init_output.tied_params
    offload_folder = policy_init_output.offload_folder

    gbs = policy.gpu_batch_size
    ngb = policy.num_gpu_batches

    # to test fwd
    for j in range(layer_nums):
        compute_device = 'cpu'
        to_flexgen_forward(model, layer_names, j, compute_device, index, weight_map,  offload_folder, ngb, gbs, dat_files, tied_params)
    
    yield model

    # to old fwd
    for j in range(layer_nums):
        to_old_forward(model, layer_names, j)
        