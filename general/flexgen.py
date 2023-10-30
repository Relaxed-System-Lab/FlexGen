# rewrite layer forward function
import os 
import shutil
import functools 

import torch

from minibatch import get_type_size_info, to_compute_device, to_mixed_device, load_kth_batch_inputs, concat_outputs
from model import ModelPolicyLoader 
from utils import logging, get_module_from_name, Policy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FlexGen(ModelPolicyLoader):
    """
    override the forward method for each layer (e.g. embedding layers, transformer blocks, etc.) in a CausalLM.
    example:
        >>> with FlexGen(...) as model:
        >>>     model.generate(...)
    args:
        model: 
            the model object to be override.
        layer_names: 
            ordered layer names.
        
    """
    def __init__(
        self, 
        checkpoint: str, 
        policy: Policy, 
        compute_device = 'cpu', 
        weights_offload_dir = 'weights_offload_dir', 
        args_offload_dir = 'args_offload_dir'
    ):
        super().__init__(
            checkpoint=checkpoint, 
            policy=policy, 
            weights_offload_dir=weights_offload_dir
        )

        self.K = policy.num_gpu_batches # number of minibatches
        self.compute_device = compute_device if torch.cuda.is_available() else 'cpu'
        self.args_offload_dir = args_offload_dir 
        os.makedirs(args_offload_dir, exist_ok=True) 

    def __enter__(self): 
        self.model_to_flexgen()
        return self.model 
            
    def __exit__(self, *exception_infos):
        self.model_reset()
        shutil.rmtree(self.args_offload_dir) 
        os.makedirs(self.args_offload_dir, exist_ok=True) 

    def layer_reset(self, j):
        """
        reset a layer's forward method to its original version.
        """
        layer_name = self.layer_names[j]
        layer = get_module_from_name(self.model, layer_name) 

        if hasattr(layer, "_flexgen_old_forward"):
            layer.forward = layer._flexgen_old_forward
            delattr(layer, "_flexgen_old_forward")
            logger.debug(f'{layer_name} from flexgen to old.')

    def model_reset(self):
        for j, _ in enumerate(self.layer_names):
            self.layer_reset(j) 

    def get_flexgen_forward(self, old_forward, layer_name, next_layer_name):
        """
        override the j-th layer's forward function to FlexGen version.

        pre forward:
            1) load current layer's weights (to compute device)
            2) load next layer's weights (to compute device)
        call forward:
            1) split the input databatch to K minibatches
                a) input databatch: *args and **kwargs
                b) minibatches: lists of *args_k and **kwargs_k
            2) call layer.forward for each minibatch, and for each mini-call:
                a) pre mini-call:
                    * load current minibatch's data (to compute device)
                    * load next minibatch's data (to compute device)
                b) mini-call forward:
                    * output_k = layer.forward(*args_k, **kwargs_k)
                c) post mini-call:
                    * offload current minibatch's output to mixed devices (by policy)
        post forward:
            1) offload current layer's weights to mixed devices (by policy)
        """
        
        @torch.no_grad()
        @functools.wraps(old_forward)
        def flexgen_forward(*args, **kwargs):
            logger.debug(f'args: {get_type_size_info(args)}')
            logger.debug(f'kwargs: {get_type_size_info(kwargs)}')

            # load current and next layers' weights
            self.load_layer_weights(layer_name, self.compute_device) 
            self.load_layer_weights(next_layer_name, self.compute_device) 
            
            outputs = []
            for k in range(self.K):

                # 'pre' fwd: load curr & next inputs (activations, KV cache) to compute device
                args_k = load_kth_batch_inputs(args, k, self.K) # TODO: CUDA stream
                kwargs_k = load_kth_batch_inputs(kwargs, k, self.K) # TODO: CUDA stream 

                # the k-th fwd pass
                output = old_forward(*args_k, **kwargs_k)

                # post fwd: 1) output: to mix, 2) args_k, kwargs_k: free (TODO?)
                output = to_mixed_device(output, self.policy, prefix=f'{self.args_offload_dir}/{layer_name}.batch.{k}.output')
                logger.debug(
                    f'layer: {layer_name}, '
                    f'batch: {k}, '
                    f'args: {get_type_size_info(args_k)}, '
                    f'kwargs: {get_type_size_info(kwargs_k)}, '
                    f'output: {get_type_size_info(output)}'
                )
                outputs.append(output) 
                
            output = concat_outputs(outputs) 

            # for the last layer (e.g. lm_head in OPT), 
            # send its output (e.g. a token's logits) to compute device 
            # to get the generated id (e.g. by a torch.argmax operation)
            if layer_name == self.layer_names[-1]: 
                output = to_compute_device(output)

            logger.debug(f'outputs after concat: {get_type_size_info(output)}')  

            # post fwd: free curr weights
            self.offload_layer_weights(layer_name)

            return output
        
        return flexgen_forward

    def layer_to_flexgen(self, j):
        # get current and next layers' names
        layer_name = self.layer_names[j]
        next_layer_name = self.layer_names[(j + 1) % self.num_layers]

        # get current layer module, and save its old forward 
        layer = get_module_from_name(self.model, layer_name)  
        if hasattr(layer, "_flexgen_old_forward"): return  
        layer._flexgen_old_forward = layer.forward 
        
        # override layer forward  
        layer.forward = self.get_flexgen_forward(
            old_forward=layer.forward, 
            layer_name=layer_name, 
            next_layer_name=next_layer_name
        )
        logger.debug(f'{layer_name} to flexgen forward')

    def model_to_flexgen(self):
        for j, _ in enumerate(self.layer_names):
            self.layer_to_flexgen(j)
    


