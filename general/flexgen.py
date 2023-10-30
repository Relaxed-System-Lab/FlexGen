# rewrite layer forward function
import os 
import shutil
import functools 

import torch

from minibatch import get_info, to_compute_device, to_mixed_device, get_kth_batch_inputs, concat_outputs
from model import ModelPolicyLoader 
from utils import logging, get_module_from_name, Policy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class BlockPolicyLoader:
    def __init__(self, block): 
        self.block = block 

    


    

class FlexGen:
    """
    override the forward method for each layer (e.g. embedding layers, transformer blocks, etc.) of a CausalLM.
    example:
        >>> with FlexGen(checkpoint, policy) as model:
        >>>     model.generate(...)
    """
    def __init__(
        self, 
        checkpoint: str, 
        policy: Policy, 
        compute_device = 'cpu', 
        weights_offload_dir = 'weights_offload_dir', 
        args_offload_dir = 'args_offload_dir'
    ):
        self.checkpoint = checkpoint
        self.policy = policy
        self.weights_offload_dir = weights_offload_dir
        self.compute_device = compute_device if torch.cuda.is_available() else 'cpu'
        self.args_offload_dir = args_offload_dir 
        os.makedirs(args_offload_dir, exist_ok=True) # in args offloader

        # mpl 
        mpl = ModelPolicyLoader(
            checkpoint=checkpoint, 
            policy=policy, 
            weights_offload_dir=weights_offload_dir
        )
        self.model = mpl.model
        self.layer_names = mpl.layer_names 
        self.num_layers = mpl.num_layers
        self.mpl = mpl

        self.K = policy.num_gpu_batches # number of minibatches

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

    def get_flexgen_forward(
        self, 
        old_forward, 
        prev_layer_name, 
        curr_layer_name, 
        next_layer_name
    ):
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
                    * free current minibatch's data 
                    * offload current minibatch's output (to mixed devices by policy)
        post forward:
            1) offload current layer's weights (to mixed devices by policy)
        """
        
        @torch.no_grad()
        @functools.wraps(old_forward)
        def flexgen_forward(*args, **kwargs):
            # load current and next layers' weights, offload prev layer's weights (TODO: in parallel)
            self.mpl.offload_layer_weights(prev_layer_name) 
            self.mpl.load_layer_weights(curr_layer_name, self.compute_device) 
            self.mpl.load_layer_weights(next_layer_name, self.compute_device) 

            # nested tuple/list/dict/Tensor/BatchMixTensor
            logger.debug(f'args: {get_info(args)}')
            logger.debug(f'kwargs: {get_info(kwargs)}')
            
            outputs = []
            for k in range(self.K):
                # 'pre' fwd: load curr & next batchs' inputs (activations, KV cache) to compute device
                #            store (offload) prev batch's outputs
                args_k, kwargs_k = get_kth_batch_inputs((args, kwargs), k, self.K) # TODO: CUDA stream
                logger.debug(f'layer: {curr_layer_name}, batch: {k}')
                logger.debug(f'args_k: {get_info(args_k)}, kwargs_k: {get_info(kwargs_k)}')

                args_k, kwargs_k = to_compute_device((args_k, kwargs_k))

                # 'pre' fwd: load next inputs (activations, KV cache) to compute device
                if k < self.K - 1:
                    args_kp1, kwargs_kp1 = get_kth_batch_inputs((args, kwargs), k + 1, self.K) # TODO: CUDA stream
                else: 
                    pass 
                    # outputs[0] = to_compute_device(outputs[0])

                # the k-th fwd pass
                output = old_forward(*args_k, **kwargs_k)

                # 'post' fwd: 1) output: to mix, 2) args_k, kwargs_k: offload / free (TODO)
                output = to_mixed_device(output, self.policy, prefix=f'{self.args_offload_dir}/{curr_layer_name}.batch.{k}.output')
                logger.debug(f'output: {get_info(output)}')

                outputs.append(output) 

                # sync: CUDA, Disk
                
            output = concat_outputs(outputs) 

            # for the last layer (e.g. lm_head in OPT), 
            # send its output (e.g. a token's logits) to compute device 
            # to get the generated id (e.g. by a torch.argmax operation)
            if curr_layer_name == self.layer_names[-1]: 
                output = to_compute_device(output)

            logger.debug(f'outputs after concat: {get_info(output)}')  

            # post fwd: free curr weights
            # self.mpl.offload_layer_weights(curr_layer_name)

            return output
        
        return flexgen_forward

    def layer_to_flexgen(self, j):
        # get prev, curr and next layers' names
        prev_layer_name = self.layer_names[(j - 1) % self.num_layers]
        curr_layer_name = self.layer_names[j]
        next_layer_name = self.layer_names[(j + 1) % self.num_layers]

        # get current layer module, and save its old forward 
        layer = get_module_from_name(self.model, curr_layer_name)  
        if hasattr(layer, "_flexgen_old_forward"): return  
        layer._flexgen_old_forward = layer.forward 
        
        # override layer forward  
        layer.forward = self.get_flexgen_forward(
            old_forward=layer.forward, 
            prev_layer_name=prev_layer_name,
            curr_layer_name=curr_layer_name, 
            next_layer_name=next_layer_name
        )
        logger.debug(f'{curr_layer_name} to flexgen forward')

    def model_to_flexgen(self):
        for j, _ in enumerate(self.layer_names):
            self.layer_to_flexgen(j)
    
