# rewrite layer forward function
import os 
import shutil
import functools 

import torch

from model import ModelPolicyLoader 
from block import get_info, to_compute_device, BlockPolicyLoader
from utils import logging, get_module_from_name, Policy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

        # bpl
        self.bpl = BlockPolicyLoader(
            policy=policy,
            args_offload_dir=args_offload_dir
        )
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
            # 1. load current and next layers' weights, offload prev layer's weights (TODO: in parallel)
            self.mpl.offload_layer_weights(prev_layer_name) 
            self.mpl.load_layer_weights(next_layer_name, self.compute_device) 
            self.mpl.load_layer_weights(curr_layer_name, self.compute_device) # curr load -> curr compute
            logger.debug(f'layer: {curr_layer_name}')
            logger.debug(f'args: {get_info(args)}')
            logger.debug(f'kwargs: {get_info(kwargs)}')

            self.bpl.layer_init(
                inputs=(args, kwargs), 
                layer_name=curr_layer_name
            )
            args_k, kwargs_k = self.bpl.get_kth_input(1)
            logger.debug(f'args_k: {get_info(args_k)}') 
            logger.debug(f'kwarg_k: {get_info(kwargs_k)}')
                    
            for k in range(self.K):
                # 2. store prev batch output (act, kv)
                if k > 0:
                    self.bpl.offload_kth_output(k - 1)
                    logger.debug(f'batch: {k - 1}, output: {get_info(self.bpl.get_kth_output(k - 1))}')
                elif k == 0: # corner case
                    self.bpl.offload_kth_input(-1) 
                    logger.debug(f'last layer, batch: {self.K - 1}, as curr layer\'s input: {get_info(self.bpl.get_kth_input(-1))}')
                
                # 3. load curr batch input (act, kv)
                self.bpl.load_kth_input(k) 

                # 4. load next batch input (act, kv)
                assert self.K >= 2 
                if k < self.K - 1:
                    self.bpl.load_kth_input(k + 1) 
                else: # corner case
                    self.bpl.load_kth_output(0) 

                # 5. compute the k-th forward pass
                args_k, kwargs_k = self.bpl.get_kth_input(k)
                output = old_forward(*args_k, **kwargs_k)
                self.bpl.set_kth_output(k, output)

                # (TODO) 6. sync: CUDA, Disk 

                
            output = self.bpl.concat_outputs() 

            # for the last layer (e.g. lm_head in OPT), 
            # send its output (e.g. a token's logits) to compute device 
            # to get the generated id (e.g. by a torch.argmax operation)
            if curr_layer_name == self.layer_names[-1]: 
                output = to_compute_device(output)

            logger.debug(f'outputs after concat: {get_info(output)}\n\n')  

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
    
