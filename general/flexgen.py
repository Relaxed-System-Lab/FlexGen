# rewrite layer forward function
import os
import shutil
import functools

import torch

from loaders.model import ModelPolicyLoader
from loaders.block import BlockPolicyLoader
from utils import logging, Policy, get_module_from_name, get_info, to_compute_device

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
        compute_device="cpu",
        weights_offload_dir="weights_offload_dir",
        args_offload_dir="args_offload_dir",
    ):
        self.checkpoint = checkpoint
        self.policy = policy
        self.weights_offload_dir = weights_offload_dir
        self.compute_device = compute_device if torch.cuda.is_available() else "cpu"
        self.args_offload_dir = args_offload_dir

        # mpl
        mpl = ModelPolicyLoader(
            checkpoint=checkpoint,
            policy=policy,
            weights_offload_dir=weights_offload_dir,
        )
        self.model = mpl.model
        self.layer_names = mpl.layer_names
        self.num_layers = mpl.num_layers
        self.mpl = mpl

        # bpl
        bpl = BlockPolicyLoader(policy=policy, args_offload_dir=args_offload_dir)
        self.K = bpl.K
        self.bpl = bpl

        # streams: prev/curr/next layers
        self.use_streams = torch.cuda.is_available()
        self.streams = {}
        self.streams['prev_layer'] = torch.cuda.Stream() if self.use_streams else None
        self.streams['next_layer'] = torch.cuda.Stream() if self.use_streams else None
        self.streams['prev_batch'] = torch.cuda.Stream() if self.use_streams else None
        self.streams['next_batch'] = torch.cuda.Stream() if self.use_streams else None
        self.streams['curr_layer'] = torch.cuda.current_stream() if self.use_streams else None
        self.streams['curr_batch'] = torch.cuda.current_stream() if self.use_streams else None
        self.stream_names = list(self.streams.keys())

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
            logger.debug(f"{layer_name} from flexgen to old.")

    def model_reset(self):
        for j, _ in enumerate(self.layer_names):
            self.layer_reset(j)

    # load next layers' weights: TODO: profiling decorator @profile
    def load_next_layer(self, layer_name):
        stream = self.streams["next_layer"]
        with torch.cuda.stream(stream):
            self.mpl.load_layer_weights(layer_name, self.compute_device)

    # offload prev layer's weights
    def offload_prev_layer(self, layer_name):
        stream = self.streams["prev_layer"]
        with torch.cuda.stream(stream):
            self.mpl.offload_layer_weights(layer_name)

    # load current weights (it should be already loaded except for the very 1st layer),
    # prepare layer inputs but do not load
    def prepare_curr_layer(self, layer_name, inputs):
        stream = self.streams["curr_layer"]
        with torch.cuda.stream(stream):
            self.mpl.load_layer_weights(layer_name, self.compute_device)
            self.bpl.layer_init(inputs=inputs, layer_name=layer_name)

            # debug infos
            args_k, kwargs_k = self.bpl.get_kth_input(1)
            logger.debug(f"args_k: {get_info(args_k)}")
            logger.debug(f"kwarg_k: {get_info(kwargs_k)}")

    def _store_prev_batch(self, k):
        # k: current batch, to store: prev batch
        assert self.K >= 2
        if k > 0:
            self.bpl.offload_kth_output(k - 1)
            logger.debug(
                f"batch: {k - 1}, offloaded output: {get_info(self.bpl.get_kth_output(k - 1))}"
            )
        elif k == 0:  # corner case
            self.bpl.offload_kth_input(-1)
            logger.debug(f"output of last layer, as curr layer's input")
            logger.debug(
                f"batch: {self.K - 1}, offloaded input: {get_info(self.bpl.get_kth_input(-1))}"
            )

    # store prev batch output (act, kv)
    def store_prev_batch(self, k):
        stream = self.streams["prev_batch"]
        with torch.cuda.stream(stream):
            self._store_prev_batch(k)

    # load curr batch input (act, kv) and compute the k-th forward pass
    def compute_curr_batch(self, k, old_forward):
        stream = self.streams["curr_batch"]
        with torch.cuda.stream(stream):
            self.bpl.load_kth_input(k)
            args_k, kwargs_k = self.bpl.get_kth_input(k)
            output = old_forward(*args_k, **kwargs_k)
            self.bpl.set_kth_output(k, output)

    def _load_next_batch(self, k):
        assert self.K >= 2
        if k < self.K - 1:
            self.bpl.load_kth_input(k + 1)
        else:  # corner case
            self.bpl.load_kth_output(0)

    # load next batch input (act, kv)
    def load_next_batch(self, k):
        stream = self.streams["next_batch"]
        with torch.cuda.stream(stream):
            self._load_next_batch(k)

    def batch_sync(self):
        if self.use_streams:
            self.streams["prev_batch"].synchronize()
            self.streams["next_batch"].synchronize()
            self.streams["curr_batch"].synchronize()

    def layer_sync(self):
        if self.use_streams:
            self.streams["prev_layer"].synchronize()
            self.streams["next_layer"].synchronize()
            self.streams["curr_layer"].synchronize()

    def concat_outputs(self):
        return self.bpl.concat_outputs()

    def get_flexgen_forward(
        self, old_forward, prev_layer_name, curr_layer_name, next_layer_name
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
            logger.debug(f"layer: {curr_layer_name}")
            logger.debug(f"args: {get_info(args)}")
            logger.debug(f"kwargs: {get_info(kwargs)}")

            # steps of FlexGen Alg.1
            self.offload_prev_layer(layer_name=prev_layer_name)
            self.load_next_layer(layer_name=next_layer_name)
            self.prepare_curr_layer(layer_name=curr_layer_name, inputs=(args, kwargs))

            for k in range(self.K):
                self.store_prev_batch(k)
                self.load_next_batch(k)
                self.compute_curr_batch(k, old_forward)

                self.batch_sync()

            # concatenate outputs of K batches.
            # And for the last layer (e.g. lm_head in OPT),
            # send its output (e.g. a token's logits) to compute device
            # to get the generated id (e.g. by a torch.argmax operation)
            output = self.concat_outputs()
            if curr_layer_name == self.layer_names[-1]:
                output = to_compute_device(output)
            logger.debug(f"outputs after concat: {get_info(output)}")

            self.layer_sync()
            logger.debug("over.\n\n")

            return output

        return flexgen_forward

    def layer_to_flexgen(self, j):
        # get prev, curr and next layers' names
        prev_layer_name = self.layer_names[(j - 1) % self.num_layers]
        curr_layer_name = self.layer_names[j]
        next_layer_name = self.layer_names[(j + 1) % self.num_layers]

        # get current layer module, and save its old forward
        layer = get_module_from_name(self.model, curr_layer_name)
        if hasattr(layer, "_flexgen_old_forward"):
            return
        layer._flexgen_old_forward = layer.forward

        # override layer forward
        layer.forward = self.get_flexgen_forward(
            old_forward=layer.forward,
            prev_layer_name=prev_layer_name,
            curr_layer_name=curr_layer_name,
            next_layer_name=next_layer_name,
        )
        logger.debug(f"{curr_layer_name} to flexgen forward")

    def model_to_flexgen(self):
        for j, _ in enumerate(self.layer_names):
            self.layer_to_flexgen(j)
