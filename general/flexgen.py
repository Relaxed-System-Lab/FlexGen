# rewrite layer forward function
import os
import shutil
import functools

import torch

from loaders import ModelPolicyLoader, BlockPolicyLoader
from utils import logging, Policy, get_module_from_name, get_info, to_compute_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class FlexGenAssets:
    """
    Assets that FlexGen need to utilize: loaders (by policy), streams, etc.
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

        # streams: prev/next layers/batches, curr layer/batch: current stream
        self.use_streams = torch.cuda.is_available() and policy.overlap
        self.streams = {}
        self.streams["prev_layer"] = torch.cuda.Stream() if self.use_streams else None
        self.streams["next_layer"] = torch.cuda.Stream() if self.use_streams else None
        self.streams["prev_batch"] = torch.cuda.Stream() if self.use_streams else None
        self.streams["next_batch"] = torch.cuda.Stream() if self.use_streams else None
        self.stream_names = list(self.streams.keys())


class NextLayerMixin:
    """
    load next layers' weights/buffers.
    """

    # TODO: profiling decorator @profile
    def _load_next_layer(self, layer_name):
        self.mpl.load_layer(layer_name, self.compute_device)

    def load_next_layer_log(self, layer_name):
        logger.debug(
            f"load_layer: {self.mpl.model_name}.{layer_name} to {self.compute_device}"
        )

    def load_next_layer(self, layer_name):
        stream = self.streams["next_layer"]
        with torch.cuda.stream(stream):
            self._load_next_layer(layer_name)


class PrevLayerMixin:
    """
    offload prev layer's weights/buffers.
    """

    def _offload_prev_layer(self, layer_name):
        self.mpl.offload_layer(layer_name)

    def offload_prev_layer_log(self, layer_name):
        logger.debug(
            f"offload_layer: {self.mpl.model_name}.{layer_name} by policy."
        )

    def offload_prev_layer(self, layer_name):
        stream = self.streams["prev_layer"]
        with torch.cuda.stream(stream):
            self._offload_prev_layer(layer_name)


class CurrLayerMixin:
    """
    1. load current weights/buffers (it should be already loaded except for the very 1st layer)
    2. prepare layer input args, kwargs (but do not load them)
    3. concat mini-outputs after computing all mini-batches
    """

    def _prepare_curr_layer(self, layer_name, inputs):
        self.mpl.load_layer(layer_name, self.compute_device)
        self.bpl.layer_init(inputs=inputs, layer_name=layer_name)

    def prepare_curr_layer_log(self, layer_name, inputs):
        # debug infos
        logger.debug(f"weights and inputs of {layer_name} are prepared")
        args_k, kwargs_k = self.bpl.get_kth_input(1)
        logger.debug(f"args_k: {get_info(args_k)}")
        logger.debug(f"kwarg_k: {get_info(kwargs_k)}")

    def prepare_curr_layer(self, layer_name, inputs):
        self._prepare_curr_layer(layer_name, inputs)

    def concat_outputs(self):
        return self.bpl.concat_outputs()


class PrevBatchMixin:
    """
    store prev batch output (act, kv).
    """

    def _store_prev_batch(self, k):
        # k: current batch, to store: prev batch
        assert self.K >= 2
        if k > 0:
            self.bpl.offload_kth_output(k - 1)
        elif k == 0:  # corner case
            exists_mix = self.bpl.exists_mix_input()
            if exists_mix:
                self.bpl.offload_kth_input(-1)

    def store_prev_batch_log(self, k):
        assert self.K >= 2
        if k > 0:
            logger.debug(
                f"offloaded output of batch {k - 1}: {get_info(self.bpl.get_kth_output(k - 1))}"
            )
        elif k == 0:  # corner case
            exists_mix = self.bpl.exists_mix_input()
            logger.debug(
                f"output from last layer is the input of curr layer, exists MixTensor: {exists_mix}"
            )
            logger.debug(
                f"offloaded input of batch {self.K - 1}: {get_info(self.bpl.get_kth_input(-1))}"
            )

    def store_prev_batch(self, k):
        stream = self.streams["prev_batch"]
        with torch.cuda.stream(stream):
            self._store_prev_batch(k)


class CurrBatchMixin:
    """
    load curr batch input (act, kv) and compute the k-th forward pass.
    """

    def _compute_curr_batch(self, k, old_forward):
        self.bpl.load_kth_input(k)
        args_k, kwargs_k = self.bpl.get_kth_input(k)
        output = old_forward(*args_k, **kwargs_k)
        self.bpl.set_kth_output(k, output)

    def compute_curr_batch_log(self, k, old_forward):
        logger.debug(f"computed batch {k}")

    def compute_curr_batch(self, k, old_forward):
        self._compute_curr_batch(k, old_forward)


class NextBatchMixin:
    """
    load next batch input (act, kv).
    """

    def _load_next_batch(self, k):
        assert self.K >= 2
        if k < self.K - 1:
            self.bpl.load_kth_input(k + 1)
        else:  # corner case
            self.bpl.load_kth_output(0)

    def load_next_batch_log(self, k):
        assert self.K >= 2
        if k < self.K - 1:
            logger.debug(
                f"loaded input of batch {k + 1}: {get_info(self.bpl.get_kth_input(k + 1))}"
            )
        else:  # corner case
            logger.debug(f"curr layer's output is next layer's input")
            logger.debug(
                f"loaded output of batch {0}: {get_info(self.bpl.get_kth_output(0))}"
            )

    def load_next_batch(self, k):
        stream = self.streams["next_batch"]
        with torch.cuda.stream(stream):
            self._load_next_batch(k)


class SyncMixin:
    """
    Synchronizations.
    """

    def batch_sync(self):
        return
        if self.use_streams:
            torch.cuda.current_stream().synchronize()
            self.streams["prev_batch"].synchronize()
            self.streams["next_batch"].synchronize()

    def curr_sync(self):
        return
        torch.cuda.current_stream().synchronize()

    def layer_sync(self):
        return
        if self.use_streams:
            torch.cuda.synchronize()


class FlexGen(
    PrevLayerMixin,
    CurrLayerMixin,
    NextLayerMixin,
    PrevBatchMixin,
    CurrBatchMixin,
    NextBatchMixin,
    SyncMixin,
    FlexGenAssets,
):
    """
    override the forward method for each layer (e.g. embedding layers, transformer blocks, etc.) of a CausalLM.
    example:
        >>> with FlexGen(checkpoint, policy) as model:
        >>>     model.generate(...)
    """

    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.stop_event = torch.cuda.Event(enable_timing=True)

        self.verbose = verbose

    def __enter__(self):
        self.model_to_flexgen()
        self.start_event.record()
        return self.model

    def __exit__(self, *exception_infos):
        # elasped time
        self.stop_event.record()
        elasped_time = self.start_event.elapsed_time(self.stop_event)
        logger.info(f"elasped time: {elasped_time}ms")

        torch.cuda.empty_cache()

        # self.model_reset()
        shutil.rmtree(self.args_offload_dir)
        os.makedirs(self.args_offload_dir, exist_ok=True)
        # import sys, traceback, threading
        # thread_names = {t.ident: t.name for t in threading.enumerate()}
        # for thread_id, frame in sys._current_frames().items():
        #     print("Thread %s:" % thread_names.get(thread_id, thread_id))
        #     traceback.print_stack(frame)
        #     print()

    def get_flexgen_forward(
        self, old_forward, prev_layer_name, curr_layer_name, next_layer_name
    ):
        """
        override the j-th layer's forward function to FlexGen version.

        'pre' forward:
            1) load current layer's weights/buffers (to compute device)
            2) load next layer's weights/buffers (to compute device)
        'call' forward:
            1) split the input databatch to K minibatches
                a) input databatch: *args and **kwargs
                b) minibatches: lists of *args_k and **kwargs_k
            2) call layer.forward for each minibatch, and for each mini-call:
                a) pre mini-call:
                    * load current minibatch's input (to compute device)
                    * load next minibatch's input (to compute device)
                b) mini-call forward:
                    * output_k = layer.forward(*args_k, **kwargs_k)
                c) post mini-call:
                    * free current minibatch's input
                    * offload current minibatch's output (to mixed devices by policy)
        'post' forward:
            1) offload current layer's weights (to mixed devices by policy)

        The 'pre', 'call', and 'post' forward above are executed in parallel.
        """

        @torch.no_grad()
        @functools.wraps(old_forward)
        def flexgen_forward(*args, **kwargs):
            logger.debug(
                f"layer: {self.mpl.model_name}.{curr_layer_name} calls forward"
            )
            if self.verbose:
                logger.debug(f"args: {get_info(args)}")
                logger.debug(f"kwargs: {get_info(kwargs)}")

            # steps of FlexGen Alg.1
            self.offload_prev_layer(layer_name=prev_layer_name) #
            self.load_next_layer(layer_name=next_layer_name) #
            self.prepare_curr_layer(layer_name=curr_layer_name, inputs=(args, kwargs)) # helper
            self.curr_sync() # current stream sync

            # log after sync
            if self.verbose:
                self.prepare_curr_layer_log(
                    layer_name=curr_layer_name, inputs=(args, kwargs)
                )

            for k in range(self.K):
                torch.cuda.nvtx.range_push(f'{curr_layer_name}-batch-{k}')

                self.store_prev_batch(k) #
                self.load_next_batch(k) #
                self.compute_curr_batch(k, old_forward) # 
                self.batch_sync() # 

                # log after sync
                if self.verbose:
                    self.store_prev_batch_log(k)
                    self.compute_curr_batch_log(k, old_forward)
                    self.load_next_batch_log(k)
                    logger.debug("")

                torch.cuda.nvtx.range_pop()

            # concatenate outputs of K batches.
            # And for the last layer (e.g. lm_head in OPT),
            # send its output (e.g. a token's logits) to compute device
            # to get the generated id (e.g. by a torch.argmax operation)
            output = self.concat_outputs()
            if curr_layer_name == self.layer_names[-1]:
                output = to_compute_device(output)
            self.layer_sync() # 
            torch.cuda.empty_cache() 

            # log after sync
            logger.debug(f"outputs after concat: {get_info(output)}")
            if self.verbose:
                self.offload_prev_layer_log(layer_name=prev_layer_name)
                self.load_next_layer_log(layer_name=next_layer_name)
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
