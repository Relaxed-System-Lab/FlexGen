# block operations: split/merge layer input/output data structures to mini-batch/large-batch

import os
from utils import (
    logging,
    Policy,
    get_kth_batch_inputs,
    to_compute_device,
    to_mixed_device,
    concat_outputs,
    any_is_mix,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = ["BlockPolicyLoader"]


class BlockPolicyLoader:
    """
    block:
        input/output data block of a layer,
        a block consist of multiple GPU batches.
    """

    def __init__(self, policy: Policy, args_offload_dir="args_offload_dir"):
        self.policy = policy
        self.K = policy.num_gpu_batches
        self.args_offload_dir = args_offload_dir
        os.makedirs(args_offload_dir, exist_ok=True)  # in args offloader

    def layer_init(self, inputs, layer_name):
        self.inputs = inputs
        self.layer_name = layer_name
        self.input_batches = [
            get_kth_batch_inputs(self.inputs, k, self.K) for k in range(self.K)
        ]
        self.output_batches = [None for _ in range(self.K)]

    # for input
    def get_kth_input(self, k):
        return self.input_batches[k]

    def load_kth_input(self, k):
        self.input_batches[k] = to_compute_device(self.input_batches[k])

    def free_kth_input(self, k):
        self.input_batches[k] = None

    def offload_kth_input(self, k):
        self.input_batches[k] = to_mixed_device(
            self.input_batches[k],
            self.policy,
            prefix=f"{self.args_offload_dir}/{self.layer_name}.batch.{k}.input",
        )

    def exists_mix_input(self):
        return any_is_mix(self.input_batches)

    # for output
    def set_kth_output(self, k, output):
        self.output_batches[k] = output

    def get_kth_output(self, k):
        return self.output_batches[k]

    def load_kth_output(self, k):
        self.output_batches[k] = to_compute_device(self.output_batches[k])

    def offload_kth_output(self, k):
        self.output_batches[k] = to_mixed_device(
            self.output_batches[k],
            self.policy,
            prefix=f"{self.args_offload_dir}/{self.layer_name}.batch.{k}.output",
        )

    def concat_outputs(self):
        return concat_outputs(self.output_batches)
