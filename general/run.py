import argparse

from utils import logging
from utils.test import test_hf_gen
from flexgen import FlexGen, Policy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# argparse
parser = argparse.ArgumentParser(description="Test-EOS_LLM")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="facebook/opt-13b",
    metavar="S",
    help="HF model name, e.g. "
    "facebook/opt-125m "
    "facebook/opt-1.3b "
    "facebook/opt-13b "
    "Salesforce/codegen-350M-mono "
    "bigscience/bloom-560m "
    "NousResearch/Llama-2-7b-chat-hf "
    "huggyllama/llama-7b s",
)
parser.add_argument(
    "--compute-device",
    type=str,
    default="cuda:0",  # multi GPUs
    metavar="S",
    help="compute device (cpu or cuda)",
)
parser.add_argument(
    "--normal-loop",
    action="store_true",
    help="no overlap",
)
parser.add_argument(
    "--prompt-len",
    type=int,
    default=128,
)
parser.add_argument(
    "--gen-len",
    type=int,
    default=32,
)
parser.add_argument(
    "--percent",
    nargs="+",
    type=int,
    default=[20, 30, 0, 25, 100, 0],
    help="Six numbers. They are "
    "the percentage of weight on GPU, "
    "the percentage of weight on CPU, "
    "the percentage of attention cache on GPU, "
    "the percentage of attention cache on CPU, "
    "the percentage of activations on GPU, "
    "the percentage of activations on CPU",
)
parser.add_argument(
    "--gpu-batch-size",
    type=int,
    default=16,
)
parser.add_argument(
    "--num-gpu-batches",
    type=int,
    default=10,
)
args = parser.parse_args()

# flexgen config
checkpoint = args.checkpoint
compute_device = args.compute_device
overlap = not args.normal_loop
prompt_len = args.prompt_len
gen_len = args.gen_len
(
    weights_gpu_percent,
    weights_cpu_percent,
    cache_gpu_percent,
    cache_cpu_percent,
    act_gpu_percent,
    act_cpu_percent,
) = [p / 100 for p in args.percent]
gpu_batch_size = args.gpu_batch_size
num_gpu_batches = args.num_gpu_batches

policy = Policy(
    gpu_batch_size=gpu_batch_size,
    num_gpu_batches=num_gpu_batches,
    weights_gpu_percent=weights_gpu_percent,
    weights_cpu_percent=weights_cpu_percent,
    cache_gpu_percent=cache_gpu_percent,
    cache_cpu_percent=cache_cpu_percent,
    act_gpu_percent=act_gpu_percent,
    act_cpu_percent=act_cpu_percent,
    overlap=overlap,
    pin_weight=True,
)  # TODO: policy solver

logger.info(args)
logger.info(policy)

# flexgen test
with FlexGen(
    checkpoint=checkpoint, policy=policy, compute_device=compute_device, verbose=False
) as model:
    num_prompts = policy.gpu_batch_size * policy.num_gpu_batches
    test_hf_gen(
        checkpoint,
        model,
        num_prompts,
        compute_device,
        prompt_len=prompt_len,
        gen_len=gen_len,
    )
