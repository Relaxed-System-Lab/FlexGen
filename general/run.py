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
    default="facebook/opt-2.7b",
    metavar="S",
    help="HF model name, e.g. "
    "facebook/opt-125m "
    "facebook/opt-1.3b "
    "facebook/opt-13b "
    "Salesforce/codegen-350M-mono "
    "bigscience/bloom-560m "
    "NousResearch/Llama-2-7b-chat-hf"
)
parser.add_argument(
    "--compute_device",
    type=str,
    default="cuda:0", # multi GPUs
    metavar="S",
    help="compute device (cpu or cuda)",
)
parser.add_argument(
    "--normal_loop",
    action="store_true",
    help="no overlap",
)
args = parser.parse_args()

# flexgen config
checkpoint = args.checkpoint
compute_device = args.compute_device
overlap = not args.normal_loop
policy = Policy(
    gpu_batch_size=32,  # batch load, compute time both are proportional to gbs?
    num_gpu_batches=4,
    weights_gpu_percent=0.1,
    weights_cpu_percent=0.3,
    cache_gpu_percent=0.1,
    cache_cpu_percent=0.3,
    act_gpu_percent=0,
    act_cpu_percent=0.5,
    overlap=overlap,
    pin_weight=True,
) # TODO: policy solver

logger.info(args)
logger.info(policy)

# flexgen test
with FlexGen(
    checkpoint=checkpoint, policy=policy, compute_device=compute_device, verbose=True
) as model:
    num_prompts = policy.gpu_batch_size * policy.num_gpu_batches
    test_hf_gen(checkpoint, model, num_prompts, compute_device)
