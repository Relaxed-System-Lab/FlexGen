from utils import Policy, logging
from flexgen import FlexGen
from utils.test import test_hf_gen

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

checkpoint = "facebook/opt-125m"  
# checkpoint = "facebook/opt-1.3b"  
# checkpoint = "facebook/opt-13b"
# checkpoint = "Salesforce/codegen-350M-mono"
# checkpoint = 'bigscience/bloom-560m' #

policy = Policy(
    gpu_batch_size=1,
    num_gpu_batches=4,
    weights_gpu_percent=0.2,
    weights_cpu_percent=0.3,
    cache_gpu_percent=0.3,
    cache_cpu_percent=0.2,
    act_gpu_percent=0.4,
    act_cpu_percent=0.5,
    overlap=True,
    pin_weight=True,
)
compute_device = "cuda:0"

with FlexGen(checkpoint, policy, compute_device=compute_device) as model:
    num_prompts = policy.gpu_batch_size * policy.num_gpu_batches
    test_hf_gen(checkpoint, model, num_prompts, compute_device)
