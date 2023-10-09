from utils import Policy, logging
from forward import flexgen
from test import test_hf_gen

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# checkpoint = "facebook/opt-125m" # 125m 6.7b 13b 30b
# checkpoint = "Salesforce/codegen-350M-mono"
checkpoint = 'bigscience/bloom-560m'

policy = Policy(
    gpu_batch_size=2, 
    num_gpu_batches=4, 
    weights_gpu_percent=0.0, 
    weights_cpu_percent=0.3, 
    cache_gpu_percent=0.0, 
    cache_cpu_percent=0.2, 
    act_gpu_percent=0.0, 
    act_cpu_percent=0.5, 
    overlap=True, 
    pin_weight=True,
)

with flexgen(checkpoint, policy) as model:
    num_prompts = policy.gpu_batch_size * policy.num_gpu_batches
    test_hf_gen(checkpoint, model, num_prompts)
