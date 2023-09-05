import logging

logging.basicConfig(
    style='{',
    format='{asctime} [{filename}:{lineno} in {funcName}] {levelname} - {message}',
    handlers=[
        logging.FileHandler(".log", 'w'),
        logging.StreamHandler()
    ],
    level=logging.DEBUG
)
logging.info('importing...')

import os
import json
import shutil
import argparse
import dataclasses

import torch
from transformers import AutoTokenizer, OPTForCausalLM
from safetensors.torch import save_model

logging.info('imported!')


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    # sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    # cpu_cache_compute: bool

    # Sparsity of attention weights
    # attn_sparsity: float

    # Compress weights with group-wise quantization
    # compress_weight: bool
    # comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    # compress_cache: bool
    # comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent


def init_model_weight(model_name): #?
    model_path = f"./io_cache_dir/eos{model_name.replace('/', '.')}"
    # if os.path.isdir(model_path):
    #     logging.info("Path exists, remove previous model dir.")
    #     shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)

    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    save_model(model, f"{model_path}/weight.safetensors")
    
    return model, f"{model_path}/weight.safetensors"


class EOS:
    """
    example: 
        eos_model = eos(model, policy)
        output_ids = eos_model.generate(...)
    """
    def __init__(self, model, policy):
        self.model = model
        self.policy = policy

    @torch.no_grad()
    def generate(
        self, 
        *args, 
        **kwargs
    ) -> torch.LongTensor:
        
        return self.model.generate(*args, **kwargs) 


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Efficient Offload Scheduling of Large Language Models.')
    parser.add_argument('--model-name', type=str, default='facebook/opt-350m', metavar='S',
                        help='HF model name')
    parser.add_argument('--compute-device', type=str, default='cpu', metavar='S',
                        help='compute device (cpu or cuda)')
    args = parser.parse_args()

    if args.compute_device == 'cpu':
        compute_device = torch.device("cpu")
    elif args.compute_device == 'cuda':
        compute_device = torch.device("cuda", 0)
    else:
        raise ValueError("Illegal device")

    # model
    logging.info('loading model...')
    model, model_path = init_model_weight(args.model_name)
    policy = Policy(1, 4, 0.0, 0.5, 0.0, 0.3, 0.0, 0.5, True, True)
    model = EOS(model, policy)
    logging.info('loaded!')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    # generate
    contexts = "Who is Joseph?"
    inputs = tokenizer(contexts, return_tensors="pt").to(compute_device)   
    
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=30)
    
    output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    logging.info(f"input: {contexts}, output: {output_text}")
