
# Support for general models (instead of hard-coded OPT)

TODO:
- policy solver
- multiple GPUs, tensor parallelism
- various inference algorithms (including: speculative)

Requirements:
```shell
accelerate==0.23.0
numpy==1.24.3
torch==2.2.0.dev20231122+cu118
tqdm==4.65.0
transformers==4.33.2

# upgrade to torch 2.2
pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
pip uninstall transformer-engine
```

Example:
```shell
python3 run.py --checkpoint facebook/opt-1.3b --compute-device cuda:0 --prompt-len 32 --gen-len 10 --gpu-batch-size 8 --num-gpu-batches 10 --percent 50 50 100 0 100 0 --verbose --normal-loop
```

