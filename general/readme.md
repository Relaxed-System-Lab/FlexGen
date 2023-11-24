
# Support for general models (instead of hard-coded OPT)

TODO:
- policy solver
- profiler: cuda event
- multiple GPU / tensor parallelism: devices manager
- various inference algorithms (including: ?)

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
python3 run.py --checkpoint facebook/opt-1.3b --compute_device cuda:0
```

