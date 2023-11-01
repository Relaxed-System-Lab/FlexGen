
# Support for general models (instead of hard-coded OPT)

TODO:
- policy solver
- profiler: cuda event
- multiple GPU / tensor parallelism: devices manager
- various inference algorithms (including: ?)

Example:
```shell
python3 run.py --checkpoint facebook/opt-1.3b --compute_device cuda:0
```