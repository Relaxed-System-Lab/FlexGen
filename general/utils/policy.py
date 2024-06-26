from dataclasses import dataclass


__all__ = ["FlexPolicy"]

# TODO: policy solver

@dataclass(frozen=True)
class FlexPolicy:
    prompt_len: int 
    gen_len: int 

    gpu_batch_size: int
    num_gpu_batches: int

    # percent of weights/cache/activations on GPU/CPU/Disk %
    weights_gpu_percent: float
    weights_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    @property
    def block_size(self):
        return self.gpu_batch_size * self.num_gpu_batches

    @property
    def weights_disk_percent(self):
        return 1.0 - self.weights_gpu_percent - self.weights_cpu_percent

    @property
    def cache_disk_percent(self):
        return 1.0 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 1.0 - self.act_gpu_percent - self.act_cpu_percent
