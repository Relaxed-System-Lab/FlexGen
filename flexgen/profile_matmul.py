"""
Usage:
python3 profile_matmul.py
"""

import numpy as np
import torch

from flexgen.profile_bandwidth import benchmark_func


def bench_matmul():
    for device in ["cuda"]:
        for dtype in [torch.float16, torch.float32]:
            for n in [1024, 2048, 7184]:
                # if device == "cuda":
                #     dtype = torch.float16
                # else:
                #     dtype = torch.float32

                a = torch.rand(n, n).to(dtype).to(device)
                b = torch.rand(n, 64 * 2024).to(dtype).to(device)

                def func():
                    return torch.matmul(a, b)

                cost = np.mean(benchmark_func(func, number=1, repeat=1, warmup=3)) 

                tflops = 2 * a.numel() * b.shape[-1] / cost / 1e12
                print(f"dtype: {dtype}, device: {device}, N: {n}, latency: {cost*1e3:.2f} ms, TFLOPS: {tflops:.3f}")
        print()


if __name__ == "__main__":
    bench_matmul()
