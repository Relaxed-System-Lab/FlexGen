import logging
from datetime import datetime
import os

__all__ = ["logging", "logging_config"]

def logging_config(args):
    log_dir = args.log_dir
    ckpt_str = args.checkpoint.replace('/', '.') 
    time_info = datetime.now().strftime("%Y-%m-%d,%H:%M:%S,%f")
    exp_dir = f'{log_dir}/{ckpt_str}/{time_info}'

    os.makedirs(f'{exp_dir}', exist_ok=True)

    logging.basicConfig(
        style="{",
        format="{asctime} [{filename}:{lineno} in {funcName}] {levelname} | {message}",
        handlers=[
            logging.FileHandler(f"{exp_dir}/info.log", "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
    )

    return exp_dir
