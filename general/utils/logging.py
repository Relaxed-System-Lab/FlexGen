import logging
from datetime import datetime
import os

__all__ = ["logging", "logging_config"]

def logging_config(log_dir):
    time_info = datetime.now().strftime("%Y-%m-%d,%H:%M:%S,%f")
    os.makedirs(f'{log_dir}/{time_info}', exist_ok=True)

    logging.basicConfig(
        style="{",
        format="{asctime} [{filename}:{lineno} in {funcName}] {levelname} | {message}",
        handlers=[
            logging.FileHandler(f"{log_dir}/{time_info}/info.log", "w"),
            logging.StreamHandler(),
        ],
        level=logging.INFO,
    )

    return f'{log_dir}/{time_info}'
