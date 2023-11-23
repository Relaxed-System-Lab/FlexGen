import logging
from datetime import datetime
import os

__all__ = ["logging"]

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
time_info = datetime.now().strftime("%Y-%m-%d,%H:%M:%S,%f")

logging.basicConfig(
    style="{",
    format="{asctime} [{filename}:{lineno} in {funcName}] {levelname} \n {message}",
    handlers=[
        logging.FileHandler(f"{log_dir}/{time_info}.log", "w"),
        logging.StreamHandler(),
    ],
    level=logging.DEBUG,
)
