import logging

__all__ = ["logging"]

logging.basicConfig(
    style="{",
    format="{asctime} [{filename}:{lineno} in {funcName}] {levelname} - {message}",
    handlers=[logging.FileHandler("run.log", "w"), logging.StreamHandler()],
    level=logging.DEBUG,
)
