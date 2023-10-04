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
