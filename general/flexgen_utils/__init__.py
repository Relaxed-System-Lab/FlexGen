import logging 

logging.basicConfig(
    style='{',
    format='{asctime} [{filename}:{lineno} in {funcName}] {levelname} - {message}',
    handlers=[
        logging.FileHandler("flexgen.log", 'w'),
        logging.StreamHandler()
    ],
    level=logging.INFO
)


from .policy import Policy


class AttrDict(dict):
    __slots__ = () 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


import numpy as np 

def get_device(cur_percent, percents, choices):
    # choose a device (gpu / cpu / disk) for a weight tensor by its percent of size
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 1.0) < 1e-5, f'{percents}'

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]