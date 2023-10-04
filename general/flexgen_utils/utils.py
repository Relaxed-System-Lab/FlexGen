import numpy as np 

def get_device(cur_percent, percents, choices):
    # choose a device (gpu / cpu / disk) for a weight tensor by its percent of size
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 1.0) < 1e-5, f'{percents}'

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]


def get_module_from_name(lm_model, name):
    splits = name.split('.')
    module = lm_model
    for split in splits:
        if split == '': 
            continue 

        new_module = getattr(module, split)
        if new_module is None:
            raise ValueError(f"{module} has no attribute {split}.")
        module = new_module
    return module 


def get_tied_target(tensor_name, tied_params, dat_files):
    # if tensor_name is tied and without a .dat file, if it is not tied, return itself
    for group in tied_params:
        if tensor_name in group:
            for name in group:
                if name + '.dat' in dat_files:
                    return name 
    return tensor_name

