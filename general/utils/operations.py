import torch
from accelerate.utils import honor_type
from typing import Mapping
from utils import logging
from tensor import MixTensor, BatchListTensor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = [
    "get_module_from_name",
    "get_info",
    "to_compute_device",
    "to_mixed_device",
    "concat_outputs",
    "get_kth_batch_inputs",
    "any_is_mix"
]


def get_module_from_name(lm_model, name):
    splits = name.split(".")
    module = lm_model
    for split in splits:
        if split == "":
            continue

        new_module = getattr(module, split)
        if new_module is None:
            raise ValueError(f"{module} has no attribute {split}.")
        module = new_module
    return module


def get_info(obj):
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (get_info(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k: get_info(v) for k, v in obj.items()})
    elif isinstance(obj, (torch.Tensor, MixTensor, BatchListTensor)):
        return f"{obj.__class__.__name__}(shape={tuple(obj.size())}, dtype={obj.dtype})"
    elif isinstance(obj, (int, bool, type(None))):
        return f"{obj}"
    else:
        logger.warning(f"inputs: {obj} of type '{type(obj)}' is not implemented.")
        return f"{obj.__class__.__name__}: {obj}"

def _is_mix(obj):
    if isinstance(obj, (tuple, list)):
        for elem in obj:
            yield from _is_mix(elem)
    elif isinstance(obj, Mapping):
        for _, v in obj.items():
            yield from _is_mix(v)
    elif isinstance(obj, BatchListTensor):
        for elem in obj.batches:
            yield from _is_mix(elem) 
    elif isinstance(obj, MixTensor):
        yield True 
    elif isinstance(obj, torch.Tensor):
        yield False 

def any_is_mix(obj):
    return any(_is_mix(obj))


def to_compute_device(obj):
    if isinstance(obj, (tuple, list)):
        return honor_type(obj, (to_compute_device(o) for o in obj))
    elif isinstance(obj, Mapping):
        return type(obj)({k: to_compute_device(v) for k, v in obj.items()})
    elif isinstance(obj, torch.Tensor):
        return obj
    elif isinstance(obj, (MixTensor, BatchListTensor)):
        return obj.to_tensor()
    elif isinstance(obj, (int, bool, type(None))):
        return obj
    else:
        logger.warning(f"inputs: {obj} of type '{type(obj)}' is not implemented.")
        return obj


def to_mixed_device(obj, policy, prefix):
    if (
        isinstance(obj, tuple)
        and len(obj) == 2
        and isinstance(obj[0], torch.Tensor)
        and isinstance(obj[1], torch.Tensor)
    ):
        # KV cache
        m0 = MixTensor.from_tensor(
            obj[0],
            percents={
                "cuda": policy.cache_gpu_percent,
                "cpu": policy.cache_cpu_percent,
                "disk": policy.cache_disk_percent,
            },
            file_path=f"{prefix}.key.dat",
        )
        m1 = MixTensor.from_tensor(
            obj[1],
            percents={
                "cuda": policy.cache_gpu_percent,
                "cpu": policy.cache_cpu_percent,
                "disk": policy.cache_disk_percent,
            },
            file_path=f"{prefix}.value.dat",
        )
        return (m0, m1)
    elif isinstance(obj, torch.Tensor): 
        # activations / attention mask
        return MixTensor.from_tensor(
            obj,
            percents={
                "cuda": policy.act_gpu_percent,
                "cpu": policy.act_cpu_percent,
                "disk": policy.act_disk_percent,
            },
            file_path=f"{prefix}.dat",
        )
    elif isinstance(obj, tuple):
        return tuple(
            to_mixed_device(o, policy, f"{prefix}.{i}") for i, o in enumerate(obj)
        )
    elif isinstance(obj, Mapping):
        return type(obj)(
            {
                key: to_mixed_device(value, policy, f"{prefix}.{key}")
                for key, value in obj.items()
            }
        )
    elif isinstance(obj, (int, bool, type(None))):
        return obj
    else:
        logger.warning(f"inputs: {obj} of type '{type(obj)}' is not implemented.")
        return obj


def concat_outputs(outputs):  # concatenate K outputs to one output
    assert len(outputs), "empty outputs."
    assert isinstance(
        outputs[0], (MixTensor, torch.Tensor, tuple)
    ), f"not supported type: {type(outputs[0])}."

    if isinstance(outputs[0], (torch.Tensor, MixTensor)):
        return BatchListTensor(outputs)
    elif isinstance(outputs[0], tuple):

        def f(outputs):
            ans = []
            for elem in zip(*outputs):
                if isinstance(elem[0], (torch.Tensor, MixTensor)):
                    ans.append(BatchListTensor(elem))
                elif isinstance(elem[0], tuple):
                    ans.append(f(elem))
                else:
                    logger.warning(
                        f"outputs: {elem[0]} of type '{type(elem[0])}' is not implemented."
                    )
                    ans.append(elem[0])
            return tuple(ans)

        return f(outputs)


def get_kth_batch_inputs(inputs, k, ngb):
    """
    get minibatch inputs with a nested structure of tuple/list/dict/Tensor/Block
    here is no data I/O costs.
    """
    if isinstance(inputs, (tuple, list)):  # e.g. args
        return honor_type(inputs, (get_kth_batch_inputs(inp, k, ngb) for inp in inputs))
    elif isinstance(inputs, Mapping):  # e.g. kwargs
        return type(inputs)(
            {key: get_kth_batch_inputs(value, k, ngb) for key, value in inputs.items()}
        )
    elif isinstance(inputs, torch.Tensor):
        mini_size = inputs.size(0) // ngb
        return inputs[k * mini_size : (k + 1) * mini_size]
    elif isinstance(inputs, BatchListTensor):
        mini_batch = inputs.batches[k]
        return mini_batch
    elif isinstance(inputs, (int, bool, type(None))):
        return inputs
    else:
        logger.warning(f"inputs: {inputs} of type '{type(inputs)}' is not implemented.")
        return inputs
