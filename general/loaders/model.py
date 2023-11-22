# model: 1) load/offload layer weights/buffers, 2) init weights by policy

import os
import shutil
import numpy as np
import json
from tqdm import tqdm
import contextlib
import functools
import copy

import torch
from torch.nn import Module, ModuleList
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module
from accelerate.utils import (
    find_tied_parameters,
    named_module_tensors,
    set_module_tensor_to_device,
    send_to_device,
)

from utils import logging, Policy
from utils import get_module_from_name, set_module_from_name


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)

__all__ = ["ModelPolicyLoader", "MetaModel"]


class MetaModel:
    """
    functions:
        1) download model weights
        2) analyze model structure
            a) layer calling order
            b) weight device map (by FlexGen policy)
            c) tied parameters
        3) weights offloading
            a) load layer weights/buffers
            b) offload layer weights/buffers

    args:
        checkpoint:
            1) download *.bin weights from Hugging Face.
            2) get empty model (on meta device) by AutoConfig.from_pretrained(checkpoint)
        policy:
            FlexGen policy: gpu_batch_size, num_gpu_batches, percents, etc.
        offload_dir:
            weights save dir
    """

    def __init__(
        self,
        checkpoint: str,
        policy: Policy,
        weights_offload_dir: str = "weights_offload_dir",
        dtype=torch.float16,
    ):
        # download weights
        self.checkpoint = checkpoint
        self.dtype = dtype
        self.policy = policy
        self.offload_dir = weights_offload_dir
        self.offload_folder = os.path.join(
            weights_offload_dir, checkpoint.replace("/", ".")
        )
        self.download()
        logger.info(f"weights offload folder: {self.offload_folder}")

        # analyze model structure
        with open(os.path.join(self.offload_folder, "index.json"), "r") as f:
            self.index = json.load(f)  # {name: {dtype, shape}}
        self.dat_files = [
            f for f in os.listdir(self.offload_folder) if f.endswith(".dat")
        ]
        self.model = self.get_empty_model()
        self.model_name = self.model.__class__.__name__
        self.tied_params = find_tied_parameters(self.model)
        logger.info(f"tied_params: {self.tied_params}")

        self.layers_dict = self.get_layer_module_dict(self.model)
        self.num_layers = len(self.layers_dict.keys())
        self.device_map = self.get_policy_weight_map()

        # offloading
        self._backups = {name: None for name in self.layers_dict}

        # test run to get layer names by calling order
        self.layer_names = self.test_run(device="cuda:0")
        assert len(self.layer_names) == self.num_layers

    def is_on_disk(self):
        # check if the model has a complete copy on disk.
        if not os.path.isdir(self.offload_folder):
            return False
        model = self.get_empty_model()
        tensor_names = [
            n
            for n, _ in named_module_tensors(model, include_buffers=False, recurse=True)
        ]
        dat_file_names = [
            file[:-4]
            for file in os.listdir(self.offload_folder)
            if file.endswith(".dat")
        ]
        # logger.info(f'{sorted(list(set(tensor_names) - set(dat_file_names)))}, {sorted(list(set(dat_file_names) - set(tensor_names)))}')
        return len(set(tensor_names) - set(dat_file_names)) == 0

    def download(self):
        if not self.is_on_disk():
            try:
                if os.path.exists(self.offload_folder):
                    shutil.rmtree(self.offload_folder)

                logger.info("downloading from hugging face...")
                AutoModelForCausalLM.from_pretrained(
                    self.checkpoint,
                    torch_dtype=self.dtype,
                    device_map={"": "disk"},
                    offload_folder=self.offload_folder,
                    offload_state_dict=True,
                    use_safetensors=False,  # download .bin files, for now
                )
                logger.info("downloaded")
            except:
                pass

        # check the model on disk
        if not self.is_on_disk():
            err_msg = "Mismatch between offload folder and model"
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        logger.info(
            f"The whole model has been downloaded an processed to offload_folder: '{self.offload_folder}'"
        )

    def get_empty_model(self):
        config = AutoConfig.from_pretrained(self.checkpoint, torch_dtype=self.dtype)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, torch_dtype=self.dtype)
        model.tie_weights()
        model.eval()
        remove_hook_from_module(model, recurse=True)
        return model

    @staticmethod
    def get_device(cur_percent, percents, choices):
        # choose a device (gpu / cpu / disk) for a weight tensor by its percent of size
        percents = np.cumsum(percents)
        assert np.abs(percents[-1] - 1.0) < 1e-5, f"{percents}"

        for i in range(len(percents)):
            if cur_percent < percents[i]:
                return choices[i]
        return choices[-1]

    @staticmethod
    def get_layer_module_dict(model: Module, prefix: str = "") -> dict:
        # return a dict of {layer_name : layer_module ('meta')}
        # of emb / norm layers & transformer layers
        res = {}
        for name, module in model.named_children():
            # leaf nodes: emb / norm layers
            if len(list(module.named_children())) == 0:
                res[prefix + name] = module
            # ModuleList: transformer
            elif isinstance(module, ModuleList):
                for block_name, block_module in module.named_children():
                    res[prefix + name + "." + block_name] = block_module
            else:
                res.update(MetaModel.get_layer_module_dict(module, prefix + name + "."))
        return res

    def get_policy_weight_map(self):
        # dict of device assignment for each tensor in the model
        weight_assign_dict = {}
        devices = ["cuda", "cpu", "disk"]
        percents_target = np.array(
            [
                self.policy.weights_gpu_percent,
                self.policy.weights_cpu_percent,
                self.policy.weights_disk_percent,
            ]
        )

        # model size (just parameters, no buffers), here we do not repeatly sum the tied paramters
        size_total = sum(
            np.prod(tensor.shape)
            for _, tensor in named_module_tensors(
                self.model, include_buffers=False, recurse=True
            )
        )
        size_done, size_todo = 0, size_total
        percents_done, percents_todo = 0 * percents_target, percents_target

        for layer_name, layer_module in self.layers_dict.items():
            # current layer
            tensor_sizes = [
                np.prod(tensor.shape)
                for _, tensor in named_module_tensors(
                    layer_module, include_buffers=False, recurse=True
                )
            ]
            tensor_sizes_cumsum = np.cumsum(tensor_sizes)

            # to balance the percents
            device_allo_size_dict = {device: 0 for device in devices}
            for i, (tensor_name, tensor) in enumerate(
                named_module_tensors(layer_module, include_buffers=False, recurse=True)
            ):
                abs_tensor_name = layer_name + "." + tensor_name

                def find_processed_tied(abs_tensor_name, weight_assign_dict):
                    # find the processed parameter (in weight_assign_dict) of the tied parameters.
                    for tp in self.tied_params:
                        if abs_tensor_name in tp:
                            for p in tp:
                                if p in weight_assign_dict:
                                    return p, tuple(tp)
                    return None

                processed_tied = find_processed_tied(
                    abs_tensor_name, weight_assign_dict
                )
                if processed_tied:  # this tensor is tied and processed.
                    p, tp = processed_tied
                    weight_assign_dict[abs_tensor_name] = {
                        # 'shape':  tensor.shape,
                        "assigned_device": weight_assign_dict[p]["assigned_device"],
                        "tied": tp,
                    }
                else:
                    # tensor mid size percent
                    mid_percent = (
                        tensor_sizes_cumsum[i] - tensor_sizes[i] / 2
                    ) / tensor_sizes_cumsum[-1]
                    device = self.get_device(mid_percent, percents_todo, devices)
                    weight_assign_dict[abs_tensor_name] = {
                        "shape": tensor.shape,
                        "assigned_device": device,
                    }

                    device_allo_size_dict[device] += tensor_sizes[i]

            # update percents_todo
            size_layer = sum(device_allo_size_dict.values())
            if size_layer > 0:
                device_allo_percents = (
                    np.array(
                        [device_allo_size_dict[device] * 1.0 for device in devices]
                    )
                    / size_layer
                )
                percents_done = (
                    percents_done * size_done + device_allo_percents * size_layer
                ) / (size_done + size_layer)
            size_done += size_layer
            size_todo -= size_layer
            if size_todo > 0:
                percents_todo = (
                    size_total * percents_target - size_done * percents_done
                ) / size_todo

            logger.debug(f"{layer_name}, {percents_done}, size_todo: {size_todo}")

        self.weight_assign_dict = weight_assign_dict
        self.device_map = {
            k: v["assigned_device"] for k, v in weight_assign_dict.items()
        }
        logger.info("device_map is prepared!")

        mem_g = (
            sum(
                [
                    np.prod(v["shape"])
                    for _, v in weight_assign_dict.items()
                    if "cuda" in v["assigned_device"] and "shape" in v
                ]
            )
            * 2
            / (2**30)
        )
        mem_c = (
            sum(
                [
                    np.prod(v["shape"])
                    for _, v in weight_assign_dict.items()
                    if v["assigned_device"] == "cpu" and "shape" in v
                ]
            )
            * 2
            / (2**30)
        )
        mem_d = (
            sum(
                [
                    np.prod(v["shape"])
                    for _, v in weight_assign_dict.items()
                    if v["assigned_device"] == "disk" and "shape" in v
                ]
            )
            * 2
            / (2**30)
        )
        mem = mem_d + mem_c + mem_g
        logger.info(
            f"CausalLM {self.checkpoint} is to be loaded on: "
            f"\nGPU Mem {mem_g:.2f} GiB ({mem_g / mem:.2%}), "
            f"CPU Mem {mem_c:.2f} GiB ({mem_c / mem:.2%}), "
            f"Disk Mem {mem_d:.2f} Gib ({mem_d / mem:.2%})"
        )

        return self.device_map

    def get_tied_target(self, tensor_name):
        if tensor_name + ".dat" in self.dat_files:
            return tensor_name

        for group in self.tied_params:
            if tensor_name in group:
                for name in group:
                    if name + ".dat" in self.dat_files:
                        return name

    def tensor_device_load(self, tensor_name, device="cpu"):
        actual_tensor_name = self.get_tied_target(tensor_name)

        metadata = self.index[actual_tensor_name]

        # copied from accelerate.utils.offload
        shape = tuple(metadata["shape"])
        if shape == ():
            # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
            shape = (1,)

        dtype = metadata["dtype"]
        if dtype == "bfloat16":
            # NumPy does not support bfloat16 so this was saved as a int16
            dtype = "int16"

        # load .dat file to device
        load_path = os.path.join(self.offload_folder, actual_tensor_name + ".dat")
        np_memmap = np.memmap(load_path, dtype=dtype, shape=shape, mode="r")
        value = torch.from_numpy(np_memmap)
        set_module_tensor_to_device(self.model, tensor_name, device, value)

    def tensor_offload(self, tensor_name):
        set_module_tensor_to_device(self.model, tensor_name, "meta")

    def layer_device_load(self, layer_name, device):
        logger.debug(f"load_layer: {self.model_name}.{layer_name} to {device}")
        layer_module = get_module_from_name(self.model, layer_name)
        weight_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, False, True)
        ]
        all_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, True, True)
        ]
        buffer_names = list(set(all_names) - set(weight_names))

        # backup
        if self._backups[layer_name] is None:
            backup = copy.deepcopy(layer_module)
            self._backups[layer_name] = backup

        # weights
        layer_dat_files = [
            os.path.join(self.offload_folder, self.get_tied_target(w) + ".dat")
            for w in weight_names
        ]
        assert all(
            [os.path.isfile(f) for f in layer_dat_files]
        ), f"dat file error, {self.dat_files}"

        for w in weight_names:
            self.tensor_device_load(w, device=device)

        # buffers
        for b in buffer_names:
            value = get_module_from_name(self.model, b)
            set_module_tensor_to_device(self.model, b, device, value)

    def layer_offload(self, layer_name):
        logger.debug(f"offload_layer: {self.model_name}.{layer_name} to meta\n\n")
        layer_module = get_module_from_name(self.model, layer_name)
        weight_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, False, True)
        ]
        all_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, True, True)
        ]
        buffer_names = list(set(all_names) - set(weight_names))

        # weights
        if self._backups[layer_name] is not None:
            set_module_from_name(self.model, layer_name, self._backups[layer_name])
            self._backups[layer_name] = None
            layer_module.to('meta')
        else:
            for w in weight_names:
                self.tensor_offload(w)

        # buffers
        for b in buffer_names:
            value = get_module_from_name(self.model, b)
            set_module_tensor_to_device(self.model, b, "cpu", value)

    def to_layer_offloading_forward(self, layer_name, device, recorder):
        layer = get_module_from_name(self.model, layer_name)
        layer._layer_offloading_forward = old_forward = layer.forward

        @functools.wraps(old_forward)
        def new_forward(*args, **kwargs):
            self.layer_device_load(layer_name, device)

            # record layer calling order
            recorder.append(layer_name)

            with torch.no_grad():
                output = old_forward(*args, **kwargs)

            self.layer_offload(layer_name)
            # torch.cuda.empty_cache()

            return output

        layer.forward = new_forward
        logger.debug(f"{layer_name} to layer-offloading forward")

    def reset_forward(self, layer_name):
        layer = get_module_from_name(self.model, layer_name)

        if hasattr(layer, "_layer_offloading_forward"):
            layer.forward = layer._layer_offloading_forward
            delattr(layer, "_layer_offloading_forward")
            logger.debug(f"{layer_name} from layer-offloading to old.")

    @contextlib.contextmanager
    def layer_offloading(self, device, recorder):
        """recording layer calling order by layer-offloading execution"""
        try:
            # unordered layer names
            layer_names = list(self.layers_dict.keys())

            # every layer to test forward
            for layer_name in layer_names:
                self.to_layer_offloading_forward(layer_name, device, recorder)

            yield

        finally:
            # every layer to old forward
            for layer_name in layer_names:
                self.reset_forward(layer_name)

    def test_run(self, device="cuda:0"):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # eos padding

        prompts = ["a"]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        recorder = []
        with self.layer_offloading(device, recorder):
            self.model.generate(inputs.input_ids, max_new_tokens=1)
        return recorder


class ModelPolicyLoader(MetaModel):
    def __init__(
        self,
        checkpoint: str,
        policy: Policy,
        weights_offload_dir: str = "weights_offload_dir",
        dtype=torch.float16,
    ):
        super().__init__(
            checkpoint=checkpoint,
            policy=policy,
            weights_offload_dir=weights_offload_dir,
            dtype=dtype,
        )
        self.init_all_weights()

    def load_module_tensor(self, tensor_name, device):
        tensor = get_module_from_name(self.model, tensor_name)
        if tensor.device == device:
            return

        actual_tensor_name = self.get_tied_target(tensor_name)

        metadata = self.index[actual_tensor_name]

        # copied from accelerate.utils.offload
        shape = tuple(metadata["shape"])
        if shape == ():
            # NumPy memory-mapped arrays can't have 0 dims so it was saved as 1d tensor
            shape = (1,)

        dtype = metadata["dtype"]
        if dtype == "bfloat16":
            # NumPy does not support bfloat16 so this was saved as a int16
            dtype = "int16"

        # load .dat file to device
        load_path = os.path.join(self.offload_folder, actual_tensor_name + ".dat")
        np_memmap = np.memmap(load_path, dtype=dtype, shape=shape, mode="r")
        value = torch.from_numpy(np_memmap)
        set_module_tensor_to_device(self.model, tensor_name, device, value)

    def offload_module_tensor(self, tensor_name):
        tensor = get_module_from_name(self.model, tensor_name)

        device = self.device_map[tensor_name]
        if device == "disk":
            device = "meta"
        device = torch.device(device)  # destination

        if tensor.device != device:
            set_module_tensor_to_device(self.model, tensor_name, device, tensor)

    def load_layer(self, layer_name, compute_device):
        layer_module = get_module_from_name(self.model, layer_name)

        # backup
        if self._backups[layer_name] is None:
            backup = copy.deepcopy(layer_module)
            self._backups[layer_name] = backup
        
        # logger.info(f'{layer_name}: {set(p.device for _, p in named_module_tensors(self._backups[layer_name], False, True))}')
        # input()

        weight_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, False, True)
        ]
        all_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, True, True)
        ]
        buffer_names = list(set(all_names) - set(weight_names))

        # weights
        layer_dat_files = [
            os.path.join(self.offload_folder, self.get_tied_target(w) + ".dat")
            for w in weight_names
        ]
        assert all(
            [os.path.isfile(f) for f in layer_dat_files]
        ), f"dat file error, {self.dat_files}"

        for w in weight_names:
            self.load_module_tensor(w, compute_device)

        # buffers
        for b in buffer_names:
            value = get_module_from_name(self.model, b)
            set_module_tensor_to_device(self.model, b, compute_device, value)


    def offload_layer(self, layer_name):
        layer_module = get_module_from_name(self.model, layer_name)
        weight_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, False, True)
        ]  
        all_names = [
            layer_name + "." + name
            for name, _ in named_module_tensors(layer_module, True, True)
        ]
        buffer_names = list(set(all_names) - set(weight_names))

        # weights
        if self._backups[layer_name] is not None:
            set_module_from_name(self.model, layer_name, self._backups[layer_name])
            # logger.info(f'{layer_name}: {set(p.device for _, p in named_module_tensors(self._backups[layer_name], False, True))}')
            self._backups[layer_name] = None
            layer_module.to('meta')
            # input()
        else:
            for w in weight_names:
                self.offload_module_tensor(w)

        # buffers
        for b in buffer_names:
            value = get_module_from_name(self.model, b)
            set_module_tensor_to_device(self.model, b, "cpu", value)

    def init_all_weights(self):
        # load weights
        logger.debug("init all weights...")
        for tensor_name, device in tqdm(
            self.device_map.items(), desc="model init: loading by policy..."
        ):
            if device != "disk":
                self.load_module_tensor(tensor_name, device)

    def __del__(self):
        if hasattr(self, "device_map"):
            for tensor_name, _ in tqdm(self.device_map.items()):
                self.load_module_tensor(tensor_name, "meta")


if __name__ == "__main__":
    checkpoint = "facebook/opt-125m"  # 125m 6.7b 13b 30b
    # checkpoint = "Salesforce/codegen-350M-mono"
    # checkpoint = 'bigscience/bloom-560m' #

    policy = Policy(
        gpu_batch_size=1,
        num_gpu_batches=4,
        weights_gpu_percent=0.0,
        weights_cpu_percent=0.3,
        cache_gpu_percent=0.0,
        cache_cpu_percent=0.2,
        act_gpu_percent=0.0,
        act_cpu_percent=0.5,
        overlap=True,
        pin_weight=True,
    )

    m = MetaModel(checkpoint, policy)
    print(m.layer_names)

    ###############

    mpl = ModelPolicyLoader(checkpoint, policy)
    mpl.init_all_weights()
    print(mpl.layer_names)
