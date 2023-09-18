from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, snapshot_download


checkpoint = "facebook/opt-6.7b"
# checkpoint = "facebook/opt-30b"
# checkpoint = 'intlsy/opt-175b-hyperparam'
# checkpoint = "facebook/opt-125m"
config = AutoConfig.from_pretrained(checkpoint)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# model.tie_weights()

# Download the Weights
weights_location = snapshot_download(checkpoint, allow_patterns=["*.bin", 'pytorch_model.bin.index.json'])#hf_hub_download(checkpoint, "*.bin")

# m = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", offload_folder='./offload_folder')
model = load_checkpoint_and_dispatch(
    model, device_map="auto",
    offload_folder='./offload_folder',
    checkpoint=weights_location
)

print(model.hf_device_map)