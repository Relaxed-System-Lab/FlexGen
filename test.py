from transformers import AutoTokenizer, OPTForCausalLM, BloomForCausalLM, CodeGenForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

prompts = [
    'Who are you? Are you conscious?',
    'Where is Deutschland?',
    'How is Huawei Mate 60 Pro?'
] * 4

prompt_len = max(max([len(p) for p in prompts]), 16)

inputs = tokenizer(prompts, padding="max_length", max_length=prompt_len, return_tensors="pt")

# Generate
generate_ids = model.generate(
    inputs.input_ids, 
    max_length=30 + prompt_len,
    # num_beams=2,
    # num_beam_groups=2,
    # diversity_penalty=0.1,
    do_sample=True,
)

output_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
for output_text in output_texts:
    print(output_text)
    print('-' * 10)