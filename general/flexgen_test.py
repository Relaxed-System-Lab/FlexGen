# generate test
from transformers import AutoTokenizer
from flexgen_utils import logging

def test_hf_gen(checkpoint, model, gbs, ngb, prompt_len=10, gen_len=30, prompts=None):
    # test .generate() for huggingface CausalLM models.

    if prompts is None: # get default prompts
        prompts = [
            'for i in range(10): ',
            'Who are you? Are you conscious?',
            'Where is Deutschland?',
            'How is Huawei Mate 60 Pro?'
        ] 
    prompts = prompts * (gbs * ngb // len(prompts)) + prompts[:(gbs * ngb % len(prompts))]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # eos padding 
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    # Generate
    generate_ids = model.generate(
        inputs.input_ids, 
        max_length=prompt_len + gen_len, # ?
        # num_beams=2, #
        # num_beam_groups=2, #
        # diversity_penalty=0.1, #
        # do_sample=True, #
    )

    output_texts = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for output_text in output_texts:
        logging.info(output_text)
        logging.info('-' * 10)