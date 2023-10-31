# generate test
from transformers import AutoTokenizer
from utils import logging


def test_hf_gen(checkpoint, model, num_prompts, compute_device, gen_len=30, prompts=None):
    # test .generate() for huggingface CausalLM models.

    # prompts
    if prompts is None:  # get default prompts
        prompts = [
            "for i in range(10): ",
            "Who are you? Are you conscious?",
            "Where is Deutschland?",
            "How is Huawei Mate 60 Pro?",
        ]
    prompts = (
        prompts * (num_prompts // len(prompts))
        + prompts[: (num_prompts % len(prompts))]
    )

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # eos padding

    # inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(compute_device)

    # generate
    generate_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=gen_len,  # max_lengths
        # num_beams=2, #
        # num_beam_groups=2, #
        # diversity_penalty=0.1, #
        # do_sample=True, #
    )

    # outputs
    output_texts = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    for output_text in output_texts:
        logging.info(output_text)
        logging.info("-" * 10)
