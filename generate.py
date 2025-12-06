#!/usr/bin/env python3
import argparse
import torch
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np


def main(args):
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    input_ids = tokenizer.encode(args.seed_text, return_tensors="pt")[:, :-1]
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=1,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(f"== {args.model}")
    wrapped = textwrap.fill(
        generated_text,
        width=80,
        initial_indent=" " * 4,
        subsequent_indent=" " * 4,
    )
    print(wrapped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from a Hugging Face model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name from Hugging Face Hub",
    )
    parser.add_argument(
        "--seed_text",
        type=str,
        required=True,
        help="Seed text to start generation",
    )
    args = parser.parse_args()
    main(args)
