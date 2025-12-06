#!/usr/bin/env python3
__author__ = "Gustaf Gren"
__doc__ = """written in bash cli style, to allow both local and SLURM etc"""

import argparse

from datetime import datetime
from typing import List
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset


def build_segments_token_chunks(
    tokenizer,
    lines: List[str],
    chunk_len: int,
    sep_text: str,
    pre_token_id: int | None = None,
) -> List:
    """
    Concatenate all lines, inserting sep_text at the end of each line, tokenize
    once, then split into chunks of exactly 'chunk_len' tokens.

    We pad the last chunk if needed.
    """

    big_text_parts = []
    for line in lines:
        big_text_parts.append(line)
        big_text_parts.append(sep_text)
    big_text = "".join(big_text_parts)
    tok = tokenizer(
        [big_text],
        add_special_tokens=False,
        return_offsets_mapping=False,
    )
    all_ids = tok["input_ids"][0]

    pre = pre_token_id if pre_token_id else tokenizer.cls_token_id
    pid = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    segments: List = []
    n = len(all_ids)
    i = 0
    while i < n:
        if (i + (chunk_len - 1)) > n:
            j = n
            seg_text = [pre] + all_ids[i:j] + [pid] * ((i + chunk_len - 1) - n)
            attention_mask = [1] * (j - i) + [0] * ((i + chunk_len) - n)
            print(len(seg_text), seg_text)
            print(len(attention_mask), attention_mask)
        else:
            j = i + chunk_len - 1
            seg_text = [pre] + all_ids[i:j]
            attention_mask = [1] * chunk_len
        segments.append(
            {
                "input_ids": seg_text,
                "attention_mask": attention_mask,
                "labels": seg_text,
            }
        )
        i = j
    return segments


def finetune(model_spec: str, output_dir: Path, data: dict, train_args: dict):
    """finetunes a model, given data containing lines of text, and saves to output_dir"""
    tokenizer = AutoTokenizer.from_pretrained(model_spec)
    model = AutoModelForCausalLM.from_pretrained(model_spec)
    print(f"loading dataset from {str(data)}")
    lines_train = []
    lines_valid = []
    for file in data["train"]:
        lines_train += [x for x in Path(file).read_text().strip().split("\n")]
    for file in data["valid"]:
        lines_valid += [x for x in Path(file).read_text().strip().split("\n")]

    segments_train = build_segments_token_chunks(
        tokenizer=tokenizer,
        lines=lines_train,
        chunk_len=model.config.max_position_embeddings,
        sep_text=tokenizer.sep_token,
    )
    segments_valid = build_segments_token_chunks(
        tokenizer=tokenizer,
        lines=lines_valid,
        chunk_len=model.config.max_position_embeddings,
        sep_text=tokenizer.sep_token,
    )
    train_dataset = Dataset.from_list(segments_train)
    valid_dataset = Dataset.from_list(segments_valid)
    del segments_train, lines_train
    del segments_valid, lines_valid

    finetune_log_dir = output_dir / "finetune_training_log"
    finetune_log_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        logging_dir=str(finetune_log_dir),
        push_to_hub=False,
        # we skip saving checkpoints since we're running experiments not
        # choosing best checkpoints
        save_strategy="no",
        save_steps=9999999999999,
        output_dir=str(output_dir),
        learning_rate=train_args["learning_rate"],
        weight_decay=train_args["weight_decay"],
        per_device_train_batch_size=train_args["batch_size"],
        # we train using steps instead of epochs since we might vary the
        # training dataset size in our experiments (multi/mono-lingual)
        num_train_epochs=0,
        max_steps=train_args["steps"],
        seed=42,  # meaning of life
    )
    print(args)

    print(f"initializing trainer for {model}")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    print(f"starting finetuning for {model}")
    trainer.train()

    print(f"finished training, saving to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # in case where we are dealing with a model which has had tokenizer
    # transfer, we should update the transfer_info.json to include the
    # finetuning parameter (mostly for clarity)
    if (transfer_info_fp := Path(model_spec) / "transfer_info.json").exists():
        transfer_info = json.loads(transfer_info_fp.read_text())
        transfer_info["finetuned"] = True
        with open(output_dir / "transfer_info.json", "w") as f:
            json.dump(transfer_info, f)


def fmt_base(base_model: str, tgt_lang: str) -> str:
    return f"{base_model.split('/')[-1]}_{tgt_lang}_finetuned"


def fmt_data(transfer_info: dict, combine_data: bool, tgt: bool = False) -> dict:
    if combine_data and tgt:
        return {
            "train": [
                transfer_info["tgt_finetune_data"]["train"],
                transfer_info["src_finetune_data"]["train"],
            ],
            "valid": [
                transfer_info["tgt_finetune_data"]["valid"],
                transfer_info["src_finetune_data"]["valid"],
            ],
        }
    elif tgt:
        return {
            "train": [
                transfer_info["src_finetune_data"]["train"],
            ],
            "valid": [
                transfer_info["src_finetune_data"]["valid"],
            ],
        }
    if combine_data:
        return {
            "train": [
                transfer_info["src_finetune_data"]["train"],
                transfer_info["tgt_finetune_data"]["train"],
            ],
            "valid": [
                transfer_info["src_finetune_data"]["valid"],
                transfer_info["tgt_finetune_data"]["valid"],
            ],
        }
    else:
        return {
            "train": [
                transfer_info["tgt_finetune_data"]["train"],
            ],
            "valid": [
                transfer_info["tgt_finetune_data"]["valid"],
            ],
        }


def main(
    src_lang: str,
    model_dir: Path,
    output_dir: Path,
    train_args: dict,
    combine_data: bool = False,
    overwrite: bool = False,
):
    # will contain tuples corresponding to args in finetune, i.e. each model
    # for this src_lang we have to finetune
    training_pairs = []

    # first we need all possible unique combinations of models with that src_lang
    pairs = []
    for model in [x for x in model_dir.iterdir() if x.is_dir()]:
        if not model / "transfer_info.json":
            print(f"WARN: {str(model)} does not contain transfer info. skipping ...")
            continue
        info = json.loads((model / "transfer_info.json").read_text())
        if info["src_lang"] != src_lang:
            continue
        pairs.append((str(model), info))

    # we create base pairs for all base goldfish models, 100mb and 10mb
    base_pairs = {}
    # since the src model does not differ between transplanted models, we just
    # use the first one
    src_model = pairs[0][1]["src_model"]
    for _, i in pairs:
        out = fmt_base(src_model, i["tgt_lang"])
        if out in base_pairs:
            continue
        if (output_dir / out).exists() and not overwrite:
            print(f"{str(out)} already exists and overwrite not specified. skipping...")
            continue
        data = fmt_data(i, combine_data)
        base_pairs[out] = (src_model, output_dir / out, data, train_args)

    for _, i in pairs:
        tgt_model = i["tgt_model"]
        out = fmt_base(tgt_model, i["src_lang"])
        if out in base_pairs:
            continue
        if (output_dir / out).exists() and not overwrite:
            print(f"{str(out)} already exists and overwrite not specified. skipping...")
            continue
        data = fmt_data(i, combine_data, tgt=True)
        base_pairs[out] = (tgt_model, output_dir / out, data, train_args)
    training_pairs += [x for x in base_pairs.values()]

    # next, we iterate over pairs to create training arguments for finetune function
    for model_spec, i in pairs:
        if len(model_spec.split("/")):
            out = output_dir / f"{Path(model_spec).name}_finetuned"
        else:
            out = output_dir / f"{model_spec}_finetuned"
        if out.exists() and not overwrite:
            print(f"{str(out)} already exists and overwrite not specified. skipping...")
            continue
        data = fmt_data(i, combine_data)
        training_pairs.append((model_spec, out, data, train_args))

    for tp in training_pairs:
        print(f"{datetime.now()} STARTING FINETUNING OF {tp[0]}")
        finetune(*tp)
        print(f"{datetime.now()} FINISHED FINETUNING OF {tp[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_lang", type=str, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--training_args", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--combine_data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    main(
        src_lang=args.src_lang,
        model_dir=args.model_dir,
        train_args=json.loads(args.training_args.read_text()),
        output_dir=args.output_dir,
        combine_data=args.combine_data,
    )
