#!/usr/bin/env python3
__author__ = "Gustaf Gren"
__doc__ = "tokenization transfer strategies helper functions and pipeline logic"

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm
from click.testing import CliRunner
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yaml

from utils import create_mappings

from fvt.fvt import FastVocabularyTransfer
from mergekit.scripts.tokensurgeon import main as tokensurgeon


@dataclass
class Pair:
    tgt_lang: str
    tgt_model: str
    tgt_finetune_data: dict[str]
    src_lang: str
    src_model: str
    src_finetune_data: dict[str]
    output_dir: Path


def fmt_model_out(pair: Pair, method: str) -> Path:
    """given pair and method returns Path with formatted output model path"""
    return pair.output_dir / f"{pair.src_lang}-{pair.tgt_lang}-{method}"


def export_transfer_info(pair, method) -> None:
    info = {
        "src_lang": pair.src_lang,
        "tgt_lang": pair.tgt_lang,
        "src_model": pair.src_model,
        "tgt_model": pair.tgt_model,
        "src_finetune_data": pair.src_finetune_data,
        "tgt_finetune_data": pair.tgt_finetune_data,
        "method": method,
        "finetuned": False,
    }
    with open(Path(str(fmt_model_out(pair, method))) / "transfer_info.json", "w") as f:
        json.dump(info, f)


def fvt(pair: Pair):
    """perform tokenization transfer using fast vocabulary transfer"""
    src_tokenizer = AutoTokenizer.from_pretrained(pair.src_model)
    src_model = AutoModelForSequenceClassification.from_pretrained(pair.src_model)
    tgt_tokenizer = AutoTokenizer.from_pretrained(pair.tgt_model)

    # named like this simply because I don't want to deal with variable scope,
    # my mind is tired enough as it is
    strategy = FastVocabularyTransfer()
    transfered_model = strategy.transfer(
        in_tokenizer=tgt_tokenizer,
        gen_tokenizer=src_tokenizer,
        gen_model=src_model,
    )
    transfered_model.save_pretrained(fmt_model_out(pair, "fvt"))
    tgt_tokenizer.save_pretrained(fmt_model_out(pair, "fvt"))


def main(config: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    mappings = create_mappings(config)
    run = CliRunner()
    for i, (tgt_lang, tgt_map) in enumerate(mappings.items()):
        pairs = [
            Pair(
                tgt_lang,
                tgt_map["target"],
                tgt_map["finetune_data"],
                src_lang,
                src_map["source"],
                src_map["finetune_data"],
                output_dir,
            )
            for src_lang, src_map in mappings.items()
            if src_lang != tgt_lang
        ]
        for pair in tqdm(pairs, desc=f"{i+1}/{len(mappings)} {tgt_lang}"):
            fvt(pair)
            export_transfer_info(pair, "fvt")

            # couldn't get the python API working... just calling the cli instead
            run.invoke(
                tokensurgeon,
                [
                    pair.src_model,
                    pair.tgt_model,
                    str(fmt_model_out(pair, "omp")),
                    "--approximation-method",
                    "omp",
                    "--k",
                    64,
                    "--cuda",
                    "--random-seed",
                    42,
                ],
            )
            export_transfer_info(pair, "omp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config_file.read_text())
    main(config, args.output_dir)
