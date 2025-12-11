#!/usr/bin/env python3
__author__ = "Gustaf Gren"
__doc__ = """
    tokenization transfer evaluation helper functions and pipeline
    logic. adapted from original goldfish/multiblimp scripts.
"""
import argparse
import logging
import json
from pathlib import Path

import codecs
from datasets import load_dataset
import numpy as np
from minicons import scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import pandas as pd

from pprint import pprint

from finetune import fmt_base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("run.log")],
)


def ppl(
    data: Path,
    model_spec: str,
    device: torch.device = torch.device("cpu"),
    only_second_half: bool = True,
) -> list[float]:
    """calculates perplexity for each line in flores, returns them as a list of floats"""
    model = AutoModelForCausalLM.from_pretrained(model_spec).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_spec)
    max_seq_len = model.config.max_position_embeddings

    with codecs.open(data, "rb", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    pid = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    uid = tokenizer.unk_token_id
    spe_id = tokenizer.convert_tokens_to_ids("▁")
    # TODO: fix to include model specific bos token behavior for evaluation on other models
    prepend_token_id = tokenizer.cls_token_id

    loss = nn.CrossEntropyLoss(ignore_index=pid, reduction="none")
    surprisals, norm_surprisals, uid_props = [], [], []
    for line in lines:
        inputs = tokenizer([line], add_special_tokens=False)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        # NOTE: why is this done? bos thing?
        input_ids[0].insert(0, prepend_token_id)
        attention_mask[0].insert(0, 1)
        if len(input_ids[0]) > max_seq_len:
            input_ids[0] = input_ids[0][:max_seq_len]
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(input_ids.clone().detach()).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs["logits"].detach()
        del outputs

        labels = input_ids[:, 1:]  # (n_examples, seq_len-1)

        logits = logits[:, :-1, :]
        logits = torch.transpose(logits, 1, 2)
        losses = loss(logits, labels).cpu()
        losses = losses * np.log2(np.e)
        # NOTE: for some reason my spider sense is tingling here
        losses[labels == uid] = np.log2(tokenizer.vocab_size)

        # we calculate proportion of unknown tokens
        mask = (labels == uid) & (labels != spe_id)
        non_underscore_count = (labels != spe_id).sum().item()
        uid_props.append(mask.sum().item() / non_underscore_count)

        # goldfish authors explain this by saying the first half of the lines
        # in multilingual LMs are usually concerned with language identification
        if only_second_half:
            halfline = line[: (len(line) // 2)]
            halfline_len_tokens = len(
                tokenizer([halfline], add_special_tokens=False)["input_ids"][0]
            )
            losses[0, :halfline_len_tokens] = 0.0
        surprisal = torch.sum(losses, dim=-1).item()
        surprisals.append(surprisal)
        if only_second_half:
            byte_count = len(line[(len(line) // 2) :].encode("utf-8"))
        else:
            byte_count = len(line.encode("utf-8"))
        norm_surprisals.append(surprisal / byte_count)
    return (
        float(np.array(surprisals).flatten().mean()),
        float(np.array(norm_surprisals).flatten().mean()),
        float(np.array(uid_props).flatten().mean()),
    )


def belebele(
    data: Path,
    model_spec: str,
    device: torch.device = torch.device("cpu"),
    max_seq_len: int = 512,
) -> float:
    model = AutoModelForCausalLM.from_pretrained(model_spec).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_spec)

    dataset = []
    infile = codecs.open(data, "rb", encoding="utf-8")
    for line in infile:
        dataset.append(json.loads(line.strip()))
    infile.close()
    assert len(dataset) == 900

    # TODO: fix to include model specific bos token behavior
    prepend_token_id = tokenizer.cls_token_id
    pid = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    corr = []
    loss = nn.CrossEntropyLoss(ignore_index=pid, reduction="none")

    for r in dataset:
        prefix = r["flores_passage"].strip() + " " + r["question"].strip() + " "
        mc_texts = []
        for ans_i in range(4):
            mc_texts.append(prefix + r[f"mc_answer{ans_i+1}"].strip())
        corr_i = int(r["correct_answer_num"]) - 1

        option_losses = torch.zeros(len(mc_texts))
        for mc_i, mc_text in enumerate(mc_texts):
            inputs = tokenizer([mc_text], add_special_tokens=False)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            input_ids[0].insert(0, prepend_token_id)
            attention_mask[0].insert(0, 1)  # NOTE: not really sure what this does?
            if len(input_ids[0]) > max_seq_len:
                input_ids[0] = input_ids[0][:max_seq_len]
                attention_mask[0] = attention_mask[0][:max_seq_len]
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"].detach()
            del outputs
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            logits = torch.transpose(logits, 1, 2)
            losses = loss(logits, labels).cpu()
            option_loss = torch.sum(losses, dim=-1).item()
            option_losses[mc_i] = option_loss

        pred_i = int(torch.argmin(option_losses).item())
        corr.append(pred_i == corr_i)
    return float(np.mean(np.array(corr)))


def token_compression(tokenizer_spec: str, dataset: str | Path) -> float:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    total_bytes, total_tokens = 0, 0
    with open(dataset) as f:
        for line in (x.strip() for x in f):
            total_bytes += len(line.encode("utf-8"))
            total_tokens += len(tokenizer([line])["input_ids"][0])
    return total_bytes / total_tokens


def multiblimp(model_spec: str, test_lang: str, device: str = "cuda") -> float:
    # load model
    model = AutoModelForCausalLM.from_pretrained(model_spec)
    tokenizer = AutoTokenizer.from_pretrained(model_spec)
    ilm_model = scorer.IncrementalLMScorer(model, device, tokenizer=tokenizer)

    # load dataset
    dataset_name = "jumelet/multiblimp"
    dataset = load_dataset(dataset_name, test_lang)["train"].to_pandas()

    # score model
    def score_pair(sen, wrong_sen):
        sen_len = len(ilm_model.tokenizer.tokenize(sen))
        wrong_sen_len = len(ilm_model.tokenizer.tokenize(wrong_sen))
        if (max_length is not None) and (
            (sen_len >= max_length) or (wrong_sen_len >= max_length)
        ):
            return 0.0, 0.0
        stimuli = [sen, wrong_sen]
        return ilm_model.sequence_score(stimuli, reduction=lambda x: x)

    dataset["sen_prob"] = pd.Series(dtype=object).astype(object)
    dataset["wrong_prob"] = pd.Series(dtype=object).astype(object)
    max_length = ilm_model.model.transformer.config.n_ctx
    for idx, row in dataset.iterrows():
        sen_prob, wrong_prob = score_pair(row.sen, row.wrong_sen)
        sen_nll = -sen_prob.sum().item()
        wrong_nll = -wrong_prob.sum().item()
        dataset.at[idx, "sen_prob"] = sen_prob.tolist()
        dataset.at[idx, "wrong_prob"] = wrong_prob.tolist()
        dataset.loc[idx, "sen_nll"] = sen_nll
        dataset.loc[idx, "wrong_nll"] = wrong_nll
        dataset.loc[idx, "delta"] = wrong_nll - sen_nll
    return float(np.mean(dataset["delta"] > 0))


def main(transfered_model_dir: Path, config: dict) -> dict:
    models = [x for x in transfered_model_dir.iterdir() if x.is_dir()]
    flores_dir = Path(config["paths"]["flores_dir"])
    device = torch.device(config["trainargs"].get("device", "cpu"))

    # no defaultdict here since we have to serialize it later anyway
    export_metrics = {}
    for model_dir in models:
        if not (model_dir / "transfer_info.json").exists():
            continue
        with open(model_dir / "transfer_info.json") as f:
            info = json.load(f)
            src_lang = info["src_lang"]
            tgt_lang = info["tgt_lang"]
        if src_lang not in export_metrics:
            export_metrics[src_lang] = {}
        if tgt_lang not in export_metrics[src_lang]:
            export_metrics[src_lang][tgt_lang] = {
                "fvt": {},
                "omp": {},
                "100mb_src": {},
                "10mb_src": {},
                "100mb_tgt": {},
                "10mb_tgt": {},
            }

    def evaluate_model(model_spec: str, src_lang: str, tgt_lang: str, label: str):
        logging.info(
            f"starting evaluation of {model_spec} ({src_lang=}, {tgt_lang=}, {label=})"
        )
        src_data = str(flores_dir / f"{src_lang}.devtest")
        tgt_data = str(flores_dir / f"{tgt_lang}.devtest")

        # -- PERPLEXITY --
        logging.info(
            f"calculating perplexity of {model_spec} ({src_lang=}, {tgt_lang=}, {label=})"
        )
        srcp, srcp_norm, src_uid_prop = ppl(
            data=src_data,
            model_spec=model_spec,
            device=device,
        )
        tgtp, tgtp_norm, tgt_uid_prop = ppl(
            data=tgt_data,
            model_spec=model_spec,
            device=device,
        )
        norm_sum = srcp_norm + tgtp_norm

        # -- TOKEN COMPRESSION --
        logging.info(
            f"calculating token compression of {model_spec} ({src_lang=}, {tgt_lang=}, {label=})"
        )
        src_bpt = token_compression(model_spec, src_data)
        tgt_bpt = token_compression(model_spec, tgt_data)

        # -- MULTIBLIMP --
        logging.info(
            f"calculating multiblimp for {model_spec} ({src_lang=}, {tgt_lang=}, {label=})"
        )
        src_mbp = multiblimp(model_spec, src_lang[:3])
        tgt_mbp = multiblimp(model_spec, tgt_lang[:3])

        export_metrics[src_lang][tgt_lang][label]["src_log_ppl"] = srcp
        export_metrics[src_lang][tgt_lang][label]["tgt_log_ppl"] = tgtp
        export_metrics[src_lang][tgt_lang][label]["sum_log_ppl"] = srcp + tgtp
        export_metrics[src_lang][tgt_lang][label]["src_norm_ppl"] = srcp_norm
        export_metrics[src_lang][tgt_lang][label]["tgt_norm_ppl"] = tgtp_norm
        export_metrics[src_lang][tgt_lang][label]["sum_norm_ppl"] = norm_sum

        export_metrics[src_lang][tgt_lang][label]["tgt_uid_prop"] = tgt_uid_prop
        export_metrics[src_lang][tgt_lang][label]["src_uid_prop"] = src_uid_prop

        export_metrics[src_lang][tgt_lang][label]["src_bpt"] = src_bpt
        export_metrics[src_lang][tgt_lang][label]["tgt_bpt"] = tgt_bpt

        export_metrics[src_lang][tgt_lang][label]["src_mbp"] = src_mbp
        export_metrics[src_lang][tgt_lang][label]["tgt_mbp"] = tgt_mbp
        export_metrics[src_lang][tgt_lang][label]["sum_mbp"] = src_mbp + tgt_mbp

        logging.info(f"finished evaluating {model_spec}")
        logging.info(f"\t{srcp_norm=:.2f} {srcp=:.2f}")
        logging.info(f"\t{tgtp_norm=:.2f} {tgtp=:.2f}")
        logging.info(f"\t{src_mbp=:.2f} {tgt_mbp=:.2f}")
        logging.info(f"\t{src_uid_prop=:.2f} {tgt_uid_prop=:.2f}")
        logging.info(f"\t{src_bpt=:.2f} {tgt_bpt=:.2f}")

    for model_dir in tqdm(models):
        if not (model_dir / "transfer_info.json").exists():
            continue
        with open(model_dir / "transfer_info.json") as f:
            info = json.load(f)
            src_lang = info["src_lang"]
            tgt_lang = info["tgt_lang"]
            src_model = info["src_model"]
            tgt_model = info["tgt_model"]
            method = info["method"]
            finetuned = info["finetuned"]

        # --- 100MB SRC BASELINE ---
        # since we only have to do these once for src/tgt w/o transfer strategy
        if not export_metrics[src_lang][tgt_lang]["100mb_src"]:
            # we need this since a finetuned model is on disk, while the
            # non-finetuned is a model spec for a HF model
            if finetuned:
                base_model = model_dir.parent / fmt_base(src_model, tgt_lang)
                if not base_model.exists():
                    raise FileNotFoundError(
                        f"finetuned model requires finetuned base model. Not found: {base_model}"
                    )
            else:
                base_model = src_model
            evaluate_model(
                model_spec=base_model,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                label="100mb_src",
            )

        """
        # --- 100MB TARGET BASELINE ---
        # NOTE: this is a bit confusing to read since k/v are switched but trust me bro
        # NOTE: this is never finetuned since that wouldn't make much sense
        # NOTE: would duplicates of this metric mess up median/std?
        if not export_metrics[tgt_lang][src_lang]["100mb_tgt"]:
            evaluate_model(
                model_spec=src_model,
                src_lang=tgt_lang,
                tgt_lang=src_lang,
                label="100mb_tgt",
            )
        """

        """
        # --- 10MB SRC BASELINE ---
        if not export_metrics[tgt_lang][src_lang]["10mb_src"]:
            evaluate_model(
                model_spec=tgt_model,
                src_lang=tgt_lang,
                tgt_lang=src_lang,
                label="10mb_src",
            )
        """

        # --- 10MB TGT BASELINE ---
        if not export_metrics[src_lang][tgt_lang]["10mb_tgt"]:
            evaluate_model(
                model_spec=tgt_model,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                label="10mb_tgt",
            )

        # --- TRANSFER MODEL EVALUATION ---
        evaluate_model(
            model_spec=model_dir,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            label=method,
        )
    return export_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    parser.add_argument("input_dir", type=Path, required=True)
    parser.add_argument("export_json", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config_file.read_text())
    metrics, mean_metrics = main(args.input_dir, config)

    if args.export_json:
        with open(args.export_json, "w") as f:
            args.export_json.parent.mkdir(exist_ok=True)
            json.dump(metrics, f)
        with open(
            args.export_json.with_name(args.export_json.stem + "_mean.json"), "w"
        ) as f:
            json.dump(mean_metrics, f)
    else:
        pprint(mean_metrics)
