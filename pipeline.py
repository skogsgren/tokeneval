import argparse
import json
from pathlib import Path
import logging
import time
from datetime import timedelta

import yaml

from transfer import main as tokenization_transfer
from finetune import main as finetune_model_dir
from evaluate import main as evaluate_model_dir


def main(
    config: dict,
    transfer: bool,
    finetune: bool,
    evaluate: bool,
    overwrite: bool = False,
):
    start = time.time()
    n_lang = len(config["languages"])

    transfer_out = Path(config["paths"]["tokenization_transfer_out"])
    mono_out = Path(config["paths"]["mono_finetuned_out"])
    mixed_out = Path(config["paths"]["mixed_finetuned_out"])

    logging.info(" ### START OF RUN")
    logging.info(f"CONFIG: {config}")

    if transfer:
        logging.info(f"starting tokenization transfer for {n_lang} languages")
        tokenization_transfer(config, transfer_out)
        logging.info(f"finished tokenization transfer for {n_lang} languages")

    if finetune:
        logging.info(f"finetuning using monolingual target data for {n_lang} languages")
        for lang in config["languages"]:
            finetune_model_dir(
                src_lang=lang,
                model_dir=transfer_out,
                output_dir=mono_out,
                train_args=config["trainargs"],
                combine_data=False,
            )
        logging.info(
            f"finished finetuning using mono target data for {n_lang} languages"
        )

        logging.info(f"finetuning using mixed target data for {n_lang} languages")
        for lang in config["languages"]:
            finetune_model_dir(
                src_lang=lang,
                model_dir=transfer_out,
                output_dir=mixed_out,
                train_args=config["trainargs"],
                combine_data=True,
            )
        logging.info(
            f"finished finetuning using mixed target data for {n_lang} languages"
        )

    if evaluate:
        logging.info(f"evaluating non-finetuned models for {n_lang} languages")
        metrics_transfer = evaluate_model_dir(transfer_out, config)
        metrics_transfer_fn = transfer_out / "metrics.json"
        with open(metrics_transfer_fn, "w") as f:
            json.dump(metrics_transfer, f)
        logging.info(f"finished evaluating non-finetuned models for {n_lang} languages")

        logging.info(f"evaluating finetuned mono models for {n_lang} languages")
        metrics_mono = evaluate_model_dir(mono_out, config)
        metrics_mono_fn = mono_out / "metrics.json"
        with open(metrics_mono_fn, "w") as f:
            json.dump(metrics_mono, f)
        logging.info(
            f"finished evaluating finetuned mono models for {n_lang} languages"
        )

        logging.info(f"evaluating finetuned mixed models for {n_lang} languages")
        metrics_mixed = evaluate_model_dir(mixed_out, config)
        metrics_mixed_fn = mixed_out / "metrics.json"
        with open(metrics_mixed_fn, "w") as f:
            json.dump(metrics_mixed, f)
        logging.info(
            f"finished evaluating finetuned mixed models for {n_lang} languages"
        )

    logging.info(" ### END OF RUN")
    logging.info(f" ### {timedelta(seconds=time.time() - start)} ELAPSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler("run.log")],
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml", type=Path, help="path to configuration yaml")
    parser.add_argument("--no_finetune", action="store_false")
    parser.add_argument("--no_transfer", action="store_false")
    parser.add_argument("--no_eval", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(
        yaml.safe_load(args.config_yaml.read_text()),
        args.no_transfer,
        args.no_finetune,
        args.no_eval,
        args.overwrite,
    )
