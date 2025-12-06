#!/usr/bin/env python3
__doc__ = "simple data sampling script using byte premiums"
import argparse
from pathlib import Path
import time

import yaml


def sample_dataset(input_dir: Path, output_dir: Path, byte_premiums: Path):
    """100MB/10MB split of text&byte files from input_dir to output_dir."""
    bp = yaml.safe_load(byte_premiums.read_text())
    for file in (x for x in input_dir.iterdir() if x.suffix == ".txt"):
        start = time.time()
        train_max_size = 1e7 * bp[file.stem]  # 10mb scaled by byte premiums
        valid_max_size = 1e6 * bp[file.stem]  # 1mb scaled by byte premiums
        train_byte_size = 0
        valid_byte_size = 0
        train_out = (output_dir / "train") / f"{file.stem}.txt"
        valid_out = (output_dir / "valid") / f"{file.stem}.txt"
        train_out.parent.mkdir(parents=True, exist_ok=True)
        valid_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"--- reading {file}, sampling {train_max_size / 1024**2:.2f}MB")
        with (
            open(file, "r") as inp,
            open(train_out, "w") as train_out,
            open(valid_out, "w") as valid_out,
        ):
            for line in inp:
                line = line.strip()  # since we don't want \n in byte calculations
                if train_byte_size <= train_max_size:
                    train_byte_size += len(line.encode("utf-8"))
                    train_out.write(line + "\n")
                elif valid_byte_size <= valid_max_size:
                    valid_byte_size += len(line.encode("utf-8"))
                    valid_out.write(line + "\n")
        print(f"sampled {train_byte_size / 1024**2:.2f}MB for train")
        print(f"sampled {valid_byte_size / 1024**2:.2f}MB for valid")
        print(f"finished processing {file}. took {int(time.time() - start)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path, help="Goldfish data dir")
    parser.add_argument("byte_premium", type=Path, help="path to byte premium yaml")
    parser.add_argument("output_dir", type=Path, help="dir for sampled files")
    args = parser.parse_args()
    assert args.input_dir.exists()
    assert args.byte_premium.exists() and args.byte_premium.suffix == ".yaml"
    sample_dataset(args.input_dir, args.output_dir, args.byte_premium)
