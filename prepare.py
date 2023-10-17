import json
import argparse

from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_config_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--subsample_size_mb", type=int, default=1024)
    parser.add_argument("--valid_percentage", type=int, default=10)
    args = parser.parse_args()
    return args


def prepare_dataset():
    args = parse_args()
    # Adapted from https://github.com/CPJKU/wechsel/blob/main/legacy/prepare.py
    subsample_size = 1024 * 1024 * args.subsample_size_mb  # in bytes

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset(
        args.dataset_name, args.dataset_config_name, split="train", streaming=True
    )
    dataset_iter = iter(dataset)

    with open(output_dir / "train.json", "w") as f:
        size = 0
        bar = tqdm(total=subsample_size)

        while size < subsample_size:
            entry = next(dataset_iter)

            entry_size = len(entry["text"].encode("utf-8"))
            size += entry_size

            bar.update(entry_size)

            f.write(f"{json.dumps(entry)}\n")

    with open(output_dir / "valid.json", "w") as f:
        size = 0
        bar = tqdm(total=subsample_size * args.valid_percent)

        while size < subsample_size * args.valid_percent:
            entry = next(dataset_iter)

            entry_size = len(entry["text"].encode("utf-8"))
            size += entry_size

            bar.update(entry_size)

            f.write(f"{json.dumps(entry)}\n")


if __name__ == "__main__":
    prepare_dataset()
