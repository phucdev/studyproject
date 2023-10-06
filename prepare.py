import json
import fire
import sys

from datasets import load_dataset
from tqdm.auto import tqdm
from pathlib import Path


def prepare_dataset(
    dataset_name: str,
    dataset_config_name: str,
    output_dir: str,
    subsample_size_mb: int = 1024,
    valid_percent = 0.1
):
    # Adapted from https://github.com/CPJKU/wechsel/blob/main/legacy/prepare.py
    subsample_size = 1024 * 1024 * subsample_size_mb  # in bytes

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset(
        dataset_name, dataset_config_name, split="train", streaming=True
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
        bar = tqdm(total=subsample_size * valid_percent)

        while size < subsample_size * valid_percent:
            entry = next(dataset_iter)

            entry_size = len(entry["text"].encode("utf-8"))
            size += entry_size

            bar.update(entry_size)

            f.write(f"{json.dumps(entry)}\n")


if __name__ == "__main__":
    fire.Fire(prepare_dataset)
    sys.exit(0)
