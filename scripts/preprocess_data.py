from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


def build_balanced_subset(sample_fraction: float, seed: int):
    """Load BANKING77 and create a label-balanced subset.

    The subset size is approximately sample_fraction of the full dataset,
    while keeping the same number of examples for every label.
    """
    if not 0.10 <= sample_fraction <= 1.00:
        raise ValueError("sample_fraction must be between 0.10 and 1.00")

    dataset_dict = load_dataset("banking77")
    full_dataset = concatenate_datasets([dataset_dict["train"], dataset_dict["test"]])

    label_feature = full_dataset.features["label"]
    label_ids = list(range(label_feature.num_classes))

    indices_by_label: dict[int, list[int]] = {label_id: [] for label_id in label_ids}
    for idx, label in enumerate(full_dataset["label"]):
        indices_by_label[int(label)].append(idx)

    min_count = min(len(indices) for indices in indices_by_label.values())
    samples_per_label = max(1, int(round(min_count * sample_fraction)))

    rng = random.Random(seed)
    selected_indices: list[int] = []
    for label_id in label_ids:
        label_indices = indices_by_label[label_id]
        if len(label_indices) < samples_per_label:
            raise ValueError(
                f"Not enough samples in label {label_id} for balanced sampling. "
                f"Needed {samples_per_label}, found {len(label_indices)}."
            )
        selected_indices.extend(rng.sample(label_indices, samples_per_label))

    rng.shuffle(selected_indices)
    subset = full_dataset.select(selected_indices)

    # Add a human-readable label column before exporting.
    subset = subset.map(lambda x: {"label_text": label_feature.int2str(x["label"])})

    return subset


def split_and_save(subset, output_dir: Path, seed: int):
    """Split subset into train/test and save as CSV files."""
    split = subset.train_test_split(test_size=0.2, seed=seed, stratify_by_column="label")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    split["train"].to_csv(str(train_path), index=False)
    split["test"].to_csv(str(test_path), index=False)

    print(f"Saved train set to: {train_path}")
    print(f"Saved test set to: {test_path}")
    print(f"Train size: {len(split['train'])}")
    print(f"Test size: {len(split['test'])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create balanced BANKING77 sample data and export train/test CSV files."
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=1.0,
        help="Fraction in [0.10, 1.00] for balanced subset extraction (default: 1.00).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "sample_data"

    subset = build_balanced_subset(sample_fraction=args.sample_fraction, seed=args.seed)
    split_and_save(subset=subset, output_dir=output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
