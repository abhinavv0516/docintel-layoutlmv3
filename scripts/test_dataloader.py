"""
Test PyTorch DataLoader with DocumentDataset.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.dataset import DocumentDataset


def main():

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    print("\nCreating dataset...")

    dataset = DocumentDataset(
        "data/train"
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    print("\nCreating DataLoader...")

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    print("DataLoader created.")

    print("\nLoading one batch...")

    batch = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("BATCH INFORMATION")
    print("=" * 60)

    for key, value in batch.items():

        print(
            f"{key:20} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}"
        )

    print("\nLabels:")

    print(batch["labels"])

    print(
        "\nLabel values:",
        batch["labels"].tolist()
    )

    print("\n" + "=" * 60)
    print("DATALOADER TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()