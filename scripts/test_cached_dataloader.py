"""
Test DataLoader with cached LayoutLMv3 inputs.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.cached_dataset import (
    CachedDocumentDataset,
)


def main():

    print("=" * 60)
    print("CACHED DATALOADER + GPU TEST")
    print("=" * 60)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    dataset = CachedDocumentDataset(
        "data/processed/train"
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nLoading one batch...")

    batch = next(iter(dataloader))

    print("\n" + "=" * 60)
    print("CPU BATCH")
    print("=" * 60)

    for key, value in batch.items():

        if hasattr(value, "shape"):
            print(
                f"{key:20} "
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )

    # Move model inputs to GPU
    gpu_batch = {
        key: value.to(
            device,
            non_blocking=True,
        )
        if torch.is_tensor(value)
        else value
        for key, value in batch.items()
    }

    print("\n" + "=" * 60)
    print("GPU BATCH")
    print("=" * 60)

    for key, value in gpu_batch.items():

        if torch.is_tensor(value):

            print(
                f"{key:20} "
                f"device={value.device} "
                f"shape={tuple(value.shape)}"
            )

    print("\nLabels:")
    print(gpu_batch["labels"])

    print("\n" + "=" * 60)
    print("GPU BATCH TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()