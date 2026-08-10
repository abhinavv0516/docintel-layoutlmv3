"""
Test the DocumentDataset with one document.
"""

import sys
from pathlib import Path

# Add project root
sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.dataset import DocumentDataset


def main():

    print("=" * 60)
    print("DOCUMENT DATASET TEST")
    print("=" * 60)

    print("\nCreating dataset...")

    dataset = DocumentDataset(
        "data/train"
    )

    print(
        f"Dataset size: {len(dataset)}"
    )

    print("\nLoading first sample...")

    sample = dataset[0]

    print("\n" + "=" * 60)
    print("SAMPLE INFORMATION")
    print("=" * 60)

    for key, value in sample.items():

        print(
            f"{key:20} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype}"
        )

    print("\nLabel:")
    print(sample["labels"].item())

    print("\n" + "=" * 60)
    print("DOCUMENT DATASET TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()