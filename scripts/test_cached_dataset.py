"""
Test CachedDocumentDataset.
"""

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.layoutlm.cached_dataset import (
    CachedDocumentDataset,
)


def main():

    print("=" * 60)
    print("CACHED DATASET TEST")
    print("=" * 60)

    dataset = CachedDocumentDataset(
        "data/processed/train"
    )

    print(
        f"\nDataset size: {len(dataset)}"
    )

    print("\nLoading first sample...")

    sample = dataset[0]

    print("\n" + "=" * 60)
    print("SAMPLE")
    print("=" * 60)

    for key, value in sample.items():

        if hasattr(value, "shape"):
            print(
                f"{key:20} "
                f"shape={tuple(value.shape)} "
                f"dtype={value.dtype}"
            )
        else:
            print(
                f"{key:20} "
                f"{value}"
            )

    print("\n" + "=" * 60)
    print("CACHED DATASET TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()