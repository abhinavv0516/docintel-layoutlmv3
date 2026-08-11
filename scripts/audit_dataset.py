"""
Dataset leakage audit.

Checks for exact duplicate images across:
    train
    validation
    test

Uses SHA-256 hashes.
"""

import hashlib
from pathlib import Path


SPLITS = {
    "train": Path("data/clean/train"),
    "validation": Path("data/clean/validation"),
    "test": Path("data/clean/test"),
}

def file_hash(path):
    """Return SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


def collect_hashes(root):
    """Collect image hashes for a dataset split."""

    hashes = {}

    for path in root.rglob("*.png"):

        digest = file_hash(path)

        hashes.setdefault(
            digest,
            []
        ).append(path)

    return hashes


def main():

    print("=" * 60)
    print("DATASET LEAKAGE AUDIT")
    print("=" * 60)

    split_hashes = {}

    # --------------------------------------------------
    # Collect hashes
    # --------------------------------------------------

    for split_name, root in SPLITS.items():

        print(
            f"\nHashing {split_name}..."
        )

        hashes = collect_hashes(root)

        split_hashes[split_name] = hashes

        print(
            f"Images: {sum(len(v) for v in hashes.values())}"
        )

        print(
            f"Unique hashes: {len(hashes)}"
        )

    # --------------------------------------------------
    # Internal duplicates
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("INTERNAL DUPLICATES")
    print("=" * 60)

    for split_name, hashes in split_hashes.items():

        duplicates = {
            digest: paths
            for digest, paths in hashes.items()
            if len(paths) > 1
        }

        print(
            f"\n{split_name}: "
            f"{len(duplicates)} duplicate hashes"
        )

        for paths in list(
            duplicates.values()
        )[:5]:

            print("  Duplicate group:")

            for path in paths:
                print(f"    {path}")

    # --------------------------------------------------
    # Cross-split duplicates
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CROSS-SPLIT DUPLICATES")
    print("=" * 60)

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    total_cross_duplicates = 0

    for split_a, split_b in comparisons:

        hashes_a = split_hashes[split_a]
        hashes_b = split_hashes[split_b]

        common = (
            set(hashes_a.keys())
            & set(hashes_b.keys())
        )

        print(
            f"\n{split_a} ↔ {split_b}: "
            f"{len(common)} duplicate hashes"
        )

        total_cross_duplicates += len(
            common
        )

        for digest in list(common)[:10]:

            print("\n  Duplicate:")

            print(
                f"    {split_a}: "
                f"{hashes_a[digest][0]}"
            )

            print(
                f"    {split_b}: "
                f"{hashes_b[digest][0]}"
            )

    # --------------------------------------------------
    # Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIT RESULT")
    print("=" * 60)

    if total_cross_duplicates == 0:

        print(
            "\nNO EXACT CROSS-SPLIT DUPLICATES FOUND."
        )

        print(
            "The 96.77% test accuracy is not explained "
            "by exact image duplication."
        )

    else:

        print(
            f"\nWARNING: "
            f"{total_cross_duplicates} "
            f"cross-split duplicate groups found."
        )

        print(
            "The current test accuracy should NOT "
            "be treated as a clean generalization result."
        )


if __name__ == "__main__":
    main()